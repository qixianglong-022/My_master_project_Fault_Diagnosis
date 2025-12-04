# data_loader.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import pickle  # 用于保存 scaler
from scipy.stats import kurtosis
from torch.utils.data import Dataset
from scipy.interpolate import interp1d # 用于插值重构转速信号
from config import Config
from utils.feature_extractor import FeatureExtractor

class MotorDataset(Dataset):
    def __init__(self, flag='train', condition='HH'):
        """
        :param flag: 'train' | 'val'| 'test' 训练、验证、测试
        :param condition: 工况代码，如 'HH', 'FB'
        """
        self.flag = flag
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE
        self.extractor = FeatureExtractor(Config)

        # 1. 获取文件路径
        self.file_paths = self._get_file_paths(condition)

        # [逻辑修正] 即使设置了 LIMIT_FILES，也要尽量保证覆盖恒速和变速
        # 这里简单处理：如果 LIMIT_FILES 生效，截取前 N 个
        if Config.LIMIT_FILES is not None:
            self.file_paths = self.file_paths[:Config.LIMIT_FILES]
            print(f"[Info] 限制加载文件数: {len(self.file_paths)} (Config.LIMIT_FILES={Config.LIMIT_FILES})")

        # 2. 加载数据
        # self.data_x: [Total_Points, N_Channels]
        # self.data_speed: [Total_Points, 1]
        self.raw_x, self.raw_speed = self._load_data_with_cache()


        # 3. 预计算特征 (Offline Feature Extraction) ===
        # 将耗时的特征提取移到这里，训练时 __getitem__ 就会飞快
        print(f"[{flag.upper()}] Pre-computing features (this may take a while)...")
        self.data_features = self._precompute_features(self.raw_x)
        self.data_speed = self._precompute_speed(self.raw_speed)  # 确保对齐

        # 释放原始数据内存 (如果显存/内存紧张)
        del self.raw_x

        # === [核心优化] 3. Z-Score 标准化 ===
        self._apply_z_score_normalization()

        # 3. 数据集划分逻辑
        # 如果是健康数据(HH)，我们需要切分成 Train/Val/Test
        # 如果是故障数据(FB, VU...)，全部作为 Test，不参与训练和验证

        # 定义切分比例 (6:2:2)
        total_len = len(self.data_features)
        train_split = int(total_len * 0.6)
        val_split = int(total_len * 0.8)  # 60% + 20%

        if condition == 'HH':
            if flag == 'train':
                self.data_features = self.data_features[:train_split]
                self.data_speed = self.data_speed[:train_split]
                self.stride = Config.TRAIN_STRIDE_FRAMES
            elif flag == 'val':
                self.data_features = self.data_features[train_split:val_split]
                self.data_speed = self.data_speed[train_split:val_split]
            elif flag == 'test':
                self.data_features = self.data_features[val_split:]
                self.data_speed = self.data_speed[val_split:]
            else:
                raise ValueError(f"Unknown flag: {flag}")
        else:
            # 故障工况：无论 flag 是什么，都只应该在测试阶段使用
            # 为了防止误用，我们强制只在 flag='test' 时返回数据
            if flag == 'test':
                pass
            else:
                # 如果你在 train/val 阶段请求故障数据，说明代码写错了
                # 但为了代码兼容性，我们可以返回空，或者抛出警告
                # 这里为了严谨，我们让它返回空，防止故障数据泄露进训练流程
                self.data_features = np.empty((0, len(Config.COL_INDICES_X)))
                self.data_speed = np.empty((0, 1))
                print(f"[Warning] 故障数据 {condition} 不应在 {flag} 阶段使用！")

        # 4. 计算样本数量
        if len(self.data_features) < self.window_size:
            self.n_samples = 0
            print(f"[Warning] 数据量不足一个窗口 ({len(self.data_features)} < {self.window_size})")
        else:
            self.n_samples = (len(self.data_features) - self.window_size) // self.stride + 1

        print(f"[{flag.upper()}] 工况:{condition} | 文件数:{len(self.file_paths)} | "
              f"样本数:{self.n_samples} | 信号维度:{self.data_features.shape}")

        # 5. 初始化特征提取器
        self.extractor = FeatureExtractor(Config)

    def _precompute_features(self, raw_data):
        """一次性提取所有特征"""
        # 这里可以使用并行加速，或者简单的循环
        # 为了简单，我们复用之前的逻辑，但应用到整段数据上
        # 注意：FeatureExtractor 需要支持长序列处理，或者你需要分块处理
        # 建议简单的分块处理以防内存溢出

        # 简化版：复用 extractor 的逻辑
        # 假设 raw_data 是 [Total, C]
        features = []

        # A. 振动
        for i in range(len(Config.COL_INDICES_VIB)):
            f = self.extractor.extract_vib_features(raw_data[:, i])  # [Total_Frames, 2]
            features.append(f)

        # B. 声纹
        f_audio = self.extractor.extract_audio_features(raw_data[:, -1])  # [Total_Frames, 13]
        features.append(f_audio)

        # 合并
        all_feats = np.concatenate(features, axis=1)  # [Total_Frames, 21]

        # 对齐转速 (转速需要降采样到 Frame 级别)
        # 简单的做法是取每帧对应的转速均值
        # 这里需要在 __init__ 里同步处理 self.raw_speed

        return all_feats

    def _precompute_speed(self, raw_speed):
        """
        将原始高频转速点降采样对齐到特征帧
        raw_speed: [Total_Points]
        return: [Total_Frames]
        """
        # 计算帧数
        n_frames = (len(raw_speed) - Config.FRAME_SIZE) // Config.HOP_LENGTH + 1

        # 使用 stride tricks 快速切帧
        strides = raw_speed.strides
        shape = (n_frames, Config.FRAME_SIZE)
        strides = (strides[0] * Config.HOP_LENGTH, strides[0])

        # [N_Frames, Frame_Size]
        frames = np.lib.stride_tricks.as_strided(raw_speed, shape=shape, strides=strides)

        # 计算每一帧的平均转速
        speed_frames = np.mean(frames, axis=1)
        return speed_frames

    def _apply_z_score_normalization(self):
        scaler_path = os.path.join(Config.OUTPUT_DIR, 'scaler_params.pkl')

        if self.flag == 'train':
            # 训练集：计算均值和方差并保存
            print("Calculating Z-Score statistics...")
            self.mean = np.mean(self.data_features, axis=0)
            self.std = np.std(self.data_features, axis=0) + 1e-6  # 防止除零

            # 保存参数供 val/test/deploy 使用
            with open(scaler_path, 'wb') as f:
                pickle.dump({'mean': self.mean, 'std': self.std}, f)
            print(f"Scaler saved to {scaler_path}")

        else:
            # 验证/测试集：加载训练集的参数
            if not os.path.exists(scaler_path):
                raise FileNotFoundError("请先运行 flag='train' 生成 Scaler 参数！")

            with open(scaler_path, 'rb') as f:
                params = pickle.load(f)
            self.mean = params['mean']
            self.std = params['std']

        # 应用标准化 (In-place 修改以节省内存)
        self.data_features = (self.data_features - self.mean) / self.std
        print("Z-Score normalization applied.")

    def _get_file_paths(self, condition):
        folder_name = Config.DATA_DOMAINS.get(condition)
        if not folder_name:
            raise ValueError(f"Config.DATA_DOMAINS 中未定义 {condition}")

        target_dir = os.path.join(Config.DATA_ROOT, folder_name)
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"目录不存在: {target_dir}")

        # 匹配所有 .txt
        files = sorted(glob.glob(os.path.join(target_dir, "*.txt")))

        if len(files) == 0:
            raise FileNotFoundError(f"在 {target_dir} 下未找到 .txt 文件")

        return files

    def _process_speed_signal(self, raw_voltage):
        """
        将方波电压信号转换为连续的 RPM 转速曲线
        原理：寻找上升沿 -> 计算脉冲间隔 -> 转换为 RPM -> 插值
        """
        # 1. 二值化 (阈值取 2V)
        # raw_voltage shape: [N, 1] -> flatten to [N]
        sig = raw_voltage.flatten()
        binary_sig = (sig > 2.0).astype(int)

        # 2. 寻找上升沿 (从0变1的位置)
        # diff: 1为上升沿, -1为下降沿, 0为不变
        diff_sig = np.diff(binary_sig)
        # 找到所有上升沿的索引
        falling_edges = np.where(diff_sig == 1)[0]

        if len(falling_edges) < 2:
            print("[Warning] 转速信号脉冲过少，无法计算 RPM，返回默认值 0")
            return np.zeros_like(raw_voltage)

        # 3. 计算相邻脉冲的间隔 (采样点数)
        # pulses_intervals[i] 对应 rising_edges[i] 到 rising_edges[i+1] 之间的转速
        period_points = np.diff(falling_edges)

        # 4. 计算瞬时 RPM
        # RPM = 60 / T = 60 / (points / fs) = 60 * fs / points
        fs = Config.SAMPLE_RATE
        rpms = 60.0 * fs / period_points

        # 5. 插值还原为连续曲线
        # 我们知道在 rising_edges[0] 到 rising_edges[1] 这段时间内，平均转速是 rpms[0]
        # 为了平滑，我们将 RPM 值赋给区间的中心点，然后线性插值

        # 时间轴 (以采样点为单位)
        # 取区间中点作为该转速的时间锚点
        t_anchors = (falling_edges[:-1] + falling_edges[1:]) / 2

        # 使用线性插值填补所有时间点
        all_indices = np.arange(len(sig))
        f_interp = interp1d(t_anchors, rpms, kind='linear', fill_value="extrapolate")
        speed_curve = f_interp(all_indices)

        # 6. 平滑处理 (可选，去除毛刺)
        # 移动平均滤波，窗口大小取 0.1秒 (约5000点)
        # window_len = int(Config.SAMPLE_RATE * 0.1)
        # if window_len > 1:
        #     w = np.ones(window_len)/window_len
        #     speed_curve = np.convolve(speed_curve, w, mode='same')

        return speed_curve.reshape(-1, 1).astype(np.float32)

    def _parse_txt_file(self, file_path):
        """解析带非结构化 Header 的 txt 文件 (GB18030 编码)"""
        print(f"Parsing: {os.path.basename(file_path)} ...")

        # 1. 寻找数据起始行 "Time (seconds)"
        header_rows = 0
        try:
            with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
                for i, line in enumerate(f):
                    if "Time (seconds)" in line:
                        header_rows = i + 1
                        break
        except UnicodeDecodeError:
            # 兜底尝试 gbk
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                for i, line in enumerate(f):
                    if "Time (seconds)" in line:
                        header_rows = i + 1
                        break

        if header_rows == 0:
            raise ValueError(f"文件 {os.path.basename(file_path)} 未找到 'Time (seconds)' 标记")

        # 2. 读取数据 (使用 C 引擎加速)
        # 目标列 = Speed(1) + Vibs[...] + Audio[...]
        # 注意：pandas read_csv 的 usecols 如果是非排序的，可能会改变列顺序
        # 所以我们需要显式记录我们要读取的列索引
        target_col_indices = [Config.COL_INDEX_SPEED] + Config.COL_INDICES_VIB + Config.COL_INDICES_AUDIO

        try:
            df = pd.read_csv(
                file_path,
                sep='\t',
                skiprows=header_rows,
                usecols=target_col_indices,
                header=None,
                engine='c',
                dtype=np.float32,  # 强转 float32 省内存
                encoding='gb18030'
            )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None

        # 3. 拆分数据
        # df 的列现在的顺序是按照文件中的原始顺序排列的 (因为 usecols 只是筛选)
        # 我们需要根据原始索引重新组织数据：Speed在前，X在后

        # 获取 df 中各列对应的原始索引 (Pandas read_csv header=None 时，columns 就是 0,1,2...)
        # 但这里我们用了 usecols，df.columns 还是会变成 0, 1, 2... 丢失了原始含义
        # 修正：read_csv 的 usecols 行为：如果传入列表，df 列顺序可能不确定，但在 C 引擎下通常按文件顺序。
        # 最稳妥方式：全部读进来再切片太慢。
        # 实际上：df.values 的列顺序 = sorted(target_col_indices) 对应的列。

        # 让我们做一个更稳妥的映射：
        # 我们知道 target_col_indices 里的值。
        # 假设 target_col_indices = [1, 8, 10, 11, 12, 20] -> 排序后 [1, 8, 10, 11, 12, 20]
        # Speed 是 1，在第0列。
        # Vibs 是 8,10,11,12，在第1,2,3,4列。
        # Audio 是 20，在第5列。

        raw_data = df.values

        # 找到 Speed 在 raw_data 中的位置
        # 由于我们传入的 target_col_indices 不一定是排序的，我们先排序它来模拟 Pandas 的行为
        sorted_indices = sorted(target_col_indices)

        speed_idx = sorted_indices.index(Config.COL_INDEX_SPEED)
        raw_speed_volts = raw_data[:, speed_idx:speed_idx + 1]

        # 找到 X 在 raw_data 中的位置
        # Config.COL_INDICES_X = Vibs + Audio
        x_indices = []
        for original_idx in Config.COL_INDICES_X:
            if original_idx in sorted_indices:
                x_indices.append(sorted_indices.index(original_idx))

        x = raw_data[:, x_indices]

        return x, raw_speed_volts

    def _load_data_with_cache(self):
        """带缓存的加载逻辑"""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        list_x = []
        list_s = []

        for fp in self.file_paths:
            # 定义缓存文件名 (Hash文件名以防过长? 这里直接用文件名简单点)
            fname = os.path.basename(fp).replace('.txt', '')
            cache_x = os.path.join(Config.CACHE_DIR, f"{fname}_X.npy")
            # 缓存名改为 _RPM 以示区别
            cache_s = os.path.join(Config.CACHE_DIR, f"{fname}_RPM_FallingEdge.npy")

            if os.path.exists(cache_x) and os.path.exists(cache_s):
                # print(f"Loading cache: {fname}") # 减少刷屏
                x = np.load(cache_x)
                s = np.load(cache_s)
            else:
                x, raw_s_volts = self._parse_txt_file(fp)
                if x is None: continue

                # 解析转速
                print(f"Converting Square Wave to RPM for {fname}...")
                s = self._process_speed_signal(raw_s_volts)

                np.save(cache_x, x)
                np.save(cache_s, s)

            list_x.append(x)
            list_s.append(s)

        if not list_x:
            # 返回空数组防止报错
            return np.empty((0, len(Config.COL_INDICES_X))), np.empty((0, 1))

        return np.concatenate(list_x, axis=0), np.concatenate(list_s, axis=0)

    def __getitem__(self, index):
        """
        从预计算好的特征序列中切片
        index: 滑动窗口的起始帧索引 (Frame Index)
        """
        # 1. 确定切片范围
        # self.window_size 是帧数 (例如 50)
        # 注意：这里的 index 直接对应帧，不需要乘 HOP_LENGTH
        start_idx = index
        end_idx = index + self.window_size

        # 2. 获取特征窗口 (Features)
        # self.data_features 已经是 (Total_Frames, 21) 且做过 Z-Score
        x_win = self.data_features[start_idx: end_idx]  # [50, 21]

        # 3. 获取转速协变量 (Covariates)
        # 假设 self.data_speed 也是帧对齐的 [Total_Frames]
        s_win_frames = self.data_speed[start_idx: end_idx]

        # 计算该窗口的宏观工况 (取帧级别转速的平均)
        v_bar = np.mean(s_win_frames)
        v2_bar = np.mean(s_win_frames ** 2)  # E[v^2]

        # 归一化 (与 deploy 保持一致)
        norm_scale = 3000.0
        covariates = np.array([v_bar / norm_scale, v2_bar / (norm_scale ** 2)], dtype=np.float32)

        # 4. 格式转换
        # x_win 可能因为切片导致内存不连续，torch 可能会报警，用 copy() 确保安全
        return torch.from_numpy(x_win.copy()).float(), torch.from_numpy(covariates).float()

    def __len__(self):
        return self.n_samples


# ==========================================
# 调试与可视化 (Debug Visualization)
# ==========================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 解决中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print(">>> 正在进行 [波形转换与全通道] 测试...")

    try:
        # 1. 实例化 Dataset 以获取配置和方法
        # 临时设为 None 以便读取一个完整文件进行演示，不受切分影响
        ds = MotorDataset(flag='test', condition='HH')

        if len(ds.file_paths) > 0:
            # 2. 选取一个文件进行“解剖”演示
            # 我们不直接用 ds[i]，因为 ds[i] 里的转速已经是 RPM 了，看不到原始方波
            # 我们手动调用解析函数来获取原始数据
            demo_file = ds.file_paths[0]  # 取第一个文件
            print(f"\n正在深度分析文件: {os.path.basename(demo_file)}")

            # 读取原始数据 (Raw Data)
            # raw_x: [Total, Channels], raw_speed_volts: [Total, 1] (方波)
            raw_x, raw_speed_volts = ds._parse_txt_file(demo_file)

            # 计算转速曲线 (RPM Curve)
            rpm_curve = ds._process_speed_signal(raw_speed_volts)

            # 3. 截取一段典型区间进行绘图 (例如 1.0秒 ~ 1.2秒，共 0.2s)
            # 0.2s 足够看清方波的脉冲和振动的波形
            start_sec = 1.0
            plot_duration = 5.5

            fs = Config.SAMPLE_RATE
            start_idx = int(start_sec * fs)
            end_idx = int((start_sec + plot_duration) * fs)

            # 截取数据
            seg_volts = raw_speed_volts[start_idx:end_idx].flatten()
            seg_rpm = rpm_curve[start_idx:end_idx].flatten()
            seg_x = raw_x[start_idx:end_idx]

            time_axis = np.linspace(start_sec, start_sec + plot_duration, len(seg_volts))

            # 4. 动态绘图
            # 子图数量 = 1(原始方波) + 1(转换后RPM) + N(振动/声纹通道)
            n_channels = seg_x.shape[1]
            n_subplots = 2 + n_channels

            fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 2.5 * n_subplots), sharex=True)

            # 4.1 原始转速脉冲 (方波)
            ax1 = axes[0]
            ax1.plot(time_axis, seg_volts, color='green', linewidth=1.5, label='原始脉冲 (Volts)')
            ax1.set_title(f"1. 原始转速传感器信号 (方波) - {os.path.basename(demo_file)}")
            ax1.set_ylabel("电压 (V)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')

            # 4.2 转换后的 RPM
            ax2 = axes[1]
            ax2.plot(time_axis, seg_rpm, color='red', linewidth=2.0, label='计算转速 (RPM)')
            ax2.set_title(f"2. 转换后的转速曲线 (Mean: {np.mean(seg_rpm):.1f} r/min)")
            ax2.set_ylabel("转速 (RPM)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')

            # 4.3 所有监测通道
            channel_names = Config.COL_INDICES_X
            # 简单的名称映射，方便你看图
            name_map = {
                8: "电机振动", 10: "轴承X", 11: "轴承Y", 12: "轴承Z",
                18: "噪声2-3", 20: "噪声1-1"
            }

            for i in range(n_channels):
                ax = axes[i + 2]
                col_idx = channel_names[i]
                name = name_map.get(col_idx, f"Ch-{col_idx}")

                ax.plot(time_axis, seg_x[:, i], linewidth=0.8, color='blue')
                ax.set_title(f"{3 + i}. 监测通道: {name} (Idx: {col_idx})")
                ax.set_ylabel("幅值")
                ax.grid(True, alpha=0.3)

            plt.xlabel("Time (seconds)")
            plt.tight_layout()

            save_path = "debug_full_visualization.png"
            plt.savefig(save_path, dpi=150)
            print(f"\n[Success] 全通道可视化图已保存至: {os.path.abspath(save_path)}")
            print("请检查图片：")
            print("1. 第一行绿色方波是否清晰？(0V-4V跳变)")
            print("2. 第二行红色RPM是否稳定？(恒速应近似直线，略有波动是正常的计算误差)")
            print("3. 下面各通道波形是否正常？")

        else:
            print("[Warning] 未找到文件，请检查 Config.DATA_ROOT")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n[Error] {e}")