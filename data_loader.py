# data_loader.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.stats import kurtosis
from torch.utils.data import Dataset
from scipy.interpolate import interp1d # 用于插值重构转速信号
from config import Config


class MotorDataset(Dataset):
    def __init__(self, flag='train', condition='HH'):
        """
        :param flag: 'train' | 'val'| 'test' 训练、验证、测试
        :param condition: 工况代码，如 'HH', 'FB'
        """
        self.flag = flag
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE

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
        self.data_x, self.data_speed = self._load_data_with_cache()

        # 3. 数据集划分逻辑
        # 如果是健康数据(HH)，我们需要切分成 Train/Val/Test
        # 如果是故障数据(FB, VU...)，全部作为 Test，不参与训练和验证

        # 定义切分比例 (6:2:2)
        total_len = len(self.data_x)
        train_split = int(total_len * 0.6)
        val_split = int(total_len * 0.8)  # 60% + 20%

        if condition == 'HH':
            if flag == 'train':
                self.data_x = self.data_x[:train_split]
                self.data_speed = self.data_speed[:train_split]
            elif flag == 'val':
                self.data_x = self.data_x[train_split:val_split]
                self.data_speed = self.data_speed[train_split:val_split]
            elif flag == 'test':
                self.data_x = self.data_x[val_split:]
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
                self.data_x = np.empty((0, len(Config.COL_INDICES_X)))
                self.data_speed = np.empty((0, 1))
                print(f"[Warning] 故障数据 {condition} 不应在 {flag} 阶段使用！")

        # 4. 计算样本数量
        if len(self.data_x) < self.window_size:
            self.n_samples = 0
            print(f"[Warning] 数据量不足一个窗口 ({len(self.data_x)} < {self.window_size})")
        else:
            self.n_samples = (len(self.data_x) - self.window_size) // self.stride + 1

        print(f"[{flag.upper()}] 工况:{condition} | 文件数:{len(self.file_paths)} | "
              f"样本数:{self.n_samples} | 信号维度:{self.data_x.shape}")

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
        # 1. 定位原始数据的切片位置
        # 注意：这里的 stride 应该是 HOP_LENGTH (帧移)，而不是原始窗口 STRIDE
        # 论文逻辑：输入是 H 个时间步的特征序列
        # 假设我们要在数据中取一段长 raw_window 的波形，算出 H 帧特征

        # 简化逻辑：我们预先切好一个大的 raw_window，然后在内部切帧计算特征
        # 但为了效率，建议离线处理好特征。
        # 这里为了不改动太大，我们实时计算（可能会慢，但逻辑正确）。

        idx = index * self.stride
        # 这里取出的长度应该是能够计算出 seq_len 个特征帧的长度
        # Length ≈ (seq_len - 1) * hop + frame_size
        raw_len = (self.window_size - 1) * Config.HOP_LENGTH + Config.FRAME_SIZE

        # 边界检查
        if idx + raw_len > len(self.data_x):
            # 简单的 padding 或截断处理
            return torch.zeros((self.window_size, Config.ENC_IN)), torch.zeros((2,))

        raw_x_win = self.data_x[idx: idx + raw_len]  # [Raw_Len, 5]
        s_win = self.data_speed[idx: idx + raw_len]  # [Raw_Len, 1]

        # === 2. 特征提取 (Feature Extraction) ===
        feature_list = []

        # A. 振动通道 (前4列: 8,10,11,12) -> 提取 RMS, Kurtosis
        # data_x 的列顺序对应 Config.COL_INDICES_X
        n_vib = len(Config.COL_INDICES_VIB)  # 4

        # A. 振动通道 (前4列) -> 提取 RMS, Kurtosis
        for i in range(n_vib):
            sig = raw_x_win[:, i]
            # 切帧: 利用 librosa.util.frame (很方便)
            frames = librosa.util.frame(sig, frame_length=Config.FRAME_SIZE, hop_length=Config.HOP_LENGTH)
            # frames shape: [Frame_Size, Seq_Len]

            # RMS
            rms = np.sqrt(np.mean(frames ** 2, axis=0))
            # Kurtosis
            kurt = kurtosis(frames, axis=0, fisher=False)  # fisher=False对应Pearson定义(常态为3)

            feature_list.append(rms)
            feature_list.append(kurt)

        # B. 声纹通道 (第5列) -> 提取 MFCC
        audio_sig = raw_x_win[:, -1]  # 假设最后一列是 Audio
        # librosa 算 MFCC
        # n_mfcc=14 (取14个，丢第0个剩13个)
        mfcc = librosa.feature.mfcc(y=audio_sig, sr=Config.SAMPLE_RATE,
                                    n_mfcc=Config.N_MFCC + 1,
                                    n_fft=Config.FRAME_SIZE,
                                    hop_length=Config.HOP_LENGTH,
                                    center=False)
        # mfcc shape: [14, Seq_Len]
        # 丢弃第0阶 (能量)
        mfcc = mfcc[1:, :]

        feature_list.append(mfcc)  # [13, Seq_Len]

        # 堆叠所有特征 -> [Seq_Len, D_total]
        # list 中现在有: RMS(L), Kurt(L), ..., MFCC(13, L)
        # 需要统一转置并拼接
        processed_feats = []
        for f in feature_list:
            if f.ndim == 1:
                processed_feats.append(f[:, np.newaxis])  # [L, 1]
            else:
                processed_feats.append(f.T)  # [L, 13]

        x_features = np.concatenate(processed_feats, axis=1)  # [L, D]

        # # 截断或填充到标准的 window_size (防止计算误差导致的 ±1 帧)
        # if x_features.shape[0] > self.window_size:
        #     x_features = x_features[:self.window_size, :]
        # elif x_features.shape[0] < self.window_size:
        #     # Padding
        #     pad_len = self.window_size - x_features.shape[0]
        #     x_features = np.pad(x_features, ((0, pad_len), (0, 0)), mode='edge')

        # === 3. 协变量 (Speed) ===
        # 转速也需要这算到对应的帧上，或者取全局均值
        # 论文里是：每个窗口一个 Speed 向量 [v, v^2] (Scalar)
        # 你的 RDLinear 设计是输入 [B, 2]，所以这里保持不变
        v_bar = np.mean(s_win)
        v2_bar = np.mean(s_win ** 2)
        norm_scale = 3000.0
        covariates = np.array([v_bar / norm_scale, v2_bar / (norm_scale ** 2)], dtype=np.float32)

        # [修改] 打印更显眼的标记，证明是最新代码
        print(f"DEBUG [NEW]: Feature Shape: {x_features.shape}")

        # [新增] 强行断言：如果维度不对，直接在这里炸掉，不要传给模型
        assert x_features.shape[
                   1] == 21, f"Fatal Error: Data Loader is generating {x_features.shape[1]} channels instead of 21!"

        print(f"DEBUG: Output Feature Shape: {x_features.shape}")
        return torch.from_numpy(x_features).float(), torch.from_numpy(covariates).float()

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