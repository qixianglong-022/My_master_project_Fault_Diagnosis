import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from config import Config
from utils.feature_extractor import FeatureExtractor


class AtomicPreprocessor:
    def __init__(self):
        self.extractor = FeatureExtractor(Config)
        self.save_dir = "./processed_data_atomic"
        os.makedirs(self.save_dir, exist_ok=True)

    def process_speed(self, raw_voltage):
        """
        [物理核心] 将方波电压信号转换为连续的 RPM 转速曲线
        """
        # 1. 扁平化处理
        sig = raw_voltage.flatten()

        # 2. 二值化 (阈值取 2V，兼容 3.3V/5V TTL电平)
        binary_sig = (sig > 2.0).astype(int)

        # 3. 寻找上升沿 (diff=1 的位置)
        diff_sig = np.diff(binary_sig)
        rising_edges = np.where(diff_sig == 1)[0]

        # 异常处理：如果脉冲太少，无法计算转速，返回全0
        if len(rising_edges) < 2:
            # print("[Warning] 脉冲过少，无法计算 RPM")
            return np.zeros_like(sig, dtype=np.float32)

        # 4. 计算脉冲间隔 (采样点数)
        # 工业电机通常一转一个脉冲 (键相)，如果是一转 N 脉冲，需除以 N
        period_points = np.diff(rising_edges)

        # RPM = 60 / T = 60 / (points / fs) = 60 * fs / points
        fs = Config.SAMPLE_RATE
        rpms = 60.0 * fs / period_points

        # 5. 插值还原为连续曲线 (与原始信号等长)
        # 将计算出的 RPM 值赋给两个上升沿的中间时刻
        t_anchors = (rising_edges[:-1] + rising_edges[1:]) / 2

        # 线性插值填补所有时间点
        all_indices = np.arange(len(sig))

        # fill_value="extrapolate" 保证首尾两端也有数据
        f_interp = interp1d(t_anchors, rpms, kind='linear', fill_value="extrapolate")
        speed_curve = f_interp(all_indices)

        # 6. 简单的平滑 (可选，去除插值带来的尖峰)
        # window_len = 100
        # w = np.ones(window_len)/window_len
        # speed_curve = np.convolve(speed_curve, w, mode='same')

        return speed_curve.astype(np.float32)

    def _parse_filename(self, filename):
        """
        解析文件名 HH-2-1.txt -> Condition=HH, Load=2, Speed=1
        """
        name = os.path.basename(filename).replace('.txt', '')
        parts = name.split('-')
        if len(parts) < 3: return None

        cond = parts[0]  # HH or FB
        load = parts[1]  # 0, 2, 4
        speed = parts[2]  # 1~8
        return cond, load, speed

    def run(self):
        # 遍历所有可能的文件夹
        # 假设 Config.DATA_DOMAINS 包含了 HH 和 FB 的文件夹路径映射
        # 如果没有，我们需要手动指定一下根目录下的文件夹
        # 这里假设你的 DATA_ROOT 下直接有 'HH_Folder' 和 'FB_Folder'
        # 或者我们直接 glob 整个 DATA_ROOT 下的所有 txt 递归查找

        print(f">>> Scanning {Config.DATA_ROOT} ...")
        all_files = glob.glob(os.path.join(Config.DATA_ROOT, "**", "*.txt"), recursive=True)

        for fp in all_files:
            meta = self._parse_filename(fp)
            if not meta: continue

            cond, load, speed = meta
            # 过滤掉非目标文件
            if cond not in ['HH', 'FB']: continue

            print(f"Processing: {cond} | Load {load} | Speed {speed} ...")

            try:
                # 1. 读取数据 (复用之前的 _parse_txt_file 逻辑)
                # 这里简写，请确保你之前的 _parse_txt_file 能够适配编码
                raw_x, raw_s_volts = self._read_txt(fp)

                # 2. 处理转速
                rpm_curve = self.process_speed(raw_s_volts)

                # 3. 提取特征 (Frame-Level)
                vib_feats = []
                for i in range(len(Config.COL_INDICES_VIB)):
                    f = self.extractor.extract_vib_features(raw_x[:, i])
                    vib_feats.append(f)
                f_audio = self.extractor.extract_audio_features(raw_x[:, -1])

                # [N_frames, 21]
                feat_mat = np.concatenate(vib_feats + [f_audio], axis=1)

                # 4. 转速对齐 (Downsample)
                n_frames = feat_mat.shape[0]
                frame_size, hop = Config.FRAME_SIZE, Config.HOP_LENGTH
                req_len = (n_frames - 1) * hop + frame_size

                rpm_valid = rpm_curve[:req_len]
                strides = rpm_valid.strides
                shape = (n_frames, frame_size)
                strides_new = (strides[0] * hop, strides[0])
                rpm_frames = np.lib.stride_tricks.as_strided(rpm_valid, shape=shape, strides=strides_new)
                speed_mat = np.mean(rpm_frames, axis=1).reshape(-1, 1)

                # 5. 保存原子文件
                # 命名格式: {COND}_{LOAD}_{SPEED}.npy
                base_name = f"{cond}_{load}_{speed}"
                np.save(os.path.join(self.save_dir, f"{base_name}.npy"), feat_mat)
                np.save(os.path.join(self.save_dir, f"{base_name}_S.npy"), speed_mat)

            except Exception as e:
                print(f"[Error] Failed {fp}: {e}")


    def _read_txt(self, file_path):
        """
        [辅助] 读取原始 TXT，剥离 Header (自动适配编码)
        """
        # 1. 寻找 Header 行数并确定编码
        header_rows = 0
        # 优先尝试 gb18030 (最全的中文编码)，其次 gbk，最后 utf-8
        encodings = ['gb18030', 'gbk', 'utf-8']
        detected_encoding = None

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    for i, line in enumerate(f):
                        if "Time (seconds)" in line:
                            header_rows = i + 1
                            detected_encoding = enc  # 记录成功的编码
                            break
                if header_rows > 0: break
            except:
                continue

        if header_rows == 0:
            raise ValueError(f"No 'Time (seconds)' found in {os.path.basename(file_path)}")

        # 2. 读取数据 (显式传入 encoding)
        target_cols = [Config.COL_INDEX_SPEED] + Config.COL_INDICES_VIB + Config.COL_INDICES_AUDIO

        try:
            # [Fix] 这里的 encoding=detected_encoding 是关键！
            df = pd.read_csv(
                file_path, sep='\t', skiprows=header_rows, header=None,
                usecols=target_cols, engine='c', dtype=np.float32,
                encoding=detected_encoding
            )
        except UnicodeDecodeError:
            # 如果自动侦测失败，强制尝试 gb18030
            df = pd.read_csv(
                file_path, sep='\t', skiprows=header_rows, header=None,
                usecols=target_cols, engine='c', dtype=np.float32,
                encoding='gb18030'
            )

        raw_data = df.values

        # 3. 拆分 Speed 和 Sensors
        # 排序以确保索引对应正确
        sorted_indices = sorted(target_cols)
        speed_col_idx = sorted_indices.index(Config.COL_INDEX_SPEED)

        raw_s = raw_data[:, speed_col_idx]  # 转速电压

        # 创建 mask 去掉 speed 列
        sensor_mask = np.ones(raw_data.shape[1], dtype=bool)
        sensor_mask[speed_col_idx] = False
        raw_x = raw_data[:, sensor_mask]

        return raw_x, raw_s

if __name__ == "__main__":
    p = AtomicPreprocessor()
    p.run()