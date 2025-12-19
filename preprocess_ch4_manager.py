import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate
from config import Config
from utils.feature_extractor import FeatureExtractor
# [关键复用] 直接复用你验证过的瞬时转速计算函数
from preprocess_atomic import compute_rpm_from_square_wave


class CH4DualStreamProcessor:
    """
    [Chapter 4 最终版]
    特性：
    1. 滑动窗口切片 (Sliding Window) -> 保证样本量充足
    2. 瞬时转速计算 (Instantaneous RPM) -> 保证 PGFA 物理对齐
    3. 整合存储 (Merged Storage) -> 保证数据原子性
    """

    def __init__(self):
        self.original_fs = Config.SAMPLE_RATE  # 51200 Hz
        # 显微流降采样: 51200 / 50 = 1024 Hz
        self.decimation_factor = 50
        self.target_fs = self.original_fs / self.decimation_factor  # 1024 Hz

        self.fft_points = 1024  # 1秒的数据
        self.hanning_win = np.hanning(self.fft_points)

        # 滑动窗口参数 (与 config 保持一致或自定义)
        # 原始窗口长度: 1024 (显微点数) * 50 (倍数) = 51200 点 (即1秒)
        self.window_size_raw = self.fft_points * self.decimation_factor
        self.stride_raw = self.window_size_raw // 2  # 50% 重叠

        self.extractor = FeatureExtractor(Config)

        self.output_dir = os.path.join(Config.PROJECT_ROOT, "processed_data_ch4_dual_stream")
        os.makedirs(self.output_dir, exist_ok=True)

    def _read_raw_data(self, file_path):
        try:
            # 兼容复杂表头
            return pd.read_csv(file_path, sep='\t', encoding='gb18030',
                               header=None, skiprows=17, engine='python').values
        except:
            return None

    def _compute_fft(self, signal):
        """计算物理幅值谱 (归一化到真实物理量纲)"""
        # 信号长度对其
        if len(signal) < self.fft_points:
            signal = np.pad(signal, (0, self.fft_points - len(signal)))

        # 加窗 + FFT
        seg = signal[:self.fft_points] * self.hanning_win
        spec = np.abs(np.fft.rfft(seg))

        # 幅值修正: *2/N
        return spec[:self.fft_points // 2] * 2 / self.fft_points

    def process_all(self):
        print(f"--- [Chapter 4] Processing Start: Sliding Window + Instantaneous Speed ---")
        print(f"    Raw Window Size: {self.window_size_raw} points (1.0 sec)")

        tasks = []
        for domain, folder in Config.DATA_DOMAINS.items():
            path = os.path.join(Config.DATA_ROOT, folder)
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.endswith('.txt'): tasks.append((domain, path, f))

        for domain, d_path, fname in tqdm(tasks, desc="Processing"):
            parts = fname.replace('.txt', '').split('-')
            if len(parts) < 3: continue
            load_id, speed_id = parts[1], parts[2]

            # 输出文件名为 _dual.npy，里面包含该文件切分出的所有样本
            save_path = os.path.join(self.output_dir, f"{domain}_{load_id}_{speed_id}_dual.npy")
            # [重要] 如果存在则跳过，避免重复计算耗时
            if os.path.exists(save_path): continue

            raw_data = self._read_raw_data(os.path.join(d_path, fname))
            if raw_data is None: continue

            # 通道提取
            vib_raw_full = raw_data[:, Config.COL_INDICES_VIB[0]].astype(np.float32)
            spd_raw_full = raw_data[:, Config.COL_INDEX_SPEED].astype(np.float32)

            # 声纹通道 (假设第20列，需根据实际情况调整)
            audio_idx = Config.COL_INDICES_AUDIO[0] if Config.COL_INDICES_AUDIO else 20
            if audio_idx < raw_data.shape[1]:
                audio_raw_full = raw_data[:, audio_idx].astype(np.float32)
            else:
                audio_raw_full = np.zeros_like(vib_raw_full)

            # === 滑动窗口切片 ===
            samples_micro = []
            samples_macro = []
            samples_acoustic = []
            samples_speed = []

            total_len = len(vib_raw_full)
            # 物理负载 (Scalar)
            phys_load = float(load_id) * 100

            for start_idx in range(0, total_len - self.window_size_raw, self.stride_raw):
                end_idx = start_idx + self.window_size_raw

                # 1. 切片
                vib_win = vib_raw_full[start_idx:end_idx]
                spd_win = spd_raw_full[start_idx:end_idx]
                aud_win = audio_raw_full[start_idx:end_idx]

                # 2. 计算该窗口内的瞬时转速 (复用 atomic 逻辑)
                # 计算出 RPM，转换为 Hz
                rpm = compute_rpm_from_square_wave(spd_win, fs=self.original_fs)
                speed_hz = rpm / 60.0
                # 简单的异常过滤: 如果算出 0 (信号丢失)，用上一个或者 0
                if speed_hz == 0 and len(samples_speed) > 0:
                    speed_hz = samples_speed[-1]

                # 3. 处理 Micro 流 (降采样 -> FFT)
                vib_micro = decimate(vib_win, self.decimation_factor, ftype='fir')
                micro_fft = self._compute_fft(vib_micro)

                # 4. 处理 Macro 流 (直接 FFT，取低频部分或全频部分? 这里为了对齐，取前1024点FFT)
                # 注意：Macro流通常需要宽带，但输入维度要一致。
                # 策略：取原始信号的前 1024 点做 FFT (覆盖高频，但分辨率低)
                # 或者：对全长做 FFT 然后插值?
                # 这里采用最简单的：取窗口的前 1024 点原始数据 (Macro View)
                # 这样分辨率是 50Hz，带宽是 25.6kHz
                macro_fft = self._compute_fft(vib_win[:self.fft_points])

                # 5. 处理声纹 (MFCC)
                # 取均值作为该秒的声纹特征
                mfcc_seq = self.extractor.extract_audio_features(aud_win[:Config.FRAME_SIZE * 5])
                acoustic_feat = np.mean(mfcc_seq, axis=0)

                # 6. 收集
                samples_micro.append(micro_fft)
                samples_macro.append(macro_fft)
                samples_acoustic.append(acoustic_feat)
                samples_speed.append(speed_hz)

            # === 聚合保存 (Big File Strategy) ===
            # 将列表转换为 numpy 数组，极大减小文件体积和读取次数
            if len(samples_micro) > 0:
                save_data = {
                    'micro': np.array(samples_micro, dtype=np.float32),  # [N, 512]
                    'panorama': np.array(samples_macro, dtype=np.float32),  # [N, 512]
                    'acoustic': np.array(samples_acoustic, dtype=np.float32),  # [N, 26]
                    'speed': np.array(samples_speed, dtype=np.float32),  # [N]
                    'load': phys_load  # Scalar
                }
                np.save(save_path, save_data)


if __name__ == '__main__':
    processor = CH4DualStreamProcessor()
    processor.process_all()