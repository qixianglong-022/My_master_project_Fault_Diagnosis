import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate
from config import Ch4Config, Config
from utils.feature_extractor import FeatureExtractor


def robust_compute_frequency_hz(signal, fs):
    """
    [物理层修正] 基于迟滞比较与中值滤波的鲁棒转速频率计算
    :param signal: 转速脉冲信号 (电压值)
    :param fs: 采样率
    :return: 基础转频 (Hz)
    """
    # 1. 消除高频噪声 (简单的移动平均)
    signal = np.convolve(signal, np.ones(5) / 5, mode='same')

    # 2. 动态阈值 (防止电压漂移)
    _min, _max = np.min(signal), np.max(signal)
    if _max - _min < 0.5: return 0.0  # 信号太弱，视为静止

    mid = (_max + _min) / 2
    high_th = mid + 0.2 * (_max - mid)
    low_th = mid - 0.2 * (_max - mid)

    # 3. 迟滞比较器提取上升沿
    state = 0  # 0: low, 1: high
    rising_indices = []
    for i, val in enumerate(signal):
        if state == 0 and val > high_th:
            state = 1
            rising_indices.append(i)
        elif state == 1 and val < low_th:
            state = 0

    if len(rising_indices) < 2: return 0.0

    # 4. 计算间隔并剔除异常值 (中值滤波)
    intervals = np.diff(rising_indices)
    median_val = np.median(intervals)
    # 允许 30% 的抖动，剔除突变脉冲
    valid_intervals = intervals[np.abs(intervals - median_val) < 0.3 * median_val]

    if len(valid_intervals) == 0:
        avg_interval = median_val
    else:
        avg_interval = np.mean(valid_intervals)

    # 5. 转 Hz (fs / points)
    return fs / avg_interval


class CH4DualStreamProcessor:
    def __init__(self):
        self.original_fs = Config.SAMPLE_RATE  # 51200
        self.decimation = 50
        self.target_fs = self.original_fs / self.decimation  # 1024 Hz
        self.fft_points = 1024

        # 汉宁窗 (能量校正系数)
        self.hanning = np.hanning(self.fft_points)
        self.correction = 1.63

        self.win_size_raw = self.fft_points * self.decimation  # 51200
        self.stride = self.win_size_raw  # 无重叠，保证样本纯净性

        self.extractor = FeatureExtractor(Config)
        self.out_dir = Ch4Config.DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

    def _compute_fft(self, sig):
        # 截断或补零
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))
        sig = sig[:self.fft_points] * self.hanning

        # FFT 并取单边谱
        spec = np.abs(np.fft.rfft(sig)) / (self.fft_points / 2)
        spec = spec * self.correction  # 恢复能量
        return spec[:self.fft_points // 2]  # 512点

    def process_all(self):
        print(f">>> [Preprocess] Mode: Dual-Stream Physics (Decimation={self.decimation})")

        # 扫描所有相关文件
        tasks = []
        for dom, folder in Config.DATA_DOMAINS.items():
            path = os.path.join(Config.DATA_ROOT, folder)
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.endswith('.txt'): tasks.append((dom, path, f))

        for dom, d_path, fname in tqdm(tasks):
            # Parse Name: HH-2-1.txt -> Load=2, Speed=1
            try:
                parts = fname.replace('.txt', '').split('-')
                load_id, speed_id = parts[1], parts[2]
            except:
                continue

            save_name = f"{dom}_{load_id}_{speed_id}_dual.npy"
            save_path = os.path.join(self.out_dir, save_name)
            if os.path.exists(save_path): continue

            # 读取
            try:
                raw = pd.read_csv(os.path.join(d_path, fname), sep='\t', skiprows=17, header=None, engine='c').values
            except:
                continue

            # 通道获取
            vib = raw[:, Config.COL_INDICES_VIB[0]]
            spd_signal = raw[:, Config.COL_INDEX_SPEED]
            aud_idx = Config.COL_INDICES_AUDIO[0]
            aud = raw[:, aud_idx] if aud_idx < raw.shape[1] else np.zeros_like(vib)

            samples = {'micro': [], 'macro': [], 'acoustic': [], 'speed': []}

            # 滑动窗口切片
            N = len(vib)
            for i in range(0, N - self.win_size_raw + 1, self.stride):
                seg_vib = vib[i: i + self.win_size_raw]
                seg_spd = spd_signal[i: i + self.win_size_raw]
                seg_aud = aud[i: i + self.win_size_raw]

                # 1. 计算物理转速 (Hz) - 这是 PGFA 的核心输入
                hz = robust_compute_frequency_hz(seg_spd, self.original_fs)
                if hz < 1.0: hz = 15.0  # 兜底防止除0，假设最低转速

                # 2. Micro Stream (低频显微: 降采样 -> FFT)
                # 使用 scipy.signal.decimate 自动包含抗混叠滤波 (Chebyshev Type I)
                vib_low = decimate(seg_vib, self.decimation)
                fft_micro = self._compute_fft(vib_low)

                # 3. Macro Stream (高频全景: 直接FFT前段)
                fft_macro = self._compute_fft(seg_vib[:self.fft_points])

                # 4. Acoustic (MFCC)
                # 取前 0.2秒计算声纹，避免平均化过度
                mfcc = np.mean(self.extractor.extract_audio_features(seg_aud[:8192]), axis=0)

                samples['micro'].append(fft_micro)
                samples['macro'].append(fft_macro)
                samples['acoustic'].append(mfcc)
                samples['speed'].append(hz)

            if len(samples['micro']) > 0:
                data = {
                    'micro': np.array(samples['micro'], dtype=np.float32),
                    'panorama': np.array(samples['macro'], dtype=np.float32),
                    'acoustic': np.array(samples['acoustic'], dtype=np.float32),
                    'speed': np.array(samples['speed'], dtype=np.float32),  # Real Hz
                    'load': float(load_id) * 100  # Real kg: 0, 200, 400
                }
                np.save(save_path, data)


if __name__ == '__main__':
    CH4DualStreamProcessor().process_all()