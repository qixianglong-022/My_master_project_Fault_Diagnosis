# preprocess_ch4_manager.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate, cheby2, sosfilt
from config import Ch4Config, Config
from utils.feature_extractor import FeatureExtractor


def robust_compute_frequency_hz(signal, fs, default_hz=None):
    """
    [物理层修正] 鲁棒转速计算
    1. 尝试从方波脉冲计算
    2. 如果失败，回退到 default_hz (基于文件名的理论值)
    """
    # 1. 消除高频噪声
    signal = np.convolve(signal, np.ones(5) / 5, mode='same')

    _min, _max = np.min(signal), np.max(signal)
    if _max - _min < 0.5:
        return float(default_hz) if default_hz else 0.0

    # 动态阈值迟滞比较
    mid = (_max + _min) / 2
    high_th = mid + 0.2 * (_max - mid)
    low_th = mid - 0.2 * (_max - mid)

    state = 0
    rising_indices = []
    for i, val in enumerate(signal):
        if state == 0 and val > high_th:
            state = 1
            rising_indices.append(i)
        elif state == 1 and val < low_th:
            state = 0

    if len(rising_indices) < 2:
        return float(default_hz) if default_hz else 0.0

    intervals = np.diff(rising_indices)
    median_val = np.median(intervals)
    valid_intervals = intervals[np.abs(intervals - median_val) < 0.3 * median_val]

    if len(valid_intervals) == 0:
        avg_interval = median_val
    else:
        avg_interval = np.mean(valid_intervals)

    return fs / avg_interval


class CH4DualStreamProcessor:
    def __init__(self):
        self.original_fs = Config.SAMPLE_RATE  # 51200
        self.decimation = 50
        # 目标采样率 1024Hz => Nyquist 512Hz
        self.target_fs = self.original_fs / self.decimation
        self.fft_points = 1024  # 对应分辨率 1Hz

        self.hanning = np.hanning(self.fft_points)
        self.correction = 1.63

        # 窗口必须足够大才能降采样
        self.win_size_raw = self.fft_points * self.decimation  # 51200点
        self.stride = self.win_size_raw  # 无重叠

        self.extractor = FeatureExtractor(Config)
        self.out_dir = Ch4Config.DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        # 理论转速表 (用于兜底)
        self.ref_speeds = {
            '1': 15.0, '2': 30.0, '3': 45.0, '4': 60.0,
            # 变工况取中间值或最大值近似
            '5': 30.0, '6': 45.0, '7': 30.0, '8': 45.0
        }

    def _compute_fft(self, sig):
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))

        # 移除直流分量
        sig = sig - np.mean(sig)
        sig = sig[:self.fft_points] * self.hanning

        spec = np.abs(np.fft.rfft(sig)) / (self.fft_points / 2)
        spec = spec * self.correction
        # 返回前 512 点 (对应 0~512Hz)
        return spec[:self.fft_points // 2]

    def process_all(self):
        print(f">>> [Preprocess] Dual-Stream (Decimation={self.decimation}, Fs'={self.target_fs}Hz)")

        tasks = []
        for dom, folder in Config.DATA_DOMAINS.items():
            path = os.path.join(Config.DATA_ROOT, folder)
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.endswith('.txt'): tasks.append((dom, path, f))

        for dom, d_path, fname in tqdm(tasks):
            try:
                # Parse: HH-2-1.txt
                parts = fname.replace('.txt', '').split('-')
                load_id, speed_id = parts[1], parts[2]
            except:
                continue

            save_name = f"{dom}_{load_id}_{speed_id}_dual.npy"
            save_path = os.path.join(self.out_dir, save_name)
            if os.path.exists(save_path): continue

            try:
                raw = pd.read_csv(os.path.join(d_path, fname), sep='\t', skiprows=17, header=None, engine='c').values
            except:
                continue

            vib = raw[:, Config.COL_INDICES_VIB[0]]  # 取第一个振动通道
            spd_signal = raw[:, Config.COL_INDEX_SPEED]

            # 声纹处理
            if Config.COL_INDICES_AUDIO[0] < raw.shape[1]:
                aud = raw[:, Config.COL_INDICES_AUDIO[0]]
            else:
                aud = np.zeros_like(vib)

            samples = {'micro': [], 'macro': [], 'acoustic': [], 'speed': []}

            N = len(vib)
            # 理论转速兜底
            default_hz = self.ref_speeds.get(speed_id, 15.0)

            for i in range(0, N - self.win_size_raw + 1, self.stride):
                seg_vib = vib[i: i + self.win_size_raw]
                seg_spd = spd_signal[i: i + self.win_size_raw]
                seg_aud = aud[i: i + self.win_size_raw]

                # 1. 计算转速
                hz = robust_compute_frequency_hz(seg_spd, self.original_fs, default_hz)

                # 2. Micro Stream (物理抗混叠滤波 + 降采样)
                # 51200 -> 1024 Hz
                # decimate 内部包含 8阶切比雪夫滤波器
                vib_low = decimate(seg_vib, self.decimation, zero_phase=True)
                fft_micro = self._compute_fft(vib_low)

                # 3. Macro Stream (全带宽)
                # 直接取前 1024 点做 FFT (分辨率 50Hz, 带宽 25.6kHz)
                fft_macro = self._compute_fft(seg_vib[:self.fft_points])

                # 4. Acoustic
                mfcc = np.mean(self.extractor.extract_audio_features(seg_aud[:8192]), axis=0)

                samples['micro'].append(fft_micro)
                samples['macro'].append(fft_macro)
                samples['acoustic'].append(mfcc)
                samples['speed'].append(hz)

            if len(samples['micro']) > 0:
                data = {
                    'micro': np.array(samples['micro'], dtype=np.float32),
                    'macro': np.array(samples['macro'], dtype=np.float32),  # 注意 key 改为 macro 统一
                    'acoustic': np.array(samples['acoustic'], dtype=np.float32),
                    'speed': np.array(samples['speed'], dtype=np.float32),
                    'load': float(load_id) * 100 if load_id != '0' else 0.0
                }
                np.save(save_path, data)


if __name__ == '__main__':
    CH4DualStreamProcessor().process_all()