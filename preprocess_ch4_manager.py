import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate
from config import Ch4Config, Config
from utils.feature_extractor import FeatureExtractor


def robust_compute_rpm(signal, fs):
    """
    [增强版] 计算一段方波信号的平均转速
    """
    # 1. 消除高频噪声 (简单的移动平均)
    # window = 5
    # signal = np.convolve(signal, np.ones(window)/window, mode='valid')

    # 2. 动态阈值二值化
    _min, _max = np.min(signal), np.max(signal)
    if _max - _min < 0.5:  # 信号幅度太小，可能是静止或干扰
        return 0.0
    threshold = (_max + _min) / 2
    binary = (signal > threshold).astype(int)

    # 3. 提取上升沿
    edges = np.diff(binary)
    rising_indices = np.where(edges == 1)[0]

    if len(rising_indices) < 2:
        return 0.0

    # 4. 计算平均间隔 (点数)
    intervals = np.diff(rising_indices)
    # 去除异常间隔 (比如毛刺导致的极短间隔)
    median_interval = np.median(intervals)
    valid_intervals = intervals[np.abs(intervals - median_interval) < 0.5 * median_interval]

    if len(valid_intervals) == 0:
        avg_interval = median_interval
    else:
        avg_interval = np.mean(valid_intervals)

    # 5. 转 RPM -> Hz
    # RPM = (fs / interval) * 60
    # Hz = RPM / 60 = fs / interval
    if avg_interval <= 0: return 0.0

    freq_hz = fs / avg_interval
    return freq_hz


class CH4DualStreamProcessor:
    def __init__(self):
        self.original_fs = Config.SAMPLE_RATE  # 51200
        self.decimation = 50
        self.target_fs = self.original_fs / self.decimation  # 1024 Hz
        self.fft_points = 1024
        self.hanning = np.hanning(self.fft_points)

        # 滑动窗口设置
        self.win_size_raw = self.fft_points * self.decimation  # 51200点 = 1秒
        self.stride = self.win_size_raw // 2  # 50% 重叠

        self.extractor = FeatureExtractor(Config)
        self.out_dir = Ch4Config.DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

    def _compute_fft(self, sig):
        # 物理幅值谱
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))
        sig = sig[:self.fft_points] * self.hanning
        spec = np.abs(np.fft.rfft(sig))
        return spec[:self.fft_points // 2]  # [512]

    def process_all(self):
        print(">>> [Preprocess] Starting Sliding Window Processing...")
        tasks = []
        for dom, folder in Config.DATA_DOMAINS.items():
            path = os.path.join(Config.DATA_ROOT, folder)
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.endswith('.txt'): tasks.append((dom, path, f))

        for dom, d_path, fname in tqdm(tasks):
            # Parse: HH-2-1.txt
            parts = fname.replace('.txt', '').split('-')
            load_id, speed_id = parts[1], parts[2]

            save_name = f"{dom}_{load_id}_{speed_id}_dual.npy"
            save_path = os.path.join(self.out_dir, save_name)
            if os.path.exists(save_path): continue

            # Read
            try:
                raw = pd.read_csv(os.path.join(d_path, fname), sep='\t', skiprows=17, header=None,
                                  engine='python').values
            except:
                continue

            vib = raw[:, Config.COL_INDICES_VIB[0]]  # 取第一个振动通道
            spd = raw[:, Config.COL_INDEX_SPEED]
            # 声纹
            aud_idx = Config.COL_INDICES_AUDIO[0]
            aud = raw[:, aud_idx] if aud_idx < raw.shape[1] else np.zeros_like(vib)

            samples = {'micro': [], 'macro': [], 'acoustic': [], 'speed': []}

            N = len(vib)
            for i in range(0, N - self.win_size_raw, self.stride):
                seg_vib = vib[i: i + self.win_size_raw]
                seg_spd = spd[i: i + self.win_size_raw]
                seg_aud = aud[i: i + self.win_size_raw]

                # 1. Speed (Hz) - 物理对齐关键！
                hz = robust_compute_rpm(seg_spd, self.original_fs)

                # 2. Micro Stream (Low Freq, High Res)
                # 先降采样 -> 再FFT
                vib_low = decimate(seg_vib, self.decimation)
                fft_micro = self._compute_fft(vib_low)

                # 3. Macro Stream (High Freq, Low Res)
                # 直接取前1024点做FFT (模拟低内存MCU只能处理短序列)
                fft_macro = self._compute_fft(seg_vib[:self.fft_points])

                # 4. Acoustic
                mfcc = np.mean(self.extractor.extract_audio_features(seg_aud[:8192]), axis=0)

                samples['micro'].append(fft_micro)
                samples['macro'].append(fft_macro)
                samples['acoustic'].append(mfcc)
                samples['speed'].append(hz)

            if len(samples['micro']) > 0:
                # 转 float32 减小体积
                data = {k: np.array(v, dtype=np.float32) for k, v in samples.items()}
                data['load'] = float(load_id) * 100  # 0, 200, 400
                np.save(save_path, data)


if __name__ == '__main__':
    CH4DualStreamProcessor().process_all()