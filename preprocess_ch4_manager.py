import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate
from config import Ch4Config, Config
from utils.feature_extractor import FeatureExtractor


# ==========================================
# 1. 鲁棒频率计算 (保持不变)
# ==========================================
def robust_compute_frequency_hz(signal, fs, default_hz=None):
    """
    [物理层修正] 基于迟滞比较与中值滤波的鲁棒转速频率计算
    """
    # 1. 消除高频噪声
    signal = np.convolve(signal, np.ones(5) / 5, mode='same')

    _min, _max = np.min(signal), np.max(signal)
    if _max - _min < 0.5:
        return float(default_hz) if default_hz else 0.0

    mid = (_max + _min) / 2
    high_th = mid + 0.2 * (_max - mid)
    low_th = mid - 0.2 * (_max - mid)

    state = 0  # 0: low, 1: high
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


# ==========================================
# 2. 鲁棒文件读取 (保持不变)
# ==========================================
def read_raw_data_robust(file_path):
    """
    智能读取带中文表头的 txt 文件，自动处理 GB18030 编码和变长表头
    """
    data_start_line = None

    try:
        with open(file_path, 'rb') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:50]):
                try:
                    line_str = line.decode('gb18030', errors='ignore')
                except:
                    continue
                if 'Time' in line_str and 'Data Channels' in line_str:
                    data_start_line = i + 1
                    break
    except Exception as e:
        print(f"[Warn] 预扫描文件失败: {e}")
        return None

    if data_start_line is None:
        data_start_line = 17

    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='gb18030',
            header=None,
            skiprows=data_start_line,
            engine='python'
        )
        return df.values
    except Exception as e:
        try:
            df = pd.read_csv(
                file_path,
                sep='\t',
                encoding='gbk',
                header=None,
                skiprows=data_start_line,
                engine='python'
            )
            return df.values
        except:
            print(f"    [Read Error] 读取失败: {file_path}")
            return None


class CH4DualStreamProcessor:
    def __init__(self):
        # 初始化 Ch4 配置实例
        self.ch4_cfg = Ch4Config()

        self.original_fs = Config.SAMPLE_RATE  # 51200
        self.decimation = 50

        # 降采样后的频率: 51200 / 50 = 1024 Hz
        self.target_fs = self.original_fs / self.decimation

        self.fft_points = 1024
        self.hanning = np.hanning(self.fft_points)
        self.correction = 1.63

        # 原始窗口大小: 1024 * 50 = 51200 (即 1秒数据)
        self.win_size_raw = self.fft_points * self.decimation
        self.stride = self.win_size_raw

        self.extractor = FeatureExtractor(Config)
        self.out_dir = self.ch4_cfg.DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        self.ref_speeds = {
            '1': 15.0, '2': 30.0, '3': 45.0, '4': 60.0,
            '5': 30.0, '6': 45.0, '7': 30.0, '8': 45.0
        }

    def _compute_fft(self, sig):
        """通用 FFT 计算"""
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))

        # 移除直流
        sig = sig - np.mean(sig)
        sig = sig[:self.fft_points] * self.hanning

        spec = np.abs(np.fft.rfft(sig)) / (self.fft_points / 2)
        spec = spec * self.correction

        # rfft 返回 N/2 + 1 个点，截取前 N/2
        return spec[:self.fft_points // 2]

    def process_all(self):
        print(f">>> [Preprocess] Dual-Stream (Decimation={self.decimation}, Fs'={self.target_fs}Hz)")
        print(f"    [Physics] Current channel will be decimated to align resolution (1Hz)")

        tasks = []
        for dom, folder in Config.DATA_DOMAINS.items():
            path = os.path.join(Config.DATA_ROOT, folder)
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.endswith('.txt'): tasks.append((dom, path, f))

        for dom, d_path, fname in tqdm(tasks):
            # 1. 解析文件名
            try:
                parts = fname.replace('.txt', '').split('-')
                load_id, speed_id = parts[1], parts[2]
            except Exception:
                continue

            save_name = f"{dom}_{load_id}_{speed_id}_dual.npy"
            save_path = os.path.join(self.out_dir, save_name)

            # [Optional] 如果需要强制重新生成，注释掉下面这行
            # if os.path.exists(save_path): continue

            # 2. 读取数据
            file_full_path = os.path.join(d_path, fname)
            raw = read_raw_data_robust(file_full_path)

            if raw is None: continue

            # 3. 通道获取
            max_col = raw.shape[1]
            curr_idx = self.ch4_cfg.COL_INDICES_CURRENT[0]

            vib = raw[:, Config.COL_INDICES_VIB[0]]
            spd_signal = raw[:, Config.COL_INDEX_SPEED]

            # 获取电流
            if curr_idx < max_col:
                curr = raw[:, curr_idx]
            else:
                curr = np.zeros_like(vib)

            # 获取声纹
            aud_idx = Config.COL_INDICES_AUDIO[0]
            if aud_idx < max_col:
                aud = raw[:, aud_idx]
            else:
                aud = np.zeros_like(vib)

            samples = {'micro': [], 'macro': [], 'acoustic': [], 'current': [], 'speed': [], 'load_rms': []}
            N = len(vib)
            default_hz = self.ref_speeds.get(speed_id, 15.0)

            # 4. 滑动窗口处理
            for i in range(0, N - self.win_size_raw + 1, self.stride):
                # 截取 1秒 长的原始数据 (51200点)
                seg_vib = vib[i: i + self.win_size_raw]
                seg_spd = spd_signal[i: i + self.win_size_raw]
                seg_aud = aud[i: i + self.win_size_raw]
                seg_curr = curr[i: i + self.win_size_raw]

                # A. 计算物理转速
                hz = robust_compute_frequency_hz(seg_spd, self.original_fs, default_hz)

                # B. Micro Stream (振动降采样 -> 1Hz 分辨率 FFT)
                try:
                    vib_low = decimate(seg_vib, self.decimation, zero_phase=True)
                    fft_micro = self._compute_fft(vib_low)
                except Exception:
                    continue

                # C. [CRITICAL FIX] Current Stream (电流降采样 -> 1Hz 分辨率 FFT)
                # 之前代码直接取前1024点原始数据，分辨率仅50Hz，完全丢失电气故障特征。
                # 现在同样做降采样，获得 0-512Hz 范围的高分辨率频谱。
                try:
                    curr_low = decimate(seg_curr, self.decimation, zero_phase=True)
                    fft_curr_full = self._compute_fft(curr_low)
                    # 截取 Config 定义的维度 (128点, 即 0-128Hz, 覆盖 50Hz 基频及边带)
                    fft_curr = fft_curr_full[:self.ch4_cfg.CURRENT_DIM]
                except Exception:
                    # 如果电流全为0或降采样失败，补零
                    fft_curr = np.zeros(self.ch4_cfg.CURRENT_DIM, dtype=np.float32)

                # D. 负载 RMS (基于 1秒 原始数据计算)
                curr_rms = np.sqrt(np.mean(seg_curr ** 2))

                # E. Macro Stream (原始高频振动 FFT)
                fft_macro = self._compute_fft(seg_vib[:self.fft_points])

                # F. Acoustic (MFCC)
                # 使用更多数据计算 MFCC 均值，更稳定
                # 注意：FeatureExtractor 内部通常有分帧，输入长一点没关系
                mfcc = np.mean(self.extractor.extract_audio_features(seg_aud[:16384]), axis=0)

                samples['micro'].append(fft_micro)
                samples['macro'].append(fft_macro)
                samples['acoustic'].append(mfcc)
                samples['current'].append(fft_curr)
                samples['speed'].append(hz)
                samples['load_rms'].append(curr_rms)

            if len(samples['micro']) > 0:
                data = {
                    'micro': np.array(samples['micro'], dtype=np.float32),
                    'macro': np.array(samples['macro'], dtype=np.float32),
                    'acoustic': np.array(samples['acoustic'], dtype=np.float32),
                    'current': np.array(samples['current'], dtype=np.float32),
                    'speed': np.array(samples['speed'], dtype=np.float32),
                    'load_rms': np.array(samples['load_rms'], dtype=np.float32),
                    'label_domain': dom
                }
                np.save(save_path, data)


if __name__ == '__main__':
    CH4DualStreamProcessor().process_all()