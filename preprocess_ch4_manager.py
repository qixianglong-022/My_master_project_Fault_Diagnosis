import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate
from config import Ch4Config, Config
from utils.feature_extractor import FeatureExtractor


# ==========================================
# 核心修复 1: 移植 robust_compute_frequency_hz
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
# 核心修复 2: 移植 read_raw_data (解决编码报错)
# ==========================================
def read_raw_data_robust(file_path):
    """
    智能读取带中文表头的 txt 文件，自动处理 GB18030 编码和变长表头
    """
    data_start_line = None

    # 1. 预扫描：寻找数据起点 (避免 skiprows=17 硬编码错误)
    # 使用 'rb' 模式读取二进制，防止 encoding 报错阻断扫描
    try:
        with open(file_path, 'rb') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:50]):  # 只扫前50行
                # 尝试解码这一行，如果失败就跳过
                try:
                    line_str = line.decode('gb18030', errors='ignore')
                except:
                    continue

                # 寻找数据开始的特征 (通常包含 "Time" 和 "Data Channels")
                if 'Time' in line_str and 'Data Channels' in line_str:
                    data_start_line = i + 1
                    break
    except Exception as e:
        print(f"[Warn] 预扫描文件失败: {e}")
        return None

    # 兜底：如果没找到，默认17行
    if data_start_line is None:
        data_start_line = 17

    # 2. Pandas 读取 (指定 encoding='gb18030')
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='gb18030',  # [CRITICAL FIX] 解决 utf-8 报错
            header=None,
            skiprows=data_start_line,
            engine='python'  # python 引擎对分隔符容错更好
        )
        return df.values
    except Exception as e:
        # 如果 gb18030 也不行，尝试 gbk
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
            print(f"    [Read Error] 最终读取失败: {e}")
            return None


class CH4DualStreamProcessor:
    def __init__(self):
        # 初始化 Ch4 配置实例
        self.ch4_cfg = Ch4Config()

        self.original_fs = Config.SAMPLE_RATE  # 51200 (这是硬件属性，共用 Config 没问题)
        self.decimation = 50
        self.target_fs = self.original_fs / self.decimation

        self.fft_points = 1024
        self.hanning = np.hanning(self.fft_points)
        self.correction = 1.63

        self.win_size_raw = self.fft_points * self.decimation
        self.stride = self.win_size_raw

        self.extractor = FeatureExtractor(Config)  # 特征提取器依赖基础 Config

        # 输出目录指向 Ch4Config 定义的新路径
        self.out_dir = self.ch4_cfg.DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        # 理论转速表 (用于兜底)
        self.ref_speeds = {
            '1': 15.0, '2': 30.0, '3': 45.0, '4': 60.0,
            '5': 30.0, '6': 45.0, '7': 30.0, '8': 45.0
        }

    def _compute_fft(self, sig):
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))

        # 移除直流
        sig = sig - np.mean(sig)
        sig = sig[:self.fft_points] * self.hanning

        spec = np.abs(np.fft.rfft(sig)) / (self.fft_points / 2)
        spec = spec * self.correction
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
            # 1. 解析文件名
            try:
                parts = fname.replace('.txt', '').split('-')
                load_id, speed_id = parts[1], parts[2]
            except Exception as e:
                # print(f"[Skip] 文件名格式不符: {fname}")
                continue

            save_name = f"{dom}_{load_id}_{speed_id}_dual.npy"
            save_path = os.path.join(self.out_dir, save_name)
            if os.path.exists(save_path): continue

            # 2. 读取数据 (使用修复后的函数)
            file_full_path = os.path.join(d_path, fname)
            raw = read_raw_data_robust(file_full_path)

            if raw is None:
                continue

            # 3. 通道获取
            max_col = raw.shape[1]

            curr_idx = self.ch4_cfg.COL_INDICES_CURRENT[0]

            # 获取各通道数据
            vib = raw[:, Config.COL_INDICES_VIB[0]]  # 振动仍用 Config
            spd_signal = raw[:, Config.COL_INDEX_SPEED]

            # 获取电流 (带越界保护)
            if curr_idx < max_col:
                curr = raw[:, curr_idx]
            else:
                curr = np.zeros_like(vib)

            # 声纹通道保护
            aud_idx = Config.COL_INDICES_AUDIO[0]
            if aud_idx < max_col:
                aud = raw[:, aud_idx]
            else:
                aud = np.zeros_like(vib)

            samples = {'micro': [], 'macro': [], 'acoustic': [], 'current': [], 'speed': [], 'load_rms': []}

            N = len(vib)
            # 理论转速兜底
            default_hz = self.ref_speeds.get(speed_id, 15.0)

            for i in range(0, N - self.win_size_raw + 1, self.stride):
                seg_vib = vib[i: i + self.win_size_raw]
                seg_spd = spd_signal[i: i + self.win_size_raw]
                seg_aud = aud[i: i + self.win_size_raw]
                seg_curr = curr[i: i + self.win_size_raw]

                # A. 计算物理转速
                hz = robust_compute_frequency_hz(seg_spd, self.original_fs, default_hz)

                # B. Micro Stream (降采样 -> FFT)
                # 使用 zero_phase=True 避免相位偏移
                try:
                    vib_low = decimate(seg_vib, self.decimation, zero_phase=True)
                    fft_micro = self._compute_fft(vib_low)
                except Exception:
                    # 极少数情况 decimate 可能会因为数据太短报错
                    continue

                # [NEW] Current 处理
                # 1. 频谱特征 (用于分类)
                fft_curr_full = self._compute_fft(seg_curr[:self.fft_points])
                fft_curr = fft_curr_full[:self.ch4_cfg.CURRENT_DIM]  # 使用 Ch4Config 定义的维度 (128)

                # 2. 负载 RMS (用于物理引导)
                curr_rms = np.sqrt(np.mean(seg_curr ** 2))

                # C. Macro Stream (直接 FFT)
                fft_macro = self._compute_fft(seg_vib[:self.fft_points])

                # D. Acoustic (MFCC)
                mfcc = np.mean(self.extractor.extract_audio_features(seg_aud[:8192]), axis=0)

                samples['micro'].append(fft_micro)
                samples['macro'].append(fft_macro)
                samples['acoustic'].append(mfcc)
                samples['current'].append(fft_curr)  # [NEW]
                samples['speed'].append(hz)
                samples['load_rms'].append(curr_rms)  # [NEW] RMS Load

            if len(samples['micro']) > 0:
                data = {
                    'micro': np.array(samples['micro'], dtype=np.float32),
                    'macro': np.array(samples['macro'], dtype=np.float32),
                    'acoustic': np.array(samples['acoustic'], dtype=np.float32),
                    'current': np.array(samples['current'], dtype=np.float32),  # [NEW]
                    'speed': np.array(samples['speed'], dtype=np.float32),
                    'load_rms': np.array(samples['load_rms'], dtype=np.float32),  # [NEW]
                    'label_domain': dom  # 保存标签字符串
                }
                np.save(save_path, data)


if __name__ == '__main__':
    CH4DualStreamProcessor().process_all()