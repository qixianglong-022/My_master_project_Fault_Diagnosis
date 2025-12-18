import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import decimate
from config import Config


class CH4DualStreamProcessor:
    """
    [集成版] 第四章多分辨率双流预处理管理器
    功能：一键实现 51.2kHz(全景) 与 5.12kHz(显微) 特征提取与对齐
    """

    def __init__(self):
        # 物理参数定义
        self.original_fs = Config.SAMPLE_RATE  # 51200 Hz
        self.decimation_factor = 10  # 10倍降采样
        self.target_fs = self.original_fs / self.decimation_factor

        self.fft_points = 1024  # 统一FFT点数
        self.hanning_win = np.hanning(self.fft_points)

        # 路径定义
        self.output_dir = os.path.join(Config.PROJECT_ROOT, "processed_data_ch4_dual_stream")
        os.makedirs(self.output_dir, exist_ok=True)

    def _read_raw_data(self, file_path):
        """鲁棒读取工业原始数据"""
        try:
            return pd.read_csv(file_path, sep='\t', encoding='gb18030',
                               header=None, skiprows=17, engine='python').values
        except Exception as e:
            print(f"  [Read Error] {file_path}: {e}")
            return None

    def _parse_meta(self, fname):
        """解析文件名获取物理工况: [Fault]-[LoadID]-[SpeedID].txt"""
        parts = fname.replace('.txt', '').replace('.csv', '').split('-')
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return None, None, None

    def process_all(self):
        print(f"--- [Chapter 4] Dual-Stream Processing Start ---")
        print(f"Micro-Stream Resolution: {self.target_fs / self.fft_points:.2f} Hz")

        tasks = []
        for domain, folder in Config.DATA_DOMAINS.items():
            path = os.path.join(Config.DATA_ROOT, folder)
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.endswith('.txt'): tasks.append((domain, path, f))

        for domain, d_path, fname in tqdm(tasks, desc="DSP Pipeline"):
            _, load_id, speed_id = self._parse_meta(fname)
            if load_id is None: continue

            save_path = os.path.join(self.output_dir, f"{domain}_{load_id}_{speed_id}_dual.npy")
            if os.path.exists(save_path): continue

            raw_data = self._read_raw_data(os.path.join(d_path, fname))
            if raw_data is None: continue

            # 1. 信号提取 (取第一路振动和转速)
            vib_raw = raw_data[:, Config.COL_INDICES_VIB[0]].astype(np.float32)
            spd_raw = raw_data[:, Config.COL_INDEX_SPEED].astype(np.float32)

            # 2. 核心：显微流降采样 (抗混叠滤波)
            vib_micro = decimate(vib_raw, self.decimation_factor, ftype='fir')

            # 3. 频域变换与物理归一化
            # 为了“小而美”，我们抽取中间一段稳定工况进行诊断
            micro_fft = self._compute_fft(vib_micro)
            panorama_fft = self._compute_fft(vib_raw)

            # 4. 自动化存储
            # 获取实际物理负载 (用于虚拟传感器回归)
            phys_load = float(load_id) * 100

            save_data = {
                'micro': micro_fft.astype(np.float32),  # 5Hz分辨率
                'panorama': panorama_fft.astype(np.float32),  # 50Hz分辨率
                'speed': np.mean(spd_raw),  # 实时转速
                'load': phys_load  # 负载标签
            }
            np.save(save_path, save_data)

    def _compute_fft(self, signal):
        """计算物理意义明确的幅值谱"""
        if len(signal) < self.fft_points:
            return np.zeros(self.fft_points // 2)
        # 取信号中段
        mid = len(signal) // 2
        seg = signal[mid: mid + self.fft_points] * self.hanning_win
        spec = np.abs(np.fft.rfft(seg))
        return spec[:self.fft_points // 2] * 2 / self.fft_points


if __name__ == '__main__':
    processor = CH4DualStreamProcessor()
    processor.process_all()