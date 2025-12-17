import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fft import fft, rfft
from scipy.signal import decimate  # [新增] 引入降采样
import warnings

# 引入项目配置
from config import Config

# ================= 第四章：物理引导的预处理配置 =================
ORIGINAL_FS = 51200.0  # 原始采样率
DECIMATION_FACTOR = 10  # [核心调整] 降采样因子 (51.2k -> 5.12k)
TARGET_FS = ORIGINAL_FS / DECIMATION_FACTOR  # 新采样率: 5120 Hz

FFT_POINTS = 4096  # FFT点数 (保证分辨率 = 5120/4096 = 1.25 Hz)
HOP_LENGTH = FFT_POINTS // 2  # 50% 重叠

# 输出目录 (建议区分文件夹，以免混淆)
OUTPUT_DIR = os.path.join(Config.PROJECT_ROOT, "processed_data_fft_decimated")


def parse_filename(fname):
    """解析文件名: [Fault]_[Load]_[Speed].txt"""
    name_body = fname.replace('.txt', '').replace('.csv', '')
    parts = name_body.split('-')
    if len(parts) >= 3:
        return parts[1], parts[2]

    parts = name_body.split('_')
    if len(parts) >= 3:
        return parts[1], parts[2]
    return None, None


def read_raw_data_robust(file_path):
    """鲁棒读取数据"""
    try:
        # 尝试直接读取
        df = pd.read_csv(file_path, sep='\t', encoding='gb18030', header=None, skiprows=16, engine='python')
        if df.shape[1] < 2:
            df = pd.read_csv(file_path, sep=',', encoding='gb18030', header=None, skiprows=16)
        return df.values
    except:
        # 回退逻辑略... 保持你原有的即可
        return None


def process_fft_optimized():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f">>> 创建输出目录: {OUTPUT_DIR}")

    # 获取任务列表
    if not hasattr(Config, 'DATA_DOMAINS'):
        print("[Error] Config 未定义 DATA_DOMAINS")
        return

    tasks = []
    for domain_name, folder_name in Config.DATA_DOMAINS.items():
        domain_path = os.path.join(Config.DATA_ROOT, folder_name)
        if not os.path.exists(domain_path): continue
        files = [f for f in os.listdir(domain_path) if f.endswith('.txt') or f.endswith('.csv')]
        for fname in files:
            tasks.append((domain_name, domain_path, fname))

    # 显示物理参数供核对
    print("=" * 40)
    print(f"物理参数确认:")
    print(f"  - 原始采样率: {ORIGINAL_FS} Hz")
    print(f"  - 降采样因子: {DECIMATION_FACTOR} (Target Fs = {TARGET_FS} Hz)")
    print(f"  - FFT 点数  : {FFT_POINTS}")
    print(f"  - 频率分辨率: {TARGET_FS / FFT_POINTS:.4f} Hz (满足断条诊断需求)")
    print(f"  - 有效带宽  : 0 ~ {TARGET_FS / 2} Hz")
    print("=" * 40)

    # 预计算汉宁窗
    hanning_win = np.hanning(FFT_POINTS)

    process_count = 0
    skip_count = 0

    for domain, d_path, fname in tqdm(tasks, desc="DSP Processing"):
        load_id, speed_id = parse_filename(fname)
        if load_id is None: continue

        save_name = f"{domain}_{load_id}_{speed_id}_FFT.npy"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        if os.path.exists(save_path):
            skip_count += 1
            continue

        raw_data = read_raw_data_robust(os.path.join(d_path, fname))
        if raw_data is None: continue

        try:
            vib_idx = Config.COL_INDICES_VIB[0]
            spd_idx = Config.COL_INDEX_SPEED

            # 1. 获取原始信号
            raw_vib = raw_data[:, vib_idx].astype(np.float32)
            raw_spd = raw_data[:, spd_idx].astype(np.float32)

            # === 核心步骤：降采样 (Decimation) ===
            # scipy.signal.decimate 会自动应用低通滤波器防止混叠
            # 注意：降采样后数据长度变为原来的 1/10
            vib_signal = decimate(raw_vib, DECIMATION_FACTOR, ftype='fir')

            # 转速信号通常不需要复杂的抗混叠，直接取平均或降采样均可
            # 为了对应，我们简单降采样
            speed_signal = raw_spd[::DECIMATION_FACTOR][:len(vib_signal)]

        except Exception as e:
            print(f"[Error] 处理失败 {fname}: {e}")
            continue

        # 2. 滑动窗口 FFT
        n_samples = len(vib_signal)
        n_frames = (n_samples - FFT_POINTS) // HOP_LENGTH + 1

        if n_frames < 1: continue

        fft_batch = []
        speed_batch = []

        for i in range(n_frames):
            start = i * HOP_LENGTH
            end = start + FFT_POINTS

            seg_vib = vib_signal[start:end]
            seg_spd = speed_signal[start:end]

            # 加窗
            seg_vib = seg_vib * hanning_win

            # FFT (rfft 只计算正频率，效率高)
            spectrum = np.abs(np.fft.rfft(seg_vib))

            # 归一化 (关键物理修正)
            spectrum = spectrum * 2 / FFT_POINTS

            # rfft 输出长度为 N/2 + 1。
            # 我们只取前 N/2 个点作为特征，去掉直流或最后一点
            spectrum = spectrum[:FFT_POINTS // 2]

            avg_speed = np.mean(seg_spd)

            fft_batch.append(spectrum)
            speed_batch.append(avg_speed)

        # 保存
        if len(fft_batch) > 0:
            # 数据转换为 float32 节省空间 (树莓派友好)
            data_dict = {
                'x': np.array(fft_batch, dtype=np.float32),  # Shape: [N_frames, 2048]
                'speed': np.array(speed_batch, dtype=np.float32),  # Shape: [N_frames]
                'fs': TARGET_FS,
                'n_fft': FFT_POINTS
            }
            np.save(save_path, data_dict)
            process_count += 1

    print(f"\n处理完成: 新增 {process_count}, 跳过 {skip_count}")


if __name__ == '__main__':
    process_fft_optimized()