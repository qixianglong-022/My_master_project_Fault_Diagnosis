import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 配置区 =================
# 指定你生成的文件夹路径
DATA_DIR = r"processed_data_ch4"

# 选择两个代表性文件进行对比
# 建议：一个 200kg (源域/健康或故障), 一个 0kg (目标域/故障)
# 文件名格式: {Domain}_{LoadID}_{SpeedID}_dual.npy
# 例如: BR_2_1 (200kg, 15Hz), BR_0_1 (0kg, 15Hz)
TARGET_FILES = [
    os.path.join(DATA_DIR, "HH_0_1_dual.npy"),
    os.path.join(DATA_DIR, "HH_0_2_dual.npy"),
]

# 物理参数 (必须与 preprocess 保持一致)
FS_NEW = 1024.0  # 降采样后的采样率
FFT_PTS = 1024  # FFT 点数
FREQ_RES = FS_NEW / FFT_PTS  # 频率分辨率 = 1Hz


# =========================================

def set_chinese_font():
    # 尝试设置中文字体，防止乱码
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun']
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            continue


def inspect_processed_file(file_path):
    print(f"\n{'=' * 60}")
    print(f"🔍 正在检查文件: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        return

    try:
        # 加载 .npy (注意：它现在是一个字典)
        data_dict = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 1. 检查键值
    keys = data_dict.keys()
    print(f"✅ 数据键值: {list(keys)}")

    if 'current' not in keys or 'micro' not in keys:
        print("❌ 缺少关键数据 (current 或 micro)！")
        return

    # 2. 提取数据
    # Shape: [Sample_Num, Dim]
    spec_cur = data_dict['current']
    spec_vib = data_dict['micro']
    load_rms = data_dict['load_rms']
    speed_hz = data_dict['speed']

    print(f"📊 数据维度:")
    print(f"  - Current Spectrum: {spec_cur.shape} (预期: [N, 128])")
    print(f"  - Vibration Spectrum: {spec_vib.shape} (预期: [N, 512])")
    print(f"  - Load RMS (Avg): {np.mean(load_rms):.4f}")
    print(f"  - Speed (Avg): {np.mean(speed_hz):.2f} Hz")

    # 3. 计算平均频谱 (降噪以便观察)
    avg_spec_cur = np.mean(spec_cur, axis=0)
    avg_spec_vib = np.mean(spec_vib, axis=0)

    # 生成频率轴
    # Current: 0 ~ 128 Hz (分辨率 1Hz)
    freqs_cur = np.arange(len(avg_spec_cur)) * FREQ_RES
    # Vib: 0 ~ 512 Hz (分辨率 1Hz)
    freqs_vib = np.arange(len(avg_spec_vib)) * FREQ_RES

    # 4. 可视化诊断
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"数据体检报告: {os.path.basename(file_path)}", fontsize=16)

    # --- 子图 1: 电流频谱 (核心关注点) ---
    plt.subplot(3, 1, 1)
    plt.plot(freqs_cur, avg_spec_cur, color='#d62728', linewidth=1.5)
    plt.title("【核心】电流频谱 (Current Spectrum) - 寻找 50Hz 基频及边带", fontsize=12, fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)  # 重点看 0-100Hz

    # 标注 50Hz
    idx_50 = int(50 / FREQ_RES)
    if idx_50 < len(avg_spec_cur):
        val_50 = avg_spec_cur[idx_50]
        plt.annotate(f'50Hz Peak: {val_50:.2f}', xy=(50, val_50),
                     xytext=(50 + 5, val_50), arrowprops=dict(facecolor='black', shrink=0.05))

        # 简单判断
        if val_50 < 0.001:
            plt.text(60, val_50, "⚠️ 警告: 50Hz 峰值过低！可能信号丢失", color='red')
        else:
            plt.text(60, val_50, "✅ 50Hz 峰值清晰", color='green')

    # --- 子图 2: 振动频谱 (参考) ---
    plt.subplot(3, 1, 2)
    plt.plot(freqs_vib, avg_spec_vib, color='#1f77b4', linewidth=1.0)
    plt.title("振动频谱 (Micro Stream)", fontsize=12)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 200)

    # --- 子图 3: 负载与转速分布 ---
    plt.subplot(3, 2, 5)
    plt.hist(load_rms, bins=20, color='orange', alpha=0.7)
    plt.title(f"负载分布 (Load RMS)\nMean: {np.mean(load_rms):.4f}")
    plt.xlabel("Current RMS (A)")

    plt.subplot(3, 2, 6)
    plt.hist(speed_hz, bins=20, color='green', alpha=0.7)
    plt.title(f"转速分布 (Speed Hz)\nMean: {np.mean(speed_hz):.2f}")
    plt.xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_chinese_font()

    # 扫描目录下所有匹配的文件
    # 如果你想看特定的，请修改 TARGET_FILES 列表
    # 如果列表里的文件不存在，脚本会自动寻找目录下存在的 .npy

    files_to_check = []
    for f in TARGET_FILES:
        if os.path.exists(f):
            files_to_check.append(f)
        else:
            # 尝试在目录下找任意一个 dual.npy 替代
            found = glob.glob(os.path.join(DATA_DIR, "*dual.npy"))
            if found and found[0] not in files_to_check:
                files_to_check.append(found[0])

    if not files_to_check:
        print(f"⚠️ 在 {DATA_DIR} 下没找到任何 .npy 文件！请先运行 preprocess_ch4_manager.py")
    else:
        for f in list(set(files_to_check))[:2]:  # 最多看2个
            inspect_processed_file(f)