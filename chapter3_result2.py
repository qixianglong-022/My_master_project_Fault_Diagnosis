import matplotlib.pyplot as plt
import numpy as np

# 设置风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# 数据准备
snr_levels = ['Clean', '10 dB', '5 dB', '0 dB', '-5 dB']
x = np.arange(len(snr_levels))

# F1 Score 数据
f1_adaptive = [0.9064, 0.8942, 0.8903, 0.9032, 0.8893]
f1_direct =   [0.9224, 0.5343, 0.5038, 0.5294, 0.5158]
f1_vib_only = [0.8772, 0.8772, 0.8772, 0.8772, 0.8772]

# Threshold 数据
th_adaptive = [1.4173, 1.2108, 1.2903, 1.0742, 1.1804]
th_direct =   [1.5616, 760.47, 2425.59, 7499.92, 24178.69]

# ==================== 图 1: F1 Score 鲁棒性对比 ====================
plt.figure(figsize=(8, 6))

plt.plot(x, f1_adaptive, marker='o', linewidth=3, color='#d62728', label='Adaptive Fusion (Ours)')
plt.plot(x, f1_direct, marker='s', linewidth=2, linestyle='--', color='#7f7f7f', label='Direct Fusion')
plt.plot(x, f1_vib_only, marker='None', linewidth=2, linestyle=':', color='blue', label='Vibration Only Baseline')

plt.xticks(x, snr_levels)
plt.ylabel('F1-Score (Higher is Better)', fontsize=12)
plt.xlabel('Noise Level (SNR)', fontsize=12)
plt.title('Figure 4. Fault Detection Robustness under Acoustic Noise', fontsize=14)
plt.ylim(0.4, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('fig4_noise_robustness.pdf')
plt.show()

# ==================== 图 2: 阈值稳定性对比 (Log Scale) ====================
plt.figure(figsize=(8, 6))

plt.plot(x, th_adaptive, marker='o', linewidth=3, color='#d62728', label='Adaptive Fusion (Ours)')
plt.plot(x, th_direct, marker='s', linewidth=2, linestyle='--', color='#7f7f7f', label='Direct Fusion')

plt.xticks(x, snr_levels)
plt.yscale('log')  # <--- 关键：对数坐标
plt.ylabel('Decision Threshold (Log Scale)', fontsize=12)
plt.xlabel('Noise Level (SNR)', fontsize=12)
plt.title('Figure 5. Threshold Drift Analysis', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('fig5_threshold_stability.pdf')
plt.show()