# utils/rq5_kit.py
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.visualization import set_style

# === 全局设置 ===
FONT_PROP = set_style()


def add_gaussian_noise(tensor, snr_db):
    """
    向 Tensor 添加指定 SNR 的高斯白噪声
    """
    if snr_db is None: return tensor

    if not tensor.is_floating_point():
        tensor = tensor.float()

    # 计算信号功率 (P_signal)
    # dim=1 是时间/频率维度, keepdim=True 保持 [B, 1, C] 以便广播
    # 加上 1e-8 防止全0信号导致除以0
    p_signal = torch.mean(tensor ** 2, dim=1, keepdim=True) + 1e-8

    # 计算噪声功率 (P_noise)
    p_noise = p_signal / (10 ** (snr_db / 10))

    # 生成噪声
    noise_std = torch.sqrt(p_noise)
    noise = torch.randn_like(tensor) * noise_std

    return tensor + noise


def plot_noise_robustness_chart(csv_path, save_dir):
    """
    绘制噪声鲁棒性折线图
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    # 准备绘图数据 (Melt)
    plot_data = []

    # 定义映射关系
    col_map = {
        'Acc_-5dB': '-5dB',
        'Acc_0dB': '0dB',
        'Acc_5dB': '5dB',
        'Acc_10dB': '10dB',
        'Acc_Clean': 'Clean'
    }

    # 强制 X 轴顺序
    x_order = ['-5dB', '0dB', '5dB', '10dB', 'Clean']

    # 颜色映射 (Covering all models)
    model_colors = {
        'Phys-RDLinear': '#d62728',  # Red
        'Vanilla RDLinear': '#ff7f0e',  # Orange
        'TiDE': '#2ca02c',  # Green
        'FD-CNN': '#1f77b4',  # Blue
        'ResNet-18': '#7f7f7f'  # Grey
    }

    markers = {
        'Phys-RDLinear': 'o',
        'Vanilla RDLinear': 's',
        'TiDE': '^',
        'FD-CNN': 'D',
        'ResNet-18': 'X'
    }

    for idx, row in df.iterrows():
        model = row['Model']
        for csv_col, display_name in col_map.items():
            if csv_col in df.columns:
                acc = row[csv_col]
                plot_data.append({
                    'Model': model,
                    'SNR': display_name,
                    'Accuracy': acc
                })

    plot_df = pd.DataFrame(plot_data)

    # 设置 Categorical Order
    plot_df['SNR'] = pd.Categorical(plot_df['SNR'], categories=x_order, ordered=True)

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=plot_df,
        x='SNR',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=markers,
        palette=model_colors,
        linewidth=2.5,
        markersize=9
    )

    plt.title('噪声鲁棒性分析 (RQ5: 0kg工况)', fontsize=16, fontproperties=FONT_PROP, pad=15)
    plt.ylabel('诊断准确率 (%)', fontsize=14, fontproperties=FONT_PROP)
    plt.xlabel('信噪比 (SNR)', fontsize=14, fontproperties=FONT_PROP)
    plt.xticks(fontsize=12, fontproperties=FONT_PROP)
    plt.yticks(fontsize=12, fontproperties=FONT_PROP)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(prop=FONT_PROP, title='', loc='lower right')

    save_path = os.path.join(save_dir, 'RQ5_Noise_Robustness.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"    [Plot] Saved: {save_path}")
    plt.close()