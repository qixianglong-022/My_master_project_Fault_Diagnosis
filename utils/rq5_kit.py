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
    p_signal = torch.mean(tensor ** 2, dim=1, keepdim=True)
    # 计算噪声功率 (P_noise)
    p_noise = p_signal / (10 ** (snr_db / 10))
    # 生成噪声
    noise_std = torch.sqrt(p_noise)
    noise = torch.randn_like(tensor) * noise_std

    return tensor + noise


def plot_noise_robustness_chart(csv_path, save_dir):
    """
    绘制噪声鲁棒性折线图 (按用户要求排序和调整字体)
    X轴: -5dB -> 0dB -> 5dB -> 10dB -> 无噪声
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    # 准备绘图数据 (Melt)
    plot_data = []

    # 定义映射关系: CSV列名 -> 图表显示名
    col_map = {
        'Acc_-5dB': '-5dB',
        'Acc_0dB': '0dB',
        'Acc_5dB': '5dB',
        'Acc_10dB': '10dB',
        'Acc_Clean': '无噪声 (>30dB)'
    }

    # 强制 X 轴顺序
    x_order = ['-5dB', '0dB', '5dB', '10dB', '无噪声 (>30dB)']

    # 颜色映射 (涵盖所有模型)
    model_colors = {
        'Phys-RDLinear': '#d62728',  # Red (Ours)
        'Vanilla RDLinear': '#ff7f0e',  # Orange (Ablation)
        'TiDE': '#2ca02c',  # Green
        'FD-CNN': '#1f77b4',  # Blue
        'ResNet-18': '#7f7f7f'  # Grey
    }

    # 样式映射 (区分 Ours 和 Baselines)
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

    # === 绘图 ===
    plt.figure(figsize=(11, 7))

    sns.lineplot(
        data=plot_df,
        x='SNR',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=markers,
        dashes=False,
        palette=model_colors,
        linewidth=2.5,
        markersize=9
    )

    # === 字体大小控制 (Title > Axis Label > Ticks) ===
    # 1. 标题 (最大)
    plt.title('不同信噪比下的模型鲁棒性对比 (RQ5)', fontsize=20, fontproperties=FONT_PROP, pad=20, weight='bold')

    # 2. 轴标签 (中等)
    plt.ylabel('诊断准确率 (%)', fontsize=16, fontproperties=FONT_PROP)
    plt.xlabel('噪声强度 (信噪比 SNR)', fontsize=16, fontproperties=FONT_PROP)

    # 3. 刻度标签 (较小)
    plt.xticks(fontsize=12, fontproperties=FONT_PROP)
    plt.yticks(fontsize=12, fontproperties=FONT_PROP)

    # 其他装饰
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 强制设置 X 轴顺序
    plt.xlim(-0.5, 4.5)  # 5个点
    # 这一步是为了防止 Seaborn 自动乱序，这里我们通过 plot_df 的顺序结合 Hue 控制，
    # 最好是把 SNR 列转为 Categorical 类型
    plot_df['SNR'] = pd.Categorical(plot_df['SNR'], categories=x_order, ordered=True)

    # 重新画一次以应用 Categorical Order (最稳妥的方法)
    plt.clf()
    sns.lineplot(
        data=plot_df,
        x='SNR',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=markers,
        dashes=False,
        palette=model_colors,
        linewidth=2.5,
        markersize=10
    )
    # 重设装饰
    plt.title('不同信噪比下的模型鲁棒性对比 (RQ5)', fontsize=20, fontproperties=FONT_PROP, pad=20, weight='bold')
    plt.ylabel('诊断准确率 (%)', fontsize=16, fontproperties=FONT_PROP)
    plt.xlabel('噪声强度 (信噪比 SNR)', fontsize=16, fontproperties=FONT_PROP)
    plt.xticks(fontsize=13, fontproperties=FONT_PROP)  # 加大一点
    plt.yticks(fontsize=13, fontproperties=FONT_PROP)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 图例
    plt.legend(prop=FONT_PROP, title='模型架构', title_fontproperties=FONT_PROP, fontsize=12, loc='lower right')

    save_path = os.path.join(save_dir, 'RQ5_Noise_Robustness.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"    -> 图表已生成: {os.path.basename(save_path)}")