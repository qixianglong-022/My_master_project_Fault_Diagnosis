import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from utils.visualization import set_style

FONT_PROP = set_style()


def plot_covariate_analysis(csv_path, save_dir):
    """
    绘制协变量重要性分析图
    1. 准确率对比 (Bar)
    2. MMD 距离对比 (Line)
    """
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    # 排序: No -> Speed -> Load -> Full (逻辑顺序)
    # 或者 No -> Load -> Speed -> Full (按性能)
    order = ['No_Covariates', 'Load_Only', 'Speed_Only', 'Full_Covariates']
    df = df.set_index('Config').reindex(order).reset_index()

    labels = ['无协变量', '仅负载', '仅转速', '完整模型']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. 准确率柱状图
    x = np.arange(len(df))
    width = 0.4
    bars = ax1.bar(x, df['Avg_Acc'], width, color='#5f9e6e', alpha=0.85, label='平均准确率 (%)')

    ax1.set_ylabel('准确率 (%)', fontproperties=FONT_PROP, fontsize=14, color='#333333')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontproperties=FONT_PROP, fontsize=12)

    # 标数值
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 计算贡献度 (相对于 No_Covariates)
    base_acc = df.loc[0, 'Avg_Acc']
    for i in range(1, len(df)):
        diff = df.loc[i, 'Avg_Acc'] - base_acc
        sign = '+' if diff >= 0 else ''
        # 在柱子中间写提升幅度
        ax1.text(x[i], df.loc[i, 'Avg_Acc'] / 2, f"{sign}{diff:.1f}%",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # 2. MMD 折线图 (右轴)
    ax2 = ax1.twinx()
    ax2.plot(x, df['MMD'], color='#d65f5f', marker='D', markersize=8, linewidth=2, linestyle='--', label='MMD 距离')
    ax2.set_ylabel('域间 MMD 距离', fontproperties=FONT_PROP, fontsize=14, color='#d65f5f')
    ax2.tick_params(axis='y', labelcolor='#d65f5f')

    # 图例
    lines, lbls = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, lbls + lbls2, loc='upper left', prop=FONT_PROP)

    plt.title("物理协变量对泛化性能与特征分布的影响", fontproperties=FONT_PROP, fontsize=16, pad=15)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "RQ4_Covariate_Impact.pdf")
    plt.savefig(save_path, dpi=300)
    print(f"    [Plot] Saved chart to {save_path}")
    plt.close()