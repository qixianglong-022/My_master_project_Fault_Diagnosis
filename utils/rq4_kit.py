# utils/rq4_kit.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import set_style

# === 全局中文与样式 ===
FONT_PROP = set_style()


def plot_rq4_modal_comparison(csv_path, save_dir):
    """
    生成 RQ4 多模态消融对比图 (柱状图)
    """
    if not os.path.exists(csv_path):
        print(f"[Warn] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 转换数据格式用于绘图 (Melt)
    # 我们关注 0kg (最难工况) 和 400kg (正常工况)
    plot_data = []

    # 定义显示名称
    name_map = {
        'Vib_Only': '仅振动 (Vibration)',
        'Audio_Only': '仅声纹 (Acoustic)',
        'Fusion': '多模态融合 (Fusion)'
    }

    for idx, row in df.iterrows():
        cfg = row['Config']
        if cfg not in name_map: continue

        display_name = name_map[cfg]

        # 添加 0kg 数据
        plot_data.append({
            'Configuration': display_name,
            'Condition': '轻载 (0kg)',
            'Accuracy': row['Acc_0kg']
        })
        # 添加 400kg 数据
        plot_data.append({
            'Configuration': display_name,
            'Condition': '重载 (400kg)',
            'Accuracy': row['Acc_400kg']
        })

    plot_df = pd.DataFrame(plot_data)

    # === 绘图 ===
    plt.figure(figsize=(10, 6))

    # 颜色策略: 蓝色(Vib), 绿色(Audio), 红色(Fusion)
    palette = {
        '仅振动 (Vibration)': '#1f77b4',
        '仅声纹 (Acoustic)': '#2ca02c',
        '多模态融合 (Fusion)': '#d62728'
    }

    sns.barplot(
        data=plot_df,
        x='Condition',
        y='Accuracy',
        hue='Configuration',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )

    # 样式微调
    plt.ylim(0, 105)
    plt.ylabel('诊断准确率 (%)', fontsize=14, fontproperties=FONT_PROP)
    plt.xlabel('工况负载', fontsize=14, fontproperties=FONT_PROP)
    plt.title('多模态融合消融实验对比 (RQ4)', fontsize=16, fontproperties=FONT_PROP, pad=20)
    plt.legend(title='模态配置', prop=FONT_PROP, title_fontproperties=FONT_PROP, loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 标数值
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f', padding=3, fontsize=10)

    # 保存
    save_path = os.path.join(save_dir, 'RQ4_Modal_Comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"    -> 图表已生成: {os.path.basename(save_path)}")