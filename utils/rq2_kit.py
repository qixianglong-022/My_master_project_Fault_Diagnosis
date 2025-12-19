# utils/rq2_kit.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.visualization import set_style

# === 全局中文与样式 ===
FONT_PROP = set_style()

# 故障中文映射
FAULT_MAP = {
    'KA': '转子断条 (KA)',
    'RU': '转子不平衡 (RU)',
    'FB': '轴承故障 (FB)',
    'HH': '健康 (HH)',
    'RM': '转子不对中 (RM)',
    'SW': '定子绕组 (SW)',
    'VU': '电压不平衡 (VU)',
    'BR': '转子弯曲 (BR)'
}
# 重点关注的故障 (根据论文 RQ2)
FOCUS_FAULTS = ['KA', 'RU', 'FB', 'RM']


def plot_rq2_bar_chart(csv_path, save_dir):
    """
    生成 RQ2 多分辨率对比柱状图
    对比: Macro-Only (51.2k), Micro-Only (1k), Multi-Res (Ours)
    """
    if not os.path.exists(csv_path):
        print(f"[Warn] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 准备绘图数据
    # 我们需要将宽表 (Wide) 转换为长表 (Long) 以便 Seaborn 绘图
    plot_data = []

    # 定义配置显示的名称映射
    config_map = {
        'Macro_Only': '单高频 (51.2kHz)',
        'Micro_Only': '单低频 (1kHz)',
        'Multi_Res': '多分辨率融合 (本文)'
    }

    # 提取重点故障的精度 (在 0kg 目标域下，因为 RQ2 侧重微弱特征)
    target_load = 0

    for idx, row in df.iterrows():
        model_cfg = row['Config']
        if model_cfg not in config_map: continue

        display_name = config_map[model_cfg]

        # 遍历所有故障，或者只遍历 Focus Faults
        # 为了全面，我们画所有故障，但重点故障排前面
        metrics_cols = [c for c in df.columns if f'_{target_load}kg' in c and 'Acc_' in c]

        for col in metrics_cols:
            # col name example: Acc_KA_0kg
            fault_code = col.split('_')[1]  # KA
            acc = row[col]

            plot_data.append({
                'Configuration': display_name,
                'Fault Type': FAULT_MAP.get(fault_code, fault_code),
                'Accuracy': acc,
                'SortKey': FOCUS_FAULTS.index(fault_code) if fault_code in FOCUS_FAULTS else 99
            })

    plot_df = pd.DataFrame(plot_data)
    # 排序：重点故障在前，其他在后
    plot_df = plot_df.sort_values(by=['SortKey', 'Fault Type'])

    # === 绘图 ===
    plt.figure(figsize=(12, 7))

    # 颜色策略: 蓝色(Macro), 橙色(Micro), 红色(Ours)
    palette = {
        '单高频 (51.2kHz)': '#1f77b4',  # Blue
        '单低频 (1kHz)': '#ff7f0e',  # Orange
        '多分辨率融合 (本文)': '#d62728'  # Red
    }

    sns.barplot(
        data=plot_df,
        x='Fault Type',
        y='Accuracy',
        hue='Configuration',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )

    # 样式微调
    plt.ylim(0, 105)
    plt.ylabel('诊断准确率 (%)', fontsize=14, fontproperties=FONT_PROP)
    plt.xlabel('故障类型 (0kg 轻载工况)', fontsize=14, fontproperties=FONT_PROP)
    plt.title('不同分辨率策略下的故障诊断性能对比 (RQ2)', fontsize=16, fontproperties=FONT_PROP, pad=20)
    plt.xticks(rotation=30, fontsize=11, fontproperties=FONT_PROP)
    plt.legend(title='输入配置', prop=FONT_PROP, title_fontproperties=FONT_PROP)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 保存
    save_path = os.path.join(save_dir, 'RQ2_Resolution_Comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"    -> 图表已生成: {os.path.basename(save_path)}")