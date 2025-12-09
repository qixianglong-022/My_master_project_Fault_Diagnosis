import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置学术风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# 你的真实数据
data = {
    'Horizon': [24, 48, 96, 192, 336] * 5,
    'MSE': [
        # RDLinear
        0.479, 0.505, 0.529, 0.583, 0.652,
        # Transformer
        0.432, 0.461, 0.480, 0.508, 0.565,
        # DLinear
        0.469, 0.490, 0.532, 0.574, 0.637,
        # LSTMAE
        0.486, 0.494, 0.503, 0.524, 0.577,
        # TiDE
        0.432, 0.472, 0.492, 0.512, 0.534
    ],
    'Model': ['RDLinear (Ours)']*5 + ['Transformer']*5 + ['DLinear']*5 + ['LSTMAE']*5 + ['TiDE']*5
}

df = pd.DataFrame(data)

# 绘图
plt.figure(figsize=(9, 6))

# 定义颜色和线型（统一为填充标记）
palette = {
    'RDLinear (Ours)': '#d62728',  # 红色，突出
    'Transformer': '#1f77b4',      # 蓝色
    'TiDE': '#ff7f0e',             # 橙色
    'LSTMAE': '#2ca02c',           # 绿色
    'DLinear': '#7f7f7f'           # 灰色
}
markers = {
    'RDLinear (Ours)': 'o',        # 圆圈（填充）
    'Transformer': 's',            # 正方形（填充）
    'TiDE': '^',                   # 三角形（填充）
    'LSTMAE': 'D',                 # 菱形（填充）
    'DLinear': 'p'                 # 五边形（填充，替换原来的'x'）
}

# 绘制折线图
sns.lineplot(data=df, x='Horizon', y='MSE', hue='Model', style='Model',
             palette=palette, markers=markers, markersize=9, linewidth=2.5)

# 设置标题和标签
plt.title('Figure 6. Long-term Forecasting Stability Analysis', fontsize=14, y=1.02)
plt.xlabel('Prediction Horizon (H)', fontsize=12)
plt.ylabel('Prediction MSE (Mean Squared Error)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加标注箭头 (解释 Trade-off)
plt.annotate('Fitting Accuracy (Waveform)', xy=(336, 0.534), xytext=(200, 0.45),
             arrowprops=dict(facecolor='gray', shrink=0.05, alpha=0.5), fontsize=10, color='gray')
plt.annotate('Physical Robustness (Baseline)', xy=(336, 0.652), xytext=(200, 0.70),
             arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5), fontsize=10, color='#d62728')

# 调整图例位置
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

plt.tight_layout()
plt.savefig('fig6_horizon_stability.pdf', bbox_inches='tight')  # 添加bbox_inches避免图例被截断
plt.show()