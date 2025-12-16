import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os
import glob
import matplotlib.font_manager as fm
import matplotlib as mpl

# ================= 配置区域 =================
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型元数据
MODEL_META = {
    'RDLinear': {'lat': 2.5, 'type': 'Linear', 'color': '#d62728', 'marker': 'o', 'label': 'RDLinear(Ours)'},
    'TiDE': {'lat': 6.0, 'type': 'MLP', 'color': '#ff7f0e', 'marker': '^', 'label': 'TiDE'},
    'Transformer': {'lat': 18.5, 'type': 'Transformer', 'color': '#1f77b4', 'marker': 's', 'label': 'Transformer'},
    'DLinear': {'lat': 1.8, 'type': 'Linear', 'color': '#7f7f7f', 'marker': 'p', 'label': 'DLinear'},
    'LSTMAE': {'lat': 12.4, 'type': 'RNN', 'color': '#2ca02c', 'marker': 'D', 'label': 'LSTMAE'}
}


# ================= 字体管理系统 (核心修改) =================
class FontManager:
    def __init__(self):
        # 1. 寻找中文字体路径
        self.font_path = self._find_chinese_font()
        print(f">>> 使用字体路径: {self.font_path}")

        # 2. 注册字体
        if self.font_path:
            fm.fontManager.addfont(self.font_path)

        # 3. 创建不同层级的字体对象 (解决字号不生效的终极方案)
        # 基础字体
        base = fm.FontProperties(fname=self.font_path) if self.font_path else fm.FontProperties()

        # 标题字体 (大号)
        self.title = base.copy()
        self.title.set_size(16)
        self.title.set_weight('bold')

        # 坐标轴名称字体 (中号)
        self.label = base.copy()
        self.label.set_size(14)

        # 刻度/图例/雷达图标签 (小号)
        self.tick = base.copy()
        self.tick.set_size(12)

        # 标注字体
        self.note = base.copy()
        self.note.set_size(11)

    def _find_chinese_font(self):
        # 优先列表
        fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']
        # 尝试从系统查找
        for font_name in fonts:
            for f in fm.fontManager.ttflist:
                if font_name.lower() in f.name.lower():
                    return f.fname
        # Windows 默认路径兜底
        if os.path.exists(r'C:\Windows\Fonts\simhei.ttf'):
            return r'C:\Windows\Fonts\simhei.ttf'
        return None


# ================= 数据加载函数 =================
def load_all_model_data():
    pattern = os.path.join(ROOT_DIR, "Thesis_Final_*_*", "eval_results", "summary_report.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        print("[Error] 未找到CSV文件")
        return None

    df_list = []
    for file in csv_files:
        model_name = None
        path_parts = os.path.normpath(file).split(os.sep)
        for part in path_parts:
            if part.startswith("Thesis_Final") and any(m in part for m in MODEL_META.keys()):
                for m in MODEL_META.keys():
                    if m in part:
                        model_name = m;
                        break
                break
        if model_name:
            try:
                df = pd.read_csv(file)
                df['Model'] = model_name
                df_list.append(df)
            except:
                pass

    if not df_list: return None
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df[combined_df['Model'].isin(MODEL_META.keys())]
    combined_df['Speed'] = combined_df['Speed'].astype(str)
    return combined_df


# ================= 绘图函数 (应用专用字体对象) =================

def plot_radar(df, fonts):
    """图1：雷达图"""
    dimensions = {
        '稳态15Hz': 'Load == 200 and Speed == "15"',
        '稳态30Hz\n(Unseen)': 'Load == 200 and Speed == "30"',
        '稳态45Hz\n(源域)': 'Load == 200 and Speed == "45"',
        '稳态60Hz\n(Unseen)': 'Load == 200 and Speed == "60"',
        '瞬态\n15-45Hz': 'Load == 200 and Speed == "15-45"',
        '瞬态\n45-15Hz': 'Load == 200 and Speed == "45-15"',
        '瞬态\n30-60Hz': 'Load == 200 and Speed == "30-60"',
        '瞬态\n60-30Hz': 'Load == 200 and Speed == "60-30"'
    }

    radar_df = pd.DataFrame()
    for dim_name, query_str in dimensions.items():
        try:
            subset = df.query(query_str).groupby('Model')['AUC'].mean()
            radar_df[dim_name] = subset
        except:
            radar_df[dim_name] = 0

    radar_df = radar_df.fillna(0)
    if radar_df.empty: return

    categories = list(dimensions.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    for model in MODEL_META.keys():
        if model not in radar_df.index: continue
        values = radar_df.loc[model].values.flatten().tolist()
        values += values[:1]
        meta = MODEL_META[model]
        is_ours = 'RDLinear' in model

        ax.plot(angles, values, linewidth=2.5 if is_ours else 1.5,
                label=meta['label'], color=meta['color'], marker=meta['marker'])
        ax.fill(angles, values, color=meta['color'], alpha=0.15 if is_ours else 0.05)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # [关键修改] 使用 fonts.tick 确保雷达图标签大小生效 (Size 12)
    plt.xticks(angles[:-1], categories, fontproperties=fonts.tick)

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"],
               color="grey", size=10)
    plt.ylim(0, 1.05)

    # [关键修改] 使用 fonts.title 和 fonts.legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop=fonts.tick)
    plt.title("源域负载(200kg)下转速动态适应性分析", y=1.10, fontproperties=fonts.title)

    out_path = os.path.join(OUTPUT_DIR, 'Fig1_Radar_Speed_Adaptation.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f">>> 图1 保存完毕")
    plt.close()


def plot_bar_robustness(df, fonts):
    """图2：柱状图"""
    bar_df = df.groupby(['Model', 'Load'])['AUC'].mean().reset_index()
    load_map = {0: '0kg (Light/轻载)', 200: '200kg (Source/源域)', 400: '400kg (Heavy/重载)'}
    bar_df['Load Label'] = bar_df['Load'].map(load_map)

    plt.figure(figsize=(9, 6))
    palette = {model: meta['color'] for model, meta in MODEL_META.items()}
    hue_order = [m for m in MODEL_META.keys() if m in bar_df['Model'].unique()]

    sns_bar = sns.barplot(x='Load Label', y='AUC', hue='Model', data=bar_df,
                          palette=palette, hue_order=hue_order,
                          edgecolor='black', linewidth=1.0, alpha=0.9,
                          order=['0kg (Light/轻载)', '200kg (Source/源域)', '400kg (Heavy/重载)'])

    # [关键修改] 刻度字体 -> fonts.tick (12)
    plt.xticks(ticks=range(len(sns_bar.get_xticklabels())),
               labels=[label.get_text() for label in sns_bar.get_xticklabels()],
               fontproperties=fonts.tick)
    plt.yticks(fontproperties=fonts.tick)
    plt.ylim(0.4, 1.05)

    # [关键修改] 轴标签 -> fonts.label (14)
    plt.ylabel('所有转速工况平均 AUC 分数', fontproperties=fonts.label)
    plt.xlabel('负载工况域', fontproperties=fonts.label)

    # [关键修改] 标题 -> fonts.title (18)
    plt.title("跨负载工况迁移鲁棒性分析", y=1.03, fontproperties=fonts.title)

    # 标注
    plt.annotate('轻载抗噪优势\n(Ours > TiDE)',
                 xy=(-0.28, 0.90),  # 箭头尖端：从 0 左移到 -0.28
                 xytext=(-0.28, 1.00),  # 文字位置：垂直向上拉
                 ha='center', color='#d62728', fontweight='bold', fontproperties=fonts.note,
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [MODEL_META[l]['label'] if l in MODEL_META else l for l in labels]

    # [修改点 2] 图例移动到右上角空白区域，设为单列
    plt.legend(handles, new_labels,
               loc='upper right',  # 右上角
               ncol=1,  # 单列
               prop=fonts.tick,
               framealpha=0.90,  # 不透明背景防止遮挡
               borderpad=0.3)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    out_path = os.path.join(OUTPUT_DIR, 'Fig2_Bar_Load_Robustness.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f">>> 图2 保存完毕")
    plt.close()


def plot_scatter_efficiency(df, fonts):
    """图3：散点图"""
    avg_perf = df.groupby('Model')['AUC'].mean().reset_index()
    avg_perf['Latency'] = avg_perf['Model'].apply(lambda x: MODEL_META[x]['lat'])
    avg_perf['Label'] = avg_perf['Model'].apply(lambda x: MODEL_META[x]['label'])

    plt.figure(figsize=(9, 6))

    for _, row in avg_perf.iterrows():
        meta = MODEL_META[row['Model']]
        plt.scatter(row['Latency'], row['AUC'], c=meta['color'], marker=meta['marker'], s=400,
                    edgecolor='white', linewidth=1.5, zorder=5)

        xytext = (0, 15)
        if 'DLinear' in row['Model']: xytext = (0, -35)
        if 'Transformer' in row['Model']: xytext = (-20, -25)
        if 'RDLinear' in row['Model']: xytext = (0, 20)

        plt.annotate(row['Label'], (row['Latency'], row['AUC']),
                     xytext=xytext, textcoords='offset points',
                     ha='center', fontweight='bold', color='black',
                     fontproperties=fonts.note)

    plt.arrow(16, 0.70, -12, 0.20, head_width=0.02, head_length=1, fc='gray', ec='gray', alpha=0.3)
    plt.text(4, 0.92, "理想区域\n(快且准)", color='gray', fontweight='bold', fontproperties=fonts.label)

    # [关键修改] 应用层级字体
    plt.xlabel(r"推理延迟 (ms) $\leftarrow$ 更快 (边缘端实测)", fontproperties=fonts.label)
    plt.ylabel(r"全工况平均 AUC $\rightarrow$ 更准", fontproperties=fonts.label)
    plt.xticks(fontproperties=fonts.tick)
    plt.yticks(fontproperties=fonts.tick)
    plt.title("模型精度与计算效率权衡分析", y=1.03, fontproperties=fonts.title)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0.6, 1.02)
    plt.xlim(0, 22)

    out_path = os.path.join(OUTPUT_DIR, 'Fig3_Scatter_Efficiency.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f">>> 图3 保存完毕")
    plt.close()


if __name__ == "__main__":
    # 1. 初始化字体管理器 (自动加载并设置大小)
    fm_sys = FontManager()

    # 2. 设置全局样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    # 3. 加载数据并绘图
    df = load_all_model_data()
    if df is not None:
        print(">>> 开始生成图表...")
        try:
            plot_radar(df, fm_sys)
            plot_bar_robustness(df, fm_sys)
            plot_scatter_efficiency(df, fm_sys)
            print("\n>>> 所有图表生成完毕！")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"\n[Error] {e}")