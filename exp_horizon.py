import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

# ================= 新增：绘图所需的库 =================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 原有导入
from config import Config
from train import train_model
from run_evaluation import get_model_instance
from data_loader import MotorDataset

# ================= 配置区域 =================
# 定义要测试的预测步长
HORIZONS = [24, 48, 96, 192, 336]
# 也可以对比 Transformer
MODELS_TO_TEST = ['RDLinear', 'Transformer', 'DLinear', 'LSTMAE', 'TiDE']

# [新增] 字体路径配置 (请根据系统修改)
# Windows: C:\Windows\Fonts\simhei.ttf
# Mac: /System/Library/Fonts/PingFang.ttc
# Linux: /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
FONT_PATH = r'C:\Windows\Fonts\simhei.ttf'


# ================= 原有核心逻辑 (保持不变) =================
def run_horizon_exp():
    results = []

    for model_name in MODELS_TO_TEST:
        for p in HORIZONS:
            print(f"\n>>> Running Horizon Exp: Model={model_name}, P={p}")

            # 1. 动态修改配置
            Config.MODEL_NAME = model_name
            Config.PRED_LEN = p
            # 实验名区分
            Config.OUTPUT_DIR = os.path.join(Config.PROJECT_ROOT, "checkpoints", f"Horizon_{model_name}_P{p}")
            Config.SCALER_PATH = os.path.join(Config.OUTPUT_DIR, "scaler_params.pkl")

            # 2. 训练 (必须重训，因为输出头大小变了)
            # 简单起见，我们只跑 10-20 个 epoch 快速看趋势
            Config.EPOCHS = 20
            train_model()

            # 3. 评估 (只看 MSE)
            mse_score = eval_mse_only(model_name)

            results.append({
                "Model": model_name,
                "Horizon": p,
                "MSE": mse_score
            })
            print(f"    [Result] {model_name} @ P={p} -> MSE={mse_score:.6f}")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("exp_horizon_results.csv", index=False)
    print("\n>>> Horizon Experiment Done! Saved to exp_horizon_results.csv")


def eval_mse_only(model_name):
    """只计算测试集 MSE"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_instance(device)
    model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')))
    model.eval()

    # 加载测试集 (使用 Baseline 场景)
    atoms = Config.TRAIN_ATOMS  # 用训练集同分布的测试数据看拟合能力，或者用迁移数据看泛化
    ds = MotorDataset(atoms, mode='test', fault_types=['HH'])  # 只看健康数据的预测误差
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for x, y, cov, _ in dl:
            x, y, cov = x.to(device), y.to(device), cov.to(device)
            pred = model(x, cov)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)

    return total_loss / count


# ================= 新增：独立绘图函数 =================
def plot_horizon_results_pdf(csv_path="exp_horizon_results.csv"):
    """读取CSV并生成学术风格PDF，包含中文字体配置"""
    if not os.path.exists(csv_path):
        print(f"[Error] 找不到数据文件: {csv_path}，无法绘图。")
        return

    # 1. 字体配置 (解决中文和Latex混排问题)
    if os.path.exists(FONT_PATH):
        base_font = fm.FontProperties(fname=FONT_PATH)
    else:
        print("[Warn] 未找到指定中文字体，将使用默认字体。")
        base_font = fm.FontProperties()

    # 创建各层级字体属性
    fonts = {
        'title': base_font.copy(),
        'label': base_font.copy(),
        'tick': base_font.copy(),
        'legend': base_font.copy()
    }
    fonts['title'].set_size(18);
    fonts['title'].set_weight('bold')
    fonts['label'].set_size(14)
    fonts['tick'].set_size(12)
    fonts['legend'].set_size(11)

    # 2. 样式设置
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 6))

    # 3. 绘图循环
    models = df['Model'].unique()
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']  # 学术常用色
    markers = ['o', 's', '^', 'v', 'D', 'X']

    for i, model in enumerate(models):
        subset = df[df['Model'] == model].sort_values('Horizon')

        # 特殊处理：如果是 RDLinear (本文方法)，加粗红色显示
        is_ours = 'RDLinear' in model
        c = colors[0] if is_ours else colors[(i + 1) % len(colors)]
        m = markers[0] if is_ours else markers[(i + 1) % len(markers)]
        ls = '-' if is_ours else '--'
        lw = 2.5 if is_ours else 1.5
        alpha = 1.0 if is_ours else 0.8  # 其他模型稍微透明一点

        plt.plot(subset['Horizon'], subset['MSE'],
                 label=model, color=c, marker=m, linestyle=ls, linewidth=lw, markersize=8, alpha=alpha)

    # 4. 标签与装饰
    plt.title("不同预测步长下的模型预测误差对比", fontproperties=fonts['title'], y=1.02)
    plt.xlabel("预测步长 ($P$)", fontproperties=fonts['label'])  # LaTeX 符号混排
    plt.ylabel("均方误差 (MSE)", fontproperties=fonts['label'])

    # 强制 Y 轴从 0 开始
    plt.ylim(bottom=0)
    # 如果想给上方留点空间，可以解开下行注释并设置上限（可选）
    plt.ylim(0, df['MSE'].max() * 1.1)

    # 强制 X 轴显示离散的步长值
    horizons = sorted(df['Horizon'].unique())
    plt.xticks(horizons, labels=[str(h) for h in horizons], fontproperties=fonts['tick'])
    plt.yticks(fontproperties=fonts['tick'])

    plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, prop=fonts['legend'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 5. 保存
    pdf_path = "fig_horizon_comparison.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f">>> PDF 图表已成功保存至: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    # 1. 运行实验 (生成CSV)
    # 如果想跳过训练直接画图，请注释掉下面这行 run_horizon_exp()
    run_horizon_exp()

    # 2. 调用绘图函数 (读取CSV生成PDF)
    print("\n>>> 开始生成分析图表...")
    plot_horizon_results_pdf("exp_horizon_results.csv")