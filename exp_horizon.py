import os
import torch
import numpy as np
import pandas as pd
from config import Config
from train import train_model
from run_evaluation import get_model_instance
from data_loader import MotorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# 定义要测试的预测步长
HORIZONS = [24, 48, 96, 192, 336]
# 也可以对比 Transformer
MODELS_TO_TEST = ['RDLinear', 'Transformer']


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
            Config.EPOCHS = 15
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


if __name__ == "__main__":
    run_horizon_exp()