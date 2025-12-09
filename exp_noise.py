import os
import torch
import numpy as np
import pandas as pd
from config import Config
from run_evaluation import get_model_instance
from data_loader import MotorDataset
from torch.utils.data import DataLoader
from utils.anomaly import InferenceEngine

# 定义噪声水平
SNR_LEVELS = [None, 10, 5, 0, -5]  # None 表示无噪 (Clean)


def run_noise_exp():
    # 1. 确定使用哪个训练好的模型 (建议使用 Thesis_Final_Physics_Constraint)
    Config.MODEL_NAME = 'RDLinear'
    exp_source = "Thesis_Final_Physics_Constraint"
    Config.OUTPUT_DIR = os.path.join(Config.PROJECT_ROOT, "checkpoints", exp_source)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_instance(device)
    # 加载权重
    model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')))
    model.eval()

    # 2. 准备两种引擎
    # A. 自适应融合 (Adaptive) - 读取 fusion_params.json
    engine_adaptive = InferenceEngine(model)

    # B. 直接融合 (Direct/Average) - 强制 fusion_params = None
    engine_direct = InferenceEngine(model)
    engine_direct.fusion_params = None  # 强行覆盖为 None，触发 utils/anomaly.py 中的简单平均逻辑

    results = []

    # 3. 循环测试
    # 使用一个典型迁移工况，例如 load_0_speed_60 (易受噪声影响)
    test_atoms = [(0, '60')]

    for snr in SNR_LEVELS:
        print(f"\n>>> Testing SNR: {snr} dB")
        Config.TEST_NOISE_SNR = snr

        # 加载数据
        ds = MotorDataset(test_atoms, mode='test', fault_types=Config.TEST_FAULT_TYPES, noise_snr=snr)
        dl = DataLoader(ds, batch_size=64, shuffle=False)

        # 测 Adaptive
        auc_adapt, f1_adapt = eval_metrics(engine_adaptive, dl)
        results.append({"Method": "Adaptive Fusion (Ours)", "SNR": str(snr), "AUC": auc_adapt, "F1": f1_adapt})

        # 测 Direct
        auc_direct, f1_direct = eval_metrics(engine_direct, dl)
        results.append({"Method": "Direct Fusion", "SNR": str(snr), "AUC": auc_direct, "F1": f1_direct})

        print(f"    Direct   -> AUC: {auc_direct:.4f}, F1: {f1_direct:.4f}")
        print(f"    Adaptive -> AUC: {auc_adapt:.4f}, F1: {f1_adapt:.4f}")

    # 保存
    df = pd.DataFrame(results)
    df.to_csv("exp_noise_results.csv", index=False)
    print("\n>>> Noise Experiment Done!")


def eval_metrics(engine, dataloader):
    from sklearn.metrics import roc_auc_score, f1_score
    scores, labels = engine.predict(dataloader)

    # 简单的固定阈值或自适应阈值逻辑
    # 这里为了对比，直接用 AUC 最公平，F1 可以基于某个 percentile
    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.0

    # 估算 F1 (取 Top 20% 作为阈值演示，或者沿用 Config 中的阈值)
    # 这里我们简化，只返回 AUC，因为 AUC 对噪声鲁棒性对比更有说服力
    # 如果想算 F1，可以用之前训练好的阈值
    th_path = os.path.join(Config.OUTPUT_DIR, 'threshold.npy')
    if os.path.exists(th_path):
        th = float(np.load(th_path))
        preds = (scores > th).astype(int)
        f1 = f1_score(labels, preds)
    else:
        f1 = 0.0

    return auc, f1


if __name__ == "__main__":
    run_noise_exp()