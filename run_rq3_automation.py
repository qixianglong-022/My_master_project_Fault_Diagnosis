import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 导入项目模块
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
from utils.visualization import extract_features  # 复用可视化工具中的特征提取


# === MMD 计算辅助函数 ===
def rbf_mmd(X, Y, device, sigma_list=[1.0, 5.0, 10.0]):
    """
    计算 RBF 核的 MMD 距离 (衡量源域和目标域的特征分布差异)
    X: Source Features [N, D]
    Y: Target Features [M, D]
    """
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)

    m = X.size(0)
    n = Y.size(0)

    # 计算核矩阵
    xx = torch.mm(X, X.t())
    yy = torch.mm(Y, Y.t())
    zz = torch.mm(X, Y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(yy.shape).to(device),
                  torch.zeros(zz.shape).to(device))

    for sigma in sigma_list:
        XX += torch.exp(-0.5 * dxx / sigma)
        YY += torch.exp(-0.5 * dyy / sigma)
        XY += torch.exp(-0.5 * dxy / sigma)

    return torch.mean(XX + YY - 2. * XY).item()


# === 核心消融实验任务 ===
def run_ablation(model_alias, gpu_id=0):
    # 1. 初始化配置
    config = Ch4Config()
    config.MODEL_NAME = model_alias  # 这一步很关键，工厂根据这个名字加载不同配置的模型

    # 区分 Checkpoint 目录
    config.CHECKPOINT_DIR = os.path.join("checkpoints_rq3", model_alias)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    set_seed(config.SEED)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> [RQ3] Running: {model_alias}")

    # 2. 准备数据
    # Source Domain (200kg) 用于训练
    train_dl = DataLoader(Ch4DualStreamDataset(config, 'train'), batch_size=config.BATCH_SIZE, shuffle=True)

    # Target Domain (Test set 包含 0kg 和 400kg)
    test_ds = Ch4DualStreamDataset(config, 'test')
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. 初始化模型与训练器
    model = get_model(model_alias, config).to(device)
    trainer = Trainer(config, model, device)

    # 4. 训练阶段 (20 Epochs 足够收敛)
    print("    Training...")
    for epoch in range(20):
        trainer.train_epoch(train_dl)

    # 5. 评估阶段 (关注最难的 400kg 变工况)
    print("    Evaluating...")
    # 这里的 save_path 生成 csv 用于后续计算 accuracy
    eval_csv_path = os.path.join(config.CHECKPOINT_DIR, "eval_all.csv")
    trainer.evaluate(test_dl, save_path=eval_csv_path)

    # 读取 CSV 计算 Target (400kg) 的精度
    df = pd.read_csv(eval_csv_path)
    # Load_kg > 300 即为重载工况
    df_heavy = df[df['Load_kg'] > 300]

    if len(df_heavy) > 0:
        acc_target = df_heavy['Is_Correct'].mean() * 100
    else:
        acc_target = 0.0
        print("[Warn] No 400kg samples found in evaluation!")

    # 6. 计算 MMD (核心指标：衡量特征解耦程度)
    print("    Calculating MMD...")

    # A. 提取源域特征 (Source: 200kg)
    # 使用 shuffle=False 保证顺序，且只需提取一部分用于计算即可
    source_eval_dl = DataLoader(Ch4DualStreamDataset(config, 'train'), batch_size=config.BATCH_SIZE, shuffle=False)
    src_feats, _, _, _ = extract_features(model, source_eval_dl, device)

    # B. 提取目标域特征 (Target)
    tgt_feats, _, _, tgt_loads = extract_features(model, test_dl, device)

    # C. 筛选出 400kg 的特征用于计算 MMD
    # 你的 Loader 中 load 被归一化了 (load/400)，所以 400kg 对应 1.0
    # [修复点] tgt_loads 是 (N, 1)，必须 flatten 成 (N,) 才能作为 mask 索引
    tgt_mask = (tgt_loads > 0.8).flatten()
    tgt_feats_heavy = tgt_feats[tgt_mask]

    # D. 执行计算
    if len(tgt_feats_heavy) > 0:
        # 为了防止 OOM 和加快速度，只采样前 200 个样本计算 MMD 即可
        # 只要样本是随机分布的，距离就是统计有效的
        sample_n = min(200, len(src_feats), len(tgt_feats_heavy))
        mmd_val = rbf_mmd(src_feats[:sample_n], tgt_feats_heavy[:sample_n], device)
    else:
        mmd_val = 999.0
        print("[Warn] No target samples valid for MMD calculation.")

    print(f"    -> Target Acc (400kg): {acc_target:.2f}%, MMD: {mmd_val:.4f}")

    return {
        "Model": model_alias,
        "PGFA": "Yes" if "PGFA" in model_alias or "Phys" in model_alias else "No",
        "MTL": "Yes" if "MTL" in model_alias or "Phys" in model_alias else "No",
        "Target Acc (400kg)": acc_target,
        "MMD Distance": mmd_val
    }


# === 主程序 ===
if __name__ == "__main__":
    # 定义要对比的四种配置
    configs = [
        'Ablation-Base',  # 无物理引导，无MTL
        'Ablation-PGFA',  # 有物理引导，无MTL
        'Ablation-MTL',  # 无物理引导，有MTL
        'Phys-RDLinear'  # Full Model (Ours)
    ]

    results = []

    print("\n========================================================")
    print("   RQ3: Physics-Guided Mechanism Ablation (Start)")
    print("========================================================")

    for c in configs:
        # 这里串行运行，若 GPU 显存足够可尝试并行
        try:
            res = run_ablation(c, gpu_id=0)
            results.append(res)
        except Exception as e:
            print(f"[Error] Failed to run {c}: {e}")
            import traceback

            traceback.print_exc()

    # 生成最终表格
    if results:
        df = pd.DataFrame(results)
        # 保留4位小数展示 MMD，2位小数展示 Acc
        df['MMD Distance'] = df['MMD Distance'].round(4)
        df['Target Acc (400kg)'] = df['Target Acc (400kg)'].round(2)

        print("\n========================================================")
        print("   RQ3: Final Ablation Results")
        print("========================================================")
        print(df.to_string(index=False))

        df.to_csv("Table_RQ3_Mechanism_Ablation.csv", index=False)
        print("\n>>> Table saved to Table_RQ3_Mechanism_Ablation.csv")
    else:
        print("No results collected.")