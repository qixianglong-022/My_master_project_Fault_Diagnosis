# run_rq5_advanced.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
from utils.rq5_kit import add_gaussian_noise, plot_noise_robustness_chart

# === 实验配置 ===
# 加入全部模型进行全方位对比
MODELS_TO_COMPARE = [
    'Phys-RDLinear',
    'Vanilla RDLinear',  # 增加: 验证物理模块的抗噪贡献
    'TiDE',
    'FD-CNN',
    'ResNet-18'  # 增加: 验证深度模型的脆弱性
]

# 测试信噪比列表 (注意：None 代表 Clean)
# 这里的顺序只影响计算顺序，绘图顺序由 rq5_kit 控制
SNR_LEVELS = [-5, 0, 5, 10, None]

GPU_ID = 0


def evaluate_with_noise(model, dataloader, device, snr):
    """带噪声注入的评估函数 (V2 适配版)"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        # [V2 Change] Unpack 7 items
        # 注意变量名：DataLoader 吐出的是 mic, mac...
        for mic, mac, ac, cur, spd, ld, y_cls in dataloader:
            mic, mac = mic.to(device), mac.to(device)
            ac, cur = ac.to(device), cur.to(device)
            spd, ld = spd.to(device), ld.to(device)
            y_cls = y_cls.to(device)

            # --- 1. 筛选 0kg (轻载) 样本 ---
            # 还原真实负载: ld 是归一化的 (x/400)，反算
            load_real = (ld * 400).round().int()
            # 筛选条件：< 100kg 视为轻载 (兼容 0kg)
            mask_0kg = (load_real < 100).squeeze()

            if mask_0kg.sum() == 0: continue

            # 应用筛选掩码
            mic = mic[mask_0kg]
            mac = mac[mask_0kg]
            ac = ac[mask_0kg]
            cur = cur[mask_0kg]
            spd = spd[mask_0kg]
            ld = ld[mask_0kg]
            y_cls = y_cls[mask_0kg]

            # --- 2. 注入噪声 ---
            if snr is not None:
                mic = add_gaussian_noise(mic, snr)
                mac = add_gaussian_noise(mac, snr)
                ac = add_gaussian_noise(ac, snr)
                # cur = add_gaussian_noise(cur, snr) # 电流通常不加噪，如需可取消注释

            # --- 3. 推理 ---
            # [Fix] 在这里定义 is_phys
            is_phys = hasattr(model, 'pgfa') or \
                      model.__class__.__name__.startswith('Phys') or \
                      model.__class__.__name__.startswith('Ablation')

            if is_phys:
                # V2 Phys Model: 6 参数
                logits, _ = model(mic, mac, ac, cur, spd, ld)
            else:
                # Baseline: Concat Feature + Speed
                full_x = torch.cat([
                    mic.squeeze(-1),
                    mac.squeeze(-1),
                    ac,
                    cur
                ], dim=1)
                logits, _ = model(full_x, spd)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_cls.cpu().numpy())

    if not all_preds: return 0.0
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred) * 100
    return acc


def main():
    print("=========================================================")
    print("   RQ5 Automation: Noise Robustness (Full Comparison)")
    print("=========================================================")

    rq5_root = os.path.join("checkpoints_ch4", "rq5")
    os.makedirs(rq5_root, exist_ok=True)

    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 数据集
    test_dl = DataLoader(Ch4DualStreamDataset(config, mode='test'),
                         batch_size=config.BATCH_SIZE, shuffle=False)
    # 用于补训的 Train set
    train_ds = Ch4DualStreamDataset(config, mode='train')
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    results = []

    for model_name in MODELS_TO_COMPARE:
        print(f"\n>>> [Task] Evaluating Model: {model_name}")

        # 权重路径
        ckpt_dir = os.path.join("checkpoints_ch4", "rq1", model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "model.pth")

        # 初始化
        config.MODEL_NAME = model_name
        try:
            model = get_model(model_name, config).to(device)
        except Exception as e:
            print(f"    [Error] Init failed: {e}")
            continue

        # 智能加载或训练
        if os.path.exists(ckpt_path):
            print(f"    [Info] Loading existing checkpoint...")
            try:
                model.load_state_dict(torch.load(ckpt_path, weights_only=False))
            except:
                model.load_state_dict(torch.load(ckpt_path))
        else:
            print(f"    [Warn] Checkpoint missing. Training {model_name} now...")
            trainer = Trainer(config, model, device)
            epochs = 25
            for epoch in range(epochs):
                trainer.train_epoch(train_dl)
            torch.save(model.state_dict(), ckpt_path)
            print(f"    [Info] Training done.")

        row = {'Model': model_name}

        # 遍历 SNR
        for snr in SNR_LEVELS:
            snr_name = f"{snr}dB" if snr is not None else "Clean"
            acc = evaluate_with_noise(model, test_dl, device, snr)
            row[f"Acc_{snr_name}"] = acc
            print(f"    -> SNR {snr_name}: {acc:.2f}%")

        results.append(row)

    # 汇总与绘图
    if results:
        df = pd.DataFrame(results)
        # 按照用户期望的 X 轴顺序排列 CSV 列 (方便人工看)
        cols = ['Model', 'Acc_-5dB', 'Acc_0dB', 'Acc_5dB', 'Acc_10dB', 'Acc_Clean']
        # 填充缺失列防报错
        for c in cols:
            if c not in df.columns: df[c] = 0.0
        df = df[cols]

        csv_path = os.path.join(rq5_root, "RQ5_Noise_Results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> 汇总表已保存: {csv_path}")

        plot_noise_robustness_chart(csv_path, rq5_root)

    print("\n=========================================================")
    print("   RQ5 Completed! See checkpoints/rq5/")
    print("=========================================================")


if __name__ == "__main__":
    main()