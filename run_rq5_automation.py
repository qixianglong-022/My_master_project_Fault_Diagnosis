import os
import torch
import torch.nn.functional as F
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

# ==============================================================================
# 1. 实验配置
# ==============================================================================
MODELS_TO_COMPARE = [
    'Phys-RDLinear',  # Ours
    'Vanilla RDLinear',  # Ablation (No Physics)
    'TiDE',  # Transformer Baseline
    'FD-CNN',  # CNN Baseline
    'ResNet-18'  # CNN Baseline
]

# 测试信噪比列表 (None 代表 Clean)
SNR_LEVELS = [-5, 0, 5, 10, None]

GPU_ID = 0


# ==============================================================================
# 2. 专用训练器 (防止 trainer.py 未更新导致解包错误)
# ==============================================================================
class RQ5Trainer(Trainer):
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        # [适配] 解包 8 个变量
        for mic, mac, ac, cur, spd, ld, y_cls, _ in dataloader:
            mic, mac = mic.to(self.device), mac.to(self.device)
            ac, cur = ac.to(self.device), cur.to(self.device)
            spd, ld = spd.to(self.device), ld.to(self.device).float()
            y_cls = y_cls.to(self.device)

            self.optimizer.zero_grad()

            # 判断模型类型传参
            is_phys = hasattr(self.model, 'pgfa') or \
                      self.model.__class__.__name__.startswith('Phys')

            if is_phys:
                logits, pred_reg = self.model(mic, mac, ac, cur, spd, ld)
            else:
                # 基线模型只吃微观振动 + 转速
                logits = self.model(mic, speed=spd)
                pred_reg = None

            loss = F.cross_entropy(logits, y_cls)
            # 简单的辅助 Loss (如果支持)
            if pred_reg is not None:
                loss += 0.5 * F.mse_loss(pred_reg, ld)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {'loss': total_loss / len(dataloader)}


# ==============================================================================
# 3. 带噪声注入的评估函数
# ==============================================================================
def evaluate_with_noise(model, dataloader, device, snr):
    """
    RQ5 核心评估: 筛选 0kg 样本 -> 注入噪声 -> 评估
    """
    model.eval()
    all_preds, all_labels = [], []

    # 识别模型类型
    is_phys = hasattr(model, 'pgfa') or \
              model.__class__.__name__.startswith('Phys') or \
              model.__class__.__name__.startswith('Ablation')

    with torch.no_grad():
        # [修复] 解包 8 个变量
        for mic, mac, ac, cur, spd, ld_proxy, y_cls, phys_load in dataloader:

            mic, mac = mic.to(device), mac.to(device)
            ac, cur = ac.to(device), cur.to(device)
            spd, ld_proxy = spd.to(device), ld_proxy.to(device)
            y_cls = y_cls.to(device)
            # phys_load 用于筛选，留在 CPU 即可，或者转 GPU 均可
            phys_load = phys_load.to(device)

            # --- 1. 筛选 0kg (轻载) 样本 ---
            # 物理意义: 0kg 信号最微弱，加噪后最难识别，最能体现抗噪性
            # phys_load 是真实的 kg 值
            mask_0kg = (phys_load < 100)

            if mask_0kg.sum() == 0: continue

            # 应用筛选掩码
            mic = mic[mask_0kg]
            mac = mac[mask_0kg]
            ac = ac[mask_0kg]
            cur = cur[mask_0kg]
            spd = spd[mask_0kg]
            ld_proxy = ld_proxy[mask_0kg]
            y_cls = y_cls[mask_0kg]

            # --- 2. 注入噪声 (仅对波形类信号) ---
            if snr is not None:
                mic = add_gaussian_noise(mic, snr)  # 振动微观
                mac = add_gaussian_noise(mac, snr)  # 振动宏观
                ac = add_gaussian_noise(ac, snr)  # 声纹
                # 电流(cur)通常是频谱形式或抗噪性强，且在0kg下本身就是噪声，一般不额外加高斯白噪
                # 如果你想加，可以取消注释:
                # cur = add_gaussian_noise(cur, snr)

            # --- 3. 推理 ---
            if is_phys:
                # Phys Model: 全模态输入
                logits, _ = model(mic, mac, ac, cur, spd, ld_proxy)
            else:
                # [修复] Baseline: 保持与 RQ1 一致，只输入 Micro 振动
                # 这样才符合"基线是传统振动分析"的设定，且不会报维度错误
                logits = model(mic, speed=spd)
                # 注意: 如果基线返回 tuple (logits, loss), 需要解包
                if isinstance(logits, tuple):
                    logits = logits[0]

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_cls.cpu().numpy())

    if not all_preds: return 0.0
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred) * 100
    return acc


# ==============================================================================
# 4. 主流程
# ==============================================================================
def main():
    print("=========================================================")
    print("   RQ5 Automation: Noise Robustness (Full Comparison)")
    print("=========================================================")

    rq5_root = os.path.join("checkpoints_ch4", "rq5")
    os.makedirs(rq5_root, exist_ok=True)

    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 数据集: 使用 Test 集评估，Train 集用于可能的补训
    test_dl = DataLoader(Ch4DualStreamDataset(config, mode='test'),
                         batch_size=config.BATCH_SIZE, shuffle=False)
    train_dl = DataLoader(Ch4DualStreamDataset(config, mode='train'),
                          batch_size=config.BATCH_SIZE, shuffle=True)

    results = []

    for model_name in MODELS_TO_COMPARE:
        print(f"\n>>> [Task] Evaluating Model: {model_name}")

        # 尝试复用 RQ1 训练好的权重
        ckpt_dir = os.path.join("checkpoints_ch4", "rq1", model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "model.pth")

        # 初始化模型
        config.MODEL_NAME = model_name
        try:
            model = get_model(model_name, config).to(device)
        except Exception as e:
            print(f"    [Error] Init failed: {e}")
            continue

        # 加载权重
        if os.path.exists(ckpt_path):
            print(f"    [Info] Loading existing checkpoint...")
            try:
                # weights_only=False 兼容旧版
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            except:
                print("    [Warn] Load failed. Retraining...")
                os.remove(ckpt_path)

        # 如果权重不存在（或加载失败），则现场补训
        if not os.path.exists(ckpt_path):
            print(f"    [Info] Training {model_name} now...")
            trainer = RQ5Trainer(config, model, device)
            epochs = 20  # 快速补训
            for epoch in range(epochs):
                trainer.train_epoch(train_dl)
            torch.save(model.state_dict(), ckpt_path)
            print(f"    [Info] Training done.")

        row = {'Model': model_name}

        # 遍历 SNR 等级
        for snr in SNR_LEVELS:
            snr_name = f"{snr}dB" if snr is not None else "Clean"
            acc = evaluate_with_noise(model, test_dl, device, snr)
            row[f"Acc_{snr_name}"] = acc
            print(f"    -> SNR {snr_name}: {acc:.2f}%")

        results.append(row)

    # 汇总与绘图
    if results:
        df = pd.DataFrame(results)
        # 整理列顺序
        cols = ['Model', 'Acc_-5dB', 'Acc_0dB', 'Acc_5dB', 'Acc_10dB', 'Acc_Clean']
        # 补全缺失列
        for c in cols:
            if c not in df.columns: df[c] = 0.0

        csv_path = os.path.join(rq5_root, "RQ5_Noise_Results.csv")
        df[cols].to_csv(csv_path, index=False)
        print(f"\n>>> 汇总表已保存: {csv_path}")

        print(">>> Generating Charts...")
        plot_noise_robustness_chart(csv_path, rq5_root)

    print("\n=========================================================")
    print("   RQ5 Completed! See checkpoints_ch4/rq5/")
    print("=========================================================")


if __name__ == "__main__":
    main()