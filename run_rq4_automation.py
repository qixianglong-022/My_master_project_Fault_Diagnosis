import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
from utils.rq1_kit import evaluate_rq1_comprehensive
from utils.rq3_kit import compute_mmd  # 复用 MMD 计算
from utils.rq4_kit import plot_covariate_analysis

# ==============================================================================
# 1. 实验配置: 协变量消融
# ==============================================================================
# 控制变量：是否使用 转速(Speed) 和 负载(Load)
# False = 置零 (Mask), True = 正常输入
experiments = {
    'No_Covariates': {
        'use_speed': False,
        'use_load': False,
        'desc': '无协变量 (Baseline)'
    },
    'Speed_Only': {
        'use_speed': True,
        'use_load': False,
        'desc': '仅转速 (Speed)'
    },
    'Load_Only': {
        'use_speed': False,
        'use_load': True,
        'desc': '仅负载 (Load)'
    },
    'Full_Covariates': {
        'use_speed': True,
        'use_load': True,
        'desc': '双协变量 (Full)'
    }
}
GPU_ID = 0


# ==============================================================================
# 2. 协变量专用训练器
# ==============================================================================
class CovariateTrainer(Trainer):
    """
    RQ4 专用训练器
    功能：在训练过程中对 Speed 或 Load 进行 Mask (置零)
    """

    def __init__(self, config, model, device, cov_cfg):
        super().__init__(config, model, device)
        self.cov_cfg = cov_cfg

    def _apply_covariate_mask(self, spd, ld):
        """
        核心掩码逻辑
        """
        # 如果配置为不使用，则将其置为 0 (模拟缺失或未感知)
        if not self.cov_cfg['use_speed']:
            spd = torch.zeros_like(spd)

        if not self.cov_cfg['use_load']:
            ld = torch.zeros_like(ld)

        return spd, ld

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        # [适配] 解包 8 个变量
        for micro, macro, ac, cur, spd, y_load, y_cls, _ in dataloader:

            # 1. 数据迁移
            micro = micro.to(self.device)
            macro = macro.to(self.device)
            ac = ac.to(self.device)
            cur = cur.to(self.device)
            spd = spd.to(self.device)
            y_cls = y_cls.to(self.device)
            y_load = y_load.to(self.device).float()

            # 2. 应用协变量掩码 (Ablation)
            spd, y_load_input = self._apply_covariate_mask(spd, y_load.clone())

            # 注意: y_load 用于回归 Loss 时不需要 mask (它是标签)
            # 但作为 input 输入给模型时需要 mask (y_load_input)

            self.optimizer.zero_grad()

            # 3. 前向传播 (6 参数)
            logits, pred_load = self.model(micro, macro, ac, cur, spd, y_load_input)

            # 4. 计算损失
            loss_cls = F.cross_entropy(logits, y_cls)
            loss = loss_cls

            # 只有当 'use_load' 为 True 时，MTL 回归才有意义
            # 如果 Mask 了负载输入，通常也意味着无法进行负载回归(或者是盲猜)，这里为了公平，
            # 只有在 Full 或 Load_Only 模式下才计算回归 Loss
            if self.cov_cfg['use_load'] and pred_load is not None:
                loss_reg = F.mse_loss(pred_load, y_load)  # y_load 是真实标签
                loss = 0.5 * loss_cls + 0.5 * loss_reg

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {'loss': total_loss / len(dataloader)}


# ==============================================================================
# 3. 带掩码的评估函数
# ==============================================================================
def evaluate_covariates(model, dataloader, device, save_dir, cfg_name, cov_cfg):
    """
    通过 Hook 方式 Mask 协变量进行评估
    """
    print(f"    [Eval] Evaluating {cfg_name}...")

    original_forward = model.forward

    # 定义临时 Forward
    def masked_forward(micro, macro, acoustic, cur, speed, load_proxy):
        # 掩码逻辑
        if not cov_cfg['use_speed']:
            speed = torch.zeros_like(speed)
        if not cov_cfg['use_load']:
            load_proxy = torch.zeros_like(load_proxy)

        return original_forward(micro, macro, acoustic, cur, speed, load_proxy)

    model.forward = masked_forward

    try:
        # 复用 RQ1 评估
        metrics = evaluate_rq1_comprehensive(model, dataloader, device, save_dir, cfg_name)
    finally:
        model.forward = original_forward

    return metrics


def compute_mmd_covariates(model, src_dl, tgt_dl, device, cov_cfg):
    """
    带掩码的 MMD 计算
    """
    original_forward = model.forward

    def masked_forward(micro, macro, acoustic, cur, speed, load_proxy):
        if not cov_cfg['use_speed']:
            speed = torch.zeros_like(speed)
        if not cov_cfg['use_load']:
            load_proxy = torch.zeros_like(load_proxy)
        return original_forward(micro, macro, acoustic, cur, speed, load_proxy)

    model.forward = masked_forward
    try:
        val = compute_mmd(model, src_dl, tgt_dl, device)
    finally:
        model.forward = original_forward
    return val


# ==============================================================================
# 4. 主流程
# ==============================================================================
def main():
    print("=========================================================")
    print("   RQ4 Automation: Covariate Importance Analysis")
    print("=========================================================")

    rq4_root = os.path.join("checkpoints_ch4", "rq4")
    os.makedirs(rq4_root, exist_ok=True)

    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'  # 始终使用完整模型架构
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 数据准备
    train_dl = DataLoader(Ch4DualStreamDataset(config, mode='train'),
                          batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, mode='test'),
                         batch_size=config.BATCH_SIZE, shuffle=False)

    # MMD Loader
    mmd_src = DataLoader(Ch4DualStreamDataset(config, mode='train'), batch_size=32, shuffle=True)
    mmd_tgt = DataLoader(Ch4DualStreamDataset(config, mode='test'), batch_size=32, shuffle=True)

    results = []

    for cfg_name, cov_cfg in experiments.items():
        print(f"\n>>> [Task] Running: {cfg_name}")

        save_dir = os.path.join(rq4_root, cfg_name)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 初始化模型
        model = get_model('Phys-RDLinear', config).to(device)

        # 2. 训练 (带 Covariate Mask)
        trainer = CovariateTrainer(config, model, device, cov_cfg)
        ckpt_path = os.path.join(save_dir, "model.pth")

        if os.path.exists(ckpt_path):
            print("    [Info] Loading checkpoint...")
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            except:
                os.remove(ckpt_path)

        if not os.path.exists(ckpt_path):
            print("    [Info] Training with covariates masked...")
            epochs = 25
            for epoch in range(epochs):
                trainer.train_epoch(train_dl)
            torch.save(model.state_dict(), ckpt_path)

        # 3. 评估准确率
        row = evaluate_covariates(model, test_dl, device, save_dir, cfg_name, cov_cfg)

        # 4. 计算 MMD
        print("    [Eval] Calculating MMD...")
        mmd_val = compute_mmd_covariates(model, mmd_src, mmd_tgt, device, cov_cfg)

        row['Config'] = cfg_name
        row['Description'] = cov_cfg['desc']
        row['MMD'] = mmd_val
        row['Speed'] = cov_cfg['use_speed']
        row['Load'] = cov_cfg['use_load']

        results.append(row)
        print(f"    -> Acc: {row['Avg_Acc']:.2f}%, MMD: {mmd_val:.4f}")

    # 5. 汇总
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(rq4_root, "RQ4_Covariate_Analysis.csv")
        # 整理列顺序
        cols = ['Config', 'Description', 'Avg_Acc', 'MMD', 'Acc_0kg', 'Acc_400kg']
        rest = [c for c in df.columns if c not in cols]
        df[cols + rest].to_csv(csv_path, index=False)
        print(f"\n>>> Saved: {csv_path}")

        # 绘图
        plot_covariate_analysis(csv_path, rq4_root)

    print("\n=========================================================")
    print("   RQ4 Completed!")
    print("=========================================================")


if __name__ == "__main__":
    main()