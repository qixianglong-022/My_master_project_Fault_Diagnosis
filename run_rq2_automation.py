# run_rq2_advanced.py
import os
import torch
import torch.nn.functional as F  # <--- [新增] 引入函数式接口
import pandas as pd
from torch.utils.data import DataLoader

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
from utils.rq1_kit import evaluate_rq1_comprehensive  # 复用 RQ1 的评估逻辑
from utils.rq2_kit import plot_rq2_bar_chart  # RQ2 专用绘图

# === 实验配置 ===
# 定义三种对比实验配置
EXPERIMENTS = {
    'Micro_Only': {
        'desc': '仅低频 (1kHz)',
        'mask_macro': True,  # 屏蔽高频
        'mask_micro': False,
        'mask_audio': True  # 屏蔽声纹
    },
    'Macro_Only': {
        'desc': '仅高频 (51.2kHz)',
        'mask_macro': False,
        'mask_micro': True,  # 屏蔽低频
        'mask_audio': True
    },
    'Multi_Res': {
        'desc': '多分辨率融合 (Ours)',
        'mask_macro': False,
        'mask_micro': False,
        'mask_audio': True  # 全开
    }
}
GPU_ID = 0


class MaskedTrainer(Trainer):
    """
    继承 Trainer，增加数据掩码功能
    在训练和测试时将特定通道置零
    """

    def __init__(self, config, model, device, mask_cfg):
        super().__init__(config, model, device)
        self.mask_cfg = mask_cfg

    def _apply_mask(self, micro, macro, ac):
        """核心掩码逻辑"""
        # 注意：这里我们使用 clone() 避免修改原始数据，虽然在循环里通常不需要
        # 但为了安全起见，直接覆盖变量即可
        if self.mask_cfg['mask_micro']:
            micro = torch.zeros_like(micro)
        if self.mask_cfg['mask_macro']:
            macro = torch.zeros_like(macro)
        if self.mask_cfg['mask_audio']:
            ac = torch.zeros_like(ac)
        return micro, macro, ac

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for micro, macro, ac, spd, y_cls, y_load in dataloader:
            micro = micro.to(self.device)
            macro = macro.to(self.device)
            ac = ac.to(self.device)
            spd = spd.to(self.device)
            y_cls = y_cls.to(self.device)
            y_load = y_load.to(self.device).float().unsqueeze(1)

            # === Apply Mask (应用掩码) ===
            micro, macro, ac = self._apply_mask(micro, macro, ac)

            self.optimizer.zero_grad()
            logits, pred_load = self.model(micro, macro, ac, spd)

            # === Loss Calculation (修复点: 使用 F 接口) ===
            # 不再依赖 self.criterion_cls，直接调用 PyTorch 标准函数
            loss_cls = F.cross_entropy(logits, y_cls)

            loss = loss_cls  # 基础损失

            if pred_load is not None:
                loss_reg = F.mse_loss(pred_load, y_load)
                # 使用固定权重 1:1，确保消融实验的公平性
                # (自适应 Loss 在某些通道被 Mask 时可能会不稳定)
                loss = 0.5 * loss_cls + 0.5 * loss_reg

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {'loss': total_loss / len(dataloader)}


def evaluate_masked(model, dataloader, device, save_dir, cfg_name, mask_cfg):
    """
    带掩码的评估函数: 使用 Hook 机制临时修改 forward
    """
    # 保存原始 forward
    original_forward = model.forward

    # 定义带掩码的 forward
    def masked_forward(micro, macro, ac, spd):
        if mask_cfg['mask_micro']: micro = torch.zeros_like(micro)
        if mask_cfg['mask_macro']: macro = torch.zeros_like(macro)
        if mask_cfg['mask_audio']: ac = torch.zeros_like(ac)
        return original_forward(micro, macro, ac, spd)

    # 临时替换
    model.forward = masked_forward

    try:
        # 调用标准评估 (它会调用被我们替换过的 forward)
        print(f"    [Eval] Evaluating {cfg_name} on Target Domain...")
        metrics = evaluate_rq1_comprehensive(model, dataloader, device, save_dir, cfg_name)
    finally:
        # 务必还原 forward，防止污染后续实验
        model.forward = original_forward

    return metrics


def main():
    print("=========================================================")
    print("   RQ2 Automation: Resolution Ablation Study")
    print("=========================================================")

    rq2_root = os.path.join("checkpoints_ch4", "rq2")
    os.makedirs(rq2_root, exist_ok=True)

    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'  # RQ2 始终使用该模型
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_dl = DataLoader(Ch4DualStreamDataset(config, mode='train'),
                          batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, mode='test'),
                         batch_size=config.BATCH_SIZE, shuffle=False)

    final_results = []

    for cfg_name, mask_settings in EXPERIMENTS.items():
        print(f"\n>>> [Task] Running Experiment: {cfg_name} ({mask_settings['desc']})")

        save_dir = os.path.join(rq2_root, cfg_name)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 初始化模型 (每次重置，保证独立性)
        model = get_model('Phys-RDLinear', config).to(device)

        # 2. 训练 (带 Mask)
        trainer = MaskedTrainer(config, model, device, mask_settings)

        ckpt_path = os.path.join(save_dir, "model.pth")
        if os.path.exists(ckpt_path):
            print("    [Info] Loading existing checkpoint...")
            model.load_state_dict(torch.load(ckpt_path))
        else:
            print("    [Info] Training with mask...")
            # 消融实验通常收敛较快，跑 30 个 Epoch 足够
            epochs = 30
            for epoch in range(epochs):
                metrics = trainer.train_epoch(train_dl)
                if (epoch + 1) % 10 == 0:
                    print(f"        Epoch {epoch + 1}/{epochs} | Loss: {metrics['loss']:.4f}")
            torch.save(model.state_dict(), ckpt_path)

        # 3. 评估 (带 Mask)
        row = evaluate_masked(model, test_dl, device, save_dir, cfg_name, mask_settings)
        row['Config'] = cfg_name
        row['Description'] = mask_settings['desc']
        final_results.append(row)

        print(f"    -> Avg Acc: {row['Avg_Acc']:.2f}%")

    # 4. 汇总与绘图
    if final_results:
        df = pd.DataFrame(final_results)
        # 整理列
        cols = ['Config', 'Description', 'Avg_Acc', 'Acc_0kg', 'Acc_400kg']
        rest = [c for c in df.columns if c not in cols]
        df[cols + rest].to_csv(os.path.join(rq2_root, "RQ2_Resolution_Analysis.csv"), index=False)

        print("\n>>> [Plot] Generating Comparison Charts...")
        plot_rq2_bar_chart(
            os.path.join(rq2_root, "RQ2_Resolution_Analysis.csv"),
            rq2_root
        )

    print("\n=========================================================")
    print("   RQ2 Completed! See checkpoints/rq2/")
    print("=========================================================")


if __name__ == "__main__":
    main()