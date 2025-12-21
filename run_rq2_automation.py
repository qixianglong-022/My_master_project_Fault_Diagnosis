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
from utils.rq2_kit import plot_rq2_bar_chart

# ==============================================================================
# 1. 实验配置 (Ablation Study Settings)
# ==============================================================================
# 定义消融实验配置：控制哪些模态参与训练/推理
# True = 保留, False = 置零 (Mask)
experiments = {
    'Micro_Only': {
        'micro': True,
        'macro': False,
        'audio': False,
        'current': False
    },
    'Audio_Only': {
        'micro': False,
        'macro': False,
        'audio': True,
        'current': False
    },
    'Current_Only': {  # 新增：纯电流模态
        'micro': False,
        'macro': False,
        'audio': False,
        'current': True
    },
    'Vib_Plus_Audio': {  # 振动 + 声纹 (无电流)
        'micro': True,
        'macro': True,
        'audio': True,
        'current': False
    },
    'Phys-RDLinear': {  # 全模态 (完整模型)
        'micro': True,
        'macro': True,
        'audio': True,
        'current': True
    }
}
GPU_ID = 0


# ==============================================================================
# 2. Masked Trainer (带掩码的训练器)
# ==============================================================================
class MaskedTrainer(Trainer):
    """
    继承 Trainer，增加数据掩码功能
    在训练时将特定通道置零，实现模态消融
    """

    def __init__(self, config, model, device, mask_cfg, exp_name):
        super().__init__(config, model, device)
        self.mask_cfg = mask_cfg
        self.exp_name = exp_name  # 用于逻辑判断

    def _apply_mask(self, micro, macro, ac, cur):
        """
        核心掩码逻辑：根据配置将 Tensor 置零
        """
        # 注意：使用 key.get(name, True) 默认为 True (不遮蔽)
        if not self.mask_cfg.get('micro', True):
            micro = torch.zeros_like(micro)

        if not self.mask_cfg.get('macro', True):
            macro = torch.zeros_like(macro)

        if not self.mask_cfg.get('audio', True):
            ac = torch.zeros_like(ac)

        if not self.mask_cfg.get('current', True):
            cur = torch.zeros_like(cur)

        return micro, macro, ac, cur

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        # [修复] 解包 8 个变量 (适配最新 DataLoader)
        for micro, macro, ac, cur, spd, y_load, y_cls, _ in dataloader:

            # 1. 数据迁移
            micro = micro.to(self.device)
            macro = macro.to(self.device)
            ac = ac.to(self.device)
            cur = cur.to(self.device)
            spd = spd.to(self.device)
            y_cls = y_cls.to(self.device)

            # [修复] y_load 已经是 [B, 1]，不需要 unsqueeze
            y_load = y_load.to(self.device).float()

            # 2. 应用掩码 (消融实验核心)
            # 必须同时处理 cur，防止信息泄露
            micro, macro, ac, cur = self._apply_mask(micro, macro, ac, cur)

            self.optimizer.zero_grad()

            # 3. 模型前向传播 (传入所有 6 个参数)
            logits, pred_load = self.model(micro, macro, ac, cur, spd, y_load)

            # 4. 计算损失
            loss_cls = F.cross_entropy(logits, y_cls)
            loss = loss_cls

            if pred_load is not None:
                loss_reg = F.mse_loss(pred_load, y_load)
                # 固定权重，保证公平对比
                loss = 0.5 * loss_cls + 0.5 * loss_reg

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {'loss': total_loss / len(dataloader)}


# ==============================================================================
# 3. Masked Evaluation (带掩码的评估)
# ==============================================================================
def evaluate_masked(model, dataloader, device, save_dir, cfg_name, mask_settings):
    """
    带掩码的评估函数 (通过 Hook 替换 forward 实现)
    """
    print(f"    [Eval] Evaluating {cfg_name} on Target Domain...")

    # 1. 保存原始 forward
    original_forward = model.forward

    # 2. 定义带 Mask 的临时 forward (必须接收 6 个参数!)
    def masked_forward(micro, macro, acoustic, cur, speed, load_proxy):
        # 显式遮蔽逻辑
        if not mask_settings.get('micro', True):
            micro = torch.zeros_like(micro)

        if not mask_settings.get('macro', True):
            macro = torch.zeros_like(macro)

        if not mask_settings.get('audio', True):
            acoustic = torch.zeros_like(acoustic)

        if not mask_settings.get('current', True):
            cur = torch.zeros_like(cur)

        # 调用原始 forward
        return original_forward(micro, macro, acoustic, cur, speed, load_proxy)

    # 3. 临时替换 forward
    model.forward = masked_forward

    try:
        # 4. 调用通用评估流程 (复用 RQ1 代码)
        metrics = evaluate_rq1_comprehensive(model, dataloader, device, save_dir, cfg_name)
    except Exception as e:
        print(f"    [Error] Evaluation failed for {cfg_name}: {e}")
        model.forward = original_forward
        raise e
    finally:
        # 5. 务必恢复原始 forward
        model.forward = original_forward

    return metrics


# ==============================================================================
# 4. 主流程
# ==============================================================================
def main():
    print("=========================================================")
    print("   RQ2 Automation: Multi-modal Ablation Study")
    print("=========================================================")

    rq2_root = os.path.join("checkpoints_ch4", "rq2")
    os.makedirs(rq2_root, exist_ok=True)

    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'  # RQ2 始终基于该模型进行消融
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_dl = DataLoader(Ch4DualStreamDataset(config, mode='train'),
                          batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, mode='test'),
                         batch_size=config.BATCH_SIZE, shuffle=False)

    final_results = []

    for cfg_name, mask_settings in experiments.items():
        print(f"\n>>> [Task] Running Experiment: {cfg_name}")

        save_dir = os.path.join(rq2_root, cfg_name)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 初始化模型 (每次重置，保证独立性)
        model = get_model('Phys-RDLinear', config).to(device)

        # 2. 训练 (带 Mask)
        # 传入 exp_name 以便 Trainer 内部做额外判断(如果需要)
        trainer = MaskedTrainer(config, model, device, mask_settings, cfg_name)

        ckpt_path = os.path.join(save_dir, "model.pth")

        # 检查点逻辑：如果存在且完整则跳过训练
        if os.path.exists(ckpt_path):
            print("    [Info] Loading existing checkpoint...")
            # weights_only=False 是为了兼容旧版 PyTorch 习惯，虽然有安全警告但本地运行无妨
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            except:
                print("    [Warn] Checkpoint load failed, retraining...")
                os.remove(ckpt_path)

        if not os.path.exists(ckpt_path):
            print("    [Info] Training with mask...")
            epochs = 30  # 消融实验快速验证
            for epoch in range(epochs):
                metrics = trainer.train_epoch(train_dl)
                if (epoch + 1) % 10 == 0:
                    print(f"        Epoch {epoch + 1}/{epochs} | Loss: {metrics['loss']:.4f}")
            torch.save(model.state_dict(), ckpt_path)

        # 3. 评估 (带 Mask)
        row = evaluate_masked(model, test_dl, device, save_dir, cfg_name, mask_settings)

        # 记录结果
        row['Config'] = cfg_name
        # 使用配置名作为描述，不再依赖 desc 字段
        row['Description'] = cfg_name.replace('_', ' ')
        final_results.append(row)

        print(f"    -> Avg Acc: {row.get('Avg_Acc', 0):.2f}%")

    # 4. 汇总与绘图
    if final_results:
        df = pd.DataFrame(final_results)

        # 确保关键列存在
        cols = ['Config', 'Description', 'Avg_Acc', 'Acc_0kg', 'Acc_400kg']
        existing_cols = [c for c in cols if c in df.columns]
        rest_cols = [c for c in df.columns if c not in cols]

        csv_path = os.path.join(rq2_root, "RQ2_Ablation_Analysis.csv")
        df[existing_cols + rest_cols].to_csv(csv_path, index=False)
        print(f"\n    [Save] Results saved to {csv_path}")

        print("\n>>> [Plot] Generating Comparison Charts...")
        try:
            plot_rq2_bar_chart(csv_path, rq2_root)
        except Exception as e:
            print(f"    [Warn] Plotting failed: {e}")

    print("\n=========================================================")
    print("   RQ2 Completed! See checkpoints_ch4/rq2/")
    print("=========================================================")


if __name__ == "__main__":
    main()