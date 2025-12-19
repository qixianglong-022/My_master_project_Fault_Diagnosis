import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from config import Ch4Config
from utils.uncertainty_loss import UncertaintyLoss

class Trainer:
    def __init__(self, config: Ch4Config, model, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.criterion = UncertaintyLoss().to(device)

        # 优化器设置
        self.optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': self.criterion.parameters(), 'lr': 1e-3}
        ], lr=config.LEARNING_RATE)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (micro, macro, ac, spd, y_cls, y_load) in enumerate(dataloader):
            micro, macro = micro.to(self.device), macro.to(self.device)
            ac, spd = ac.to(self.device), spd.to(self.device)
            y_cls, y_load = y_cls.to(self.device), y_load.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            if self.config.MODEL_NAME.startswith('Phys') or self.config.MODEL_NAME.startswith('Ablation'):
                logits, pred_load = self.model(micro, macro, ac, spd)
            else:
                # Baselines: Concat inputs (Micro + Macro)
                # 确保基线模型也能处理 spd 输入 (我们在 models/baselines_ch4.py 中已经做了兼容)
                full_x = torch.cat([micro.squeeze(-1), macro.squeeze(-1)], dim=1)
                logits, pred_load = self.model(full_x, spd)

            # Loss Calculation
            if pred_load is not None:
                # MTL (Phys-RDLinear 或 Ablation-MTL)
                loss, l_cls, l_reg = self.criterion(logits, y_cls, pred_load, y_load)
            else:
                # Single Task (Baseline)
                loss = torch.nn.functional.cross_entropy(logits, y_cls)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # 计算准确率 (修复 KeyError: 'acc')
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

        # 返回字典，包含 loss 和 acc
        return {
            "loss": total_loss / len(dataloader),
            "acc": 100.0 * correct / total
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, save_path: str = None) -> float:
        self.model.eval()
        correct, total = 0, 0
        results = []

        for micro_x, macro_x, ac, speed, y_cls, y_load in dataloader:
            micro_x, macro_x = micro_x.to(self.device), macro_x.to(self.device)
            ac, speed = ac.to(self.device), speed.to(self.device)
            y_cls = y_cls.to(self.device)

            # Forward logic matching train_epoch
            if self.config.MODEL_NAME.startswith('Phys') or self.config.MODEL_NAME.startswith('Ablation'):
                logits, _ = self.model(micro_x, macro_x, ac, speed)
            else:
                full_x = torch.cat([micro_x.squeeze(-1), macro_x.squeeze(-1)], dim=1)
                logits, _ = self.model(full_x, speed)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

            # 收集结果
            b_true = y_cls.cpu().numpy()
            b_pred = preds.cpu().numpy()
            b_load = (y_load.cpu().numpy().flatten() * 400).round().astype(int)

            for i in range(len(b_true)):
                results.append({
                    'Load_kg': b_load[i],
                    'True_Label': b_true[i],
                    'Pred_Label': b_pred[i],
                    'Is_Correct': 1 if b_true[i] == b_pred[i] else 0
                })

        # 保存详细报表
        if save_path:
            import pandas as pd
            df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"    -> Detailed report saved to: {os.path.basename(save_path)}")

        if total == 0: return 0.0
        return 100 * correct / total