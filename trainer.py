import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from typing import Dict
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

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for micro_x, macro_x, speed, y_cls, y_load in dataloader:
            # 移动数据到设备
            micro_x, macro_x = micro_x.to(self.device), macro_x.to(self.device)
            speed, y_cls, y_load = speed.to(self.device), y_cls.to(self.device), y_load.to(self.device)

            self.optimizer.zero_grad()

            if self.config.MODEL_NAME.startswith('Phys-RDLinear') or self.config.MODEL_NAME.startswith('Ablation'):
                # 我们的模型家族
                logits, pred_load = self.model(micro_x, macro_x, speed)

                if pred_load is not None:
                    # 开启了 MTL：计算总损失
                    loss, _, _ = self.criterion(logits, y_cls, pred_load, y_load)
                else:
                    # 关闭了 MTL：只算分类损失 (CrossEntropy)
                    # 注意：criterion 是 UncertaintyLoss，它期望有 pred_load
                    # 为了简单，直接用 PyTorch 的 CE Loss
                    loss = torch.nn.functional.cross_entropy(logits, y_cls)
            else:
                # 基线模型
                full_x = torch.cat([micro_x, macro_x], dim=1)
                logits, _ = self.model(full_x, speed)
                loss = torch.nn.functional.cross_entropy(logits, y_cls)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

        return {"loss": total_loss / len(dataloader), "acc": 100 * correct / total}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, save_path: str = None) -> float:
        self.model.eval()
        correct, total = 0, 0
        results = []

        for micro_x, macro_x, speed, y_cls, y_load in dataloader:
            micro_x, macro_x = micro_x.to(self.device), macro_x.to(self.device)
            speed, y_cls = speed.to(self.device), y_cls.to(self.device)

            # === [核心修改] 修改判断条件，支持 Ablation 系列模型 ===
            if self.config.MODEL_NAME.startswith('Phys-RDLinear') or self.config.MODEL_NAME.startswith('Ablation'):
                # 我们的模型（无论是完整版还是消融版）都需要双流输入
                logits, _ = self.model(micro_x, macro_x, speed)
            else:
                # 基线模型：继续使用拼接输入
                full_x = torch.cat([micro_x, macro_x], dim=1)
                logits, _ = self.model(full_x, speed)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

            # 收集详细数据 (转回CPU)
            # 注意：这里我们需要反推 Load 和 Speed 的物理值用于人类可读
            # y_load 是归一化的，speed 是 Hz
            b_load = (y_load.cpu().numpy().flatten() * 400).round().astype(int)
            b_speed = (speed.cpu().numpy().flatten() * 60).round().astype(int)  # 假设传入的是 Hz, *60 变 RPM? 不，你之前代码是 Hz
            # 你的 Dataset 里 speed_hz = data['speed'] / 60.0，所以这里 *60 还原为 RPM
            # 或者直接存 Hz。为了报表好看，我们直接用 Dataset 里的原始信息可能更准，
            # 但这里作为验证，反推即可。

            b_true = y_cls.cpu().numpy()
            b_pred = preds.cpu().numpy()

            for i in range(len(b_true)):
                results.append({
                    'Load_kg': b_load[i],
                    'Speed_Hz': b_speed[i],  # 这里的逻辑可能需要根据你 dataset 的归一化调整
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

            # 顺便生成一个按故障类型的统计表
            cls_report = df.groupby('True_Label')['Is_Correct'].mean()
            cls_report.to_csv(save_path.replace('.csv', '_per_class.csv'))

        return 100 * correct / total