import torch
import torch.nn as nn
from config import Ch4Config
import os

class Trainer:
    def __init__(self, config: Ch4Config, model, device: torch.device):
        self.config = config
        self.model = model
        self.device = device

        # [NEW] 类别加权 CE Loss
        class_weights = torch.tensor(config.CLASS_WEIGHTS).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # [NEW] Unpack 7 items
        # 这里的 phys_load 在训练时用不到，直接忽略即可
        for batch_idx, (mic, mac, ac, cur, spd, ld_proxy, label, _) in enumerate(dataloader):
            mic, mac = mic.to(self.device), mac.to(self.device)
            ac, cur = ac.to(self.device), cur.to(self.device)
            spd, ld_proxy = spd.to(self.device), ld_proxy.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            # [修复 1] 预先初始化 pred_load，防止 UnboundLocalError
            pred_load = None

            # Forward (V2 接口)
            if self.config.MODEL_NAME == 'Phys-RDLinear':
                # Phys 模型: 接收所有物理参数
                logits, pred_load = self.model(mic, mac, ac, cur, spd, ld_proxy)
            else:
                # [修复 2] 基线模型通用调用：传入 speed (TiDE 必选，其他可选但传入无害)
                out = self.model(mic, speed=spd)

                # [修复 3] 智能解包：如果返回 (logits, aux)，只取 logits
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out

            # Loss (仅分类)
            loss = self.criterion(logits, label)

            # 只有当模型真的输出了 pred_load (即 MTL 模式开启) 时才计算回归损失
            if pred_load is not None:
                loss_reg = nn.MSELoss()(pred_load, ld_proxy)
                loss += 0.5 * loss_reg

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        return {
            "loss": total_loss / len(dataloader),
            "acc": 100.0 * correct / total
        }


    @torch.no_grad()
    def evaluate(self, dataloader, save_path: str = None) -> float:
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