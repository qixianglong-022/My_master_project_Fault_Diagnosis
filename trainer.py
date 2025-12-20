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
        # 注意这里解包出来的是 ld_proxy
        for batch_idx, (mic, mac, ac, cur, spd, ld_proxy, label) in enumerate(dataloader):
            mic, mac = mic.to(self.device), mac.to(self.device)
            ac, cur = ac.to(self.device), cur.to(self.device)

            # [Fix] 将 ld 改为 ld_proxy，保持变量名一致
            spd, ld_proxy = spd.to(self.device), ld_proxy.to(self.device)

            label = label.to(self.device)

            self.optimizer.zero_grad()

            # Forward (V2 接口)
            if self.config.MODEL_NAME == 'Phys-RDLinear':
                # 确保传入模型的也是移动到 device 后的 ld_proxy
                logits, pred_load = self.model(mic, mac, ac, cur, spd, ld_proxy)
            else:
                # 兼容基线 (需自行适配基线模型的 forward)
                logits = self.model(mic) # 或者是 ac

            # Loss (仅分类)
            loss = self.criterion(logits, label)

            # 只有当模型真的输出了 pred_load (即 MTL 模式开启) 时才计算回归损失
            # 在新的设计中，pred_load 始终为 None，所以这里只计算分类 Loss
            if pred_load is not None:
                # 如果未来想做自监督，这里 ld_proxy 也可以作为 target
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