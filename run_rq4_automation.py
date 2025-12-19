import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.phys_rdlinear import PhysRDLinearCls  # 直接调用类
from trainer import Trainer
from utils.tools import set_seed


class MultimodalTrainer(Trainer):
    def __init__(self, config, model, device, mode='fusion'):
        super().__init__(config, model, device)
        self.mode = mode  # 'vib_only', 'acoustic_only', 'fusion'

    def _mask_modalities(self, micro, macro, acoustic):
        if self.mode == 'vib_only':
            # 屏蔽声纹
            acoustic = torch.zeros_like(acoustic)
        elif self.mode == 'acoustic_only':
            # 屏蔽振动
            micro = torch.zeros_like(micro)
            macro = torch.zeros_like(macro)
        return micro, macro, acoustic

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        # [Fix] 解包 6 个
        for micro, macro, acoustic, speed, y_cls, y_load in dataloader:
            micro, macro = micro.to(self.device), macro.to(self.device)
            acoustic, speed = acoustic.to(self.device), speed.to(self.device)
            y_cls, y_load = y_cls.to(self.device), y_load.to(self.device)

            micro, macro, acoustic = self._mask_modalities(micro, macro, acoustic)

            self.optimizer.zero_grad()
            logits, pred_load = self.model(micro, macro, acoustic, speed)

            # RQ4 默认开启 MTL
            loss, _, _ = self.criterion(logits, y_cls, pred_load, y_load)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == y_cls).sum().item()
            total += y_cls.size(0)

        return {"loss": total_loss / len(dataloader), "acc": 100 * correct / total}

    @torch.no_grad()
    def evaluate(self, dataloader, save_path=None):
        self.model.eval()
        correct, total = 0, 0
        results = []

        for micro, macro, acoustic, speed, y_cls, y_load in dataloader:
            micro, macro = micro.to(self.device), macro.to(self.device)
            acoustic, speed = acoustic.to(self.device), speed.to(self.device)

            micro, macro, acoustic = self._mask_modalities(micro, macro, acoustic)

            logits, _ = self.model(micro, macro, acoustic, speed)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

            # 保存用于F1计算
            b_true = y_cls.cpu().numpy()
            b_pred = preds.cpu().numpy()
            for i in range(len(b_true)):
                results.append({'True': b_true[i], 'Pred': b_pred[i]})

        if save_path:
            pd.DataFrame(results).to_csv(save_path, index=False)
        return 100 * correct / total


def run_rq4_task(config_mode, gpu_id=0):
    config = Ch4Config()
    config.MODEL_NAME = f"Multimodal_{config_mode}"
    config.CHECKPOINT_DIR = os.path.join("checkpoints_rq4", config.MODEL_NAME)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    set_seed(config.SEED)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> [RQ4] Mode: {config_mode}")

    train_dl = DataLoader(Ch4DualStreamDataset(config, 'train'), batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, 'test'), batch_size=config.BATCH_SIZE, shuffle=False)

    # 必须开启声纹
    model = PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=True, enable_acoustic=True).to(device)

    trainer = MultimodalTrainer(config, model, device, mode=config_mode)

    for epoch in range(20):
        trainer.train_epoch(train_dl)

    csv_path = os.path.join(config.CHECKPOINT_DIR, "eval_rq4.csv")
    acc = trainer.evaluate(test_dl, save_path=csv_path)
    print(f"    Acc: {acc:.2f}%")

    return acc


if __name__ == "__main__":
    modes = ['vib_only', 'acoustic_only', 'fusion']
    for m in modes:
        run_rq4_task(m)