import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict

from config import Ch4Config
from models.phys_rdlinear import PhysRDLinearCls
from utils.uncertainty_loss import UncertaintyLoss
import os

class Trainer:
    def __init__(self, config: Ch4Config, model: PhysRDLinearCls, device: torch.device):
        self.config = config
        self.model = model
        self.device = device

        self.criterion = UncertaintyLoss().to(device)
        # Optimize both model and loss parameters (sigma)
        self.optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': self.criterion.parameters(), 'lr': 1e-3}
        ], lr=config.LEARNING_RATE)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_cls, total_reg = 0, 0, 0
        correct = 0
        total = 0

        for micro_x, speed, y_cls, y_load in dataloader:
            # Move to device
            micro_x = micro_x.to(self.device)
            speed = speed.to(self.device)
            y_cls = y_cls.to(self.device)
            y_load = y_load.to(self.device)

            # Forward
            logits, pred_load = self.model(micro_x, speed)

            # Loss Calculation
            loss, l_cls, l_reg = self.criterion(logits, y_cls, pred_load, y_load)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_cls += l_cls
            total_reg += l_reg

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

        return {
            "loss": total_loss / len(dataloader),
            "cls_loss": total_cls / len(dataloader),
            "reg_loss": total_reg / len(dataloader),
            "acc": 100 * correct / total
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0

        for micro_x, speed, y_cls, _ in dataloader:
            micro_x = micro_x.to(self.device)
            speed = speed.to(self.device)
            y_cls = y_cls.to(self.device)

            logits, _ = self.model(micro_x, speed)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

        return 100 * correct / total

    def save(self, name: str = "best_model.pth"):
        path = os.path.join(self.config.CHECKPOINT_DIR, name)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")