import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.phys_rdlinear_cls import PhysRDLinearCls
from utils.loss_ch4 import MultiTaskLoss
from data_loader_ch4 import Ch4DualStreamDataset
from config import Config

# ==============================================================================
# 1. 实验场景注册表 (SCENARIO REGISTRY) - 论文实验设计专用
# ==============================================================================

# 定义通用的训练集：200kg 下的 4 种典型工况 (覆盖低速、高速、加速、减速)
# Source Domain: Load 200kg
SOURCE_TRAIN_ATOMS = [
    (200, '15'), (200, '45'), (200, '15-45'), (200, '45-15')
]

# 定义所有 8 种工况的列表 (用于全工况测试)
ALL_SPEED_ATOMS_CODES = [
    '15', '30', '45', '60',       # 稳态
    '15-45', '30-60', '45-15', '60-30' # 变态
]

# main.py 中的场景注册表更新
SCENARIOS = {
    'challenge_transfer': {
        'description': '高难度迁移: 200kg(源域)训练 -> 0kg/400kg(目标域)泛化',
        'train_atoms': [(200, '15'), (200, '45'), (200, '15-45'), (200, '45-15')],
        'test_atoms':  [(0, s) for s in ALL_SPEED_ATOMS_CODES] + [(400, s) for s in ALL_SPEED_ATOMS_CODES]
    }
}

def run_ch4_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> [Chapter 4] Starting Leave-One-Domain-Out Experiment on {device}")

    # 1. 准备数据
    train_loader = DataLoader(Ch4DualStreamDataset(mode='train'), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Ch4DualStreamDataset(mode='test'), batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. 初始化 Phys-RDLinear
    model = PhysRDLinearCls(num_classes=Config.NUM_CLASSES, freq_dim=512).to(device)
    mtl_loss = MultiTaskLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 3. 自动化训练循环
    for epoch in range(Config.CH4_EPOCHS):
        model.train()
        for micro_x, speed, y_cls, y_load in train_loader:
            micro_x, speed, y_cls, y_load = micro_x.to(device), speed.to(device), y_cls.to(device), y_load.to(device)

            optimizer.zero_grad()
            logits, pred_load = model(micro_x, speed)
            loss, l_cls, l_reg = mtl_loss(logits, y_cls, pred_load, y_load)
            loss.backward()
            optimizer.step()

        # 4. 自动化未见域测试 (实事求是验证泛化性)
        if (epoch + 1) % 10 == 0:
            acc = evaluate_unseen_domain(model, test_loader, device)
            print(f"Epoch {epoch + 1} | Unseen Domain (200kg) Acc: {acc:.2f}% | Cls_Loss: {l_cls:.4f}")


def evaluate_unseen_domain(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for micro_x, speed, y_cls, _ in loader:
            micro_x, speed, y_cls = micro_x.to(device), speed.to(device), y_cls.to(device)
            logits, _ = model(micro_x, speed)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)
    return 100 * correct / total


if __name__ == "__main__":
    run_ch4_experiment()