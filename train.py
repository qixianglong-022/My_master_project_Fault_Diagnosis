import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from config import Config
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from utils.anomaly import InferenceEngine


def train():
    # 1. Setup
    print(f"Model: {Config.MODEL_NAME}, Device: CUDA")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 2. Data
    train_ds = MotorDataset(Config.TRAIN_ATOMS, mode='train')
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Model & Opt
    model = RDLinear().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    # 4. Loop
    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = []
        for x, y, cov, _ in train_dl:
            x, y, cov = x.cuda(), y.cuda(), cov.cuda()

            optimizer.zero_grad()
            pred = model(x, cov)
            loss = criterion(pred, y)  # Forecasting Loss

            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{Config.EPOCHS} | Loss: {avg_loss:.6f}")

    # 5. Save Model
    save_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 6. Auto-fit Threshold & Fusion Params
    # 使用训练集(的无Shuffle版)来确定统计特性
    eval_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    engine = InferenceEngine(model)
    engine.fit_threshold(eval_dl)


if __name__ == '__main__':
    train()