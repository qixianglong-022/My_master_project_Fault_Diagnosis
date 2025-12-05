import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from utils.anomaly import InferenceEngine
from utils.tools import adjust_learning_rate, EarlyStopping
from config import Config

def plot_loss(train_loss, path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def plot_weights(model, path):
    """
    绘制 RDLinear 趋势分支的权重热力图
    这能直观展示 'Speed' 对 'Vibration/Audio' 的贡献
    """
    # 获取 Trend Linear 层的权重: [Out_Len, In_Len + Cov_Dim]
    # 我们主要关心 Covariates (转速) 对输出的影响
    if not hasattr(model, 'linear_trend'): return

    weights = model.linear_trend.weight.detach().cpu().numpy()  # [Seq_Len, Seq_Len + 2]
    # 我们只看最后两列 (转速权重) 对应 输出特征 的关系
    # 但由于 RDLinear 实现中是 Channel Independence 的...
    # 等等，RDLinear 的 Linear 层是共享权重的！
    # 所以 weights 形状是 [Seq_Len, Seq_Len + 2]
    # 它表示的是：对于任意一个通道，Speed 对其 Trend 的影响。

    plt.figure(figsize=(8, 6))
    # 以此展示 时间步 vs 权重
    sns.heatmap(weights, cmap='coolwarm', center=0)
    plt.title("Trend Branch Weights (Includes Speed Covariates)")
    plt.xlabel("Input Time Steps + Covariates")
    plt.ylabel("Output Time Steps")
    plt.savefig(path)
    plt.close()


def train_model():
    print(f"Model: {Config.MODEL_NAME}, Device: CUDA")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Data
    train_ds = MotorDataset(Config.TRAIN_ATOMS, mode='train')
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = RDLinear().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, verbose=True, path=Config.OUTPUT_DIR)

    loss_history = []

    model.train()
    for epoch in range(Config.EPOCHS):
        epoch_loss = []
        for x, y, cov, _ in train_dl:
            x, y, cov = x.cuda(), y.cuda(), cov.cuda()
            optimizer.zero_grad()

            # Forward
            pred = model(x, cov)
            loss = criterion(pred, y)

            # Backward
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{Config.EPOCHS} | Loss: {avg_loss:.6f}")
            adjust_learning_rate(optimizer, epoch + 1, Config.LEARNING_RATE)

        # Early Stopping Check (这里用 Train Loss 代替 Val Loss 简化，正规应用 Val)
        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save Results
    plot_loss(loss_history, os.path.join(Config.OUTPUT_DIR, 'loss_curve.png'))
    plot_weights(model, os.path.join(Config.OUTPUT_DIR, 'weight_heatmap.png'))  # 杀手级图示

    # Fit Threshold
    print(">>> Fitting Threshold...")
    eval_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    engine = InferenceEngine(model)
    engine.fit_threshold(eval_dl)