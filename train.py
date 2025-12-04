import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings

from config import Config
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from models.rdlinear import RDLinear
# from models.baselines import LSTMAE, VanillaDLinear
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.anomaly import AnomalyMeasurer  # 引入异常检测器

warnings.filterwarnings('ignore')

def get_model(config):
    if config.MODEL_NAME == 'RDLinear':
        return RDLinear(config)
    # elif config.MODEL_NAME == 'LSTM_AE':
    #     return LSTMAE(input_dim=config.ENC_IN, hidden_dim=64)
    # elif config.MODEL_NAME == 'DLinear':
    #     return VanillaDLinear(config) # 不带 RevIN 和 Speed 的版本
    else:
        raise ValueError("Unknown Model")

def train():
    # ================= 1. 准备工作 =================
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ================= 2. 加载数据 =================
    print("Loading datasets...")
    train_dataset = MotorDataset(flag='train', condition='HH')
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_dataset = MotorDataset(flag='val', condition='HH')
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    # ================= 3. 模型初始化 =================
    print(f"Initializing RDLinear model... Seq_Len={Config.WINDOW_SIZE}, Enc_In={Config.ENC_IN}")
    model = get_model(Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 用于记录 Loss 曲线
    loss_history = {'train': [], 'val': []}

    # ================= 4. 训练循环 =================
    for epoch in range(Config.EPOCHS):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_cov) in enumerate(train_loader):
            iter_count += 1
            batch_x = batch_x.float().to(device)
            batch_cov = batch_cov.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, batch_cov)
            loss = criterion(outputs, batch_x)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"\tIt: {i + 1} | Loss: {loss.item():.7f}")

        print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
        train_loss_avg = np.average(train_loss)

        # 验证
        vali_loss = validation(model, val_loader, criterion, device)

        # 记录
        loss_history['train'].append(train_loss_avg)
        loss_history['val'].append(vali_loss)

        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}")

        early_stopping(vali_loss, model, Config.OUTPUT_DIR)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        adjust_learning_rate(optimizer, epoch + 1, Config.LEARNING_RATE)

    # ================= 5. 自动生成 POT 阈值 (关键步骤) =================
    print("\n>>> Calculating Anomaly Threshold (POT)...")
    # 加载最佳模型
    best_model_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 在验证集上计算所有样本的重构误差 (SPE)
    spe_list = []
    with torch.no_grad():
        for batch_x, batch_cov in val_loader:
            batch_x = batch_x.float().to(device)
            batch_cov = batch_cov.float().to(device)
            outputs = model(batch_x, batch_cov)

            # 计算 SPE (Batch, )
            # 简单的 MSE: mean((x - x_hat)^2)
            errors = torch.mean((batch_x - outputs) ** 2, dim=[1, 2])
            spe_list.append(errors.cpu().numpy())

    val_spe = np.concatenate(spe_list)

    # 使用极值理论 (POT) 拟合阈值
    detector = AnomalyMeasurer(q=1e-3, level=0.98)  # q=1e-3 意味着允许 0.1% 的误报
    threshold = detector.fit_pot(val_spe)

    # 保存阈值
    np.save(os.path.join(Config.OUTPUT_DIR, 'test_threshold.npy'), threshold)
    print(f"Threshold saved to: {os.path.join(Config.OUTPUT_DIR, 'test_threshold.npy')}")

    # ================= 6. 可视化诊断 =================
    print("\n>>> Generating Diagnostic Plots...")

    # 6.1 画 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train'], label='Train Loss')
    plt.plot(loss_history['val'], label='Val Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'loss_curve.png'))
    plt.close()

    # 6.2 画重构效果对比图 (取验证集最后一个 Batch 的第一个样本)
    # batch_x: [B, L, C], outputs: [B, L, C]
    # 我们画第一个通道 (通常是振动)
    sample_idx = 0
    channel_idx = 0

    orig = batch_x[sample_idx, :, channel_idx].cpu().numpy()
    recon = outputs[sample_idx, :, channel_idx].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.plot(orig, label='Original', alpha=0.7)
    plt.plot(recon, label='Reconstructed', alpha=0.7, linestyle='--')
    plt.title(f'Reconstruction Visualization (Channel {channel_idx})')
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'reconstruction_vis.png'))
    plt.close()

    print(f"Plots saved to {Config.OUTPUT_DIR}")

    # ================= 7. 导出 ONNX =================
    print("\n>>> Exporting to ONNX...")
    dummy_x = torch.randn(1, Config.WINDOW_SIZE, model.enc_in).to(device)
    dummy_cov = torch.randn(1, 2).to(device)

    onnx_path = os.path.join(Config.OUTPUT_DIR, "rdlinear_edge.onnx")
    torch.onnx.export(
        model, (dummy_x, dummy_cov), onnx_path,
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input_signal', 'input_speed_cov'],
        output_names=['reconstructed_signal'],
        dynamic_axes={'input_signal': {0: 'batch'}, 'input_speed_cov': {0: 'batch'},
                      'reconstructed_signal': {0: 'batch'}}
    )
    print(f"ONNX model saved to: {onnx_path}")


def validation(model, val_loader, criterion, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_cov in val_loader:
            batch_x = batch_x.float().to(device)
            batch_cov = batch_cov.float().to(device)
            outputs = model(batch_x, batch_cov)
            loss = criterion(outputs, batch_x)
            total_loss.append(loss.item())
    return np.average(total_loss)


if __name__ == '__main__':
    train()