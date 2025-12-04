import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import json

from config import Config
from data_loader import MotorDataset
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
    early_stopping = EarlyStopping(patience=15, verbose=True) # Loss 曲线在下降过程中会有波动，给它一点耐心，不要过早停止。

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

    # ================= 6. 计算自适应参数 =================

    print("\n>>> Calculating Adaptive Fusion Parameters...")
    # 1. 获取验证集的所有残差
    # 假设 val_spe 是 [N_Samples] 的总误差，这里我们需要分通道的误差
    # 我们需要重新跑一次验证集，获取原始的分量残差

    res_vib_list = []
    res_audio_list = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_cov in val_loader:
            batch_x = batch_x.float().to(device)
            batch_cov = batch_cov.float().to(device)
            recon = model(batch_x, batch_cov)

            # 计算残差 (x - x_hat)^2
            res = (batch_x - recon) ** 2  # [B, L, D]

            # 拆分通道 (根据 Config 里的维度)
            dim_vib = Config.FEAT_DIM_VIB

            # 振动部分误差 (取均值) -> [B]
            r_v = torch.mean(res[:, :, :dim_vib], dim=[1, 2])
            # 声纹部分误差 (这里我们需要的是方差特性，用于计算 tau_base)
            # 论文公式: sigma^2_audio(t) 是声纹残差在时间窗内的方差
            # res_audio: [B, L, D_audio]
            res_audio = batch_x[:, :, dim_vib:] - recon[:, :, dim_vib:]
            # 计算每个样本在时间轴上的方差，并对通道取平均
            r_a_var = torch.mean(torch.var(res_audio, dim=1), dim=1)  # [B]

            res_vib_list.append(r_v.cpu().numpy())
            res_audio_list.append(r_a_var.cpu().numpy())

    all_vib_spe = np.concatenate(res_vib_list)
    all_audio_var = np.concatenate(res_audio_list)

    # 2. 计算统计参数
    # tau_base: 健康声纹残差方差的中位数 (代表背景底噪水平)
    tau_base = float(np.median(all_audio_var))

    # th_vib: 健康振动 SPE 的上界 (例如 99 分位点，作为激活阈值)
    # 超过这个值，说明振动肯定异常，强制激活
    th_vib = float(np.percentile(all_vib_spe, 99))

    params = {
        "tau_base": tau_base,
        "th_vib": th_vib,
        "k1": 5.0,  # 灵敏度系数暂时保持固定，或者通过网格搜索优化
        "k2": 5.0
    }

    # 3. 保存
    with open(Config.FUSION_PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Adaptive parameters saved to: {Config.FUSION_PARAMS_PATH}")
    print(f"  tau_base (Audio Noise Floor): {tau_base:.6f}")
    print(f"  th_vib (Vibration Max Normal): {th_vib:.6f}")

    # ================= 7. 可视化诊断（loss曲线） =================
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

    # ================= 8. 可视化诊断 (重构效果图) =================
    print("\n>>> Generating Reconstruction Plots for ALL Channels...")

    # 1. 创建可视化目录
    vis_dir = os.path.join(Config.OUTPUT_DIR, "vis_recon")
    os.makedirs(vis_dir, exist_ok=True)

    # 2. 准备通道名称 (根据 Config 推断，方便看图)
    # 振动通道名 (Vib_1, Vib_2...)
    vib_names = [f"Vib_Ch{i}" for i in range(len(Config.COL_INDICES_VIB))]
    vib_feats = ["RMS", "Kurt"]  # 每个振动通道有2个特征

    feature_names = []
    for v_name in vib_names:
        for f_name in vib_feats:
            feature_names.append(f"{v_name}_{f_name}")

    # 声纹通道名 (MFCC_1 ~ MFCC_13)
    audio_names = [f"MFCC_{i + 1}" for i in range(Config.N_MFCC)]

    # 合并所有特征名
    all_names = feature_names + audio_names

    # 安全检查：确保名字数量等于模型输出维度
    if len(all_names) != model.enc_in:
        # 如果对不上，就用通用名字
        all_names = [f"Feat_{i}" for i in range(model.enc_in)]

    # 3. 取验证集的一个样本 (Batch 0, Sample 0)
    # batch_x: [B, Seq_Len, D]
    sample_idx = 0

    # 4. 循环绘制每个通道
    for ch_idx in range(model.enc_in):
        orig = batch_x[sample_idx, :, ch_idx].cpu().numpy()
        recon = outputs[sample_idx, :, ch_idx].cpu().numpy()

        feature_name = all_names[ch_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(orig, label='Original', alpha=0.7)
        plt.plot(recon, label='Reconstructed', alpha=0.7, linestyle='--')
        plt.title(f'Reconstruction: {feature_name} (Idx {ch_idx})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存文件名带上索引和特征名
        fname = f"recon_{ch_idx:02d}_{feature_name}.png"
        plt.savefig(os.path.join(vis_dir, fname))
        plt.close()  # 关掉画布释放内存

    print(f"All reconstruction plots saved to {vis_dir}")

    # ================= 9. 导出 ONNX =================
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