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
from models.baselines import LSTMAE, VanillaDLinear
from utils.tools import EarlyStopping, adjust_learning_rate, apply_moving_average

warnings.filterwarnings('ignore')

def get_model(config):
    if config.MODEL_NAME == 'RDLinear':
        return RDLinear(config)
    elif config.MODEL_NAME == 'LSTM_AE':
        return LSTMAE(input_dim=config.ENC_IN, hidden_dim=64)
    elif config.MODEL_NAME == 'DLinear':
        return VanillaDLinear(config) # 不带 RevIN 和 Speed 的版本
    else:
        raise ValueError("Unknown Model")

def train():
    # ================= 1. 准备工作 =================
    # [新增] 将输出目录改为 ./checkpoints/{MODEL_NAME}
    # 这样跑 RDLinear 就存到 ./checkpoints/RDLinear
    # 跑 LSTM_AE 就存到 ./checkpoints/LSTM_AE
    Config.OUTPUT_DIR = os.path.join(
        Config.OUTPUT_DIR,
        Config.MODEL_NAME,
        "Source_Model_200kg"
    )

    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)

    print(f">>> 实验结果将保存至: {Config.OUTPUT_DIR}")

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
    early_stopping = EarlyStopping(patience=30, verbose=True) # Loss 曲线在下降过程中会有波动，给它一点耐心，不要过早停止。

    # 用于记录 Loss 曲线
    loss_history = {'train': [], 'val': []}

    # ================= 4. 训练循环 =================
    for epoch in range(Config.EPOCHS):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_cov) in enumerate(train_loader):  # 接收3个返回值
            iter_count += 1
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # 新增 Target
            batch_cov = batch_cov.float().to(device)

            optimizer.zero_grad()

            # 模型输入 batch_x，预测 outputs
            outputs = model(batch_x, batch_cov)

            # === 核心修改：Loss 计算 ===
            # 让输出去拟合 batch_y (未来)，而不是 batch_x (现在)
            loss = criterion(outputs, batch_y)

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

    # ================= 5. 计算自适应参数 =================
    print("\n>>> Calculating Adaptive Fusion Parameters (Step 1)...")

    # 加载最佳模型
    best_model_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # --- 第一遍扫描：计算统计特性 (tau_base, th_vib) ---
    all_vib_means = []
    all_audio_vars = []

    with torch.no_grad():
        for batch_x, batch_y, batch_cov in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_cov = batch_cov.float().to(device)

            recon = model(batch_x, batch_cov)

            # 计算基础残差: (真实未来 - 预测未来)^2
            res_sq = (batch_y - recon) ** 2  # [B, L, D]

            # 1. 振动均值 (用于确定激活阈值 th_vib)
            dim_vib = Config.FEAT_DIM_VIB
            vib_mean = torch.mean(res_sq[:, :, :dim_vib], dim=[1, 2])
            all_vib_means.append(vib_mean.cpu().numpy())

            # 2. 声纹方差 (用于确定底噪 tau_base)
            # 声纹残差
            res_audio = batch_y[:, :, dim_vib:] - recon[:, :, dim_vib:]
            # 计算每个样本在时间轴上的方差
            audio_var = torch.var(res_audio.reshape(batch_x.shape[0], -1), dim=1)
            all_audio_vars.append(audio_var.cpu().numpy())

    # 合并并计算统计值
    all_vib_means = np.concatenate(all_vib_means)
    all_audio_vars = np.concatenate(all_audio_vars)

    params = {
        "tau_base": float(np.median(all_audio_vars)),  # 声纹底噪基准
        "th_vib": float(np.percentile(all_vib_means, 99)),  # 振动激活阈值 (99分位)
        "k1": 5.0,  # 灵敏度系数
        "k2": 5.0
    }

    # 保存参数
    with open(Config.FUSION_PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Adaptive parameters saved: tau={params['tau_base']:.6f}, th_vib={params['th_vib']:.6f}")

    # --- 第二遍扫描：计算融合分数并定阈值 (Step 2) ---
    print("\n>>> Calculating Anomaly Threshold (Step 2)...")

    all_scores = []

    # 定义一个临时的计算函数 (为了保持 train.py 独立，不依赖 eval_metrics)
    def calculate_score_batch(y_true, y_pred, p):
        res_sq = (y_true - y_pred) ** 2
        dim_vib = Config.FEAT_DIM_VIB

        # 1. 分量 SPE
        spe_vib = torch.mean(res_sq[:, :, :dim_vib], dim=[1, 2])
        spe_audio = torch.mean(res_sq[:, :, dim_vib:], dim=[1, 2])

        # 2. 门控权重
        res_audio_raw = y_true[:, :, dim_vib:] - y_pred[:, :, dim_vib:]
        var_audio = torch.var(res_audio_raw.reshape(y_true.shape[0], -1), dim=1)

        alpha_uncert = 1.0 / (1.0 + torch.exp(p['k1'] * (var_audio - p['tau_base'])))
        beta_activate = 1.0 / (1.0 + torch.exp(-p['k2'] * (spe_vib - p['th_vib'])))

        alpha_audio = torch.max(alpha_uncert, beta_activate)

        # 3. 融合
        w_vib = 1.0
        w_audio = 1.2 * alpha_audio
        score = (w_vib * spe_vib + w_audio * spe_audio) / (w_vib + w_audio)
        return score

    with torch.no_grad():
        for batch_x, batch_y, batch_cov in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_cov = batch_cov.float().to(device)

            recon = model(batch_x, batch_cov)

            # 使用刚刚算好的参数计算分数
            scores = calculate_score_batch(batch_y, recon, params)
            all_scores.append(scores.cpu().numpy())

    all_scores = np.concatenate(all_scores)

    # 应用移动平均滤波
    # 这一步能把那个 0.07 的离群最大值抹平，显著降低阈值
    print(f"Applying Moving Average (Window=5)...")
    all_scores = apply_moving_average(all_scores, window_size=5)

    # === 方案一：使用 99.9% 分位数作为阈值 (比 POT 更稳健) ===
    # 工业现场通常不希望阈值太敏感，99.9% 能过滤掉绝大多数正常波动
    # threshold = np.percentile(all_scores, 99.9)

    # === 方案二：改用 MAD 鲁棒阈值 ===
    # MAD 是 "Median Absolute Deviation"，抗干扰能力极强
    median = np.median(all_scores)
    mad = np.median(np.abs(all_scores - median))

    # 工业界常用 K=3 到 K=5。
    # 如果你想提高 Recall (多抓故障)，就调小 K (比如 3.5)
    # 如果你想提高 Precision (少误报)，就调大 K (比如 5.0)
    # 鉴于你的 AUC 不错但 Recall 低，建议先试 3.5 或 4.0
    K = 3.5
    threshold = median + K * mad

    np.save(os.path.join(Config.OUTPUT_DIR, 'test_threshold.npy'), threshold)
    print(f"Threshold (99.9%): {threshold:.6f} (Mean Score: {np.mean(all_scores):.6f})")
    print(f"Threshold saved to: {os.path.join(Config.OUTPUT_DIR, 'test_threshold.npy')}")

    # ================= 6. 可视化诊断 (重构效果图) =================
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
        orig = batch_y[sample_idx, :, ch_idx].cpu().numpy()  # [Fix] 改为 batch_y
        pred = recon[sample_idx, :, ch_idx].cpu().numpy()  # 这是模型预测值

        feature_name = all_names[ch_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(orig, label='True Future (Target)', alpha=0.7)  # 改标签名
        plt.plot(pred, label='Predicted Future', alpha=0.7, linestyle='--')
        plt.title(f'Prediction: {feature_name} (Idx {ch_idx})')
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
        for batch_x, batch_y, batch_cov in val_loader:  # 接收3个
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_cov = batch_cov.float().to(device)

            outputs = model(batch_x, batch_cov)
            loss = criterion(outputs, batch_y)  # 拟合 Target
            total_loss.append(loss.item())
    return np.average(total_loss)


if __name__ == '__main__':
    train()