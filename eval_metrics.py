# eval_metrics.py
import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
import seaborn as sns

from config import Config
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from utils.tools import apply_moving_average
from models.baselines import LSTMAE, VanillaDLinear

# 设置中文字体 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加噪函数
def add_noise(x, snr_db):
    """给声纹通道加噪声 (假设最后一列是声纹)"""
    if snr_db is None: return x

    # 1. 分离声纹
    audio_feat = x[:, :, -13:]  # 假设最后13维是MFCC

    # 2. 计算信号功率
    signal_power = torch.mean(audio_feat ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # 3. 生成噪声
    noise = torch.randn_like(audio_feat, device=x.device) * torch.sqrt(noise_power)

    # 4. 叠加
    x[:, :, -13:] += noise
    return x


def get_spe(model, dataloader, device, config):
    """计算重构误差 (SPE)"""
    spe_list = []
    model.eval()

    # 确定振动通道数
    vib_dim = config.FEAT_DIM_VIB

    # 加载自适应参数 (用于 RDLinear 的加权)
    # 注意：这里的 params_path 也应该从 source_dir 读取
    try:
        fusion_path = os.path.join(config.OUTPUT_DIR, "fusion_params.json")
        with open(fusion_path, 'r') as f:
            fusion_params = import_json(f)  # json.load(f)
    except:
        # 兜底参数
        fusion_params = {'tau_base': 0.5, 'th_vib': 1.0, 'k1': 5.0, 'k2': 5.0}

    import json
    if os.path.exists(os.path.join(config.OUTPUT_DIR, "fusion_params.json")):
        with open(os.path.join(config.OUTPUT_DIR, "fusion_params.json"), 'r') as f:
            fusion_params = json.load(f)

    with torch.no_grad():
        for x, y, cov in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
            cov = cov.float().to(device)

            # 模型推理
            recon = model(x, cov)

            # 计算基础残差平方
            res_sq = (y - recon) ** 2

            # 如果是 RDLinear，使用自适应融合
            if config.MODEL_NAME == 'RDLinear':
                # 1. 分量 SPE
                spe_vib = torch.mean(res_sq[:, :, :vib_dim], dim=[1, 2])
                spe_audio = torch.mean(res_sq[:, :, vib_dim:], dim=[1, 2])

                # 2. 门控逻辑
                res_audio_raw = y[:, :, vib_dim:] - recon[:, :, vib_dim:]
                var_audio = torch.var(res_audio_raw.reshape(x.shape[0], -1), dim=1)

                alpha_uncert = 1.0 / (1.0 + torch.exp(fusion_params['k1'] * (var_audio - fusion_params['tau_base'])))
                beta_activate = 1.0 / (1.0 + torch.exp(-fusion_params['k2'] * (spe_vib - fusion_params['th_vib'])))
                alpha_audio = torch.max(alpha_uncert, beta_activate)

                score = (1.0 * spe_vib + 1.2 * alpha_audio * spe_audio) / (1.0 + 1.2 * alpha_audio)
                spe_list.append(score.cpu().numpy())

            else:
                # Baseline (LSTM_AE / DLinear): 直接使用 MSE
                # 对所有通道取平均
                score = torch.mean(res_sq, dim=[1, 2])
                spe_list.append(score.cpu().numpy())

    if len(spe_list) == 0: return np.array([])
    return np.concatenate(spe_list)

# 定义模型选择函数
def get_model(config):
    if config.MODEL_NAME == 'RDLinear':
        return RDLinear(config)
    elif config.MODEL_NAME == 'LSTM_AE':
        # 注意参数匹配
        return LSTMAE(input_dim=config.ENC_IN, hidden_dim=64)
    elif config.MODEL_NAME == 'DLinear':
        return VanillaDLinear(config)
    else:
        raise ValueError(f"Unknown Model: {config.MODEL_NAME}")

def evaluate_current_task():
    task_name = Config.TEST_TASK_NAME
    atoms = Config.TEST_ATOMS
    model_name = Config.MODEL_NAME  # 获取当前模型名

    # ================= 1. 路径修正 =================
    # A. 权重从哪读？(Source Model 目录)
    # ./checkpoints/{MODEL_NAME}/Source_Model_200kg
    source_dir = os.path.join(
        "./checkpoints",
        model_name,
        "Source_Model_200kg"
    )

    # B. 结果存哪去？(Transfer Results 目录)
    # ./checkpoints/{MODEL_NAME}/Transfer_Results/{TASK_NAME}
    task_save_dir = os.path.join(
        "./checkpoints",
        model_name,
        "Transfer_Results",
        task_name
    )
    os.makedirs(task_save_dir, exist_ok=True)

    print(f"\n>>> Evaluating Task: {task_name}")
    print(f"    Model: {model_name}")
    print(f"    Loading Source from: {source_dir}")
    print(f"    Saving Results to:   {task_save_dir}")

    Config.OUTPUT_DIR = source_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= 2. 加载模型 =================
    # 必须使用 get_model 动态获取，不能写死 RDLinear
    from models.baselines import LSTMAE, VanillaDLinear  # 确保导入了

    # 定义简单的获取函数 (或从 train.py 导入)
    def get_model_class(name):
        if name == 'RDLinear': return RDLinear(Config)
        if name == 'LSTM_AE': return LSTMAE(input_dim=Config.ENC_IN)
        if name == 'DLinear': return VanillaDLinear(Config)
        raise ValueError(f"Unknown model: {name}")

    model = get_model_class(model_name).to(device)

    # 加载权重
    ckpt_path = os.path.join(source_dir, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        print(f"[Error] 权重文件不存在: {ckpt_path}")
        return None

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 加载阈值 (注意：不同模型的阈值也不同，必须从对应的 source_dir 读)
    threshold_path = os.path.join(source_dir, 'test_threshold.npy')
    if os.path.exists(threshold_path):
        threshold = float(np.load(threshold_path))
        print(f"    Loaded Threshold: {threshold:.6f}")
    else:
        print(f"[Warning] 阈值文件不存在，使用默认值 1.0 (结果可能不准)")
        threshold = 1.0

    # ================= 4. 加载数据并测试 =================
    # 实例化 Test Dataset (会自动读取 Config.OUTPUT_DIR 下的 scaler)
    # 这里的 atoms 由 run_transfer_challenge 动态修改 Config.TEST_ATOMS 传入
    ds = MotorDataset(flag='test', atoms=atoms)

    # --- 测试 HH (健康) ---
    ds.set_condition('HH')
    if len(ds) == 0:
        print(f"    [Skip] No HH data for {atoms}")
        return None
    dl_hh = DataLoader(ds, batch_size=32, shuffle=False)
    spe_hh = get_spe(model, dl_hh, device, Config)

    # --- 测试 FB (故障) ---
    ds.set_condition('FB')
    if len(ds) == 0:
        print(f"    [Skip] No FB data for {atoms}")
        return None
    dl_fb = DataLoader(ds, batch_size=32, shuffle=False)
    spe_fb = get_spe(model, dl_fb, device, Config)

    # 噪声鲁棒性测试
    # 在 Config 中定义 TEST_SNR，或者通过参数传入

    # ================= 5. 指标计算 =================
    # 平滑
    spe_hh = apply_moving_average(spe_hh, window_size=5)
    spe_fb = apply_moving_average(spe_fb, window_size=5)

    y_true = np.concatenate([np.zeros(len(spe_hh)), np.ones(len(spe_fb))])
    y_scores = np.concatenate([spe_hh, spe_fb])
    y_pred = (y_scores > threshold).astype(int)

    res = {
        'Task': task_name,
        'AUC': roc_auc_score(y_true, y_scores),
        'F1': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'HH_Mean_SPE': np.mean(spe_hh),
        'FB_Mean_SPE': np.mean(spe_fb)
    }

    print(f"    Result: AUC={res['AUC']:.4f}, Recall={res['Recall']:.4f}")

    # 保存结果到 csv 以备查
    pd.DataFrame([res]).to_csv(os.path.join(task_save_dir, "metrics.csv"), index=False)
    return res

    # 7. 绘图 (Paper Quality)
    save_dir = Config.OUTPUT_DIR

    # 7.1 SPE 分布直方图 (密度图)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(spe_normal, fill=True, color='green', label='Normal (HH)', alpha=0.3)
    sns.kdeplot(spe_fault, fill=True, color='red', label=f'Fault ({target_fault})', alpha=0.3)
    plt.axvline(threshold, color='k', linestyle='--', linewidth=2, label=f'POT Threshold ({threshold:.4f})')
    plt.title(f"重构误差分布对比 (HH vs {target_fault})")
    plt.xlabel("Squared Prediction Error (SPE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'eval_spe_distribution.png'), dpi=300)
    print("Saved eval_spe_distribution.png")

    # 7.2 ROC 曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (误报率)')
    plt.ylabel('True Positive Rate (召回率)')
    plt.title(f'ROC Curve ({target_fault})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'eval_roc_curve.png'), dpi=300)
    print("Saved eval_roc_curve.png")

    # 7.3 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_pot)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fault'],
                yticklabels=['Normal', 'Fault'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Th={threshold:.4f})')
    plt.savefig(os.path.join(save_dir, 'eval_confusion_matrix.png'), dpi=300)
    print("Saved eval_confusion_matrix.png")

    # 8. 学术可视化 (Paper Visualization)
    print("\n>>> Generating Paper-Quality Visualizations...")

    vis_save_dir = os.path.join(Config.OUTPUT_DIR, "vis_paper")
    os.makedirs(vis_save_dir, exist_ok=True)

    def plot_prediction_sample(dataloader, condition_name, color_true='black', color_pred='red'):
        """
        抽取一个 Batch，绘制 预测值 vs 真实值 的对比图
        """
        # 取一个 batch
        iter_dl = iter(dataloader)
        try:
            x, y, cov = next(iter_dl)
        except StopIteration:
            return

        x = x.float().to(device)
        y = y.float().to(device)
        cov = cov.float().to(device)

        with torch.no_grad():
            pred = model(x, cov)

        # 转回 CPU numpy
        y_np = y.cpu().numpy()  # 真实未来 [Batch, Len, Dim]
        pred_np = pred.cpu().numpy()  # 预测未来 [Batch, Len, Dim]

        # 选择一个样本 (Sample 0)
        idx = 0

        # === 画图 1: 振动通道 (比如第1个振动通道, Dim=0) ===
        plt.figure(figsize=(10, 4))
        # 画真实值 (Target)
        plt.plot(y_np[idx, :, 0], label='Ground Truth (Next Step)', color=color_true, linewidth=2, alpha=0.8)
        # 画预测值 (Prediction)
        plt.plot(pred_np[idx, :, 0], label='RDLinear Prediction', color=color_pred, linestyle='--', linewidth=2)

        plt.title(f"Vibration Prediction under {condition_name}", fontsize=14)
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("Amplitude (Normalized)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存
        fname = f"Vis_{condition_name}_Vibration.png"
        plt.savefig(os.path.join(vis_save_dir, fname), dpi=300)
        plt.close()

        # === 画图 2: 声纹通道 (比如第1个MFCC, Dim=8 假设振动有8个) ===
        # 根据 Config 计算声纹起始索引
        audio_start_idx = Config.FEAT_DIM_VIB

        plt.figure(figsize=(10, 4))
        plt.plot(y_np[idx, :, audio_start_idx], label='Ground Truth', color=color_true, linewidth=2, alpha=0.8)
        plt.plot(pred_np[idx, :, audio_start_idx], label='Prediction', color=color_pred, linestyle='--', linewidth=2)

        plt.title(f"Acoustic(MFCC-1) Prediction under {condition_name}", fontsize=14)
        plt.xlabel("Time Step (Frame)", fontsize=12)
        plt.ylabel("MFCC Value", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"Vis_{condition_name}_Audio.png"
        plt.savefig(os.path.join(vis_save_dir, fname), dpi=300)
        plt.close()

        print(f"Saved visualization for {condition_name}")

    # 1. 可视化健康数据 (HH)
    # 预期：预测曲线和真实曲线紧密贴合 (拟合得好)
    plot_prediction_sample(dl_normal, "Health_HH", color_true='green', color_pred='blue')

    # 2. 可视化故障数据 (FB)
    # 预期：预测曲线和真实曲线出现明显偏差 (Deviation)，说明模型无法预测故障突变
    plot_prediction_sample(dl_fault, f"Fault_{target_fault}", color_true='black', color_pred='red')

    print(f"可视化图片已保存至: {vis_save_dir}")

    # =================  9. 自动保存指标到文件 =================
    metrics_path = os.path.join(Config.OUTPUT_DIR, "final_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Target Fault: {target_fault}\n")
        f.write("========================================\n")
        f.write(f"Threshold : {threshold:.6f}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"AUC       : {roc_auc:.4f}\n")
        f.write("========================================\n")
        f.write(f"Normal Mean SPE: {np.mean(spe_normal):.6f}\n")
        f.write(f"Fault Mean SPE : {np.mean(spe_fault):.6f}\n")

    print(f"\n[Success] 详细评估指标已保存至: {metrics_path}")


if __name__ == "__main__":
    evaluate_model()