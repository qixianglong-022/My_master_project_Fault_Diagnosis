# eval_metrics.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, accuracy_score
import seaborn as sns

from config import Config
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from utils.anomaly import AnomalyMeasurer

# 设置中文字体 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model():
    print(">>> 正在启动验证评估流程...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    print(f"Loading model from {Config.OUTPUT_DIR}")
    model = RDLinear(Config).to(device)
    model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')))
    model.eval()

    # 2. 加载 POT 阈值
    try:
        threshold = float(np.load(os.path.join(Config.OUTPUT_DIR, 'test_threshold.npy')))
        print(f"已加载 POT 动态阈值: {threshold:.6f}")
    except:
        print("[Error] 未找到阈值文件，请先运行 train.py！")
        return

    # 3. 准备测试数据
    # 3.1 正常样本 (Test-HH) - 对应 Label 0
    print("Loading Test-HH (未见过的健康数据)...")
    ds_normal = MotorDataset(flag='test', condition='HH')
    dl_normal = DataLoader(ds_normal, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3.2 故障样本 (Test-Fault) - 对应 Label 1
    # 比如我们测试 'FB' (轴承故障)
    target_fault = 'FB'
    print(f"Loading Test-Fault ({target_fault})...")
    ds_fault = MotorDataset(flag='test', condition=target_fault)
    dl_fault = DataLoader(ds_fault, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 4. 计算 SPE 分数
    def get_spe_list(dataloader):
        spe_list = []
        with torch.no_grad():
            for x, cov in dataloader:
                x = x.float().to(device)
                cov = cov.float().to(device)

                # 模型重构
                x_hat = model(x, cov)

                # 计算 SPE: mean((x - x_hat)^2)
                # 维度: [Batch, Length, Channel] -> [Batch]
                loss = torch.mean((x - x_hat) ** 2, dim=[1, 2])
                spe_list.append(loss.cpu().numpy())
        if len(spe_list) == 0: return np.array([])
        return np.concatenate(spe_list)

    print("Calculating SPE for Normal data...")
    spe_normal = get_spe_list(dl_normal)
    print("Calculating SPE for Fault data...")
    spe_fault = get_spe_list(dl_fault)

    # 诊断信息
    print(f"\n[Diagnostic] SPE Statistics:")
    print(f"Normal (HH): Mean={np.mean(spe_normal):.6f}, Max={np.max(spe_normal):.6f}")
    print(f"Fault  ({target_fault}): Mean={np.mean(spe_fault):.6f}, Min={np.min(spe_fault):.6f}")
    print(f"Threshold  : {threshold:.6f}")

    # 5. 构建标签与预测
    # 0 = Normal, 1 = Fault
    y_true = np.concatenate([np.zeros(len(spe_normal)), np.ones(len(spe_fault))])
    y_scores = np.concatenate([spe_normal, spe_fault])

    # 5.1 使用 POT 阈值进行预测
    y_pred_pot = (y_scores > threshold).astype(int)

    # 6. 计算指标 (POT 阈值下的性能)
    acc = accuracy_score(y_true, y_pred_pot)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_pot, average='binary')

    # 6.2 计算 AUC (与阈值无关，评价模型分离能力)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print("\n" + "=" * 40)
    print(f" Evaluation Result vs {target_fault}")
    print("=" * 40)
    print(f" Accuracy : {acc:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall   : {recall:.4f}")
    print(f" F1-Score : {f1:.4f}")
    print(f" AUC      : {roc_auc:.4f}")
    print("=" * 40)

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


if __name__ == "__main__":
    evaluate_model()