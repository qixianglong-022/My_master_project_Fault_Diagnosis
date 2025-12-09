import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
# [新增] 引入更多指标
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from config import Config
from data_loader import MotorDataset
from utils.anomaly import InferenceEngine
from models.rdlinear import RDLinear
from models.baselines import LSTMAE, VanillaDLinear


def get_model_instance(device):
    name = Config.MODEL_NAME
    if name == 'RDLinear':
        return RDLinear().to(device)
    elif name == 'DLinear':
        return VanillaDLinear(Config).to(device)
    elif name == 'LSTMAE':
        return LSTMAE(input_dim=Config.ENC_IN, hidden_dim=64).to(device)
    else:
        raise ValueError(f"Unknown Model Name: {name}")


def run_evaluation_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')

    print(f">>> Loading Model: {model_path}")
    try:
        model = get_model_instance(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"[Fatal Error] {e}")
        return

    engine = InferenceEngine(model)

    # 结果容器
    results_summary = []

    # 加载阈值
    th_path = os.path.join(Config.OUTPUT_DIR, 'threshold.npy')
    if os.path.exists(th_path):
        threshold = float(np.load(th_path))
        print(f">>> Loaded Threshold: {threshold:.4f}")
    else:
        print(">>> [Warn] No threshold file found. Metrics requiring labels will be 0.")
        threshold = None

    for phys_load, phys_speed in Config.TEST_ATOMS:
        exp_name = f"load_{phys_load}_speed_{phys_speed}"
        if Config.TEST_NOISE_SNR: exp_name += f"_noise{Config.TEST_NOISE_SNR}dB"

        print(f"\n--- Running: {exp_name} ---")

        try:
            dataset = MotorDataset([(phys_load, phys_speed)], mode='test',
                                   fault_types=Config.TEST_FAULT_TYPES,
                                   noise_snr=Config.TEST_NOISE_SNR)

            if len(dataset) == 0: continue

            dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            # === [新增] 调用可视化诊断 ===
            # 创建保存目录: checkpoints/ExpName/vis_pdf/
            vis_dir = os.path.join(Config.OUTPUT_DIR, "vis_pdf")
            engine.diagnose(dataloader, save_dir=vis_dir, prefix=exp_name)
            # ==========================

            scores, labels = engine.predict(dataloader)

            # --- 核心指标计算 ---
            metrics = {
                'Experiment': exp_name,
                'Model': Config.MODEL_NAME,
                'Load': phys_load,
                'Speed': phys_speed,
                'Noise': Config.TEST_NOISE_SNR,
                'AUC': 0.0, 'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0
            }

            # 1. AUC (需要正负样本都存在)
            if len(np.unique(labels)) > 1:
                metrics['AUC'] = roc_auc_score(labels, scores)

            # 2. Hard Metrics (需要阈值)
            if threshold is not None:
                preds = (scores > threshold).astype(int)
                metrics['Accuracy'] = accuracy_score(labels, preds)
                metrics['Precision'] = precision_score(labels, preds, zero_division=0)
                metrics['Recall'] = recall_score(labels, preds, zero_division=0)
                metrics['F1'] = f1_score(labels, preds, zero_division=0)

                # 打印混淆矩阵信息
                tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
                print(f"   [Confusion] TP:{tp} | FP:{fp} | TN:{tn} | FN:{fn}")

            print(
                f"   AUC: {metrics['AUC']:.4f} | F1: {metrics['F1']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f}")

            results_summary.append(metrics)

        except Exception as e:
            print(f"[Error] {exp_name}: {e}")

    # 保存到 CSV
    if results_summary:
        df = pd.DataFrame(results_summary)
        # 调整列顺序，好看一点
        cols = ['Experiment', 'AUC', 'F1', 'Precision', 'Recall', 'Accuracy', 'Model', 'Load', 'Speed', 'Noise']
        df = df[cols]

        csv_path = os.path.join(Config.OUTPUT_DIR, "eval_results", "summary_report.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # 覆盖写入还是追加写入？建议覆盖本次运行的结果，或者带时间戳。这里演示追加
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)

        print(f"\n>>> Report Saved: {csv_path}")


if __name__ == '__main__':
    run_evaluation_loop()