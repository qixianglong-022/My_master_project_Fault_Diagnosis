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
from models.baselines import LSTMAE, VanillaDLinear, TiDE, TransformerBaseline


def get_model_instance(device):
    name = Config.MODEL_NAME
    if name == 'RDLinear':
        return RDLinear().to(device)
    elif name == 'DLinear':
        return VanillaDLinear(Config).to(device)
    elif name == 'LSTMAE':
        return LSTMAE(Config).to(device)
    elif name == 'TiDE':
        return TiDE(Config).to(device)
    elif name == 'Transformer':
        return TransformerBaseline(Config).to(device)
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

    # 1. 加载基础阈值 (源域训练得到的)
    th_path = os.path.join(Config.OUTPUT_DIR, 'threshold.npy')
    if os.path.exists(th_path):
        base_threshold = float(np.load(th_path))
        print(f">>> Loaded Base Threshold: {base_threshold:.4f}")
    else:
        print(">>> [Warn] No threshold file found. Metrics requiring labels will be 0.")
        base_threshold = None

    # 遍历所有测试工况
    for phys_load, phys_speed in Config.TEST_ATOMS:
        exp_name = f"load_{phys_load}_speed_{phys_speed}"
        if Config.TEST_NOISE_SNR: exp_name += f"_noise{Config.TEST_NOISE_SNR}dB"

        print(f"\n--- Running: {exp_name} ---")

        try:
            # 加载测试数据 (Normal + Fault)
            dataset = MotorDataset([(phys_load, phys_speed)], mode='test',
                                   fault_types=Config.TEST_FAULT_TYPES,
                                   noise_snr=Config.TEST_NOISE_SNR)

            if len(dataset) == 0: continue

            dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            # =========================================================
            # [Step A] 可视化诊断 (Diagnosis)
            # =========================================================
            # 生成 PDF 诊断图，保存到 vis_pdf 文件夹
            vis_dir = os.path.join(Config.OUTPUT_DIR, "vis_pdf")
            engine.diagnose(dataloader, save_dir=vis_dir, prefix=exp_name)

            # =========================================================
            # [Step B] 阈值自适应 (Adaptive Thresholding)
            # =========================================================
            current_threshold = base_threshold

            # 1. 判断是否为迁移工况
            # 定义源域(训练过的)转速: 15, 45, 15-45, 45-15
            source_speeds = ['15', '45', '15-45', '45-15']
            is_transfer = (phys_load != 200) or (phys_speed not in source_speeds)

            # 2. 如果是迁移工况，尝试校准阈值
            if is_transfer and base_threshold is not None:
                # 临时加载少量健康数据 (Few-shot Calibration)
                calib_ds = MotorDataset([(phys_load, phys_speed)], mode='test',
                                        fault_types=['HH'],
                                        noise_snr=Config.TEST_NOISE_SNR)

                if len(calib_ds) > 0:
                    calib_loader = DataLoader(calib_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
                    # 只预测分数，不需要 Label
                    calib_scores, _ = engine.predict(calib_loader)

                    if len(calib_scores) > 0:
                        # 使用 MAD 策略重新计算 (与训练时保持一致 eta=4.0)
                        median = np.median(calib_scores)
                        mad = np.median(np.abs(calib_scores - median))
                        current_threshold = median + 4.0 * mad
                        print(f"   [Adapt] Threshold adapted: {base_threshold:.4f} -> {current_threshold:.4f}")
                    else:
                        print("   [Warn] Calib scores empty, using base threshold.")
                else:
                    print("   [Warn] No HH data for calibration, using base threshold.")

            # =========================================================
            # [Step C] 预测与评估
            # =========================================================
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
            else:
                # 如果只有一种标签(比如全正常)，AUC无法计算，设为NaN或跳过
                metrics['AUC'] = 0.0

                # 2. Hard Metrics (需要阈值)
            if current_threshold is not None:
                preds = (scores > current_threshold).astype(int)
                metrics['Accuracy'] = accuracy_score(labels, preds)
                metrics['Precision'] = precision_score(labels, preds, zero_division=0)
                metrics['Recall'] = recall_score(labels, preds, zero_division=0)
                metrics['F1'] = f1_score(labels, preds, zero_division=0)

                # 打印混淆矩阵信息
                # labels=[0, 1] 确保即使数据里缺某一类也能正确输出
                tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
                print(f"   [Confusion] TP:{tp} | FP:{fp} | TN:{tn} | FN:{fn}")

            print(
                f"   AUC: {metrics['AUC']:.4f} | F1: {metrics['F1']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f}")

            results_summary.append(metrics)

        except Exception as e:
            print(f"[Error] {exp_name}: {e}")
            # 打印堆栈以便调试
            import traceback
            traceback.print_exc()

    # 保存到 CSV
    if results_summary:
        df = pd.DataFrame(results_summary)
        # 调整列顺序
        cols = ['Experiment', 'AUC', 'F1', 'Precision', 'Recall', 'Accuracy', 'Model', 'Load', 'Speed', 'Noise']
        # 仅选择存在的列
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        csv_path = os.path.join(Config.OUTPUT_DIR, "eval_results", "summary_report.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)

        print(f"\n>>> Report Saved: {csv_path}")

if __name__ == '__main__':
    run_evaluation_loop()