import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from utils.anomaly import InferenceEngine
from config import Config

def run_evaluation_loop():
    # 1. 准备环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')

    print(f">>> Loading Model: {model_path}")
    model = RDLinear().to(device)
    model.load_state_dict(torch.load(model_path))

    # 初始化引擎 (自动加载 fusion_params.json)
    engine = InferenceEngine(model)

    # 2. 获取测试配置
    test_atoms = Config.TEST_ATOMS
    fault_types = Config.TEST_FAULT_TYPES
    noise_snr = Config.TEST_NOISE_SNR

    print(f">>> Test Config: Faults={fault_types}, Noise={noise_snr}dB")

    results_summary = []

    # 3. 循环测试
    for load, speed in test_atoms:
        exp_name = f"load_{load}_speed_{speed}"
        if noise_snr is not None:
            exp_name += f"_noise{noise_snr}dB"

        save_dir = os.path.join(Config.OUTPUT_DIR, "eval_results", exp_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Running: {exp_name} ---")

        try:
            # 实例化 Dataset (传入噪声参数)
            dataset = MotorDataset(
                atoms_list=[(load, speed)],
                mode='test',
                fault_types=fault_types,
                noise_snr=noise_snr
            )

            if len(dataset) == 0:
                print(f"[Warn] No data. Skipping.")
                continue

            dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            # 推理
            scores, labels = engine.predict(dataloader)

            # 指标
            if len(np.unique(labels)) < 2:
                # 只有一类标签无法算 AUC
                auc = 0.5
                print("[Warn] Only one class present in labels.")
            else:
                auc = roc_auc_score(labels, scores)

            # 加载阈值
            th_path = os.path.join(Config.OUTPUT_DIR, 'threshold.npy')
            if os.path.exists(th_path):
                threshold = np.load(th_path)
                preds = (scores > threshold).astype(int)
                f1 = f1_score(labels, preds)
            else:
                f1 = 0.0
                print("[Warn] Threshold file not found.")

            print(f"   AUC: {auc:.4f} | F1: {f1:.4f}")

            # 保存结果
            np.save(os.path.join(save_dir, 'scores.npy'), scores)
            np.save(os.path.join(save_dir, 'labels.npy'), labels)

            results_summary.append({
                'Experiment': exp_name,
                'Load': load, 'Speed': speed,
                'Noise': noise_snr,
                'AUC': auc, 'F1': f1
            })

        except Exception as e:
            print(f"[Error] {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. 保存报表 (支持追加模式，防止多轮测试覆盖)
    if results_summary:
        df = pd.DataFrame(results_summary)
        csv_path = os.path.join(Config.OUTPUT_DIR, "eval_results", "summary_report.csv")

        # 检查文件是否存在
        if os.path.exists(csv_path):
            # 追加写入，不写 Header
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # 新建写入
            df.to_csv(csv_path, mode='w', header=True, index=False)

        print(f"\n>>> Done. Report updated: {csv_path}")


if __name__ == '__main__':
    run_evaluation_loop()