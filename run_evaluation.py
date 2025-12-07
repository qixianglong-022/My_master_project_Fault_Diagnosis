import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from config import Config
from data_loader import MotorDataset
from utils.anomaly import InferenceEngine

# [新增] 导入模型定义
from models.rdlinear import RDLinear
from models.baselines import LSTMAE, VanillaDLinear


def get_model_instance(device):
    """
    模型工厂函数：根据配置动态实例化模型架构
    """
    name = Config.MODEL_NAME
    if name == 'RDLinear':
        return RDLinear().to(device)
    elif name == 'DLinear':
        return VanillaDLinear(Config).to(device)
    elif name == 'LSTMAE':
        # 确保输入维度与 Config 一致
        return LSTMAE(input_dim=Config.ENC_IN, hidden_dim=64).to(device)
    else:
        raise ValueError(f"Unknown Model Name: {name}")


def run_evaluation_loop():
    # 1. 准备环境
    # 自动检测设备 (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')

    print(f">>> Loading Model: {model_path}")
    print(f"    Architecture: {Config.MODEL_NAME}")
    print(f"    Device: {device}")

    # 2. [核心修改] 动态实例化模型
    # 之前是 model = RDLinear().to(device) -> 导致报错
    try:
        model = get_model_instance(device)
        # 加载权重 (map_location 确保在 CPU 上也能加载 GPU 训练的权重)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"\n[Fatal Error] 模型权重加载失败！")
        print(f"请检查 Config.MODEL_NAME ({Config.MODEL_NAME}) 是否与 checkpoint 内的权重匹配。")
        print(f"错误详情: {e}")
        return

    # 初始化引擎 (自动加载 fusion_params.json)
    engine = InferenceEngine(model)

    # 3. 获取测试配置
    test_atoms = Config.TEST_ATOMS
    # 这里的 fault_types 可以从 Config 获取，也可以默认
    fault_types = Config.TEST_FAULT_TYPES
    noise_snr = Config.TEST_NOISE_SNR

    print(f">>> Test Config: Faults={fault_types}, Noise={noise_snr}dB")

    results_summary = []

    # 4. 循环测试
    for phys_load, phys_speed in test_atoms:
        # 构造实验名用于显示
        exp_name = f"load_{phys_load}_speed_{phys_speed}"
        if noise_snr is not None:
            exp_name += f"_noise{noise_snr}dB"

        save_dir = os.path.join(Config.OUTPUT_DIR, "eval_results", exp_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Running: {exp_name} ---")

        try:
            # 实例化 Dataset
            # 注意：传入的是单个工况的列表 [(load, speed)]
            dataset = MotorDataset(
                atoms_list=[(phys_load, phys_speed)],
                mode='test',
                fault_types=fault_types,
                noise_snr=noise_snr
            )

            if len(dataset) == 0:
                print(f"[Warn] No data found for {exp_name}. Skipping.")
                continue

            dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            # 推理
            scores, labels = engine.predict(dataloader)

            # 指标计算
            auc = 0.5
            f1 = 0.0

            # AUC 需要至少两类标签
            if len(np.unique(labels)) < 2:
                print("[Warn] Only one class present in labels. Cannot compute AUC.")
            else:
                auc = roc_auc_score(labels, scores)

            # F1 需要阈值
            th_path = os.path.join(Config.OUTPUT_DIR, 'threshold.npy')
            if os.path.exists(th_path):
                threshold = np.load(th_path)
                preds = (scores > threshold).astype(int)
                f1 = f1_score(labels, preds)
            else:
                print("[Warn] Threshold file not found. F1 is 0.0.")

            print(f"   AUC: {auc:.4f} | F1: {f1:.4f}")

            # 保存结果 (可选，如果你需要画详细图表)
            np.save(os.path.join(save_dir, 'scores.npy'), scores)
            np.save(os.path.join(save_dir, 'labels.npy'), labels)

            results_summary.append({
                'Experiment': exp_name,
                'Model': Config.MODEL_NAME,  # 记录模型名
                'Load': phys_load,
                'Speed': phys_speed,
                'Noise': noise_snr,
                'AUC': auc,
                'F1': f1
            })

        except Exception as e:
            print(f"[Error] {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # 5. 保存报表 (支持追加模式)
    if results_summary:
        df = pd.DataFrame(results_summary)
        csv_path = os.path.join(Config.OUTPUT_DIR, "eval_results", "summary_report.csv")

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)

        print(f"\n>>> Done. Report updated: {csv_path}")


if __name__ == '__main__':
    run_evaluation_loop()