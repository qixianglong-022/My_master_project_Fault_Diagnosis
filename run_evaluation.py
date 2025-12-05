import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from config import Config
from data_loader import MotorDataset
from models.rdlinear import RDLinear
from utils.anomaly import InferenceEngine


def run_evaluation_loop():
    # 1. 准备模型与推理引擎
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')

    # 加载训练好的模型
    print(f">>> Loading Model from {model_path}")
    model = RDLinear().to(device)
    model.load_state_dict(torch.load(model_path))

    # 初始化推理引擎 (负责计算 Score, 加载阈值等)
    engine = InferenceEngine(model)

    # 2. 读取要测试的工况列表
    test_atoms = Config.TEST_ATOMS
    fault_types = Config.TEST_FAULT_TYPES
    print(f">>> Target Faults: {fault_types}")
    print(f">>> Test Atoms Count: {len(test_atoms)}")

    results_summary = []

    # 3. 循环测试每个原子工况
    for load, speed in test_atoms:
        # --- (A) 构造易读的文件夹名 ---
        # 文件夹名: load_200_speed_30-60
        exp_name = f"load_{load}_speed_{speed}"

        # 结果保存路径: ./checkpoints/eval_results/load_200_speed_30-60
        save_dir = os.path.join(Config.OUTPUT_DIR, "eval_results", exp_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n==================================================")
        print(f"Running Experiment: {exp_name}")
        print(f"Condition: Load={load}, Speed={speed}, Faults={fault_types}")
        print(f"Output Dir: {save_dir}")

        # --- (B) 构造原子数据集 ---
        # 每次只加载当前这一个工况 (atoms_list 只包含一项)
        current_atom = [(load, speed)]

        try:
            dataset = MotorDataset(
                atoms_list=current_atom,
                mode='test',
                fault_types=fault_types
            )

            if len(dataset) == 0:
                print(f"[Warn] No data found for {exp_name}. Skipping.")
                continue

            dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            # --- (C) 执行推理 ---
            # predict 返回: scores, labels (0=Health, 1=Fault)
            scores, labels = engine.predict(dataloader)

            # --- (D) 计算指标 ---
            auc = roc_auc_score(labels, scores)

            # 加载自动阈值 (在 train 阶段生成的)
            threshold = np.load(os.path.join(Config.OUTPUT_DIR, 'threshold.npy'))
            preds = (scores > threshold).astype(int)
            f1 = f1_score(labels, preds)

            print(f"Result: AUC = {auc:.4f}, F1 = {f1:.4f}")

            # --- (E) 保存该工况的详细结果 ---
            # 保存 score 分布图、原始数据等，方便单独分析
            np.save(os.path.join(save_dir, 'scores.npy'), scores)
            np.save(os.path.join(save_dir, 'labels.npy'), labels)

            # 记录汇总信息
            results_summary.append({
                'Experiment': exp_name,
                'Load': load,
                'Speed': speed,
                'Faults': str(fault_types),
                'AUC': auc,
                'F1': f1
            })

            # (可选) 在这里调用绘图函数画出该工况的 Score 曲线
            # plot_score_curve(scores, labels, save_path=os.path.join(save_dir, 'score_plot.png'))

        except Exception as e:
            print(f"[Error] Failed on {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. 保存总表
    if results_summary:
        df = pd.DataFrame(results_summary)
        summary_path = os.path.join(Config.OUTPUT_DIR, "eval_results", "summary_report.csv")
        df.to_csv(summary_path, index=False)
        print(f"\n>>> All experiments done. Summary saved to {summary_path}")


if __name__ == '__main__':
    run_evaluation_loop()