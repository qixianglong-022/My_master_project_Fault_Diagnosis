# run_ablation.py
import os
import torch
from config import Config
from train import train  # 直接调用 train.py 里的函数


def run_experiment(exp_name, use_revin, use_speed):
    print(f"\n>>> Running Experiment: {exp_name}")

    # 1. 动态修改配置
    Config.USE_REVIN = use_revin
    Config.USE_SPEED = use_speed

    # 修改输出目录，避免覆盖
    Config.OUTPUT_DIR = f"./checkpoints/{exp_name}"

    # 2. 运行训练
    train()  # 这会生成该实验下的 checkpoint.pth 和 threshold.npy

    print(f">>> {exp_name} Finished. Model saved in {Config.OUTPUT_DIR}")


if __name__ == '__main__':
    # 实验 1: 完整模型 (Proposed)
    run_experiment(exp_name="Full_RDLinear", use_revin=True, use_speed=True)

    # 实验 2: 去掉转速引导 (w/o Speed) -> 验证工况解耦能力
    run_experiment(exp_name="Ablation_NoSpeed", use_revin=True, use_speed=False)

    # 实验 3: 去掉 RevIN (w/o RevIN) -> 验证抗分布漂移能力
    run_experiment(exp_name="Ablation_NoRevIN", use_revin=False, use_speed=True)