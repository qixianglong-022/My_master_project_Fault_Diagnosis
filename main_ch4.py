import torch
import argparse
import os
import pandas as pd
import json
from torch.utils.data import DataLoader
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from trainer import Trainer
from utils.tools import set_seed
from models.factory import get_model  # 确保 models/factory.py 存在


def run_ch4_experiment():
    # 1. 参数解析 (这一步是解决你报错的关键)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model Name')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    # 2. 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = Ch4Config()
    config.MODEL_NAME = args.model_name  # 动态覆盖配置

    # 根据模型名调整输出路径，避免冲突
    config.CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "checkpoints_ch4", args.model_name)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n==================================================")
    print(f">>> Running Pipeline for: {args.model_name}")
    print(f"    Output Dir: {config.CHECKPOINT_DIR}")
    print(f"==================================================")

    # 3. 准备数据
    train_ds = Ch4DualStreamDataset(config, mode='train')
    test_ds = Ch4DualStreamDataset(config, mode='test')

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Data Loaded: Train Samples={len(train_ds)}, Test Samples={len(test_ds)}")

    # 4. 初始化模型与训练器
    try:
        model = get_model(args.model_name, config).to(device)
    except Exception as e:
        print(f"[Fatal] Model initialization failed: {e}")
        exit(1)

    trainer = Trainer(config, model, device)

    # 5. 训练阶段 (Train on Source: 200kg)
    print("\n>>> [Phase 1] Training Source Model...")

    TRAIN_EPOCHS = 55

    for epoch in range(TRAIN_EPOCHS):
        metrics = trainer.train_epoch(train_dl)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1}/{TRAIN_EPOCHS} | Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.2f}%")

    # 保存权重
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "model.pth"))
    print("    Training Completed & Model Saved.")

    # 6. 评估阶段 (Evaluation)
    print("\n>>> [Phase 2] Evaluating & Generating Metrics...")

    # A. 评估源域 (Source 200kg) - 用 eval 模式
    # 复用 train_ds 但 shuffle=False
    source_eval_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    acc_source = trainer.evaluate(source_eval_dl,
                                  save_path=os.path.join(config.CHECKPOINT_DIR, "eval_source_200kg.csv"))

    # B. 评估目标域 (Target 0kg & 400kg)
    acc_target_all = trainer.evaluate(test_dl, save_path=os.path.join(config.CHECKPOINT_DIR, "eval_target_all.csv"))

    # 7. 计算拆分指标 (0kg vs 400kg)
    try:
        df_tgt = pd.read_csv(os.path.join(config.CHECKPOINT_DIR, "eval_target_all.csv"))

        # 你的 Dataset 中 target_load 是归一化的 (x/400.0)
        # Load=0kg -> 0.0, Load=400kg -> 1.0
        # 判断逻辑：Load_kg < 100 为轻载，> 300 为重载

        # 兼容逻辑：检查 CSV 列名，trainer.py 中保存的是 'Load_kg'
        acc_0kg = df_tgt[df_tgt['Load_kg'] < 100]['Is_Correct'].mean() * 100
        acc_400kg = df_tgt[df_tgt['Load_kg'] > 300]['Is_Correct'].mean() * 100
    except Exception as e:
        print(f"[Warn] Metric splitting failed: {e}. Using overall acc for sub-metrics.")
        acc_0kg = acc_target_all
        acc_400kg = acc_target_all

    print(f"\n>>> [Final Results] {args.model_name}")
    print(f"    Source (200kg): {acc_source:.2f}%")
    print(f"    Target (0kg):   {acc_0kg:.2f}%")
    print(f"    Target (400kg): {acc_400kg:.2f}%")
    print(f"    Avg Target:     {(acc_0kg + acc_400kg) / 2:.2f}%")

    # 8. 保存摘要供自动化脚本读取
    summary = {
        "Model": args.model_name,
        "Source_200kg": acc_source,
        "Target_0kg": acc_0kg,
        "Target_400kg": acc_400kg,
        "Avg_Target": (acc_0kg + acc_400kg) / 2
    }
    with open(os.path.join(config.CHECKPOINT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    run_ch4_experiment()