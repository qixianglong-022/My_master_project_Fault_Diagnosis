# run_rq3_advanced.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
# 直接引用模型定义类，以便灵活实例化
from models.phys_rdlinear import PhysRDLinearCls
from trainer import Trainer
from utils.tools import set_seed
from utils.rq1_kit import evaluate_rq1_comprehensive
from utils.rq3_kit import compute_mmd, plot_ablation_chart

# === 实验配置 ===
ABLATION_CONFIGS = {
    'Ablation_Base': {
        'pgfa': False, 'mtl': False, 'desc': '无物理引导 (Baseline)'
    },
    'Ablation_PGFA': {
        'pgfa': True, 'mtl': False, 'desc': '仅 PGFA'
    },
    'Ablation_MTL': {
        'pgfa': False, 'mtl': True, 'desc': '仅 MTL'
    },
    'Phys_RDLinear': {
        'pgfa': True, 'mtl': True, 'desc': '完整模型 (Ours)'
    }
}
GPU_ID = 0


def main():
    print("=========================================================")
    print("   RQ3 Automation: Physics Module Ablation Study")
    print("=========================================================")

    rq3_root = os.path.join("checkpoints_ch4", "rq3")
    os.makedirs(rq3_root, exist_ok=True)

    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据
    train_ds = Ch4DualStreamDataset(config, mode='train')
    test_ds = Ch4DualStreamDataset(config, mode='test')

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    # MMD 计算用的 Loader (Source 无 Shuffle, Target Shuffle 均可)
    mmd_src_dl = DataLoader(train_ds, batch_size=64, shuffle=False)
    mmd_tgt_dl = DataLoader(test_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    results = []

    for cfg_name, settings in ABLATION_CONFIGS.items():
        print(f"\n>>> [Task] Running: {cfg_name} ({settings['desc']})")

        # === [关键修复] 设置 config.MODEL_NAME ===
        config.MODEL_NAME = cfg_name
        # Trainer 需要依靠这个名字来决定 Forward 传参方式
        # ========================================

        save_dir = os.path.join(rq3_root, cfg_name)
        os.makedirs(save_dir, exist_ok=True)
        config.CHECKPOINT_DIR = save_dir  # 同时更新 Checkpoint 路径

        # 1. 初始化模型 (手动传入 flags)
        model = PhysRDLinearCls(
            config,
            enable_pgfa=settings['pgfa'],
            enable_mtl=settings['mtl'],
            enable_acoustic=True
        ).to(device)

        # 2. 训练
        trainer = Trainer(config, model, device)
        ckpt_path = os.path.join(save_dir, "model.pth")

        if os.path.exists(ckpt_path):
            print("    [Info] Loading checkpoint...")
            model.load_state_dict(torch.load(ckpt_path))
        else:
            print("    [Info] Training...")
            epochs = 25
            for epoch in range(epochs):
                trainer.train_epoch(train_dl)
            torch.save(model.state_dict(), ckpt_path)

        # 3. 评估准确率
        print("    [Eval] Evaluating Accuracy...")
        # 此时传入正确的 config.MODEL_NAME 给评估函数
        metrics = evaluate_rq1_comprehensive(model, test_dl, device, save_dir, cfg_name)

        # 4. 计算 MMD
        print("    [Eval] Calculating MMD Distance (Source <-> Target)...")
        mmd_val = compute_mmd(model, mmd_src_dl, mmd_tgt_dl, device)
        print(f"    -> MMD: {mmd_val:.4f}")

        row = {
            'Config': cfg_name,
            'Description': settings['desc'],
            'Avg_Acc': metrics.get('Avg_Acc', 0),
            'Acc_0kg': metrics.get('Acc_0kg', 0),
            'Acc_400kg': metrics.get('Acc_400kg', 0),
            'MMD_Distance': mmd_val
        }
        results.append(row)

    # 5. 汇总与绘图
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(rq3_root, "RQ3_Ablation_Results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> 汇总表已保存: {csv_path}")

        # 生成论文图表
        plot_ablation_chart(csv_path, rq3_root)

    print("\n=========================================================")
    print("   RQ3 Completed! See checkpoints/rq3/")
    print("=========================================================")


if __name__ == "__main__":
    main()