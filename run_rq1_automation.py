# run_rq1_advanced.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
# 直接引入合并后的工具箱
from utils.rq1_kit import (
    evaluate_rq1_comprehensive,
    plot_radar_chart_from_csv,
    plot_tsne_for_model,
    plot_fusion_heatmap_automated
)

MODELS_TO_RUN = [
    'ResNet-18',
    'FD-CNN',
    'TiDE',
    'Vanilla RDLinear',
    'Phys-RDLinear'
]
GPU_ID = 0


def evaluate_source_domain(model, dataloader, device):
    """源域精度计算 (修复版：兼容8返回值)"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # [修改点] 增加一个 _ 来接收第8个返回值 phys_load，或者使用 *args
        for mic, mac, ac, cur, spd, ld, y_cls, _ in dataloader:

            mic, mac = mic.to(device), mac.to(device)
            ac, cur = ac.to(device), cur.to(device)
            spd, ld = spd.to(device), ld.to(device)
            y_cls = y_cls.to(device)

            is_phys = hasattr(model, 'pgfa') or model.__class__.__name__.startswith('Phys')

            if is_phys:
                # Phys 模型: 全量输入
                out = model(mic, mac, ac, cur, spd, ld)
            else:
                # 基线模型: 只喂 Micro 和 Speed
                out = model(mic, speed=spd)

            # 智能解包
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            correct += (torch.argmax(logits, dim=1) == y_cls).sum().item()
            total += y_cls.size(0)

    return (correct / total) * 100 if total > 0 else 0.0


def main():
    print("=========================================================")
    print("   RQ1 终极自动化: 训练 -> 评估 -> 绘图")
    print("=========================================================")

    rq1_root = os.path.join("checkpoints_ch4", "rq1")
    os.makedirs(rq1_root, exist_ok=True)
    final_results = []

    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据 (只需加载一次)
    print(">>> 加载数据集...")
    train_ds = Ch4DualStreamDataset(config, mode='train')
    test_ds = Ch4DualStreamDataset(config, mode='test')

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    source_eval_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. 遍历模型训练与评估
    for model_name in MODELS_TO_RUN:
        print(f"\n>>> [Task 1] 处理模型: {model_name}...")

        config.MODEL_NAME = model_name
        config.CHECKPOINT_DIR = os.path.join(rq1_root, model_name)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # 初始化
        try:
            model = get_model(model_name, config).to(device)
        except Exception as e:
            print(f"[Error] Init failed: {e}")
            continue

        # 训练或加载
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model.pth")
        if os.path.exists(ckpt_path):
            print(f"    [Info] 加载已有权重...")
            model.load_state_dict(torch.load(ckpt_path))
        else:
            print(f"    [Info] 开始训练...")
            trainer = Trainer(config, model, device)
            epochs = 25 if 'ResNet' in model_name else config.EPOCHS
            for epoch in range(epochs):
                trainer.train_epoch(train_dl)
            torch.save(model.state_dict(), ckpt_path)

        # 评估
        print("    [Eval] 计算源域与目标域精度...")
        acc_src = evaluate_source_domain(model, source_eval_dl, device)
        # 这里的 evaluate_rq1_comprehensive 现在包含了物理门控逻辑
        row_metrics = evaluate_rq1_comprehensive(model, test_dl, device, config.CHECKPOINT_DIR, model_name)
        row_metrics['Acc_Source'] = acc_src
        final_results.append(row_metrics)
        print(f"    -> 结果: Src={acc_src:.1f}%, Tgt_Avg={row_metrics.get('Avg_Acc', 0):.1f}%")

    # 3. 保存汇总 CSV
    csv_path = os.path.join(rq1_root, "RQ1_Final_Comparison.csv")
    if final_results:
        df = pd.DataFrame(final_results)
        first_cols = ['Model', 'Acc_Source', 'Avg_Acc', 'Avg_F1', 'Acc_0kg', 'Acc_400kg']
        cols = first_cols + [c for c in df.columns if c not in first_cols]
        df[cols].to_csv(csv_path, index=False)
        print(f"\n>>> [Task 2] 汇总表已保存: {csv_path}")

        # 4. 绘制雷达图
        print("\n>>> [Task 3] 生成雷达图...")
        plot_radar_chart_from_csv(csv_path, rq1_root, load_kg=0)
        plot_radar_chart_from_csv(csv_path, rq1_root, load_kg=400)

    # 5. 生成 t-SNE (仅针对关键模型)
    print("\n>>> [Task 4] 生成 t-SNE (选取对比模型)...")
    tsne_models = ['Phys-RDLinear', 'ResNet-18']

    for m_name in tsne_models:
        if m_name not in MODELS_TO_RUN: continue

        # 重新加载模型
        config.MODEL_NAME = m_name
        config.CHECKPOINT_DIR = os.path.join(rq1_root, m_name)
        model = get_model(m_name, config).to(device)
        model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "model.pth")))

        # 绘图
        plot_tsne_for_model(model, source_eval_dl, test_dl, device, rq1_root, m_name)

    plot_fusion_heatmap_automated()

    print("\n=========================================================")
    print("   RQ1 全流程结束！请查看 checkpoints/rq1 文件夹")
    print("=========================================================")


if __name__ == "__main__":
    main()