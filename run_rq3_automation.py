import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
from utils.visualization import extract_features, set_style  # 复用


# === 新增：PGFA 物理对齐可视化工具 ===
def visualize_pgfa_alignment(model, dataloader, device, save_dir):
    """
    画出 Mask 和 Spectrum 的叠加图，证明位置对齐。
    """
    model.eval()
    font_prop = set_style()  # 设置中文字体
    os.makedirs(save_dir, exist_ok=True)

    # 抽取一个 Batch
    micro, macro, acoustic, speed, y_cls, load = next(iter(dataloader))
    micro = micro.to(device)
    speed = speed.to(device)

    # 找一个转速比较典型的样本 (例如最接近 45Hz 的)
    target_speed = 45.0
    diff = torch.abs(speed - target_speed)
    idx = torch.argmin(diff)

    # 获取该样本数据
    sample_micro = micro[idx]  # [512, 1] (Log Spectrum)
    sample_speed = speed[idx]  # Scalar (Hz)

    # 1. 计算 PGFA Mask
    # 我们需要手动调用 model.pgfa 来生成 mask
    if not hasattr(model, 'pgfa'):
        print("[Vis] Model has no PGFA module, skipping alignment plot.")
        return

    with torch.no_grad():
        # 模拟 forward 过程中的输入
        # 注意：Mask 是根据 speed 生成的，与输入内容无关，只与频率轴有关
        # model.pgfa.forward(seasonal, speed) -> return seasonal * (1 + alpha * mask)
        # 我们想单独把 mask 拿出来。

        # 重新生成 Mask (复制 PGFA 内部逻辑)
        f_axis = model.pgfa.freq_axis.to(device).view(1, -1, 1)  # [1, 512, 1]
        s = sample_speed.view(1, 1, 1)
        sigma = model.pgfa.sigma

        # Gaussian Mask 公式
        mask = torch.exp(- (f_axis - s) ** 2 / (2 * sigma ** 2)) + \
               torch.exp(- (f_axis - 2 * s) ** 2 / (2 * sigma ** 2)) + \
               torch.exp(- (f_axis - 3 * s) ** 2 / (2 * sigma ** 2))

        mask_np = mask.cpu().numpy().flatten()

    # 2. 获取频谱 (Micro Stream)
    # 输入已经是 Log 谱了，直接画
    spec_np = sample_micro.cpu().numpy().flatten()

    # 3. 绘图
    plt.figure(figsize=(10, 4))

    # 双坐标轴：左边频谱，右边 Mask 权重
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 画频谱
    freqs = np.linspace(0, 512, len(spec_np))  # 假设 0-512Hz
    ax1.plot(freqs, spec_np, color='#1f77b4', alpha=0.6, label='Micro-Stream Spectrum (Log)')
    ax1.set_xlabel('Frequency (Hz)', fontproperties=font_prop, fontsize=12)
    ax1.set_ylabel('Amplitude (Log)', fontproperties=font_prop, fontsize=12, color='#1f77b4')

    # 画 Mask
    ax2.plot(freqs, mask_np, color='#d62728', linewidth=2, linestyle='--', label='PGFA Attention Mask')
    ax2.set_ylabel('Attention Weight', fontproperties=font_prop, fontsize=12, color='#d62728')

    # 标注转速
    real_speed = sample_speed.item()
    plt.title(f"PGFA 物理对齐验证 (转速: {real_speed:.1f} Hz)", fontproperties=font_prop, fontsize=14)

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=font_prop)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Vis_PGFA_Alignment.pdf"))
    plt.close()
    print(f">>> [Vis] PGFA Alignment plot saved to {save_dir}/Vis_PGFA_Alignment.pdf")


# === 主消融实验逻辑 ===
def run_ablation(model_alias, gpu_id=0):
    config = Ch4Config()
    config.MODEL_NAME = model_alias
    config.CHECKPOINT_DIR = os.path.join("checkpoints_rq3", model_alias)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    set_seed(config.SEED)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> [RQ3] Running: {model_alias}")

    # 数据 (Shuffle=True for training)
    train_dl = DataLoader(Ch4DualStreamDataset(config, 'train'), batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, 'test'), batch_size=config.BATCH_SIZE, shuffle=False)

    # 模型
    model = get_model(model_alias, config).to(device)
    trainer = Trainer(config, model, device)

    # 训练
    for epoch in range(20):
        trainer.train_epoch(train_dl)

    # 评估
    eval_csv = os.path.join(config.CHECKPOINT_DIR, "eval_all.csv")
    trainer.evaluate(test_dl, save_path=eval_csv)

    # === [Key] 如果是 Phys-RDLinear，画物理对齐图 ===
    if model_alias == 'Phys-RDLinear':
        # 用测试集数据画，看看泛化时的 Mask 对不对
        visualize_pgfa_alignment(model, test_dl, device, config.CHECKPOINT_DIR)

    return eval_csv


if __name__ == "__main__":
    configs = ['Ablation-Base', 'Ablation-PGFA', 'Ablation-MTL', 'Phys-RDLinear']

    for c in configs:
        run_ablation(c)

    print("\n>>> RQ3 All Done.")