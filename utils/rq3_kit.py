# utils/rq3_kit.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import set_style

# === 全局设置 ===
FONT_PROP = set_style()


# === 1. MMD (最大均值差异) 计算内核 ===
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算高斯核矩阵"""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def compute_mmd(model, src_loader, tgt_loader, device):
    model.eval()
    features = []

    # Hook 位置：V2 模型的 cls_head 还是存在的
    def hook_fn(module, input, output):
        features.append(input[0])  # input to cls_head is the fused feature

    handle = model.cls_head.register_forward_hook(hook_fn)
    max_samples = 300
    src_feats, tgt_feats = [], []

    def get_feats(loader):
        collected = []
        with torch.no_grad():
            # [V2 Change] Unpack 7 items
            for mic, mac, ac, cur, spd, ld, _ in loader:
                mic, mac = mic.to(device), mac.to(device)
                ac, cur = ac.to(device), cur.to(device)
                spd, ld = spd.to(device), ld.to(device)

                is_phys = model.__class__.__name__.startswith('Phys')
                if is_phys:
                    _ = model(mic, mac, ac, cur, spd, ld)
                else:
                    full_x = torch.cat([mic.squeeze(-1), mac.squeeze(-1), ac, cur], dim=1)
                    _ = model(full_x, spd)

                collected.append(features[-1])
                features.clear()
                if len(collected) * mic.size(0) >= max_samples: break
        return collected

    # Run
    src_chunks = get_feats(src_loader)
    tgt_chunks = get_feats(tgt_loader)
    handle.remove()

    if not src_chunks or not tgt_chunks: return 1.0

    S = torch.cat(src_chunks, dim=0)[:max_samples]
    T = torch.cat(tgt_chunks, dim=0)[:max_samples]

    # 计算 MMD
    loss = 0
    kernels = guassian_kernel(S, T)
    XX = kernels[:len(S), :len(S)]
    YY = kernels[len(S):, len(S):]
    XY = kernels[:len(S), len(S):]
    YX = kernels[len(S):, :len(S)]
    loss = torch.mean(XX + YY - XY - YX)

    return loss.item()


# === 2. 可视化绘图 ===
def plot_ablation_chart(csv_path, save_dir):
    """绘制双轴图: 柱状图(Acc) + 折线图(MMD)"""
    print(">>> 正在绘制消融实验分析图...")
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    # 映射中文名称用于展示
    name_map = {
        'Ablation_Base': '基础模型\n(Base)',
        'Ablation_PGFA': '仅物理引导\n(PGFA)',
        'Ablation_MTL': '仅负载解耦\n(MTL)',
        'Phys_RDLinear': '完整方法\n(Ours)'
    }
    df['DisplayName'] = df['Config'].map(name_map)

    # 排序
    order = ['Ablation_Base', 'Ablation_PGFA', 'Ablation_MTL', 'Phys_RDLinear']
    # 确保 Config 列是 Categorical 类型以便排序
    df['Config'] = pd.Categorical(df['Config'], categories=order, ordered=True)
    df = df.sort_values('Config')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴: 准确率 (柱状图)
    color_bar = '#6baed6'  # 浅蓝
    bars = ax1.bar(df['DisplayName'], df['Avg_Acc'], color=color_bar, alpha=0.8, width=0.5, label='平均准确率 (%)')
    ax1.set_ylabel('诊断准确率 (%)', fontsize=13, fontproperties=FONT_PROP, color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(50, 100)

    # 标数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=11)

    # 右轴: MMD (折线图)
    ax2 = ax1.twinx()
    color_line = '#d62728'  # 红色
    line = ax2.plot(df['DisplayName'], df['MMD_Distance'], color=color_line, marker='o',
                    linewidth=3, markersize=10, linestyle='--', label='MMD 分布距离')
    ax2.set_ylabel('特征分布距离 (MMD)', fontsize=13, fontproperties=FONT_PROP, color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)

    # 标数值 (MMD)
    for i, txt in enumerate(df['MMD_Distance']):
        ax2.text(i, txt + 0.01, f'{txt:.3f}', ha='center', va='bottom', color=color_line, fontsize=11,
                 fontweight='bold')

    plt.title('物理引导模块消融实验分析 (RQ3)', fontsize=15, fontproperties=FONT_PROP, pad=20)

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', prop=FONT_PROP)

    save_path = os.path.join(save_dir, 'RQ3_Ablation_Analysis.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"    -> 已保存: {os.path.basename(save_path)}")