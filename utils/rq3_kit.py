import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.visualization import set_style

FONT_PROP = set_style()


# ==============================================================================
# 1. MMD 计算逻辑 (核心)
# ==============================================================================

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算 Gram 核矩阵 (用于 MMD)
    """
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


def compute_mmd_loss(source, target):
    """
    计算 Maximum Mean Discrepancy (MMD) 损失
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def extract_features_for_mmd(model, dataloader, device, max_batches=10):
    """
    从模型中提取特征 (Latent Features) 而不是 Logits
    """
    model.eval()
    features = []

    # 1. 定义 Hook 抓取特征
    # 抓取 cls_head 之前的输入特征 (融合后的特征)
    hook_data = []

    def hook_fn(module, input, output):
        # input[0] 通常是进入 Linear 层的特征 Tensor
        hook_data.append(input[0].detach())

    # 尝试 Hook 模型的 cls_head (假设是 nn.Sequential 或 Linear)
    handle = None
    if hasattr(model, 'cls_head'):
        # 注册在 cls_head 的第一层
        handle = model.cls_head.register_forward_hook(hook_fn)

    with torch.no_grad():
        count = 0
        # === [核心修改] 解包 8 个变量 (适配最新 DataLoader) ===
        for micro, macro, ac, cur, spd, ld, y_cls, _ in dataloader:
            if count >= max_batches: break

            micro, macro = micro.to(device), macro.to(device)
            ac, cur = ac.to(device), cur.to(device)
            spd, ld = spd.to(device), ld.to(device)

            # Forward (Hook 会自动捕获特征)
            # === [核心修改] 传入 6 个参数 ===
            hook_data.clear()  # 清空上一轮
            _ = model(micro, macro, ac, cur, spd, ld)

            if hook_data:
                features.append(hook_data[-1])  # 取最后一个 Hook 的结果

            count += 1

    if handle:
        handle.remove()

    if features:
        return torch.cat(features, dim=0)
    else:
        return None


def compute_mmd(model, src_loader, tgt_loader, device):
    """
    计算源域和目标域特征空间的 MMD 距离
    """
    # 提取特征
    src_feats = extract_features_for_mmd(model, src_loader, device)
    tgt_feats = extract_features_for_mmd(model, tgt_loader, device)

    if src_feats is None or tgt_feats is None:
        print("    [Warn] Feature extraction failed for MMD.")
        return 0.0

    # 对齐数量 (MMD 需要 sample 数一致或接近，这里取最小公倍数或截断)
    min_len = min(len(src_feats), len(tgt_feats))
    # 截断到相同长度，且不能太长否则 OOM
    eval_len = min(min_len, 500)

    src_sample = src_feats[:eval_len]
    tgt_sample = tgt_feats[:eval_len]

    mmd_val = compute_mmd_loss(src_sample, tgt_sample)
    return mmd_val.item()


# ==============================================================================
# 2. 绘图功能
# ==============================================================================

def plot_ablation_chart(csv_path, save_dir):
    """
    绘制消融实验结果 (条形图 + 折线图双轴)
    Bar: 准确率
    Line: MMD 距离 (越低越好)
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    # 简化名称
    df['Label'] = df['Config'].apply(lambda x: x.replace('Ablation_', '').replace('Phys_RDLinear', 'Full'))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. 柱状图 - 准确率 (左轴)
    # 使用 0kg 和 400kg 的平均值或者 Avg_Acc
    x = np.arange(len(df))
    width = 0.35

    bars = ax1.bar(x, df['Avg_Acc'], width, color='#4c72b0', alpha=0.8, label='平均准确率 (%)')
    ax1.set_ylabel('准确率 (%)', color='#4c72b0', fontproperties=FONT_PROP, fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#4c72b0')
    ax1.set_ylim(0, 100)

    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=11)

    # 2. 折线图 - MMD (右轴)
    ax2 = ax1.twinx()
    ax2.plot(x, df['MMD_Distance'], color='#c44e52', marker='o', linewidth=2, markersize=8, label='特征分布距离 (MMD)')
    ax2.set_ylabel('MMD 距离 (越低越好)', color='#c44e52', fontproperties=FONT_PROP, fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#c44e52')

    # 设置 X 轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Label'], fontproperties=FONT_PROP, fontsize=12, rotation=0)
    ax1.set_xlabel("消融配置", fontproperties=FONT_PROP, fontsize=14)

    # 标题
    plt.title("物理引导模块消融实验: 准确率 vs 域偏移距离", fontproperties=FONT_PROP, fontsize=16, pad=20)

    # 图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', prop=FONT_PROP)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'RQ3_Ablation_MMD.pdf'), dpi=300)
    plt.close()