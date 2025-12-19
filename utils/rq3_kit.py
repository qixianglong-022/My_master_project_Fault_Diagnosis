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
    """
    计算源域和目标域特征空间的 MMD 距离
    """
    model.eval()

    # 定义 Hook 获取倒数第二层特征 (Fusion Layer)
    features = []

    def hook_fn(module, input, output):
        # input[0] 是进入分类头之前的 fusion 特征
        features.append(input[0])

    # 注册 Hook 到 cls_head (假设它是最后一层)
    # Phys-RDLinear 结构: model.cls_head = nn.Sequential(...)
    # 我们Hook cls_head 的第一层
    handle = model.cls_head.register_forward_hook(hook_fn)

    # 提取特征 (采样一部分，避免 OOM)
    max_samples = 300
    src_feats, tgt_feats = [], []

    with torch.no_grad():
        # Source Loop
        for micro, macro, ac, spd, _, _ in src_loader:
            micro, macro, ac, spd = micro.to(device), macro.to(device), ac.to(device), spd.to(device)

            # === [关键修复] ===
            # 不再只检查 pgfa 属性，而是检查类名。
            # 只要是 PhysRDLinearCls (无论是否消融)，都必须传 4 个参数
            is_phys = model.__class__.__name__.startswith('Phys')

            if is_phys:
                _ = model(micro, macro, ac, spd)  # 触发 Hook
            else:
                # 兼容 RQ1 的基线模型
                x = torch.cat([micro.squeeze(-1), macro.squeeze(-1)], dim=1)
                _ = model(x, spd)

            src_feats.append(features[-1])  # 取最近一次 Hook 的结果
            features.clear()  # 清空
            if len(src_feats) * micro.size(0) >= max_samples: break

        # Target Loop
        for micro, macro, ac, spd, _, _ in tgt_loader:
            micro, macro, ac, spd = micro.to(device), macro.to(device), ac.to(device), spd.to(device)

            # 同样的判断逻辑
            is_phys = model.__class__.__name__.startswith('Phys')

            if is_phys:
                _ = model(micro, macro, ac, spd)
            else:
                x = torch.cat([micro.squeeze(-1), macro.squeeze(-1)], dim=1)
                _ = model(x, spd)

            tgt_feats.append(features[-1])
            features.clear()
            if len(tgt_feats) * micro.size(0) >= max_samples: break

    handle.remove()  # 移除 Hook

    # 拼接
    if len(src_feats) == 0 or len(tgt_feats) == 0: return 1.0  # Error

    S = torch.cat(src_feats, dim=0)[:max_samples]
    T = torch.cat(tgt_feats, dim=0)[:max_samples]

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