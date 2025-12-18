# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os
import matplotlib.font_manager as fm


# === 1. 全局配置 ===
def set_style():
    # [修复] 增加样式兼容性处理
    try:
        # 尝试使用新版样式
        plt.style.use('seaborn-v0_8-whitegrid')
    except (OSError, FileNotFoundError):
        try:
            # 回退到旧版样式
            plt.style.use('seaborn-whitegrid')
        except (OSError, FileNotFoundError):
            # 如果都失败，使用默认样式
            plt.style.use('seaborn')

    # 字体配置 (保持原逻辑)
    font_candidates = [r'C:\Windows\Fonts\simhei.ttf', r'/System/Library/Fonts/PingFang.ttc']
    font_prop = None
    for f in font_candidates:
        if os.path.exists(f):
            font_prop = fm.FontProperties(fname=f)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            break
    return font_prop

FONT_PROP = set_style()
CLASS_NAMES = ['HH', 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']

# 强制颜色映射 (确保 200kg 是红色，0kg 蓝色，400kg 黄色/橙色)
DOMAIN_COLORS = {
    200: '#d62728',  # Red (Source)
    0: '#1f77b4',  # Blue (Target Light)
    400: '#ff7f0e'  # Orange (Target Heavy)
}


def extract_features(model, loader, device):
    """提取特征通用函数 (适配 Ablation 实验)"""
    model.eval()
    feats, preds, labs, loads = [], [], [], []

    with torch.no_grad():
        for micro_x, macro_x, speed, y_cls, y_load in loader:
            micro_x = micro_x.to(device)
            macro_x = macro_x.to(device)
            speed = speed.to(device)

            # --- [核心修改] 复现 PhysRDLinear 的 Forward 逻辑 ---
            # 必须判断当前模型是否有对应的模块 (适配消融实验)

            # 1. Stream 1: Micro
            if hasattr(model, 'revin_micro'):
                x1 = model.revin_micro(micro_x, 'norm')
                sea, trend = model.decomp(x1)

                # [Fix] 检查是否存在 pgfa 模块
                if hasattr(model, 'pgfa') and model.pgfa is not None:
                    sea_guided = model.pgfa(sea, speed)
                else:
                    sea_guided = sea  # 没有 PGFA 就直通

                f_trend = model.lin_trend(trend.squeeze(-1))
                f_sea = model.lin_sea(sea_guided.squeeze(-1))

                # 2. Stream 2: Macro
                x2 = model.revin_macro(macro_x, 'norm')
                f_macro = model.lin_macro(x2.squeeze(-1))

                # Fusion Feature
                fusion = torch.cat([f_trend, f_sea, f_macro], dim=1)

                # Logits
                logits = model.cls_head(fusion)

            else:
                # 兼容基线模型 (如果以后要用这个函数跑 ResNet 的特征提取)
                # 基线模型没有 decomp/revin 等结构，直接取倒数第二层比较麻烦
                # 这里简单处理：直接 forward 拿 logits，特征暂不可用
                # 或者针对特定基线写特定逻辑。
                # 针对 RQ3，我们只跑 PhysRDLinear 家族，上面的 if 分支就够了。
                full_x = torch.cat([micro_x, macro_x], dim=1)
                logits, _ = model(full_x, speed)
                fusion = torch.zeros(logits.size(0), 1)  # 占位符

            preds_batch = torch.argmax(logits, dim=1)

            feats.append(fusion.cpu().numpy())
            preds.append(preds_batch.cpu().numpy())
            labs.append(y_cls.cpu().numpy())
            loads.append(y_load.cpu().numpy())

    return (np.concatenate(feats), np.concatenate(preds),
            np.concatenate(labs), np.concatenate(loads))


def plot_tsne_domain(features, loads, save_path):
    """绘制域分布 t-SNE (带源域)"""
    print(">>> [Vis] Calculating t-SNE...")
    # 降维
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(features)

    # 还原真实负载 (归一化值 -> 物理值)
    # 假设 load 归一化是 /400.0。注意 0kg 可能会有浮点误差
    real_loads = (loads * 400).round().astype(int).flatten()

    plt.figure(figsize=(10, 8))

    # 遍历画图，确保颜色固定
    unique_loads = np.unique(real_loads)
    for ld in unique_loads:
        mask = (real_loads == ld)
        # 如果有未知负载，给个默认灰
        color = DOMAIN_COLORS.get(ld, '#7f7f7f')
        label = f"{ld} kg ({'Source' if ld == 200 else 'Target'})"

        plt.scatter(X_emb[mask, 0], X_emb[mask, 1],
                    label=label, c=color, s=50, alpha=0.7, edgecolors='w')

    plt.legend(fontsize=12)
    plt.title("Domain Generalization Analysis (t-SNE)", fontproperties=FONT_PROP, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(9, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Target Domain)", fontproperties=FONT_PROP, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_visualization_pipeline(model, target_loader, source_loader, device, output_dir, acc):
    """
    参数:
        source_loader: 用于提供源域数据 (200kg) 画 t-SNE
    """
    os.makedirs(output_dir, exist_ok=True)

    # === 1. 生成优雅的文件名前缀 ===
    # 格式: PhysRDLinear_Src200_Tgt0-400_AccXX.X
    prefix = f"PhysRDLinear_Src200_Tgt0-400_Acc{acc:.1f}"

    print(f"\n>>> [Vis] Starting Visualization: {prefix}")

    # === 2. 提取数据 ===
    # A. 目标域 (用于混淆矩阵 & t-SNE)
    tgt_feats, tgt_preds, tgt_labs, tgt_loads = extract_features(model, target_loader, device)

    # B. 源域 (仅用于 t-SNE 对比，不需要全部，采样一部分即可)
    if source_loader is not None:
        src_feats, _, _, src_loads = extract_features(model, source_loader, device)
        # 合并特征用于 t-SNE
        combined_feats = np.concatenate([src_feats, tgt_feats])
        combined_loads = np.concatenate([src_loads, tgt_loads])
    else:
        combined_feats, combined_loads = tgt_feats, tgt_loads

    # === 3. 绘图 ===
    # 图1: 混淆矩阵 (只画目标域的，因为源域肯定是 100% 没什么意义)
    cm_path = os.path.join(output_dir, f"{prefix}_ConfusionMatrix.pdf")
    plot_confusion_matrix(tgt_labs, tgt_preds, cm_path)

    # 图2: 域泛化 t-SNE (画合并后的数据)
    tsne_path = os.path.join(output_dir, f"{prefix}_tSNE_Domain.pdf")
    plot_tsne_domain(combined_feats, combined_loads, tsne_path)

    print(f"    -> Saved: {os.path.basename(cm_path)}")
    print(f"    -> Saved: {os.path.basename(tsne_path)}")
    print(">>> [Vis] Done.\n")