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
    plt.style.use('seaborn-v0_8-whitegrid')
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
    """提取特征通用函数"""
    model.eval()
    feats, preds, labs, loads = [], [], [], []

    with torch.no_grad():
        for micro_x, speed, y_cls, y_load in loader:
            micro_x = micro_x.to(device)
            speed = speed.to(device)

            # 复现前向过程
            x = model.revin(micro_x, 'norm')
            seasonal, trend = model.decomp(x)
            seasonal_guided = model.pgfa(seasonal, speed)
            t_feat = model.linear_trend(trend.squeeze(-1))
            s_feat = model.linear_seasonal(seasonal_guided.squeeze(-1))

            # 融合特征
            fusion = torch.cat([t_feat, s_feat], dim=1)

            logits, _ = model(micro_x, speed)
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