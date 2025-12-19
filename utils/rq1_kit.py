# utils/rq1_kit.py
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

# 引用你的样式设置
from utils.visualization import set_style

# === 1. 全局配置 (中文与样式) ===
FONT_PROP = set_style()

# 故障类别映射
CLASS_NAMES_EN = ['HH', 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']
FAULT_MAP_CN = {
    'HH': '健康', 'RU': '转子不平衡', 'RM': '转子不对中', 'SW': '定子绕组',
    'VU': '电压不平衡', 'BR': '转子弯曲', 'KA': '转子断条', 'FB': '轴承故障'
}
FAULT_LABELS_SHORT = [FAULT_MAP_CN[c] for c in CLASS_NAMES_EN]

# 颜色盘
COLORS = ['#7f7f7f', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
MARKERS = ['o', 's', '^', 'D', '*']


# === 2. 核心评估函数 ===

def compute_detailed_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES_EN)))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)
    metrics = {'Acc': acc * 100, 'F1': f1 * 100}
    for i, name in enumerate(CLASS_NAMES_EN):
        metrics[f'Acc_{name}'] = per_class_acc[i] * 100
    return metrics


def plot_cm(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=FAULT_LABELS_SHORT, yticklabels=FAULT_LABELS_SHORT,
                cbar=False, annot_kws={"size": 9})
    plt.title(title, fontsize=14, fontproperties=FONT_PROP, pad=15)
    plt.ylabel('真实标签', fontsize=12, fontproperties=FONT_PROP)
    plt.xlabel('预测标签', fontsize=12, fontproperties=FONT_PROP)
    plt.xticks(rotation=45, fontproperties=FONT_PROP)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_rq1_comprehensive(model, dataloader, device, save_dir, model_name):
    """
    RQ1 核心评估 (含 Bug 修复版)
    """
    model.eval()
    all_preds, all_labels, all_loads = [], [], []

    ENABLE_PEG = True
    PEG_THRESHOLD = 0.05

    with torch.no_grad():
        for micro, macro, ac, spd, y_cls, y_load in dataloader:
            micro, macro = micro.to(device), macro.to(device)
            ac, spd = ac.to(device), spd.to(device)

            # === [关键修复] 判定逻辑增强 ===
            # 如果是 Phys 系列或其消融变体(Ablation)，都必须用 4 参数调用
            is_phys_model = hasattr(model, 'pgfa') or \
                            model_name.startswith('Phys') or \
                            model_name.startswith('Ablation')

            if is_phys_model:
                logits, _ = model(micro, macro, ac, spd)
            else:
                # 基线模型 (ResNet, TiDE等)
                full_x = torch.cat([micro.squeeze(-1), macro.squeeze(-1)], dim=1)
                logits, _ = model(full_x, spd)

            preds = torch.argmax(logits, dim=1)

            if ENABLE_PEG:
                input_rms = torch.sqrt(torch.mean(micro.squeeze(-1) ** 2, dim=1))
                is_noise = input_rms < PEG_THRESHOLD
                preds[is_noise] = 0

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_cls.cpu().numpy())
            all_loads.append(y_load.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    loads_real = (np.concatenate(all_loads).flatten() * 400).round().astype(int)

    pd.DataFrame({'Load': loads_real, 'True': y_true, 'Pred': y_pred}).to_csv(
        os.path.join(save_dir, 'raw_predictions.csv'), index=False)

    summary_row = {'Model': model_name}
    for ld in [0, 400]:
        mask = (loads_real == ld)
        if np.sum(mask) == 0: continue
        sub_true, sub_pred = y_true[mask], y_pred[mask]
        m = compute_detailed_metrics(sub_true, sub_pred)
        for k, v in m.items(): summary_row[f'{k}_{ld}kg'] = v
        plot_cm(sub_true, sub_pred, os.path.join(save_dir, f"CM_{ld}kg.pdf"), f"{model_name} - {ld}kg工况")

    if f'Acc_0kg' in summary_row and f'Acc_400kg' in summary_row:
        summary_row['Avg_Acc'] = (summary_row['Acc_0kg'] + summary_row['Acc_400kg']) / 2
        summary_row['Avg_F1'] = (summary_row['F1_0kg'] + summary_row['F1_400kg']) / 2

    return summary_row


# === 3. 高级可视化函数 ===

def plot_radar_chart_from_csv(csv_path, save_dir, load_kg):
    print(f"    [Plot] 生成雷达图 ({load_kg}kg)...")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    models = df['Model'].unique()

    N = len(CLASS_NAMES_EN)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    for i, model in enumerate(models):
        values = []
        for cat in CLASS_NAMES_EN:
            col = f'Acc_{cat}_{load_kg}kg'
            val = df.loc[df['Model'] == model, col].values[0] if col in df.columns else 0
            values.append(val)
        values += values[:1]

        is_ours = 'Phys' in model
        color = '#d62728' if is_ours else COLORS[i % len(COLORS)]
        lw = 3.5 if is_ours else 1.5
        alpha = 0.15 if is_ours else 0.05
        label = model.replace('Phys-RDLinear', 'Phys-RDLinear (本文)')

        ax.plot(angles, values, linewidth=lw, linestyle='solid', label=label, color=color,
                marker='o' if not is_ours else '*')
        ax.fill(angles, values, color=color, alpha=alpha)

    plt.xticks(angles[:-1], FAULT_LABELS_SHORT, size=14, fontproperties=FONT_PROP)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop=FONT_PROP)
    plt.title(f"各故障类型诊断精度 - {'轻载' if load_kg == 0 else '重载'}", y=1.1, fontproperties=FONT_PROP, size=16,
              weight='bold')
    plt.savefig(os.path.join(save_dir, f'RQ1_Radar_{load_kg}kg.pdf'), bbox_inches='tight')
    plt.close()


def plot_tsne_for_model(model, dataloader_src, dataloader_tgt, device, save_dir, model_name):
    print(f"    [Plot] 生成 t-SNE ({model_name})...")
    model.eval()
    MAX_PER_CLASS = 150

    def extract_feats(loader, domain_label):
        feats, labels, domains = [], [], []
        counts = {i: 0 for i in range(8)}
        with torch.no_grad():
            for micro, macro, ac, spd, y_cls, _ in loader:
                micro, macro, ac, spd = micro.to(device), macro.to(device), ac.to(device), spd.to(device)

                # 兼容性调用
                is_phys = hasattr(model, 'pgfa') or model_name.startswith('Phys') or model_name.startswith('Ablation')
                if is_phys:
                    out, _ = model(micro, macro, ac, spd)
                else:
                    full_x = torch.cat([micro.squeeze(-1), macro.squeeze(-1)], dim=1)
                    out, _ = model(full_x, spd)

                curr_f, curr_l = out.cpu().numpy(), y_cls.cpu().numpy()
                for f, l in zip(curr_f, curr_l):
                    if counts[l] < MAX_PER_CLASS:
                        feats.append(f);
                        labels.append(l);
                        domains.append(domain_label)
                        counts[l] += 1
        return np.array(feats), np.array(labels), np.array(domains)

    f_s, l_s, d_s = extract_feats(dataloader_src, 0)
    f_t, l_t, d_t = extract_feats(dataloader_tgt, 1)
    if len(f_s) == 0: return

    X = np.concatenate([f_s, f_t])
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    markers = ['o', 'x']

    for cls_id in range(8):
        c = cmap(cls_id)
        for dom_id in [0, 1]:
            mask = (np.concatenate([l_s, l_t]) == cls_id) & (np.concatenate([d_s, d_t]) == dom_id)
            if np.sum(mask) == 0: continue
            plt.scatter(X_emb[mask, 0], X_emb[mask, 1], color=c, marker=markers[dom_id],
                        label=FAULT_LABELS_SHORT[cls_id] if dom_id == 0 else None, alpha=0.7,
                        s=40 if dom_id == 0 else 60)

    plt.title(f"特征空间 t-SNE 可视化 - {model_name}", fontsize=16, fontproperties=FONT_PROP)
    plt.xticks([]);
    plt.yticks([])
    h, l = plt.gca().get_legend_handles_labels()
    l1 = plt.legend(h, l, loc='upper right', title="故障类型", prop=FONT_PROP, bbox_to_anchor=(1.15, 1))
    plt.setp(l1.get_title(), fontproperties=FONT_PROP)
    plt.gca().add_artist(l1)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='源域', markersize=10),
                       Line2D([0], [0], marker='x', color='gray', linestyle='None', label='目标域', markersize=10)]
    l2 = plt.legend(handles=legend_elements, loc='lower right', title="工况域", prop=FONT_PROP,
                    bbox_to_anchor=(1.15, 0))
    plt.setp(l2.get_title(), fontproperties=FONT_PROP)

    plt.savefig(os.path.join(save_dir, f'RQ1_tSNE_{model_name}.pdf'), bbox_inches='tight')
    plt.close()