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
from utils.visualization import set_style

FONT_PROP = set_style()
CLASS_NAMES_EN = ['HH', 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']
FAULT_MAP_CN = {
    'HH': '健康', 'RU': '转子不平衡', 'RM': '转子不对中', 'SW': '定子绕组',
    'VU': '电压不平衡', 'BR': '转子弯曲', 'KA': '转子断条', 'FB': '轴承故障'
}
FAULT_LABELS_SHORT = [FAULT_MAP_CN[c] for c in CLASS_NAMES_EN]
COLORS = ['#7f7f7f', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']


def compute_detailed_metrics(y_true, y_pred):
    # ... (保持原有逻辑不变) ...
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
    # ... (保持原有绘图逻辑不变) ...
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
    RQ1 核心评估 (V2 适配版)
    适配: 7项 unpack, 6参 forward
    """
    model.eval()
    all_preds, all_labels, all_loads = [], [], []

    with torch.no_grad():
        # [V2 Change] Unpack 7 items
        for mic, mac, ac, cur, spd, ld, y_cls in dataloader:
            mic, mac = mic.to(device), mac.to(device)
            ac, cur = ac.to(device), cur.to(device)
            spd, ld = spd.to(device), ld.to(device)
            y_cls = y_cls.to(device)  # Label

            # === 模型调用分流 ===
            is_phys_model = model_name.startswith('Phys') or \
                            model_name.startswith('Ablation') or \
                            hasattr(model, 'pgfa')

            if is_phys_model:
                # V2 Phys Model: 接收所有物理量
                logits, _ = model(mic, mac, ac, cur, spd, ld)
            else:
                # Baselines (ResNet, TiDE等): 拼接所有特征
                # mic/mac: [B, 512, 1] -> [B, 512]
                # ac: [B, 15]
                # cur: [B, 128]
                full_x = torch.cat([
                    mic.squeeze(-1),
                    mac.squeeze(-1),
                    ac,
                    cur
                ], dim=1)  # [B, 512+512+15+128] = [B, 1167]

                # Baselines forward (通常只接受 feature + speed)
                logits, _ = model(full_x, spd)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_cls.cpu().numpy())
            all_loads.append(ld.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    # ld 是归一化的，还原成大概的物理值用于统计 (假设最大400kg)
    loads_real = (np.concatenate(all_loads).flatten() * 400).round().astype(int)

    # 保存原始预测结果
    pd.DataFrame({'Load': loads_real, 'True': y_true, 'Pred': y_pred}).to_csv(
        os.path.join(save_dir, 'raw_predictions.csv'), index=False)

    summary_row = {'Model': model_name}

    # 分负载统计 (0kg, 400kg)
    # 注意：由于 RMS 估算有误差，使用范围判定
    for target_load, name in [(0, '0kg'), (400, '400kg')]:
        if target_load == 0:
            mask = loads_real < 100  # 轻载区间
        else:
            mask = loads_real > 300  # 重载区间

        if np.sum(mask) == 0: continue

        sub_true, sub_pred = y_true[mask], y_pred[mask]
        m = compute_detailed_metrics(sub_true, sub_pred)
        for k, v in m.items(): summary_row[f'{k}_{name}'] = v

        # 画混淆矩阵
        plot_cm(sub_true, sub_pred, os.path.join(save_dir, f"CM_{name}.pdf"), f"{model_name} - {name}工况")

    # 计算平均
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
    MAX_PER_CLASS = 150  # 限制采样数，防止 t-SNE 跑太慢

    def extract_feats(loader, domain_label):
        feats, labels, domains = [], [], []
        counts = {i: 0 for i in range(8)}  # 8类故障计数器

        with torch.no_grad():
            # [V2 Change] Unpack 7 items
            for mic, mac, ac, cur, spd, ld, y_cls in loader:
                mic, mac = mic.to(device), mac.to(device)
                ac, cur = ac.to(device), cur.to(device)
                spd, ld = spd.to(device), ld.to(device)

                # --- 模型调用 (获取倒数第二层特征) ---
                # 注意：这里我们需要修改 forward 让他返回 feature，或者使用 hook
                # 既然 Phys-RDLinear 的 forward 返回的是 logits, pred_load
                # 我们可以临时使用 hook 或者简单修改 forward
                # 为了不改动 model 文件，这里使用 hook 方式最稳妥

                # 定义 Hook
                batch_feats = []

                def hook_fn(module, input, output):
                    # input[0] 是进入 cls_head 的融合特征 (B, Hidden*2)
                    batch_feats.append(input[0])

                # 注册 Hook (假设 cls_head 是最后一层分类器)
                # 如果是基线模型 (ResNet/TiDE)，通常 fc 层前是特征
                if hasattr(model, 'cls_head'):
                    target_layer = model.cls_head
                elif hasattr(model, 'fc'):
                    target_layer = model.fc
                elif hasattr(model, 'backbone'):  # ResNet 封装
                    target_layer = model.backbone.fc
                else:
                    # 兜底：如果没有明确层，跳过 t-SNE
                    return np.array([]), np.array([]), np.array([])

                handle = target_layer.register_forward_hook(hook_fn)

                # Forward
                is_phys = hasattr(model, 'pgfa') or \
                          model_name.startswith('Phys') or \
                          model_name.startswith('Ablation')

                if is_phys:
                    _ = model(mic, mac, ac, cur, spd, ld)
                else:
                    full_x = torch.cat([
                        mic.squeeze(-1),
                        mac.squeeze(-1),
                        ac,
                        cur
                    ], dim=1)
                    _ = model(full_x, spd)

                # 获取 Hook 到的特征
                if len(batch_feats) > 0:
                    out_feat = batch_feats[-1]
                else:
                    out_feat = torch.zeros(mic.size(0), 2)  # Fallback

                handle.remove()  # 移除 Hook 防止内存泄漏

                # 收集数据
                curr_f = out_feat.cpu().numpy()
                curr_l = y_cls.cpu().numpy()

                for f, l in zip(curr_f, curr_l):
                    if counts[l] < MAX_PER_CLASS:
                        feats.append(f)
                        labels.append(l)
                        domains.append(domain_label)
                        counts[l] += 1

        if len(feats) == 0:
            return np.array([]), np.array([]), np.array([])

        return np.array(feats), np.array(labels), np.array(domains)

    # 提取源域和目标域特征
    f_s, l_s, d_s = extract_feats(dataloader_src, 0)  # Domain 0 = Source
    f_t, l_t, d_t = extract_feats(dataloader_tgt, 1)  # Domain 1 = Target

    if len(f_s) == 0:
        print("    [Warn] t-SNE: No features extracted.")
        return

    # 合并
    X = np.concatenate([f_s, f_t])

    # 运行 t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)

    # --- 绘图 ---
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    markers = ['o', 'x']  # o: Source, x: Target

    # 遍历每个类别画散点
    for cls_id in range(8):
        c = cmap(cls_id)
        for dom_id in [0, 1]:
            # 掩码筛选
            mask = (np.concatenate([l_s, l_t]) == cls_id) & \
                   (np.concatenate([d_s, d_t]) == dom_id)

            if np.sum(mask) == 0: continue

            # 画点
            plt.scatter(
                X_emb[mask, 0], X_emb[mask, 1],
                color=c,
                marker=markers[dom_id],
                label=FAULT_LABELS_SHORT[cls_id] if dom_id == 0 else None,  # 图例只显示一次类别名
                alpha=0.7,
                s=40 if dom_id == 0 else 60
            )

    plt.title(f"特征空间 t-SNE 可视化 - {model_name}", fontsize=16, fontproperties=FONT_PROP)
    plt.xticks([])
    plt.yticks([])

    # 图例 1: 故障类别
    h, l = plt.gca().get_legend_handles_labels()
    l1 = plt.legend(h, l, loc='upper right', title="故障类型", prop=FONT_PROP, bbox_to_anchor=(1.15, 1))
    plt.setp(l1.get_title(), fontproperties=FONT_PROP)
    plt.gca().add_artist(l1)

    # 图例 2: 域 (手动构造)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='源域 (200kg)', markersize=10),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', label='目标域 (0/400kg)', markersize=10)
    ]
    l2 = plt.legend(handles=legend_elements, loc='lower right', title="工况域", prop=FONT_PROP,
                    bbox_to_anchor=(1.15, 0))
    plt.setp(l2.get_title(), fontproperties=FONT_PROP)

    plt.savefig(os.path.join(save_dir, f'RQ1_tSNE_{model_name}.pdf'), bbox_inches='tight')
    plt.close()