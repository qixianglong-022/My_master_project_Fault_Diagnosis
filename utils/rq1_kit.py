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
from config import Ch4Config
from models.factory import get_model

# ==============================================================================
# 1. 全局配置与样式 (Small & Beautiful)
# ==============================================================================
FONT_PROP = set_style()
CLASS_NAMES_EN = ['HH', 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']
FAULT_MAP_CN = {
    'HH': '健康', 'RU': '转子不平衡', 'RM': '转子不对中', 'SW': '定子绕组',
    'VU': '电压不平衡', 'BR': '转子弯曲', 'KA': '转子断条', 'FB': '轴承故障'
}
# FAULT_LABELS_SHORT = [FAULT_MAP_CN[c] for c in CLASS_NAMES_EN] # 使用中文标签
FAULT_LABELS_SHORT = CLASS_NAMES_EN # 使用字母标签
COLORS = ['#7f7f7f', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']


# ==============================================================================
# 2. 核心工具函数 (Metrics & Plotting)
# ==============================================================================

def compute_detailed_metrics(y_true, y_pred):
    """计算详细指标：Acc, F1, 以及各类别 Acc"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 混淆矩阵归一化计算各列精度
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES_EN)))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)  # 填补 NaN

    metrics = {'Acc': acc * 100, 'F1': f1 * 100}
    for i, name in enumerate(CLASS_NAMES_EN):
        metrics[f'Acc_{name}'] = per_class_acc[i] * 100
    return metrics


def plot_cm(y_true, y_pred, save_path, title):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    # 归一化
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


# ==============================================================================
# 3. RQ1 核心评估流程 (Logic Fixed)
# ==============================================================================

def evaluate_rq1_comprehensive(model, dataloader, device, save_dir, model_name):
    """
    RQ1 综合评估:
    1. 跑通模型 (区分 Phys 和 Baseline 输入)
    2. 计算 0kg/200kg/400kg 下的详细指标
    3. 绘制混淆矩阵
    """
    model.eval()
    all_preds, all_labels, all_loads = [], [], []

    # 1. 确定是否为物理引导模型 (提取到循环外定义，提高效率)
    is_phys = hasattr(model, 'pgfa') or \
              model_name.startswith('Phys') or \
              model_name.startswith('Ablation')

    with torch.no_grad():
        # === [核心修复] 解包 8 个变量 (适配 DataLoader 的修改) ===
        # 注意：这里最后接收 phys_load (真实的kg值)
        for mic, mac, ac, cur, spd, ld_proxy, y_cls, phys_load in dataloader:

            # 数据迁移到 GPU
            mic, mac = mic.to(device), mac.to(device)
            ac, cur = ac.to(device), cur.to(device)
            spd, ld_proxy = spd.to(device), ld_proxy.to(device)  # 模型要吃 ld_proxy
            y_cls = y_cls.to(device)

            # 模型推理
            if is_phys:
                # Phys 模型: 输入包含软感知负载 ld_proxy
                out = model(mic, mac, ac, cur, spd, ld_proxy)
            else:
                # Baseline 模型: 只输入 mic 和 speed
                out = model(mic, speed=spd)

            # 智能解包 logits
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_cls.cpu().numpy())

            # === [核心修复] 保存真实的 phys_load 用于分桶 ===
            all_loads.append(phys_load.cpu().numpy())

    # 拼接数据
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # === [核心修复] 直接使用物理负载 (不需要再 *400 了，因为 data_loader 直接传回了 kg 值) ===
    loads_real = np.concatenate(all_loads).flatten().astype(int)

    # 保存原始预测 (用于 debug)
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({'Load_kg': loads_real, 'True': y_true, 'Pred': y_pred}).to_csv(
        os.path.join(save_dir, f'raw_predictions_{model_name}.csv'), index=False)

    summary_row = {'Model': model_name}

    # === 定义要统计的负载点 ===
    target_loads_list = [(0, '0kg'), (200, '200kg'), (400, '400kg')]
    acc_values = []
    f1_values = []

    # === 分负载统计 ===
    for target_load, name in target_loads_list:
        # 根据真实的 load_real 进行分桶
        if target_load == 0:
            mask = loads_real < 100  # 0kg 组
        elif target_load == 400:
            mask = loads_real > 300  # 400kg 组
        else:
            mask = (loads_real >= 100) & (loads_real <= 300)  # 200kg 组

        if np.sum(mask) == 0:
            continue

        sub_true, sub_pred = y_true[mask], y_pred[mask]

        # 计算指标
        m = compute_detailed_metrics(sub_true, sub_pred)
        for k, v in m.items():
            summary_row[f'{k}_{name}'] = v

        acc_values.append(m['Acc'])
        f1_values.append(m['F1'])

        # 绘制混淆矩阵
        cm_path = os.path.join(save_dir, f"CM_{model_name}_{name}.pdf")
        plot_cm(sub_true, sub_pred, cm_path, f"{model_name} - {name}工况")

    # 动态计算平均值
    if len(acc_values) > 0:
        summary_row['Avg_Acc'] = sum(acc_values) / len(acc_values)
        summary_row['Avg_F1'] = sum(f1_values) / len(f1_values)

    return summary_row


# ==============================================================================
# 4. 高级可视化 (Radar & t-SNE)
# ==============================================================================

def plot_radar_chart_from_csv(csv_path, save_dir, load_kg):
    """从 summary.csv 读取数据绘制雷达图"""
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

        # 样式区分
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
    """绘制 t-SNE 特征分布图"""
    print(f"    [Plot] 生成 t-SNE ({model_name})...")
    model.eval()
    MAX_PER_CLASS = 150

    # 1. 确定是否为物理引导模型
    is_phys = hasattr(model, 'pgfa') or \
              model_name.startswith('Phys') or \
              model_name.startswith('Ablation')

    def extract_feats(loader, domain_label):
        feats, labels, domains = [], [], []
        counts = {i: 0 for i in range(8)}

        with torch.no_grad():
            # === [核心修复] 解包 8 个变量 ===
            for mic, mac, ac, cur, spd, ld_proxy, y_cls, phys_load in loader:

                mic, mac = mic.to(device), mac.to(device)
                ac, cur = ac.to(device), cur.to(device)
                spd, ld_proxy = spd.to(device), ld_proxy.to(device)

                # --- 核心：使用 Hook 抓取倒数第二层特征 ---
                batch_feats = []

                def hook_fn(module, input, output):
                    # 通常 input[0] 是进入分类头的特征
                    if isinstance(input, tuple):
                        batch_feats.append(input[0])
                    else:
                        batch_feats.append(input)

                # 智能选择 Hook 层
                target_layer = None
                if hasattr(model, 'cls_head'):
                    target_layer = model.cls_head
                elif hasattr(model, 'fc'):
                    target_layer = model.fc
                elif hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
                    target_layer = model.backbone.fc  # ResNet

                if target_layer is None: return [], [], []  # 没法画

                handle = target_layer.register_forward_hook(hook_fn)

                # === [一致性修复] 使用定义好的 is_phys ===
                if is_phys:
                    _ = model(mic, mac, ac, cur, spd, ld_proxy)
                else:
                    _ = model(mic, speed=spd)

                handle.remove()  # 移除 Hook

                if len(batch_feats) > 0:
                    out_feat = batch_feats[-1]
                else:
                    continue

                curr_f = out_feat.cpu().numpy()
                curr_l = y_cls.cpu().numpy()

                for f, l in zip(curr_f, curr_l):
                    if counts[l] < MAX_PER_CLASS:
                        feats.append(f)
                        labels.append(l)
                        domains.append(domain_label)
                        counts[l] += 1

            if len(feats) == 0: return [], [], []
            return np.array(feats), np.array(labels), np.array(domains)

    # 提取特征
    f_s, l_s, d_s = extract_feats(dataloader_src, 0)
    f_t, l_t, d_t = extract_feats(dataloader_tgt, 1)

    if len(f_s) == 0 or len(f_t) == 0:
        print("    [Warn] t-SNE 特征提取失败，跳过绘图。")
        return

    # t-SNE 降维
    X = np.concatenate([f_s, f_t])
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)

    # 绘图
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    markers = ['o', 'x']

    for cls_id in range(8):
        c = cmap(cls_id)
        for dom_id in [0, 1]:
            mask = (np.concatenate([l_s, l_t]) == cls_id) & \
                   (np.concatenate([d_s, d_t]) == dom_id)
            if np.sum(mask) == 0: continue

            plt.scatter(
                X_emb[mask, 0], X_emb[mask, 1],
                color=c, marker=markers[dom_id],
                label=FAULT_LABELS_SHORT[cls_id] if dom_id == 0 else None,
                alpha=0.7, s=40 if dom_id == 0 else 60
            )

    plt.title(f"特征空间 t-SNE - {model_name}", fontsize=16, fontproperties=FONT_PROP)
    plt.xticks([]);
    plt.yticks([])

    # 双图例
    h, l = plt.gca().get_legend_handles_labels()
    l1 = plt.legend(h, l, loc='upper right', title="故障类型", prop=FONT_PROP, bbox_to_anchor=(1.15, 1))
    plt.setp(l1.get_title(), fontproperties=FONT_PROP)
    plt.gca().add_artist(l1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='源域 (200kg)', markersize=10),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', label='目标域 (0/400kg)', markersize=10)
    ]
    l2 = plt.legend(handles=legend_elements, loc='lower right', title="工况域", prop=FONT_PROP,
                    bbox_to_anchor=(1.15, 0))
    plt.setp(l2.get_title(), fontproperties=FONT_PROP)

    plt.savefig(os.path.join(save_dir, f'RQ1_tSNE_{model_name}.pdf'), bbox_inches='tight')
    plt.close()


GPU_ID = 0
FONT_PROP = set_style()  # 复用你现有的字体设置


def plot_fusion_heatmap_automated():
    print("=========================================================")
    print("   自动化融合机理分析 (Fusion Mechanism Analysis)")
    print("   目标: 可视化不同工况(转速/负载)下各模态权重的动态变化")
    print("=========================================================")

    # 1. 初始化环境
    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'  # 必须是此模型
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    vis_dir = config.CHECKPOINT_DIR
    os.makedirs(vis_dir, exist_ok=True)

    # 2. 加载模型
    try:
        model = get_model(config.MODEL_NAME, config).to(device)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME, "model.pth")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f">>> 模型已加载: {ckpt_path}")
        else:
            print(f"[Warn] 未找到训练权重，将使用随机初始化权重演示功能 (仅供代码测试)")
            # 实际跑论文图时，请确保权重文件存在
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        return

    model.eval()

    # 3. 构建虚拟工况网格 (Simulation Grid)
    # 转速范围: 0Hz ~ 60Hz
    # 负载范围: 0.0 (空载) ~ 2.0 (重载/超载) - 对应 Current RMS Ratio
    resolution = 100
    speeds = np.linspace(0, 60, resolution)
    loads = np.linspace(0, 2.5, resolution)

    # 网格化
    S, L = np.meshgrid(speeds, loads)

    # 展平以进行批量推理
    s_flat = torch.tensor(S.flatten(), dtype=torch.float32).unsqueeze(1).to(device)  # [N, 1]
    l_flat = torch.tensor(L.flatten(), dtype=torch.float32).unsqueeze(1).to(device)  # [N, 1]

    # 4. 提取权重 (核心魔法)
    # 我们不需要完整的 forward，只需要调用 fusion_layer 的 weight_net
    # 只要模型里的 fusion_layer 结构没变，这招就有效
    print(">>> 计算全工况域权重分布...")
    with torch.no_grad():
        # 对应 phys_rdlinear.py 中的 AdaptiveFusion 逻辑:
        # cov = torch.cat([speed / 100.0, load_proxy], dim=1)
        cov = torch.cat([s_flat / 100.0, l_flat], dim=1)

        # 直接调用 fusion_layer 内部的 MLP
        # model.fusion_layer.weight_net 输出 logits
        # model.fusion_layer.softmax 输出 归一化权重
        weights = model.fusion_layer.softmax(model.fusion_layer.weight_net(cov))
        weights = weights.cpu().numpy()  # [N, 3] -> [vib, aud, cur]

    # 5. 重塑回网格形状
    W_vib = weights[:, 0].reshape(resolution, resolution)
    W_aud = weights[:, 1].reshape(resolution, resolution)
    W_cur = weights[:, 2].reshape(resolution, resolution)

    # 6. 绘图 (Paper Quality)
    print(">>> 正在绘图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # 定义绘图元数据
    maps = [
        ('振动模态 (Vibration)', W_vib, 'Blues'),
        ('声纹模态 (Acoustic)', W_aud, 'Greens'),
        ('电流模态 (Current)', W_cur, 'Reds')
    ]

    # 物理刻度标签
    extent = [0, 60, 0, 2.5]  # [x_min, x_max, y_min, y_max]

    for ax, (title, data, cmap_name) in zip(axes, maps):
        # 使用 imshow 绘制热力图
        im = ax.imshow(
            data,
            origin='lower',
            extent=extent,
            aspect='auto',
            cmap=cmap_name,
            vmin=0, vmax=1.0  # 权重在 0~1 之间
        )

        ax.set_title(title, fontsize=16, fontproperties=FONT_PROP, weight='bold', pad=10)
        ax.set_xlabel("转速 (Hz)", fontsize=14, fontproperties=FONT_PROP)
        ax.set_ylabel("负载率 (Current Proxy)", fontsize=14, fontproperties=FONT_PROP)

        # 添加等高线，增强可读性
        CS = ax.contour(data, levels=[0.3, 0.5, 0.7], extent=extent, colors='white', linewidths=0.8, alpha=0.5)
        ax.clabel(CS, inline=True, fontsize=10, fmt='%.1f')

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label('自适应权重', fontproperties=FONT_PROP)

    # 添加总标题
    fig.suptitle(f"变工况下的多模态自适应融合机理 (Fusion Weights)", fontsize=20, fontproperties=FONT_PROP, y=1.05)

    # 保存
    save_path = os.path.join(vis_dir, "RQ1_fusion_Mechanism_Heatmap.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"    -> [Success] 热力图已保存: {save_path}")
    print("    -> 分析提示: 请检查是否符合物理规律 (例如: 低速时声纹权重大，重载时振动/电流权重大)")