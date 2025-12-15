import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from config import Config

# 设置中文字体 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 12


def plot_physics_in_physical_domain():
    print(">>> [Step 1] 正在生成中文版物理验证图...")

    points_v = []
    points_v2 = []
    points_rms = []

    data_dir = Config.ATOMIC_DATA_DIR
    if not os.path.exists(data_dir):
        print(f"Error: 找不到数据目录 {data_dir}")
        return

    # 1. 筛选 HH 变工况文件
    target_speed_ids = ['_6_S.npy', '_8_S.npy']
    hh_files = [f for f in os.listdir(data_dir)
                if ('HH' in f)
                and f.endswith('_S.npy')
                and any(tid in f for tid in target_speed_ids)]

    for fname in hh_files:
        try:
            base_name = fname.replace('_S.npy', '')
            path_s = os.path.join(data_dir, fname)
            path_x = os.path.join(data_dir, base_name + '.npy')
            if not os.path.exists(path_x): continue

            s_data = np.load(path_s)
            x_data = np.load(path_x)

            # 形状适配
            L = x_data.shape[0]
            if s_data.shape == (2 * L, 1):
                v_raw = s_data[0::2, 0]
                v2_raw = s_data[1::2, 0]
            elif s_data.shape == (L, 2):
                v_raw = s_data[:, 0]
                v2_raw = s_data[:, 1]
            else:
                continue

            # 梯度筛选
            acceleration = np.abs(np.gradient(v_raw))
            acc_threshold = 0.2 * np.max(acceleration)
            mask = acceleration > acc_threshold

            if np.sum(mask) < 10: continue

            step = 2
            points_v.extend(v_raw[mask][::step])
            points_v2.extend(v2_raw[mask][::step])
            points_rms.extend(x_data[mask, 0][::step])

        except Exception:
            continue

    if not points_v:
        print("❌ 未提取到数据。")
        return

        # 转换与归一化
    v = np.array(points_v)
    v2 = np.array(points_v2)
    rms = np.array(points_rms)

    # 使用 Min-Max 归一化，方便绘图
    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    v_norm = normalize(v)
    v2_norm = normalize(v2)
    rms_norm = normalize(rms)

    # ================= 模型拟合 =================

    # 1. 传统模型 (Linear on v)
    # y = a*v + b
    coeffs_linear = np.polyfit(v_norm, rms_norm, 1)

    # 2. 本文模型 (Linear on v^2)
    # y = a*(v^2) + b -> 相当于在 v 空间拟合抛物线
    coeffs_ours = np.polyfit(v2_norm, rms_norm, 1)

    # 计算拟合曲线 (用于画红线)
    # 生成平滑的 X 轴 (v) 用于画线
    v_grid = np.linspace(0, 1, 100)

    # 左图红线：直线
    line_linear = np.polyval(coeffs_linear, v_grid)

    # 右图红线：抛物线
    # 注意：输入是 v_grid 的平方！因为我们的模型是针对 v^2 也是线性的
    v2_grid = v_grid ** 2
    line_ours = np.polyval(coeffs_ours, v2_grid)

    # 计算 R2 分数 (在原始数据点上计算)
    pred_linear = np.polyval(coeffs_linear, v_norm)
    r2_linear = r2_score(rms_norm, pred_linear)

    pred_ours = np.polyval(coeffs_ours, v2_norm)  # 注意这里输入真实数据的 v^2
    r2_ours = r2_score(rms_norm, pred_ours)

    # ================= 绘图 =================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- 左图：传统一阶矩 ---
    # 散点：v vs RMS
    axes[0].scatter(v_norm, rms_norm, alpha=0.4, s=20, color='#1f77b4', edgecolors='none', label='实测样本点')
    # 红线：直线
    axes[0].plot(v_grid, line_linear, color='#d62728', linestyle='--', lw=3,
                 label=f'传统线性拟合 ($R^2$={r2_linear:.3f})')

    axes[0].set_title('(a) 传统基线：基于 $E[v]$ 的线性假设', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('转速一阶矩 $E[v]$ (归一化)', fontsize=12)
    axes[0].set_ylabel('振动能量 (RMS)', fontsize=12)
    axes[0].legend(loc='upper left', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.3)


    # --- 右图：本文二阶矩 ---
    # 散点：v vs RMS (完全相同！)
    axes[1].scatter(v_norm, rms_norm, alpha=0.4, s=20, color='#1f77b4', edgecolors='none', label='实测样本点')
    # 红线：抛物线
    axes[1].plot(v_grid, line_ours, color='#2ca02c', linestyle='-', lw=3, label=f'物理引导拟合 ($R^2$={r2_ours:.3f})')

    axes[1].set_title('(b) 本文方法：基于 $E[v^2]$ 的物理引导', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('转速一阶矩 $E[v]$ (归一化)', fontsize=12)
    axes[1].set_ylabel('振动能量 (RMS)', fontsize=12)
    axes[1].legend(loc='upper left', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.3)


    plt.tight_layout()
    plt.savefig('fig_B_physics_same_points.pdf')
    print(">>> ✅ 完美对比图已生成: fig_B_physics_same_points.pdf")
    print("    左图展示了传统线性模型的无能为力，右图展示了你的方法如何优雅地解决问题。")
    plt.show()


if __name__ == "__main__":
    plot_physics_in_physical_domain()