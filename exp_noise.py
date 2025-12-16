import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm  # 字体管理
import matplotlib as mpl  # 新增：全局配置
# 注意：以下导入根据你的实际环境调整，若不存在需注释/补全
from config import Config
from run_evaluation import get_model_instance
from data_loader import MotorDataset
from torch.utils.data import DataLoader
from utils.anomaly import InferenceEngine

# ================= 配置区域 =================
SNR_LEVELS = [None, 10, 5, 0, -5]

TEST_ATOMS_LIST = [
    (0, '15'), (0, '45'), (0, '15-45'), (0, '45-15'),
    (0, '30'), (0, '60'), (0, '30-60'), (0, '60-30'),

    (200, '15'), (200, '45'), (200, '15-45'), (200, '45-15'),
    (200, '30'), (200, '60'), (200, '30-60'), (200, '60-30'),

    (400, '15'), (400, '45'), (400, '15-45'), (400, '45-15'),
    (400, '30'), (400, '60'), (400, '30-60'), (400, '60-30')
]

TEST_FAULTS = ['RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']

# ================= 新增：字体文件路径配置（关键！）=================
# 手动指定Windows系统中SimHei.ttf的路径（必改：确认你的系统字体路径）
# Windows：C:\Windows\Fonts\simhei.ttf
# macOS：/System/Library/Fonts/PingFang.ttc
# Linux：/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
FONT_PATH = r'C:\Windows\Fonts\simhei.ttf'  # Windows系统黑体路径（优先推荐）
# FONT_PATH = r'/System/Library/Fonts/PingFang.ttc'  # macOS系统苹方路径
# FONT_PATH = r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux系统文泉驿路径

# ================= 类定义 =================
class RobustMotorDataset(MotorDataset):
    def _add_noise(self, data, snr_db):
        vib_dim = Config.FEAT_DIM_VIB
        vib_data = data[:, :vib_dim]
        audio_data = data[:, vib_dim:]
        if audio_data.shape[1] > 0:
            signal_power = np.mean(audio_data ** 2)
            if signal_power == 0: return data
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), audio_data.shape)
            return np.concatenate([vib_data, audio_data + noise], axis=1)
        return data

class VibOnlyEngine(InferenceEngine):
    """纯振动基线 - 修正版"""
    def _calculate_batch_score(self, y_true, y_pred, x_input=None):
        dim_vib = Config.FEAT_DIM_VIB
        n_sensors = len(Config.COL_INDICES_VIB)
        n_feat_vib = Config.N_VIB_FEAT  # 2

        res_sq = (y_true - y_pred) ** 2
        vib_res = res_sq[:, :, :dim_vib]

        # [Critical Fix] 必须与 anomaly.py 中的逻辑完全一致：先 Mean 再 Max
        vib_res_reshaped = vib_res.view(vib_res.shape[0], vib_res.shape[1], n_sensors, n_feat_vib)
        spe_per_sensor = torch.mean(vib_res_reshaped, dim=[1, 3])  # [Batch, Sensors]
        spe_vib_max, _ = torch.max(spe_per_sensor, dim=1)  # [Batch] (Max-Dominant)

        return spe_vib_max

# ================= 核心逻辑 =================
def run_noise_exp_final():
    print(">>> Starting Comprehensive Robustness Experiment (Fixed Logic)...")

    # 初始化配置 (确保 Model Name 正确)
    Config.MODEL_NAME = 'RDLinear'
    exp_source = f"Thesis_Final_Physics_Constraint_AllFault_{Config.MODEL_NAME}"

    Config.OUTPUT_DIR = os.path.join(Config.PROJECT_ROOT, "checkpoints", exp_source)
    Config.USE_REVIN = False
    Config.SCALER_PATH = os.path.join(Config.OUTPUT_DIR, "scaler_params.pkl")
    Config.FUSION_PARAMS_PATH = os.path.join(Config.OUTPUT_DIR, "fusion_params.json")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 路径回退逻辑
        ckpt_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            # 尝试不带 AllFault 的旧命名
            alt_source = f"Thesis_Final_Physics_Constraint_AllFault_{Config.MODEL_NAME}"
            alt_path = os.path.join(Config.PROJECT_ROOT, "checkpoints", alt_source, "checkpoint.pth")
            if os.path.exists(alt_path):
                ckpt_path = alt_path
                Config.OUTPUT_DIR = os.path.dirname(alt_path)
                Config.SCALER_PATH = os.path.join(Config.OUTPUT_DIR, "scaler_params.pkl")
                Config.FUSION_PARAMS_PATH = os.path.join(Config.OUTPUT_DIR, "fusion_params.json")

        model = get_model_instance(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f">>> Model Loaded form {Config.OUTPUT_DIR}")
    except Exception as e:
        print(f"[Error] {e}")
        return

    # 准备引擎
    engine_adaptive = InferenceEngine(model)
    if engine_adaptive.fusion_params is None:
        print("   [Warn] Fusion params NOT found. Using defaults.")
        engine_adaptive.fusion_params = {'tau_base': 0.5, 'th_vib': 1.0, 'k1': 5.0, 'k2': 5.0}
    else:
        print(f"   [Info] Fusion params loaded: {engine_adaptive.fusion_params}")

    engine_direct = InferenceEngine(model)
    engine_direct.fusion_params = None
    engine_vib = VibOnlyEngine(model)

    engines = {
        "Adaptive Fusion (Ours)": engine_adaptive,
        "Direct Fusion": engine_direct,
        "Vibration Only": engine_vib
    }

    all_results = []

    # 在 run_noise_exp_final 循环开始前，先跑一个 batch 的 clean data
    print("\n>>> [DEBUG] Analyzing Magnitude...")
    debug_loader = DataLoader(RobustMotorDataset(TEST_ATOMS_LIST, mode='test', noise_snr=None), batch_size=32)
    x, y, cov, _ = next(iter(debug_loader))
    x, y, cov = x.to(device), y.to(device), cov.to(device)
    with torch.no_grad():
        pred = model(x, cov)
        res_sq = (y - pred) ** 2

        # 振动部分残差
        vib_dim = Config.FEAT_DIM_VIB
        spe_vib = torch.mean(res_sq[:, :, :vib_dim]).item()

        # 声纹部分残差
        spe_audio = torch.mean(res_sq[:, :, vib_dim:]).item()

        print(f"    Mean SPE Vibration: {spe_vib:.6f}")
        print(f"    Mean SPE Audio    : {spe_audio:.6f}")
        print(f"    Ratio (Vib/Audio) : {spe_vib / (spe_audio + 1e-8):.2f}")

    # 循环测试
    for snr in SNR_LEVELS:
        snr_label = str(snr) if snr is not None else "Clean"
        print(f"\n--- Testing SNR: {snr_label} dB ---")
        Config.TEST_NOISE_SNR = snr

        ds = RobustMotorDataset(TEST_ATOMS_LIST, mode='test', fault_types=TEST_FAULTS, noise_snr=snr)
        if len(ds) == 0: continue
        dl = DataLoader(ds, batch_size=64, shuffle=False)

        for method_name, engine in engines.items():
            res = eval_safe(engine, dl, snr_label, method_name)
            all_results.append(res)
            print(f"    {method_name:25s} | F1: {res['F1']:.4f} | AUC: {res['AUC']:.4f}")

    # 保存与绘图
    df = pd.DataFrame(all_results)
    df.to_csv("exp_noise_robustness_final.csv", index=False)
    # 初始化字体并传递给绘图函数
    font_prop = set_chinese_font()
    plot_paper_figure(df, font_prop)

    print(">>> Plotting Figure C (Threshold Drift)...")
    plot_threshold_drift(df, font_prop)  # <--- 调用新函数

def get_sliding_threshold_preds(scores, window_size=100, z=3.0):
    """
    [修正版] 带背景锁定机制的动态阈值
    关键改进：当检测到异常时，停止更新滑动窗口，防止阈值被故障信号"同化"。
    """
    preds = []
    thresholds = []

    # 初始化缓冲区
    history = list(scores[:window_size])

    # 冷却计数器：防止一旦报警就永远不更新（死锁）
    # 如果连续报警超过 max_freeze_steps，强制更新一次，适应新环境
    freeze_counter = 0
    MAX_FREEZE = 200

    for score in scores:
        # 1. 计算当前窗口统计量
        median = np.median(history)
        mad = np.median(np.abs(history - median))

        # 2. 计算阈值 (z=3.0 是经验值，对应约 99.7% 的置信度)
        th = median + z * (mad + 1e-6)

        # 3. 判定
        is_anomaly = score > th
        preds.append(1 if is_anomaly else 0)
        thresholds.append(th)

        # 4. 智能更新策略 (Background Locking)
        # 只有当样本是"正常"的，或者虽然异常但已经"冻结"太久了，才更新背景模型
        if (not is_anomaly) or (freeze_counter > MAX_FREEZE):
            history.pop(0)
            history.append(score)
            freeze_counter = 0  # 重置计数
        else:
            # 如果是异常，保持历史窗口不变（锁定背景），让阈值维持在低位
            freeze_counter += 1

    return np.array(preds), np.array(thresholds)

def eval_safe(engine, dataloader, snr_label, method_name):
    """安全评估函数，返回包含指标的字典"""
    # 预测逻辑
    scores, labels = engine.predict(dataloader)
    scores = np.nan_to_num(scores, nan=0.0)

    # 切换为动态阈值
    # 使用滑动窗口生成动态预测
    preds, dynamic_ths = get_sliding_threshold_preds(scores, window_size=100, z=2.5)

    # 计算指标
    from sklearn.metrics import f1_score, roc_auc_score
    try:
        auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    except Exception as e:
        print(f"[Warn] AUC计算失败：{e}")
        auc = 0.0

    try:
        f1 = f1_score(labels, preds, zero_division=0)
    except Exception as e:
        print(f"[Warn] F1计算失败：{e}")
        f1 = 0.0

    # 记录平均阈值，作为“抗噪代价”的证据
    avg_th = np.mean(dynamic_ths)

    # 打印一下，看看 Direct Fusion 是不是阈值飙升了
    if snr_label in ['Clean', '-5', '-10', '10', '5', '0']:
        print(f"    [Analysis] {method_name} ({snr_label}) Avg TH: {avg_th:.4f}")

    return {"SNR_Label": snr_label, "Method": method_name, "AUC": auc, "F1": f1, "Avg_Threshold": avg_th}

# ================= 中文字体配置函数（终极版）=================
def set_chinese_font():
    """
    终极方案：直接加载字体文件，强制覆盖所有Matplotlib字体配置
    返回字体属性对象，供绘图元素显式使用
    """
    # 1. 检查字体文件是否存在
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"字体文件不存在：{FONT_PATH}\n请检查路径是否正确，或替换为系统中存在的中文字体路径")

    # 2. 加载并注册字体
    font_prop = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    font_name = font_prop.get_name()

    # 3. 强制全局字体配置（覆盖所有层级）
    mpl.rcParams['font.family'] = font_name  # 全局字体
    mpl.rcParams['font.sans-serif'] = [font_name]  # 无衬线字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    mpl.rcParams['font.size'] = 12  # 全局字体大小

    # ========== 新增：强制设置标题字体大小 ==========
    mpl.rcParams['axes.titlesize'] = 20  # 标题默认大小（可改为18/22）
    mpl.rcParams['axes.titleweight'] = 'bold'  # 标题默认加粗

    print(f">>> 成功强制加载字体文件：{FONT_PATH}")
    print(f">>> 字体名称：{font_name}")
    return font_prop

# ================= 绘图函数（显式指定字体）=================
# ================= 绘图函数（修复字体大小失效问题）=================
def plot_paper_figure(df, font_prop):
    """绘图函数：为每个元素显式指定字体，确保中文正常显示"""
    # 替换Matplotlib样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    # 处理SNR数值映射
    df['SNR_Value'] = df['SNR_Label'].apply(lambda x: 30 if x == 'Clean' else int(x))

    method_name_map = {
        "Adaptive Fusion (Ours)": "自适应融合 (本文方法)",
        "Direct Fusion": "直接融合",
        "Vibration Only": "仅振动数据"
    }
    df['Method_CN'] = df['Method'].map(method_name_map)

    plt.figure(figsize=(9, 6))

    styles = {
        "自适应融合 (本文方法)": {'color': '#d62728', 'marker': 'o', 'lw': 2.5, 'ls': '-'},
        "直接融合": {'color': '#7f7f7f', 'marker': 's', 'lw': 2.0, 'ls': '--'},
        "仅振动数据": {'color': '#1f77b4', 'marker': '^', 'lw': 2.0, 'ls': '-.'}
    }

    for method in df['Method_CN'].unique():
        subset = df[df['Method_CN'] == method].sort_values('SNR_Value')
        s = styles.get(method, {})
        plt.plot(subset['SNR_Value'], subset['F1'],
                 label=method,
                 color=s.get('color'),
                 marker=s.get('marker'),
                 linewidth=s.get('lw'),
                 linestyle=s.get('ls'))

    # ========== 关键修改区域 Start ==========

    # 1. 创建标题专用的字体属性副本，并强制设置大小
    title_font_prop = font_prop.copy()
    title_font_prop.set_size(16)  # 强制设置为 24 (比 20 更明显)
    title_font_prop.set_weight('bold')  # 强制加粗

    # 2. 创建坐标轴标签专用的字体属性副本
    label_font_prop = font_prop.copy()
    label_font_prop.set_size(16)  # 坐标轴标签设置为 16

    # 3. 创建刻度专用的字体属性副本
    tick_font_prop = font_prop.copy()
    tick_font_prop.set_size(14)  # 刻度数字设置为 14

    # 应用字体配置
    # 注意：此时仅传递 fontproperties，不再传递 fontsize，避免冲突
    plt.title("声学噪声下的故障检测鲁棒性分析", fontproperties=title_font_prop, y=1.05)
    plt.xlabel("信噪比 (dB)", fontproperties=label_font_prop)
    plt.ylabel("F1分数", fontproperties=label_font_prop)

    # 横轴刻度
    xticks_labels = ['-5 dB', '0 dB', '5 dB', '10 dB', '无噪声\n(>30 dB)']
    plt.xticks([-5, 0, 5, 10, 30], xticks_labels, fontproperties=tick_font_prop)

    # 纵轴刻度
    plt.yticks(fontproperties=tick_font_prop)

    # 图例 (图例的大小设置比较特殊，通常在 prop 字典里设置 size)
    legend_font_prop = font_prop.copy()
    legend_font_prop.set_size(12)
    plt.legend(frameon=True, loc='lower left', prop=legend_font_prop)

    # ========== 关键修改区域 End ==========

    plt.xlim(-6, 32)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig("fig_robustness_curve_cn.pdf", dpi=300, bbox_inches='tight')
    print(">>> 中文版本图表已保存：fig_robustness_curve_cn.pdf")
    plt.close()


def plot_threshold_drift(df, font_prop):
    """
    绘制图C：阈值漂移分析 (Threshold Drift Analysis)
    修复：统一映射中文名，确保与图B的线条颜色、样式完全一致
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    # 1. 数据预处理 & 映射 (关键修复步骤)
    # 将 SNR 标签转换为数值用于排序 (Clean -> 30)
    df['SNR_Value'] = df['SNR_Label'].apply(lambda x: 30 if x == 'Clean' else int(x))
    df = df.sort_values('SNR_Value')

    # === [关键修复] 必须先把英文名映射为中文，才能匹配 styles 字典的键 ===
    method_name_map = {
        "Adaptive Fusion (Ours)": "自适应融合 (本文方法)",
        "Direct Fusion": "直接融合",
        "Vibration Only": "仅振动数据"
    }
    df['Method_CN'] = df['Method'].map(method_name_map)

    plt.figure(figsize=(9, 6))

    # 2. 样式定义 (与 plot_paper_figure 完全一致)
    styles = {
        "自适应融合 (本文方法)": {'color': '#d62728', 'marker': 'o', 'lw': 2.5, 'ls': '-'},
        "直接融合": {'color': '#7f7f7f', 'marker': 's', 'lw': 2.0, 'ls': '--'},
        "仅振动数据": {'color': '#1f77b4', 'marker': '^', 'lw': 2.0, 'ls': '-.'}
    }

    # 3. 字体属性副本 (保持字号统一)
    title_font = font_prop.copy()
    title_font.set_size(16);
    title_font.set_weight('bold')

    label_font = font_prop.copy()
    label_font.set_size(14)

    tick_font = font_prop.copy()
    tick_font.set_size(12)

    legend_font = font_prop.copy()
    legend_font.set_size(12)

    # 4. 绘图循环 (遍历中文名)
    for method in df['Method_CN'].unique():
        subset = df[df['Method_CN'] == method]
        s = styles.get(method, {})

        # 使用 semilogy (对数坐标) 展示指数级漂移
        plt.semilogy(subset['SNR_Value'], subset['Avg_Threshold'],
                     label=method,
                     color=s.get('color'),
                     marker=s.get('marker'),
                     linewidth=s.get('lw'),
                     linestyle=s.get('ls'),
                     markersize=8)

    # 5. 装饰与标签 (应用 font_prop)
    plt.xlabel("信噪比 (dB)", fontproperties=label_font)
    plt.ylabel("POT 自适应阈值 (Log Scale)", fontproperties=label_font)
    plt.title("噪声环境下的报警阈值漂移分析", fontproperties=title_font, y=1.02)

    # X轴刻度
    plt.xticks([-5, 0, 5, 10, 30],
               ['-5 dB', '0 dB', '5 dB', '10 dB', '无噪声\n(>30 dB)'],
               fontproperties=tick_font)

    # Y轴刻度 (Log坐标下通常不手动设刻度，只设字体)
    plt.yticks(fontproperties=tick_font)
    plt.xlim(-6, 32)

    # 添加注解 (指着 Ours 的线)
    plt.annotate('自适应屏蔽\n(Stable)', xy=(5, 1.5), xytext=(5, 0.5),
                 color='#d62728', fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
                 fontproperties=legend_font)  # 借用 legend 字体

    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 图例
    plt.legend(frameon=True, loc='best', prop=legend_font)

    plt.tight_layout()
    plt.savefig("fig_threshold_drift.pdf", dpi=300, bbox_inches='tight')
    print(">>> [Figure C] 阈值漂移图已生成: fig_threshold_drift.pdf")

if __name__ == "__main__":
    run_noise_exp_final()