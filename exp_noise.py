import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        # 之前版本只用了 Mean，导致基线分数偏低
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
    plot_paper_figure(df)


def eval_safe(engine, dataloader, snr_label, method_name):
    from sklearn.metrics import roc_auc_score, f1_score
    scores, labels = engine.predict(dataloader)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

    try:
        auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    except:
        auc = 0.0

    # 阈值加载
    th_path = os.path.join(Config.OUTPUT_DIR, 'threshold.npy')
    if os.path.exists(th_path):
        th = float(np.load(th_path))
    else:
        th = np.median(scores) + 3 * np.median(np.abs(scores - np.median(scores)))

    preds = (scores > th).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)

    return {"SNR_Label": snr_label, "Method": method_name, "AUC": auc, "F1": f1}


def plot_paper_figure(df):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    df['SNR_Value'] = df['SNR_Label'].apply(lambda x: 30 if x == 'Clean' else int(x))

    plt.figure(figsize=(8, 6))
    styles = {
        "Adaptive Fusion (Ours)": {'color': '#d62728', 'marker': 'o', 'lw': 2.5, 'ls': '-'},
        "Direct Fusion": {'color': '#7f7f7f', 'marker': 's', 'lw': 2.0, 'ls': '--'},
        "Vibration Only": {'color': '#1f77b4', 'marker': '^', 'lw': 2.0, 'ls': '-.'}
    }

    for method in df['Method'].unique():
        subset = df[df['Method'] == method].sort_values('SNR_Value')
        s = styles.get(method, {})
        plt.plot(subset['SNR_Value'], subset['F1'], label=method,
                 color=s.get('color'), marker=s.get('marker'),
                 linewidth=s.get('lw'), linestyle=s.get('ls'))

    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=14, fontweight='bold')
    plt.ylabel("F1-Score", fontsize=14, fontweight='bold')
    plt.title("Robustness Analysis under Acoustic Noise", fontsize=16, y=1.02)
    plt.xticks([-5, 0, 5, 10, 30], ['-5', '0', '5', '10', 'Clean\n(>30)'])
    plt.xlim(-6, 32)
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=True, fontsize=11, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig_robustness_curve.pdf", dpi=300)
    print(">>> Plot saved.")


if __name__ == "__main__":
    run_noise_exp_final()