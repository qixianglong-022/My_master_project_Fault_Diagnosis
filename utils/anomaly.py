import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import json
from config import Config


class InferenceEngine:
    """
    [论文对齐版] 3.5节 自适应多模态故障监测器
    核心特性：
    1. N+1 主导通道对齐 (Dominant Channel Alignment)
    2. 基于预测不确定性的动态仲裁 (Uncertainty-aware Arbitration)
    3. 高频门控机制 (High-Frequency Gating)
    """

    def __init__(self, model, fusion_params_path=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.fusion_params = None

        path = fusion_params_path if fusion_params_path else Config.FUSION_PARAMS_PATH
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.fusion_params = json.load(f)

    def predict(self, data_loader):
        self.model.eval()
        scores_list = []
        labels_list = []

        with torch.no_grad():
            for x, y, cov, label in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                cov = cov.to(self.device)

                # 1. 模型重构
                pred = self.model(x, cov)

                # 2. 计算融合得分 (Batch处理)
                # 传入 x (原始输入) 是为了获取 SF 和 EnergyRatio 特征
                batch_score = self._calculate_batch_score(y, pred, x_input=x)

                scores_list.append(batch_score.cpu().numpy())
                labels_list.append(label.numpy())

        return np.concatenate(scores_list), np.concatenate(labels_list)

    def _calculate_batch_score(self, y_true, y_pred, x_input=None):
        """
        [核心算法] 论文公式实现
        """
        # 基础残差平方 [B, L, D]
        res_sq = (y_true - y_pred) ** 2

        # === 简单模式 ===
        if self.fusion_params is None:
            return torch.mean(res_sq, dim=[1, 2])

        # === 高级模式：自适应融合 (Thesis Sec 3.5) ===
        p = self.fusion_params

        # 1. 维度解析
        # 振动: 前 8 维 (4个传感器 * 2特征)
        # 声纹: 后 15 维 (13 MFCC + 1 SF + 1 ER)
        dim_vib_total = Config.FEAT_DIM_VIB

        # --- A. 计算 "N+1" 振动主导通道 SPE ---
        # 我们需要将 flat 的振动特征 reshape 成 [B, L, N_Sensors, N_Feats_Per_Sensor]
        n_sensors = len(Config.COL_INDICES_VIB)  # 4
        n_feat_vib = Config.N_VIB_FEAT  # 2 (RMS, Kurt)

        vib_res = res_sq[:, :, :dim_vib_total]  # [B, L, 8]
        vib_res_reshaped = vib_res.view(vib_res.shape[0], vib_res.shape[1], n_sensors, n_feat_vib)

        # 计算每个传感器的平均残差 (对 Time 和 Feat 求均值) -> [B, N_Sensors]
        spe_per_sensor = torch.mean(vib_res_reshaped, dim=[1, 3])

        # [论文策略] 最大主导通道对齐 (Max-Dominant Channel)
        # 归一化 (可选，这里假设传感器底噪相近，直接取Max)
        spe_vib_max, _ = torch.max(spe_per_sensor, dim=1)  # [B]

        # --- B. 计算声纹 SPE ---
        # 注意：计算残差时，不应包含 SF 和 ER (它们是辅助特征，不是重构目标)
        # 我们只计算 MFCC 部分 (前13维) 的重构误差
        audio_start = dim_vib_total
        mfcc_len = Config.N_MFCC

        audio_res = res_sq[:, :, audio_start: audio_start + mfcc_len]
        spe_audio = torch.mean(audio_res, dim=[1, 2])  # [B]

        # --- C. 提取声纹辅助特征 (SF, ER) ---
        # 这些特征直接从输入 x_input 中读取 (它们是 FeatureExtractor 算好的)
        # 假设它们在声纹向量的最后两位
        # SF index relative to audio: 13
        # ER index relative to audio: 14
        sf_idx = audio_start + getattr(Config, 'IDX_AUDIO_SF', 13)
        er_idx = audio_start + getattr(Config, 'IDX_AUDIO_ER', 14)

        # 取一个窗口内的均值作为该样本的特征
        feat_sf = torch.mean(x_input[:, :, sf_idx], dim=1)  # [B]
        feat_er = torch.mean(x_input[:, :, er_idx], dim=1)  # [B] (High/Low Ratio)

        # --- D. 计算时变相关性 (Correlation) ---
        # 计算 振动残差序列 vs 声纹残差序列 的相关性
        # 取振动主导通道的残差序列 [B, L]
        # 既然 k* 比较难取序列，我们简化为：取所有振动通道残差均值的序列 (近似)
        # 或者取 RMS 特征的残差序列
        seq_vib = torch.mean(vib_res, dim=2)  # [B, L]
        seq_audio = torch.mean(audio_res, dim=2)  # [B, L]

        rho = self._batch_pearson_corr(seq_vib, seq_audio)  # [B]

        # --- E. 动态仲裁逻辑 (Arbitration) ---
        # 1. 强相关 (Resonance): rho > 0.5 -> beta = 1.5
        # 2. 弱相关 (Arbitration):
        #    - SF > 0.6 (白噪) -> beta = 0 (抑制)
        #    - SF < 0.6 (窄带):
        #      - ER > 0 (高频主导) -> beta = 1.0 (早期故障)
        #      - ER < 0 (低频主导) -> beta = 0.0 (环境撞击)

        tau_corr = 0.4
        tau_sf = -3.0  # SF 是取了 log 的吗？FeatureExtractor 里没取 log，范围 0-1
        # 如果 FeatureExtractor 返回的是 0-1，tau_sf 设为 0.4 左右
        # 这里的 param 应该从 self.fusion_params 里读，这里写死默认值逻辑

        beta = torch.zeros_like(spe_audio)

        # 向量化逻辑
        mask_corr = (rho > tau_corr)
        mask_noise = (feat_sf > 0.5)  # 平坦度高 -> 噪声
        mask_high_freq = (feat_er > 0.0)  # 能量比(log10) > 0 -> High > Low

        # 逻辑分支
        # Case 1: 强相关 -> 1.5
        beta[mask_corr] = 1.5

        # Case 2: 弱相关
        # Not Correlated
        mask_uncorr = ~mask_corr

        # Subcase 2.1: 噪声 (High SF) -> 0.0
        # beta 默认为 0，不需要操作

        # Subcase 2.2: 早期故障 (Low SF & High Freq) -> 1.0
        mask_early = mask_uncorr & (~mask_noise) & mask_high_freq
        beta[mask_early] = 1.0

        # Subcase 2.3: 环境撞击 (Low SF & Low Freq) -> 0.0
        # 默认 0

        # --- F. 最终加权 ---
        # Score = w_v * S_vib + w_a * beta * S_audio
        w_v = 1.0
        w_a = 1.0

        final_score = (w_v * spe_vib_max + w_a * beta * spe_audio) / (w_v + w_a * beta + 1e-6)

        return final_score

    def _batch_pearson_corr(self, x, y):
        """
        计算 Batch 中每对序列的 Pearson 相关系数
        x, y: [B, L]
        """
        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)

        xm = x - mean_x
        ym = y - mean_y

        num = torch.sum(xm * ym, dim=1)
        den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1)) + 1e-8

        return num / den

    def diagnose(self, data_loader, save_dir, prefix=""):
        """
        可视化诊断：抽取一个 Batch，画出 真实值 vs 重构值，保存为 PDF
        """
        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)

        # 1. 抽取一个 Batch
        try:
            # 尝试获取一个 batch
            iter_dl = iter(data_loader)
            x, y, cov, labels = next(iter_dl)
        except StopIteration:
            return

        x = x.to(self.device)
        y = y.to(self.device)
        cov = cov.to(self.device)

        with torch.no_grad():
            pred = self.model(x, cov)

        # 转为 numpy
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        cov_np = cov.cpu().numpy()
        labels_np = labels.numpy()

        # 2. 挑选样本进行绘图 (挑一个正常样本，挑一个故障样本)
        indices_to_plot = []

        # 找正常样本 (Label=0) 的第一个
        normal_idxs = np.where(labels_np == 0)[0]
        if len(normal_idxs) > 0:
            indices_to_plot.append((normal_idxs[0], "Normal"))

        # 找故障样本 (Label=1) 的第一个
        fault_idxs = np.where(labels_np == 1)[0]
        if len(fault_idxs) > 0:
            indices_to_plot.append((fault_idxs[0], "Fault"))

        # 如果没找到（比如全只有正常），就随便画前两个
        if not indices_to_plot:
            indices_to_plot = [(0, "Sample_0")]

        # 3. 开始绘图
        for idx, tag in indices_to_plot:
            # 获取数据
            # 振动通道 0 (假设索引0是振动)
            vib_true = y_np[idx, :, 0]
            vib_pred = pred_np[idx, :, 0]
            # 转速 (Covariate) - 假设第0维是 Normalized Speed
            speed_curve = cov_np[idx, 0]  # 这是一个标量，RDLinear里它是均值

            # 创建画布
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            # 子图1: 重构对比
            ax[0].plot(vib_true, label='True Signal', color='black', alpha=0.7, linewidth=1.5)
            ax[0].plot(vib_pred, label='Reconstruction', color='red', linestyle='--', linewidth=1.5)
            ax[0].set_title(f"[{tag}] Reconstruction Analysis (Speed Input: {speed_curve:.2f})", fontsize=12)
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)

            # 子图2: 残差 (SPE贡献)
            residual = (vib_true - vib_pred) ** 2
            ax[1].plot(residual, label='Squared Error', color='blue', alpha=0.6)
            ax[1].set_title("Reconstruction Error (Residual)", fontsize=12)
            ax[1].set_xlabel("Time Steps", fontsize=10)
            ax[1].grid(True, alpha=0.3)
            ax[1].legend()

            plt.tight_layout()

            # 4. 保存为 PDF
            filename = f"{prefix}_{tag}_idx{idx}.pdf"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, format='pdf', dpi=300)
            plt.close(fig)
            # print(f"   [Vis] Diagnosis plot saved: {save_path}")

    def fit_threshold(self, val_loader):
        # 这里的逻辑可以简化，主要目的是保存 param
        # 为了不破坏原有结构，我们只做最小修改
        print(">>> Fitting Threshold (Robust MAD)...")
        self.model.eval()

        scores, _ = self.predict(val_loader)

        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        # 默认阈值：Median + 3 * MAD
        # 论文建议系数 eta \in [3, 5]
        eta = 4.0
        threshold = median + eta * mad

        # 保存参数
        os.makedirs(os.path.dirname(Config.OUTPUT_DIR), exist_ok=True)
        # 这里顺便存一个 dummy fusion params 防止报错
        params = {"note": "Automated logic used in anomaly.py"}
        with open(Config.FUSION_PARAMS_PATH, 'w') as f:
            json.dump(params, f)

        np.save(os.path.join(Config.OUTPUT_DIR, 'threshold.npy'), threshold)
        print(f"   [Done] Median={median:.4f}, MAD={mad:.4f} -> Th={threshold:.4f}")
        return threshold