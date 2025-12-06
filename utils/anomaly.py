import torch
import numpy as np
import os
import json
from config import Config


class InferenceEngine:
    """
    封装模型推理、自适应融合与阈值计算逻辑。
    实现 'Small & Beautiful' 的核心组件。
    """

    def __init__(self, model, fusion_params_path=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.fusion_params = None

        # 尝试加载自适应融合参数 (tau_base, th_vib 等)
        # 如果路径未传，尝试从 Config 默认路径加载
        path = fusion_params_path if fusion_params_path else Config.FUSION_PARAMS_PATH
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.fusion_params = json.load(f)
                # print(f"[Info] Loaded fusion params: {self.fusion_params}")

    def predict(self, data_loader):
        """
        执行推理并计算融合后的异常得分。
        Returns:
            scores (np.array): [N, ] 融合后的异常得分
            labels (np.array): [N, ] 真实标签
        """
        self.model.eval()
        scores_list = []
        labels_list = []

        with torch.no_grad():
            for x, y, cov, label in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                cov = cov.to(self.device)

                # 1. 模型重构/预测
                pred = self.model(x, cov)

                # 2. 计算 Batch 的异常得分
                batch_score = self._calculate_batch_score(y, pred)

                scores_list.append(batch_score.cpu().numpy())
                labels_list.append(label.numpy())

        return np.concatenate(scores_list), np.concatenate(labels_list)

    def _calculate_batch_score(self, y_true, y_pred):
        """
        核心物理融合逻辑 (对应论文 4.3 节)
        """
        # 计算平方残差 [B, L, D]
        res_sq = (y_true - y_pred) ** 2

        # === 简单模式：如果没有融合参数，退化为 Mean SPE ===
        if self.fusion_params is None:
            return torch.mean(res_sq, dim=[1, 2])

        # === 高级模式：自适应融合 ===
        p = self.fusion_params
        dim_vib = Config.FEAT_DIM_VIB

        # 1. 分量 SPE (振动 vs 声纹)
        # [B]
        spe_vib = torch.mean(res_sq[:, :, :dim_vib], dim=[1, 2])
        spe_audio = torch.mean(res_sq[:, :, dim_vib:], dim=[1, 2])

        # 2. 计算声纹不确定性 (Variance)
        # [B]
        diff_audio = y_true[:, :, dim_vib:] - y_pred[:, :, dim_vib:]
        var_audio = torch.var(diff_audio.reshape(y_true.size(0), -1), dim=1)

        # 3. 双重门控系数
        # Alpha (不确定性抑制): 方差越大，信度越低
        alpha_uncert = 1.0 / (1.0 + torch.exp(p['k1'] * (var_audio - p['tau_base'])))

        # Beta (故障强制激活): 振动越大，强制拉高声纹权重
        beta_activate = 1.0 / (1.0 + torch.exp(-p['k2'] * (spe_vib - p['th_vib'])))

        # 最终声纹权重
        w_audio_dynamic = torch.max(alpha_uncert, beta_activate)

        # 4. 加权融合
        # 振动权重恒为 1.0，声纹权重动态调整 (并乘灵敏度系数 lambda=1.2)
        w_vib = 1.0
        w_audio = 1.2 * w_audio_dynamic

        score = (w_vib * spe_vib + w_audio * spe_audio) / (w_vib + w_audio + 1e-6)
        return score

    def fit_threshold(self, val_loader):
        """
        利用验证集/训练集计算 MAD 阈值，并保存融合参数。
        """
        print(">>> Fitting Adaptive Threshold & Fusion Params...")
        self.model.eval()

        # --- Step 1: 扫描统计特性 (确定底噪和激活阈值) ---
        all_vib_spe = []
        all_audio_vars = []

        with torch.no_grad():
            for x, y, cov, _ in val_loader:
                x, y, cov = x.to(self.device), y.to(self.device), cov.to(self.device)
                pred = self.model(x, cov)

                # 振动 SPE
                res_sq = (y - pred) ** 2
                dim_vib = Config.FEAT_DIM_VIB
                spe_vib = torch.mean(res_sq[:, :, :dim_vib], dim=[1, 2])
                all_vib_spe.append(spe_vib.cpu().numpy())

                # 声纹 Variance
                diff_audio = y[:, :, dim_vib:] - pred[:, :, dim_vib:]
                var_audio = torch.var(diff_audio.reshape(x.size(0), -1), dim=1)
                all_audio_vars.append(var_audio.cpu().numpy())

        all_vib_spe = np.concatenate(all_vib_spe)
        all_audio_vars = np.concatenate(all_audio_vars)

        # 自动计算参数
        params = {
            "tau_base": float(np.median(all_audio_vars)),  # 声纹底噪
            "th_vib": float(np.percentile(all_vib_spe, 99)),  # 振动激活阈值 (99分位)
            "k1": 5.0,  # Sigmoid 陡度
            "k2": 5.0
        }

        # 保存参数
        os.makedirs(os.path.dirname(Config.FUSION_PARAMS_PATH), exist_ok=True)
        with open(Config.FUSION_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"   [Params Saved] tau_base={params['tau_base']:.4f}, th_vib={params['th_vib']:.4f}")

        # 更新当前的 engine 实例
        self.fusion_params = params

        # --- Step 2: 1. 获取所有验证样本的 SPE
        scores, _ = self.predict(val_loader)

        # 2. 鲁棒统计量计算
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        # 3. 这里的 Trick：不要死板地用 3.5
        # 我们保存 median 和 mad，推理时允许动态调整 sensitivity
        threshold_params = {'median': float(median), 'mad': float(mad)}

        # 默认推荐阈值 (保守型)
        default_th = median + 3.0 * mad

        print(f"   Median={median:.4f}, MAD={mad:.4f} -> Default Th={default_th:.4f}")

        # 保存参数而不是直接保存阈值
        np.save(os.path.join(Config.OUTPUT_DIR, 'threshold_params.npy'), threshold_params)
        return default_th