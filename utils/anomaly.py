import numpy as np
import torch
import torch.nn as nn
from scipy.stats import genpareto


class AnomalyMeasurer:
    """
    异常检测器：计算 SPE 并利用 POT (Peaks-Over-Threshold) 极值理论自动确定阈值。
    """

    def __init__(self, q=1e-3, level=0.98):
        """
        :param q: 风险系数 (Risk probability), 例如 1e-3 代表允许 0.1% 的误报
        :param level: POT 算法中用于筛选峰值的百分位，默认 98%
        """
        self.q = q
        self.level = level
        self.threshold = None

    def calculate_spe(self, preds, trues):
        """
        计算平方预测误差 (Squared Prediction Error)
        """
        # preds, trues: [Batch, Length, Channel]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(trues, torch.Tensor):
            trues = trues.detach().cpu().numpy()

        errors = (preds - trues) ** 2
        # 在时间维和通道维求均值 -> [Batch]
        spe_scores = np.mean(errors, axis=(1, 2))
        return spe_scores

    def fit_pot(self, train_spe_scores):
        """
        在验证集（健康数据）上拟合 POT 阈值
        """
        threshold_init = np.percentile(train_spe_scores, 100 * self.level)
        peaks = train_spe_scores[train_spe_scores > threshold_init] - threshold_init

        # 广义帕累托分布拟合
        c, loc, scale = genpareto.fit(peaks)

        # 计算动态阈值
        n = len(train_spe_scores)
        Nt = len(peaks)

        # 避免除零
        if c == 0:
            self.threshold = threshold_init - scale * np.log(self.q * n / Nt)
        else:
            self.threshold = threshold_init + (scale / c) * (((self.q * n / Nt) ** (-c)) - 1)

        # 安全系数
        self.threshold = self.threshold * 1.5
        print(f"[POT] Initial Th: {threshold_init:.5f}, Final Th: {self.threshold:.5f}")
        return self.threshold

    def detect(self, test_spe_scores):
        if self.threshold is None:
            raise ValueError("Run fit_pot first!")
        return (test_spe_scores > self.threshold).astype(int)