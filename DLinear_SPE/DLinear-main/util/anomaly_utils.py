import numpy as np
import torch
import torch.nn as nn
from scipy.stats import genpareto


class AnomalyMeasurer:
    """
    异常检测度量器：负责计算 SPE 并应用 POT 动态阈值
    """

    def __init__(self, q=1e-3, level=0.98):
        """
        :param q: 风险系数 (Risk probability)，例如 1e-3 表示我们允许 0.1% 的误报率
        :param level: POT 算法中用于初步筛选峰值的百分位，默认 98%
        """
        self.q = q
        self.level = level
        self.threshold = None
        self.criterion = nn.MSELoss(reduction='none')

    def calculate_spe(self, preds, trues):
        """
        计算平方预测误差 (SPE)
        :param preds: 模型预测值 [Batch, Length, Channel]
        :param trues: 真实值 [Batch, Length, Channel]
        :return: 每个样本的 SPE 分数 (numpy array)
        """
        # 1. 计算逐点平方误差
        # shape: [Batch, Length, Channel]
        errors = (preds - trues) ** 2

        # 2. 在时间维和通道维求均值 (或者求和，取决于具体需求)
        # 如果是融合模型，这里会自动把振动和声纹的误差综合起来
        # shape: [Batch]
        spe_scores = torch.mean(errors, dim=[1, 2])

        return spe_scores.detach().cpu().numpy()

    def fit_pot(self, train_spe_scores):
        """
        利用正常数据的 SPE 分数来拟合 POT 阈值
        注意：传入的应该是“验证集”或“训练集”中正常样本的 SPE
        """
        # 1. 确定初始阈值 (Initial Threshold)，通常取 98% 分位数
        threshold_init = np.percentile(train_spe_scores, 100 * self.level)

        # 2. 提取超过该阈值的峰值 (Peaks)
        peaks = train_spe_scores[train_spe_scores > threshold_init] - threshold_init

        # 3. 利用广义帕累托分布 (GPD) 拟合这些峰值
        # fit 返回 (shape, loc, scale)
        params = genpareto.fit(peaks)
        c, loc, scale = params  # c is shape parameter (kappa), scale is sigma

        # 4. 根据极值理论公式计算最终的动态阈值 (z_q)
        # Formula: Th = u + (sigma/k) * ((q * n / Nt)^(-k) - 1)
        # n: total samples, Nt: number of peaks
        n = len(train_spe_scores)
        Nt = len(peaks)

        # 处理除零或参数边界情况，这里使用标准 POT 公式
        if c == 0:
            self.threshold = threshold_init - scale * np.log(self.q * n / Nt)
        else:
            self.threshold = threshold_init + (scale / c) * (((self.q * n / Nt) ** (-c)) - 1)

        self.threshold = self.threshold * 1.2  # 强制提高 20% 的余量
        print(f"[POT Auto-Threshold] Initial Th: {threshold_init:.4f}, Final Dynamic Th: {self.threshold:.4f}")
        return self.threshold

    def detect(self, test_spe_scores):
        """
        输入测试集的 SPE，返回是否异常 (0/1)
        """
        if self.threshold is None:
            raise ValueError("Run fit_pot() on normal data first!")

        preds = (test_spe_scores > self.threshold).astype(int)
        return preds, self.threshold