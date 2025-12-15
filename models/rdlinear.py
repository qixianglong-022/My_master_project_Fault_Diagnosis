import torch
import torch.nn as nn
from config import Config
from models.layers import RevIN, SeriesDecomp


class RDLinear(nn.Module):
    """
    [论文对齐版 - 修正后] RDLinear-AD
    Sec 3.4: 转速引导的解耦线性预测模型

    修正说明 (导师指导):
    1. 移除了致命的 "物理阻断" (Trend History Zeroing)，防止模型欠拟合导致阈值虚高。
    2. 保留转速协变量 (Speed Covariates) 作为物理引导 (Physics-Guided)。
    3. 让 Linear 层自动学习历史趋势与当前工况(Speed)的权重分配。
    """

    def __init__(self):
        super().__init__()
        self.seq_len = Config.WINDOW_SIZE
        # 获取预测长度，如果没设置或者是0，就等于输入长度 (重构任务)
        pred_len_cfg = getattr(Config, 'PRED_LEN', 0)
        self.pred_len = pred_len_cfg if pred_len_cfg > 0 else Config.WINDOW_SIZE
        self.enc_in = Config.ENC_IN

        # 1. RevIN: 应对分布漂移 (强烈建议开启，否则无法处理非平稳工况)
        self.use_revin = Config.USE_REVIN
        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=True)

        # 2. 序列分解
        self.decomp = SeriesDecomp(kernel_size=25)

        # 3. Seasonal 分支: 拟合高频振动/周期性冲击
        # 输入: Seq_Len -> 输出: Pred_Len
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

        # 4. Trend 分支: 拟合低频基线 (物理引导核心)
        # 输入维度: seq_len (历史趋势) + cov_dim (转速协变量)
        # 我们将转速拼接到时间维度上，让模型同时看到 "过去的趋势" 和 "当前的工况"
        cov_dim = 2 if Config.USE_SPEED else 0  # Speed Mean, Speed Sq
        self.linear_trend = nn.Linear(self.seq_len + cov_dim, self.pred_len)

    def forward(self, x, cov):
        """
        :param x: [Batch, Seq_Len, Channels] 输入时序数据
        :param cov: [Batch, 2] 协变量 (Normalized Speed Mean & Square)
        """

        # A. 归一化 (RevIN Norm)
        if self.use_revin:
            x = self.revin(x, 'norm')

        # B. 序列分解
        seasonal, trend = self.decomp(x)
        # seasonal, trend: [Batch, Seq_Len, Channels]

        # --- Seasonal Branch ---
        # [Batch, Seq_Len, D] -> [Batch, D, Seq_Len]
        seasonal = seasonal.permute(0, 2, 1)
        seasonal = self.linear_seasonal(seasonal)
        # -> [Batch, D, Pred_Len]

        # --- Trend Branch ---
        # [Batch, Seq_Len, D] -> [Batch, D, Seq_Len]
        trend = trend.permute(0, 2, 1)

        if Config.USE_SPEED:
            # [物理引导逻辑修正 - 方案A]
            # 1. 扩展协变量维度: [Batch, 2] -> [Batch, 1, 2] -> [Batch, D, 2]
            cov_exp = cov.unsqueeze(1).repeat(1, self.enc_in, 1)

            # 2. [CRITICAL FIX] 拼接历史趋势与物理工况
            # 之前错误做法: trend_history = torch.zeros_like(trend) (导致 Recall=0)
            # 正确做法: 直接拼接，保留历史信息，由线性层权重决定依赖程度
            trend_input = torch.cat([trend, cov_exp], dim=-1)
            # trend_input shape: [Batch, D, Seq_Len + 2]
        else:
            # 无物理引导，仅使用历史趋势
            trend_input = trend

        # 线性映射
        trend = self.linear_trend(trend_input)
        # -> [Batch, D, Pred_Len]

        # C. 分支融合
        # [Batch, D, Pred_Len] -> [Batch, Pred_Len, D]
        out = (seasonal + trend).permute(0, 2, 1)

        # D. 反归一化 (RevIN Denorm)
        if self.use_revin:
            out = self.revin(out, 'denorm')

        return out