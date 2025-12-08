import torch
import torch.nn as nn
from config import Config
from models.layers import RevIN, SeriesDecomp


class RDLinear(nn.Module):
    """
    [论文对齐版] RDLinear-AD
    Sec 3.4: 转速引导的解耦线性预测模型

    关键机制：
    1. RevIN: 消除转速引起的分布漂移 (Distribution Shift)
    2. Trend分支: 学习 f(Speed) -> Baseline 的物理映射
    3. CI策略: 支持 N+1+1 弹性架构
    """

    def __init__(self):
        super().__init__()
        self.seq_len = Config.WINDOW_SIZE
        self.pred_len = Config.WINDOW_SIZE
        self.enc_in = Config.ENC_IN

        self.use_revin = Config.USE_REVIN
        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=True)

        self.decomp = SeriesDecomp(kernel_size=25)

        # Seasonal: 拟合高频振动纹理 (Allow History Access)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

        # Trend: 拟合能量基线 (Physical Constrained)
        # 输入维度: seq_len (历史) + cov_dim (转速协变量)
        cov_dim = 2 if Config.USE_SPEED else 0
        self.linear_trend = nn.Linear(self.seq_len + cov_dim, self.pred_len)

    def forward(self, x, cov):
        # x: [B, L, D]
        if self.use_revin:
            x = self.revin(x, 'norm')

        seasonal, trend = self.decomp(x)

        # --- Seasonal Branch ---
        # 学习周期性故障冲击
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        # --- Trend Branch ---
        trend = trend.permute(0, 2, 1)  # [B, D, L]

        if Config.USE_SPEED:
            # [物理约束]
            # 如果没有 RevIN，我们甚至应该把 Trend 的历史信息抹除，强制模型只看转速。
            # 但在有 RevIN 的情况下，Trend 分量已经平稳化，
            # 此时拼接 Covariates 是为了提供 "Context" (当前是加速还是减速?)

            # cov: [B, 2] -> 扩展到所有通道 [B, D, 2]
            cov_exp = cov.unsqueeze(1).repeat(1, self.enc_in, 1)

            # 拼接: [B, D, L+2]
            trend = torch.cat([trend, cov_exp], dim=-1)

        trend = self.linear_trend(trend).permute(0, 2, 1)

        # 融合
        out = seasonal + trend

        if self.use_revin:
            out = self.revin(out, 'denorm')

        return out