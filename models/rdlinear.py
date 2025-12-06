import torch
import torch.nn as nn
from config import Config
from models.layers import RevIN, SeriesDecomp


class RDLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = Config.WINDOW_SIZE
        self.pred_len = Config.WINDOW_SIZE
        self.enc_in = Config.ENC_IN

        self.use_revin = Config.USE_REVIN
        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=True)

        self.decomp = SeriesDecomp(kernel_size=25)

        # Seasonal: 负责拟合周期性纹理 (保留输入)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

        # Trend: 负责拟合能量基线 (物理引导)
        cov_dim = 2 if Config.USE_SPEED else 0
        self.linear_trend = nn.Linear(self.seq_len + cov_dim, self.pred_len)

    def forward(self, x, cov):
        # x: [B, L, D]
        if self.use_revin: x = self.revin(x, 'norm')

        seasonal, trend = self.decomp(x)

        # --- Seasonal Branch (正常工作) ---
        # 拟合高频振动纹理，这部分允许看历史数据
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        # --- Trend Branch (核心手术) ---
        trend = trend.permute(0, 2, 1)  # [B, D, L]

        if Config.USE_SPEED:
            # === 核心修正：物理强制截断 ===
            # 如果不使用 RevIN，为了防止模型直接复制输入的巨大能量(Identity Mapping)，
            # 我们必须将 Trend 分支的“历史输入”抹零，强制模型只学习 f(Speed) -> Baseline
            if not self.use_revin:
                trend = torch.zeros_like(trend)

                # 注入转速协变量
            # cov: [B, 2] -> [B, D, 2]
            cov_exp = cov.unsqueeze(1).repeat(1, self.enc_in, 1)
            trend = torch.cat([trend, cov_exp], dim=-1)

        trend = self.linear_trend(trend).permute(0, 2, 1)

        out = seasonal + trend
        if self.use_revin: out = self.revin(out, 'denorm')
        return out