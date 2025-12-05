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

        # Seasonal
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

        # Trend (Covariate Injection)
        cov_dim = 2 if Config.USE_SPEED else 0
        self.linear_trend = nn.Linear(self.seq_len + cov_dim, self.pred_len)

    def forward(self, x, cov):
        # x: [B, L, D]
        if self.use_revin: x = self.revin(x, 'norm')

        seasonal, trend = self.decomp(x)

        # Seasonal Branch
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        # Trend Branch
        trend = trend.permute(0, 2, 1)  # [B, D, L]
        if Config.USE_SPEED:
            # cov: [B, 2] -> [B, D, 2]
            cov_exp = cov.unsqueeze(1).repeat(1, self.enc_in, 1)
            trend = torch.cat([trend, cov_exp], dim=-1)

        trend = self.linear_trend(trend).permute(0, 2, 1)

        out = seasonal + trend
        if self.use_revin: out = self.revin(out, 'denorm')
        return out