# models/baselines.py
import torch
import torch.nn as nn
import math
from models.layers import SeriesDecomp


class LSTMAE(nn.Module):
    """
    [Baseline 1] LSTM Autoencoder
    经典的基于重构的异常检测模型。
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super(LSTMAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enc_in = input_dim  # 兼容接口

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, cov=None):
        batch_size, seq_len, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        z = h_n[-1]  # [Batch, Hidden]
        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(decoder_input)
        recon = self.projection(dec_out)
        return recon


class VanillaDLinear(nn.Module):
    """
    [Baseline 2] Vanilla DLinear (无 RevIN, 无协变量)
    """

    def __init__(self, config):
        super(VanillaDLinear, self).__init__()
        self.seq_len = config.WINDOW_SIZE
        self.pred_len = config.WINDOW_SIZE
        self.enc_in = config.ENC_IN

        self.decomp = SeriesDecomp(kernel_size=25)
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cov=None):
        # 忽略 cov
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.seasonal_linear(seasonal_init)
        trend_output = self.trend_linear(trend_init)

        x_out = seasonal_output + trend_output
        return x_out.permute(0, 2, 1)


class TiDE(nn.Module):
    """
    [Baseline 3] TiDE (Time-series Dense Encoder)
    谷歌提出的基于 MLP 的 SOTA 模型。
    特点：显式利用协变量 (Covariates)，但核心是 MLP 而非 Linear。
    用于对比：证明 Linear 在物理基线拟合上比 MLP 更鲁棒。
    """

    def __init__(self, config, hidden_dim=256, dropout=0.1):
        super(TiDE, self).__init__()
        self.seq_len = config.WINDOW_SIZE
        self.pred_len = config.WINDOW_SIZE
        self.enc_in = config.ENC_IN
        self.cov_dim = 2 if config.USE_SPEED else 0  # Speed Mean & Speed Sq

        # 1. Feature Projection
        # 将历史序列展平 + 协变量
        self.feature_dim = self.enc_in * self.seq_len + self.cov_dim

        # 2. Encoder (MLP)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. Decoder (MLP) -> 输出展平的序列
        self.decoder = nn.Linear(hidden_dim, self.enc_in * self.pred_len)

        # 4. Global Residual (类似 ResNet，帮助收敛)
        self.global_residual = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cov):
        batch_size = x.shape[0]

        # Flatten input: [B, L, D] -> [B, L*D]
        x_flat = x.reshape(batch_size, -1)

        # Concat Covariates: [B, L*D + Cov]
        if self.cov_dim > 0:
            encoder_input = torch.cat([x_flat, cov], dim=1)
        else:
            encoder_input = x_flat

        # MLP Encoding
        hidden = self.encoder(encoder_input)

        # MLP Decoding
        out_flat = self.decoder(hidden)
        out = out_flat.reshape(batch_size, self.pred_len, self.enc_in)

        # Global Residual (Channel Independent)
        # 对每个通道独立做线性残差
        res = self.global_residual(x.permute(0, 2, 1)).permute(0, 2, 1)

        return out + res


class TransformerBaseline(nn.Module):
    """
    [Baseline 4] Standard Transformer
    代表 Informer/Autoformer 类模型。
    用于对比：证明 Transformer 架构在机械信号上的过拟合与低效。
    """

    def __init__(self, config, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerBaseline, self).__init__()
        self.enc_in = config.ENC_IN

        # 1. Input Embedding
        self.embedding = nn.Linear(self.enc_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 3. Output Projection
        self.projection = nn.Linear(d_model, self.enc_in)

    def forward(self, x, cov=None):
        # x: [B, L, D]
        # Embedding
        x = self.embedding(x)  # [B, L, d_model]
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)

        # Projection
        out = self.projection(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)