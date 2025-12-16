# models/baselines.py
import torch
import torch.nn as nn
import math
from models.layers import SeriesDecomp


class LSTMAE(nn.Module):
    """
    [Baseline 1] LSTM Autoencoder / Seq2Seq
    修改后：支持变长预测 (Forecasting)。
    如果 pred_len > 0，则 Decoder 解码 pred_len 步；否则重构 seq_len 步。
    """

    def __init__(self, config, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        统一改为传入 config 对象，方便获取 ENC_IN, WINDOW_SIZE, PRED_LEN
        """
        super(LSTMAE, self).__init__()
        self.input_dim = config.ENC_IN
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 获取序列长度配置
        self.seq_len = config.WINDOW_SIZE
        pred_len_cfg = getattr(config, 'PRED_LEN', 0)
        self.pred_len = pred_len_cfg if pred_len_cfg > 0 else config.WINDOW_SIZE

        # Encoder
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Projection
        self.projection = nn.Linear(hidden_dim, self.input_dim)

    def forward(self, x, cov=None):
        # x: [Batch, Seq_Len, Dim]
        batch_size = x.shape[0]

        # 1. Encode
        _, (h_n, _) = self.encoder(x)
        z = h_n[-1]  # [Batch, Hidden] 取最后一层的状态作为 Context Vector

        # 2. Repeat Context
        # [关键修改] 根据 pred_len 重复 z，而不是 seq_len
        decoder_input = z.unsqueeze(1).repeat(1, self.pred_len, 1)  # [Batch, Pred_Len, Hidden]

        # 3. Decode
        dec_out, _ = self.decoder(decoder_input)

        # 4. Project
        recon = self.projection(dec_out)  # [Batch, Pred_Len, Dim]

        return recon


class VanillaDLinear(nn.Module):
    """
    [Baseline 2] Vanilla DLinear
    修改后：支持变长预测。
    """

    def __init__(self, config):
        super(VanillaDLinear, self).__init__()
        self.seq_len = config.WINDOW_SIZE

        # [关键修改] 获取预测长度，如果没设置则默认为重构(输入长度)
        pred_len_cfg = getattr(config, 'PRED_LEN', 0)
        self.pred_len = pred_len_cfg if pred_len_cfg > 0 else config.WINDOW_SIZE

        self.enc_in = config.ENC_IN

        self.decomp = SeriesDecomp(kernel_size=25)

        # [关键修改] 线性层映射从 seq_len -> pred_len
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cov=None):
        # x: [Batch, L, D]
        seasonal_init, trend_init = self.decomp(x)

        # Permute: [B, D, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # Linear Projection: [B, D, L] -> [B, D, P]
        seasonal_output = self.seasonal_linear(seasonal_init)
        trend_output = self.trend_linear(trend_init)

        x_out = seasonal_output + trend_output

        # Permute back: [B, P, D]
        return x_out.permute(0, 2, 1)


class TiDE(nn.Module):
    """
    [Baseline 3] TiDE
    修改后：支持变长预测。
    """

    def __init__(self, config, hidden_dim=256, dropout=0.1):
        super(TiDE, self).__init__()
        self.seq_len = config.WINDOW_SIZE

        # [关键修改] 获取预测长度
        pred_len_cfg = getattr(config, 'PRED_LEN', 0)
        self.pred_len = pred_len_cfg if pred_len_cfg > 0 else config.WINDOW_SIZE

        self.enc_in = config.ENC_IN
        self.cov_dim = 3 if config.USE_SPEED else 0

        # 1. Feature Projection
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

        # 3. Decoder (MLP)
        # [关键修改] 输出维度适配 pred_len
        self.decoder = nn.Linear(hidden_dim, self.enc_in * self.pred_len)

        # 4. Global Residual
        # [关键修改] 残差连接适配 pred_len
        self.global_residual = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cov):
        batch_size = x.shape[0]

        # Flatten input
        x_flat = x.reshape(batch_size, -1)

        if self.cov_dim > 0:
            encoder_input = torch.cat([x_flat, cov], dim=1)
        else:
            encoder_input = x_flat

        hidden = self.encoder(encoder_input)

        # Decode & Reshape: [B, P*D] -> [B, P, D]
        out_flat = self.decoder(hidden)
        out = out_flat.reshape(batch_size, self.pred_len, self.enc_in)

        # Residual: [B, L, D] -> [B, D, L] -> Linear(L->P) -> [B, D, P] -> [B, P, D]
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

        # 获取预测长度
        pred_len_cfg = getattr(config, 'PRED_LEN', 0)
        self.pred_len = pred_len_cfg if pred_len_cfg > 0 else config.WINDOW_SIZE
        self.seq_len = config.WINDOW_SIZE

        # 1. Input Embedding
        self.embedding = nn.Linear(self.enc_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 3. Output Projection
        self.projection = nn.Linear(d_model, self.enc_in)

        # 时间维度的投影 (Seq_Len -> Pred_Len)
        if self.pred_len != self.seq_len:
            self.time_proj = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cov=None):
        # x: [B, L, D]
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        x_enc = self.transformer_encoder(x_emb)  # [B, L, d_model]

        out = self.projection(x_enc)  # [B, L, D]

        # [新增] 如果预测长度不等于输入长度，进行维度变换
        if self.pred_len != self.seq_len:
            # [B, L, D] -> [B, D, L] -> Linear(L->P) -> [B, D, P] -> [B, P, D]
            out = out.permute(0, 2, 1)
            out = self.time_proj(out)
            out = out.permute(0, 2, 1)

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