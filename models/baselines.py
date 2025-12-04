# models/baselines.py
import torch
import torch.nn as nn
from models.layers import SeriesDecomp


class LSTMAE(nn.Module):
    """
    LSTM Autoencoder (Baseline 1)
    经典的基于重构的异常检测模型。
    结构：Encoder (LSTM) -> Latent Vector (Last Hidden) -> Repeat -> Decoder (LSTM) -> Projection
    特点：不具备解耦趋势的能力，也无法利用转速协变量。
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        :param input_dim: 输入特征维度 (config.ENC_IN)
        :param hidden_dim: LSTM 隐藏层维度
        """
        super(LSTMAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: [Batch, Seq, Dim] -> (h_n, c_n)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder: [Batch, Seq, Hidden] -> [Batch, Seq, Hidden]
        # 输入是重复后的 Latent Vector
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Projection: [Batch, Seq, Hidden] -> [Batch, Seq, Dim]
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, cov=None):
        """
        :param x: [Batch, Seq_Len, Channels]
        :param cov: [Batch, Cov_Dim] (LSTMAE 无法使用协变量，此处仅为了兼容接口)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Encode
        # _, (h_n, _) = self.encoder(x)
        # h_n shape: [num_layers, batch, hidden]
        # 取最后一层的 hidden state 作为潜变量 z
        _, (h_n, _) = self.encoder(x)
        z = h_n[-1]  # [Batch, Hidden]

        # 2. Repeat
        # 将 z 在时间维度复制，作为 Decoder 的输入
        # [Batch, Hidden] -> [Batch, Seq_Len, Hidden]
        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)

        # 3. Decode
        dec_out, _ = self.decoder(decoder_input)

        # 4. Project
        recon = self.projection(dec_out)

        return recon


class VanillaDLinear(nn.Module):
    """
    Vanilla DLinear (Baseline 2)
    原始 DLinear 模型 [Zeng et al., 2023]。
    结构：Series Decomposition + Linear Layers (Trend & Seasonal)
    区别于 RDLinear：
    1. 没有 RevIN (无法处理非平稳分布漂移)。
    2. Trend 分支不接受转速协变量 (无法解耦工况)。
    """

    def __init__(self, config):
        super(VanillaDLinear, self).__init__()

        self.seq_len = config.WINDOW_SIZE
        self.pred_len = config.WINDOW_SIZE  # 重构任务
        self.enc_in = config.ENC_IN

        # 分解模块
        self.decomp = SeriesDecomp(kernel_size=25)

        # 线性层 (Channel Independence, 权重共享)
        self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)

        # 注意：这里的 Trend 分支输入仅仅是 seq_len，没有 +2 (Speed Covariates)
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cov=None):
        """
        :param x: [Batch, Seq_Len, Channels]
        :param cov: 忽略，原始 DLinear 不处理协变量
        """
        # 1. 分解
        seasonal_init, trend_init = self.decomp(x)

        # 2. 维度变换 [B, L, D] -> [B, D, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # 3. 线性预测
        seasonal_output = self.seasonal_linear(seasonal_init)
        trend_output = self.trend_linear(trend_init)

        # 4. 合成
        x_out = seasonal_output + trend_output

        # 5. 维度还原 [B, D, L] -> [B, L, D]
        x_out = x_out.permute(0, 2, 1)

        return x_out