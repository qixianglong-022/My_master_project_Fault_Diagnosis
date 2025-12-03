# models/layers.py
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)
    论文公式 (6)-(7) 的实现。
    作用：消除由转速变化引起的输入信号统计分布漂移 (Distribution Shift)。
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: 输入通道数 (enc_in)
        :param eps: 防止除零的微小量
        :param affine: 是否学习仿射变换参数 (gamma, beta)
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def _init_params(self):
        # 初始化可学习参数 gamma 和 beta
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        # 计算时间维度 (dim=1) 上的均值和方差
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def normalize(self, x):
        # 1. 保存统计量用于反归一化
        self._get_statistics(x)
        # 2. 归一化 (x - mu) / sigma
        x = (x - self.mean) / self.stdev
        # 3. 仿射变换 gamma * x + beta
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        # 反归一化：用于将模型输出恢复到真实物理量纲
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight)
        x = x * self.stdev + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            return self.denormalize(x)
        else:
            raise NotImplementedError


class MovingAvg(nn.Module):
    """
    移动平均滤波器
    用于提取趋势项 (Trend)。
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        # Padding 以保持序列长度不变
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # Permute for Pooling: [B, C, L]
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        # Permute back: [B, L, C]
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    序列分解模块
    论文公式 (8)-(10)
    将序列分解为: Seasonal (周期/高频) + Trend (趋势/低频)
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 提取 Trend
        res = x - moving_mean  # 提取 Seasonal (残差)
        return res, moving_mean