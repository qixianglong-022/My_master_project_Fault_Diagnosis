import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    移动平均模块，用于提取时间序列的趋势成分。
    参数:
        kernel_size (int): 平均池化窗口大小。
        stride (int): 池化步长。
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        前向传播过程，在时间序列两端进行填充以保持长度不变，并应用移动平均滤波。
        参数:
            x (Tensor): 输入的时间序列张量，形状为 [Batch, Length, Channel]。
        返回:
            Tensor: 经过移动平均处理后的输出张量。
        """
        # 在时间序列的前后端复制边界值进行填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    时间序列分解模块，将输入序列分解为季节性和趋势成分。

    参数:
        kernel_size (int): 趋势成分提取所使用的移动平均核大小。
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        对输入序列执行分解操作。

        参数:
            x (Tensor): 输入的时间序列张量，形状为 [Batch, Length, Channel]。

        返回:
            tuple: 包含两个元素：
                   - res (Tensor): 季节性成分（原始序列减去趋势）。
                   - moving_mean (Tensor): 提取的趋势成分。
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    分解线性模型（DLinear），先对时间序列进行分解再分别建模预测。

    参数:
        seq_len (int): 输入序列的长度。
        pred_len (int): 预测序列的长度。
        individual (bool): 是否对每个通道单独使用不同的线性层。
        enc_in (int): 输入特征的维度/通道数。
        **kwargs: 其他可选参数。
    """

    model_name = 'Decomposition-Linear'

    def __init__(self, seq_len, pred_len, individual, enc_in, **kwargs):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 使用固定大小的卷积核进行序列分解
        kernel_size = 301  # 原来是25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # 可视化权重时启用以下两行
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # 可视化权重时启用以下两行
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        """
        执行前向传播计算，包括序列分解、独立或共享线性变换以及结果融合。

        参数:
            x (Tensor): 输入的时间序列张量，形状为 [Batch, Input length, Channel]。

        返回:
            Tensor: 输出的预测序列张量，形状为 [Batch, Output length, Channel]。
        """
        # 将输入序列分解为季节性和趋势成分
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            # 初始化输出张量
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # 对每个通道分别进行线性映射
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # 使用共享线性层进行映射
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 合并季节性和趋势部分的结果作为最终输出
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # 调整输出格式为 [Batch, Output length, Channel]
