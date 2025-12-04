# models/rdlinear.py
import torch
import torch.nn as nn
from models.layers import RevIN, SeriesDecomp


class RDLinear(nn.Module):
    """
    RDLinear (RevIN + Decomposition + Linear) with Speed Covariate Injection

    论文核心创新点:
    1. 使用 RevIN 消除转速变化引起的非平稳性。
    2. 将转速协变量 [v, v^2] 注入 Trend 分支，引导模型学习 "Baseline(Speed)" 映射。
    3. 采用 Channel Independence (CI) 策略，支持任意数量的振动/声纹通道。
    """

    def __init__(self, config):
        super(RDLinear, self).__init__()

        # 从配置中读取参数
        self.seq_len = config.WINDOW_SIZE
        self.pred_len = config.WINDOW_SIZE  # 自监督/重构任务：输入=输出
        # 如果是预测未来，这里 pred_len 可以不同

        self.enc_in = len(config.COL_INDICES_VIB) + len(config.COL_INDICES_AUDIO)  # 通道总数

        # 1. RevIN 模块 (归一化层)
        # 带RevIN 开关
        self.use_revin = config.USE_REVIN
        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=True)

        # 转速注入开关
        self.use_speed = config.USE_SPEED
        self.covariate_dim = 2 if self.use_speed else 0

        # 2. 分解模块 (Decomposition)
        # kernel_size 决定了平滑程度，通常取 25 左右用于滤除工频干扰
        self.decomp = SeriesDecomp(kernel_size=25)

        # 3. 线性预测层 (Linear Layers)
        # 采用 Channel Independence 策略: 权重共享
        # 这是一个针对单个通道的线性层

        # Seasonal 分支: 纯粹处理周期性波动
        # Input: seq_len -> Output: pred_len
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

        # Trend 分支: 处理趋势 + 转速影响
        # Input: seq_len + 2 (Speed, Speed^2) -> Output: pred_len
        # 这里的 +2 是论文中的核心设计：显式引入物理协变量
        self.covariate_dim = 2
        self.linear_trend = nn.Linear(self.seq_len + self.covariate_dim, self.pred_len)

    def forward(self, x, speed_cov):
        """
        :param x: [Batch, Seq_Len, Channels] 输入信号 (振动+声纹)
        :param speed_cov: [Batch, 2] 转速协变量 [v, v^2]
        """
        # ================= 1. 归一化 (RevIN) =================
        # 消除工况引起的幅值差异 (e.g. 60Hz 能量 > 15Hz 能量)
        # x: [B, L, D]
        if self.use_revin:
            x = self.revin(x, 'norm')

        # ================= 2. 序列分解 =================
        # seasonal: [B, L, D], trend: [B, L, D]
        seasonal_init, trend_init = self.decomp(x)

        # 转换维度以适应 Linear 层 (CI 策略)
        # [B, L, D] -> [B, D, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # ================= 3. Seasonal 分支预测 =================
        # 直接通过线性层
        # [B, D, L] -> [B, D, P]
        seasonal_output = self.linear_seasonal(seasonal_init)

        # ================= 4. Trend 分支预测 (含协变量注入) =================
        # 这是论文最关键的一步：将 Speed 信息融合进 Trend

        # speed_cov: [B, 2] -> 扩展为 [B, D, 2] 以匹配通道数
        if self.use_speed:
            # 有转速：拼接
            batch_size, channels, _ = trend_init.shape
            speed_cov_expanded = speed_cov.unsqueeze(1).repeat(1, channels, 1)
            trend_with_cov = torch.cat([trend_init, speed_cov_expanded], dim=-1)
            trend_output = self.linear_trend(trend_with_cov)
        else:
            # 无转速：直接过 Linear
            trend_output = self.linear_trend(trend_init)

        x_out = seasonal_output + trend_output
        x_out = x_out.permute(0, 2, 1)

        if self.use_revin:
            x_out = self.revin(x_out, 'denorm')

        return x_out