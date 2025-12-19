import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FD_CNN(nn.Module):
    def __init__(self, num_classes=8, freq_dim=512):  # 注意这里 freq_dim 会传入 1024
        super().__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, speed=None):
        # x: [B, L, 1] -> [B, 1, L]
        x = x.permute(0, 2, 1)
        feat = self.conv_pool(x).squeeze(-1)
        return self.fc(feat), None


class TiDE_Cls(nn.Module):
    def __init__(self, num_classes=8, freq_dim=512):
        super().__init__()
        # 线性层输入维度自动适配
        self.encoder = nn.Sequential(
            nn.Linear(freq_dim + 1, 256), nn.ReLU(),  # freq_dim=1024
            nn.Linear(256, 256), nn.ReLU()
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, speed):
        # x: [B, L, 1], speed: [B, 1]
        x_flat = x.squeeze(-1)
        combined = torch.cat([x_flat, speed], dim=1)
        feat = self.encoder(combined)
        return self.fc(feat), None


class ResNet18_Thin(nn.Module):
    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(num_classes=num_classes)

        # 修改第一层：接受 2通道 (Micro + Macro) 的 1D 信号，通过 Conv 变成 2D
        # 方法：将 input (B, 2, 512) 视为 图像的高度为1
        # 但 ResNet 需要 H, W 都有值。
        # 策略：使用 Conv1d 把 (2, 512) 映射到 (64, 512)，然后 unsqueeze 成 (64, 1, 512) ? 不行。

        # 简单有效策略：把 Spectrum 重排成矩阵
        # 1024 -> 32x32
        self.side = 32
        # 输入通道为 2 (Micro流 + Macro流)
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, speed=None):
        # x: [B, 1024, 1] -> 我们在 trainer 里把 micro/macro 拼起来了
        # 但最好在 trainer 里改一下，分开传。
        # 假设这里 x 是拼接好的 [B, 1024]

        # Reshape to [B, 2, 16, 32] ?
        # 为了简单，假设 input_len=1024，我们 reshape 成 [B, 1, 32, 32] (单通道)
        # 或者如果有双流，就是 [B, 2, 32, 16] (假设 Micro 512, Macro 512)

        B = x.shape[0]
        # x: [B, 1024]
        img = x.view(B, 1, 32, 32)  # 单通道灰度图
        # 复制成3通道适配 ResNet 预训练 (可选，如果不加载预训练权重就改 conv1)
        # img = img.repeat(1, 3, 1, 1)

        return self.model(img), None

class Vanilla_RDLinear_Cls(nn.Module):
    def __init__(self, num_classes=8, freq_dim=512):
        super().__init__()
        from models.layers import SeriesDecomp
        self.decomp = SeriesDecomp(kernel_size=25)
        # 线性层适配输入维度
        self.fc = nn.Linear(freq_dim, num_classes)

    def forward(self, x, speed=None):
        # 即使输入变长，分解和线性映射依然适用
        seasonal, trend = self.decomp(x)
        logits = self.fc(x.squeeze(-1))
        return logits, None