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
    def __init__(self, num_classes=8, input_len=1024):  # 增加 input_len 参数
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(num_classes=num_classes)
        # 单通道输入
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 动态计算 reshape 的形状 (H, W)
        # 尽量让 H * W = input_len 且接近正方形
        self.side = int(math.sqrt(input_len))
        if self.side * self.side != input_len:
            # 如果不能开方，就简单处理 (例如 512 -> 32*16)
            # 这里为了简单，假设 input_len 是 1024 -> 32*32, 512 -> 32*16
            self.h = 32
            self.w = input_len // 32
        else:
            self.h = self.side
            self.w = self.side

    def forward(self, x, speed=None):
        # x: [B, L, 1] -> [B, 1, H, W]
        # 强制 reshape
        B, L, _ = x.shape
        x_img = x.view(B, 1, self.h, self.w)
        return self.model(x_img), None


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