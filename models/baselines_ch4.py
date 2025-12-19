import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18_1D(nn.Module):
    """
    [公平基线] 将 ResNet-18 修改为 1D 卷积版本，
    或者将输入视为 (1, H, W) 的单通道图像 (Spectrogram)。
    这里为了最简单且公平，我们使用 1D 卷积重写第一层，
    保留后续 2D 结构（将 Time 视为 Width, Feature=1 视为 Height）。
    """

    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        # 加载预训练权重
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 修改第一层：适应单通道输入
        # 原始: Conv2d(3, 64, 7x7)
        # 修改: Conv2d(1, 64, 7x7)
        old_w = self.backbone.conv1.weight.data
        new_w = old_w.mean(dim=1, keepdim=True)  # RGB 平均
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = new_w

        # 修改全连接层
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x, speed=None):
        # x: [B, 1024] -> Reshape to [B, 1, 32, 32]
        # 虽然这破坏了物理邻域，但这是 CV 模型处理 1D 信号最常见的 Baseline 方法
        B = x.size(0)
        # 补齐到 1024 (如果不足)
        if x.size(1) != 1024:
            x = torch.nn.functional.pad(x, (0, 1024 - x.size(1)))

        x_img = x.view(B, 1, 32, 32)
        return self.backbone(x_img), None


class FD_CNN(nn.Module):
    """纯频域 1D-CNN"""

    def __init__(self, num_classes=8, freq_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # 大核卷积提取宽带特征
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, speed=None):
        # x: [B, 1024] -> [B, 1, 1024]
        x = x.unsqueeze(1)
        feat = self.net(x).squeeze(-1)
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