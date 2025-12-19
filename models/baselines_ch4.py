import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FD_CNN(nn.Module):
    def __init__(self, num_classes=8, freq_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # 大卷积核提取宽带特征
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=8),
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


class ResNet18_Thin(nn.Module):
    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        # 1. 加载预训练权重 (关键!)
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 2. 修改第一层 Conv
        # 假设输入 reshape 成 32x32 的单通道图
        # 原始: Conv2d(3, 64, k=7, s=2, p=3)
        # 修改: Conv2d(1, 64, ...)
        # 为了保留预训练权重，我们取 RGB 通道的平均值作为新权重
        old_weight = self.model.conv1.weight.data  # [64, 3, 7, 7]
        new_weight = torch.mean(old_weight, dim=1, keepdim=True)  # [64, 1, 7, 7]

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data = new_weight

        # 3. 修改 FC 层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, speed=None):
        # x: [B, 1024] (Micro + Macro)
        # Reshape: [B, 1, 32, 32]
        B = x.shape[0]
        # 确保能开方，或者 Padding
        img = x.view(B, 1, 32, 32)
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