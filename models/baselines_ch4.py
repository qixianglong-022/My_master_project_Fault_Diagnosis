# models/baselines_ch4.py
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18_1D(nn.Module):
    """
    [公平基线] 原生 1D ResNet-18
    不做 Reshape，直接在频域上卷。
    """

    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, speed=None):
        # x: [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x), None


class FD_CNN(nn.Module):
    """
    [基线修正] 增强版 1D-CNN
    增加 Dropout 和 Global Average Pooling，防止在小样本上过拟合。
    """

    def __init__(self, num_classes=8, freq_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 大感受野，提取宽带特征
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),  # [新增]

            # Layer 2
            nn.Conv1d(16, 32, kernel_size=16, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),  # [新增]

            # Layer 3
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Global Avg Pool: 替代 Flatten，允许输入长度变化，且参数更少
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, speed=None):
        # x: [B, 1024] -> [B, 1, 1024]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.net(x).squeeze(-1)  # [B, 64]
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