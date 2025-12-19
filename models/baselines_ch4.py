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


class STFTLayer(nn.Module):
    """
    将 1D 振动信号转换为 2D 时频图
    输入: [B, L] -> 输出: [B, 3, H, W] (适配 ImageNet 预训练权重)
    """

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

    def forward(self, x):
        # x: [B, L]
        # 注意：STFT 需要 Tensor 在 CPU 或 GPU 上一致，register_buffer 可以自动处理 device
        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        # STFT: [B, Freq, Time, 2]
        x_stft = torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        x_mag = torch.abs(x_stft)  # [B, F, T]

        # Log 变换增强特征
        x_mag = torch.log1p(x_mag)

        # 归一化到 0-1 (Instance Norm 风格)
        B, F, T = x_mag.shape
        x_mag = x_mag.view(B, -1)
        x_mag -= x_mag.min(dim=1, keepdim=True)[0]
        x_mag /= (x_mag.max(dim=1, keepdim=True)[0] + 1e-6)
        x_mag = x_mag.view(B, F, T)

        # 扩展为 3 通道 (RGB)
        x_img = x_mag.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, F, T]

        # Resize 到 ResNet 标准输入 224x224 (可选，或者保持小尺寸用小 Kernel)
        # 这里为了保留物理分辨率，建议 Resize 到 64x64 或 128x128
        x_img = F.interpolate(x_img, size=(224, 224), mode='bilinear', align_corners=False)

        return x_img


class ResNet18_2D(nn.Module):
    """
    [修正版] 2D-ResNet18 (SOTA Standard)
    输入 1D 信号 -> 内部转 STFT 图 -> 2D CNN
    """

    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        # 1. 前端：时频变换
        self.stft = STFTLayer(n_fft=128, hop_length=32)

        # 2. 骨干：ResNet18 (加载 ImageNet 预训练权重效果更好，但这里先用随机初始化保证公平)
        # 如果样本极少 (32个)，强烈建议 pretrained=True
        self.backbone = resnet18(pretrained=True)

        # 3. 替换最后的全连接层
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x, speed=None):
        # x: [B, L]
        # 如果输入是 [B, L, 1] 或 [B, 1, L]，先 flatten
        if x.dim() > 2:
            x = x.squeeze()

        # 1. 生成图像
        img = self.stft(x)  # [B, 3, 224, 224]

        # 2. CNN 推理
        logits = self.backbone(img)

        return logits, None  # 保持接口一致 (logits, regression)


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