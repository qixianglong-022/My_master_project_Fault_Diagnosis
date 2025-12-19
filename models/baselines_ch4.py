import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 辅助模块 (STFT, ResBlock)
# ==============================================================================

class STFTLayer(nn.Module):
    """
    将 1D 振动信号转换为 2D 时频图
    输入: [B, L] -> 输出: [B, 3, 224, 224] (适配 ResNet 输入)
    """

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        # x: [B, L]
        # STFT: [B, Freq, Time, 2] (Complex)
        x_stft = torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        x_mag = torch.abs(x_stft)  # [B, F, T]

        # Log 变换增强特征
        x_mag = torch.log1p(x_mag)

        # [修复点] 使用 Freq_Dim 避免覆盖 F (torch.nn.functional)
        B, Freq_Dim, T = x_mag.shape

        # === [CRITICAL FIX] ===
        # 原代码: x_mag = x_mag.view(B, -1) -> 报错 RuntimeError
        # 修改为: .reshape(B, -1)
        x_mag = x_mag.reshape(B, -1)
        # ======================

        x_mag -= x_mag.min(dim=1, keepdim=True)[0]
        x_mag /= (x_mag.max(dim=1, keepdim=True)[0] + 1e-6)

        # 还原回 [B, F, T]
        x_mag = x_mag.reshape(B, Freq_Dim, T)

        # 扩展为 3 通道 (RGB)
        x_img = x_mag.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, F, T]

        # Resize 到 224x224 (为了适配标准 ResNet 感受野)
        x_img = F.interpolate(x_img, size=(224, 224), mode='bilinear', align_corners=False)

        return x_img


# ==============================================================================
# 2. 手写标准 ResNet-18 (移除 torchvision 依赖)
# ==============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18_custom(num_classes=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ==============================================================================
# 3. 封装好的模型类 (供 factory 调用)
# ==============================================================================

class ResNet18_2D(nn.Module):
    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        self.stft = STFTLayer(n_fft=128, hop_length=32)
        self.backbone = resnet18_custom(num_classes=num_classes)

    def forward(self, x, speed=None):
        if x.dim() > 2: x = x.reshape(x.size(0), -1)
        img = self.stft(x)
        logits = self.backbone(img)
        return logits, None


class FD_CNN(nn.Module):
    def __init__(self, num_classes=8, freq_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(16, 32, kernel_size=16, stride=1, padding='same'),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, speed=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.net(x).squeeze(-1)
        return self.fc(feat), None


class TiDE_Cls(nn.Module):
    def __init__(self, num_classes=8, freq_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(freq_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, speed):
        x_flat = x.reshape(x.size(0), -1)
        combined = torch.cat([x_flat, speed], dim=1)
        feat = self.encoder(combined)
        return self.fc(feat), None


class Vanilla_RDLinear_Cls(nn.Module):
    def __init__(self, num_classes=8, freq_dim=512):
        super().__init__()
        from models.layers import SeriesDecomp
        self.decomp = SeriesDecomp(kernel_size=25)
        self.fc = nn.Linear(freq_dim, num_classes)

    def forward(self, x, speed=None):
        if x.dim() == 3: x = x.squeeze(-1)
        logits = self.fc(x)
        return logits, None