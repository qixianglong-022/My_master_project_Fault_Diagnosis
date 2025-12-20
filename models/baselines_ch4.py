import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 辅助模块 (STFT, ResBlock)
# ==============================================================================

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

# models/baselines_ch4.py

class ResNet18_2D(nn.Module):
    def __init__(self, num_classes=8, input_len=1024):
        super().__init__()
        # 论文中 ResNet 输入是 2D 频谱图 (32x16=512)
        # 假设 input_len 是 512+512=1024 (Micro+Macro)，这里我们只取 Micro 512 或者都需要适配
        # 按照论文，通常只用 Vibrartion Spectrum (512)

        # 这里的 backbone 第一层 conv1 接受 3通道，我们需要适配一下
        self.backbone = resnet18_custom(num_classes=num_classes)

        # 替换第一层卷积以接受 1通道 (灰度图) 或者是将输入复制为3通道
        # 简单起见，我们在 forward 里把输入复制成 3 通道

    def forward(self, x, speed=None):
        # [修复开始] -------------------------
        # 1. 检查输入维度
        # 如果输入是 [Batch, Length, 1] (3D)，则压缩掉最后一个维度变成 [Batch, Length]
        if x.dim() == 3:
            x = x.squeeze(-1)

        # 2. 现在 x 的形状不仅是安全的 [Batch, Length]，而且可以直接解包
        B, L = x.shape
        # [修复结束] -------------------------

        # 以下逻辑保持原样
        # 假设我们只取前 512 点构建图像 (32x16)
        # 如果长度不够 512 (例如是电流)，需要 Pad，如果超长则截断
        target_len = 512
        if L > target_len:
            x = x[:, :target_len]
        elif L < target_len:
            # 简单的补零逻辑，防止崩溃
            pad_len = target_len - L
            x = F.pad(x, (0, pad_len))

        # 重新获取截断后的长度（现在肯定是 512）
        # Reshape 为图像: [B, 1, 16, 32]
        img = x.view(B, 1, 16, 32)

        # 复制为 3 通道以适配 ResNet
        img = img.repeat(1, 3, 1, 1)

        # 上采样到 64x64 以避免特征图过小
        img = F.interpolate(img, size=(64, 64), mode='bilinear')

        logits = self.backbone(img)

        # ResNet 基线通常不返回 loss dict，只返回 logits
        return logits


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
        # x shape: [Batch, Length, 1] (例如 [64, 512, 1])
        # 或者可能是 [Batch, Length] (如果被 squeeze 过)

        # 1. 维度处理：确保形状为 [Batch, 1, Length] 以适配 Conv1d
        if x.dim() == 2:
            # 如果是 [B, L]，扩展为 [B, 1, L]
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # 如果是 [B, L, 1]，我们需要把最后一维换到中间 -> [B, 1, L]
            x = x.permute(0, 2, 1)

        # 2. 此时 x 为 [64, 1, 512]，完全符合 Conv1d 要求
        feat = self.net(x)

        # 3. 后续处理 (Flatten 等)
        # 这里的 squeeze 视你的 net 结构而定，通常 Conv1d 输出是 [B, C_out, L_out]
        # 如果最后接的是 Flatten + Linear，可能不需要 squeeze，或者 net 里包含了 Flatten
        # 假设 self.net 输出已经是 [B, Features]，则直接返回

        # 为了保险，看一眼原本代码里的操作。通常是：
        # feat = self.net(x) -> [B, 128, 1] (经过 GlobalAvgPool)
        # feat = feat.squeeze(-1) -> [B, 128]
        if feat.dim() == 3:
            feat = feat.squeeze(-1)

        logits = self.fc(feat)
        return logits


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