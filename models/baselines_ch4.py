import torch
import torch.nn as nn
import torch.nn.functional as F

class FD_CNN(nn.Module):
    """
    [基线1] 频域一维卷积网络 (FD-CNN)
    代表纯数据驱动的非线性特征提取能力。
    """
    def __init__(self, num_classes=8, freq_dim=512):
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
        return self.fc(feat), None # 返回None保持与多任务Phys-RDLinear接口一致

class TiDE_Cls(nn.Module):
    """
    [基线2] 基于协变量的残差多层感知器 (TiDE-Cls)
    代表目前处理变工况最先进的线性延伸模型，侧重于特征拼接。
    """
    def __init__(self, num_classes=8, freq_dim=512):
        super().__init__()
        # 协变量维度：转频(1)
        self.encoder = nn.Sequential(
            nn.Linear(freq_dim + 1, 256), nn.ReLU(),
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
    """
    [基线3] 轻量化 ResNet-18
    用于评估深层架构在边缘侧的“性能损耗比”。
    """
    def __init__(self, num_classes=8):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(num_classes=num_classes)
        # 适配单通道频域输入
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x, speed=None):
        # x: [B, L, 1] -> [B, 1, 32, 16] (伪图像化处理)
        x_img = x.view(x.size(0), 1, 32, 16)
        return self.model(x_img), None

class Vanilla_RDLinear_Cls(nn.Module):
    """
    [基线4] 朴素 RDLinear 分类模型 (消融实验组)
    剔除了多分辨率与 PGFA 物理引导，仅保留线性分解结构。
    """
    def __init__(self, num_classes=8, freq_dim=512):
        super().__init__()
        from models.layers import SeriesDecomp
        self.decomp = SeriesDecomp(kernel_size=25)
        self.fc = nn.Linear(freq_dim, num_classes)

    def forward(self, x, speed=None):
        seasonal, trend = self.decomp(x)
        # 仅使用原始分解特征进行线性映射
        logits = self.fc(x.squeeze(-1))
        return logits, None