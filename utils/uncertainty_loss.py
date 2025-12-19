import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """
    [数值修正] 稳定的同方差不确定性损失
    """

    def __init__(self):
        super().__init__()
        # 初始化为 -2.0 (对应 sigma ≈ 0.36)，给 Loss 一个较大的初始权重，加速收敛
        self.s_cls = nn.Parameter(torch.tensor(-2.0))
        self.s_reg = nn.Parameter(torch.tensor(-2.0))

        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

    def forward(self, pred_cls, target_cls, pred_load, target_load):
        # 1. 计算原始 Loss
        loss_cls = self.cls_loss(pred_cls, target_cls)
        loss_reg = self.reg_loss(pred_load, target_load)

        # 2. 计算不确定性权重
        # 使用 exp(-s) 作为精度 (Precision)
        # 这里的 s 实际上是 log(sigma^2)

        # 策略：限制 s 的范围，防止除以 0 或权重过大
        # 简单公式: Loss = exp(-s) * L + 0.5 * s

        w_cls = torch.exp(-self.s_cls)
        w_reg = torch.exp(-self.s_reg)

        # [关键] 加上 0.5 * s 是正则项，防止模型通过无限增大 s (即增大 sigma) 来作弊降低 Loss
        total_loss = (w_cls * loss_cls + 0.5 * self.s_cls) + \
                     (w_reg * loss_reg + 0.5 * self.s_reg)

        return total_loss, loss_cls.item(), loss_reg.item()