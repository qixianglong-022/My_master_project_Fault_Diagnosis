import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    [论文对齐] 4.4 节：基于同方差不确定性的自适应多任务损失函数
    解决了故障分类与负载回归任务之间的梯度竞争问题
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # 初始化两个任务的可学习噪声参数 (Learnable noise parameters)
        # 初始值设为 0，对应权重 exp(-0) = 1
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

    def forward(self, pred_cls, target_cls, pred_load, target_load):
        # 1. 计算基础损失
        loss_cls = self.cls_criterion(pred_cls, target_cls)
        loss_reg = self.reg_criterion(pred_load, target_load)

        # 2. 自适应加权计算 (参考论文公式)
        # Loss = exp(-log_var) * Task_Loss + log_var
        weighted_loss_cls = torch.exp(-self.log_var_cls) * loss_cls + self.log_var_cls
        weighted_loss_reg = torch.exp(-self.log_var_reg) * loss_reg + self.log_var_reg

        total_loss = weighted_loss_cls + weighted_loss_reg

        return total_loss, loss_cls.item(), loss_reg.item()