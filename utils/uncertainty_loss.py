import torch
import torch.nn as nn
from typing import Tuple


class UncertaintyLoss(nn.Module):
    """
    Multi-task loss with homoscedastic uncertainty weighting.
    Loss = 1/(2*sigma^2) * L + log(sigma)
    """

    def __init__(self):
        super().__init__()
        # Initialize log variance (s = log(sigma^2)) for numerical stability
        self.s_cls = nn.Parameter(torch.zeros(1))
        self.s_reg = nn.Parameter(torch.zeros(1))

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

    def forward(self,
                pred_cls: torch.Tensor, target_cls: torch.Tensor,
                pred_load: torch.Tensor, target_load: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # Calculate raw losses
        L_cls = self.cls_loss_fn(pred_cls, target_cls)
        L_reg = self.reg_loss_fn(pred_load, target_load)

        # Adaptive Weighting
        # exp(-s) represents precision (1/variance)
        loss = (torch.exp(-self.s_cls) * L_cls + 0.5 * self.s_cls) + \
               (torch.exp(-self.s_reg) * L_reg + 0.5 * self.s_reg)

        return loss, L_cls.item(), L_reg.item()

    def get_weights(self) -> Tuple[float, float]:
        """Returns current weights for monitoring: w = exp(-s)"""
        return torch.exp(-self.s_cls).item(), torch.exp(-self.s_reg).item()