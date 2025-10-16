from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss implementation.

    Taken from:
    https://github.com/SaoYan/IPMI2019-AttnMel/blob/e214a3797a04e66647844e582ee10b4833342759/loss.py#L5

    Args:
        gamma (float): Focal loss gamma parameter. Default is 2.0.
        size_average (bool): Whether to average the loss over the batch.
                             Default is True.
        weight (Optional[torch.Tensor]): Class weights. Shape is (C,).
                                         Default is None.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        size_average: bool = True,
        weight: Optional[torch.Tensor] = None,
    ):
        super(FocalLoss, self).__init__()
        """
        weight: size(C)
        """
        self.gamma = gamma
        self.size_average = size_average
        self.weight = weight

    def forward(self, inputs, targets):
        """
        inputs: size(N,C)
        targets: size(N)
        """
        log_P = -F.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none"
        )
        P = torch.exp(log_P)
        batch_loss = -torch.pow(1 - P, self.gamma).mul(log_P)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
