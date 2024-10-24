import torch
from torch import nn
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad

class MemoryEfficientTverskyLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = True,
    ):
        super(MemoryEfficientTverskyLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.beta = beta
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        # Apply non-linearity
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Determine axes for summation
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            # Ensure y has correct shape
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            # One-hot encode y using float tensors
            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros_like(x)
                y_onehot.scatter_(1, y.long(), 1)

            # Exclude background class if needed
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # Exclude background class from x if needed
        if not self.do_bg:
            x = x[:, 1:]

        # Compute Tversky components
        if loss_mask is None:
            tp = (x * y_onehot).sum(dim=axes)
            fp = (x * (1 - y_onehot)).sum(dim=axes)
            fn = ((1 - x) * y_onehot).sum(dim=axes)
        else:
            loss_mask = loss_mask.unsqueeze(1)
            tp = (x * y_onehot * loss_mask).sum(dim=axes)
            fp = (x * (1 - y_onehot) * loss_mask).sum(dim=axes)
            fn = ((1 - x) * y_onehot * loss_mask).sum(dim=axes)

        # Handle batch dice
        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            else:
                tp = tp.sum(0)
                fp = fp.sum(0)
                fn = fn.sum(0)

        # Compute Tversky coefficient
        tversky_coef = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Compute loss
        loss = 1 - tversky_coef
        loss = loss.mean()

        return loss