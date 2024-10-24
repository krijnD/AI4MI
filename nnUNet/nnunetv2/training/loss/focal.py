import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification, suitable for nnUNetV2.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int, list, np.ndarray)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha  # Tensor of shape (C,)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits with shape (N, C, H, W) or (N, C, D, H, W)
            targets: Ground truth labels with shape (N, H, W) or (N, D, H, W)
        """
        targets = targets.long()

        # Remove the channel dimension if present
        if targets.dim() == inputs.dim():
            # This means targets have a channel dimension of size 1
            targets = targets.squeeze(1)

        # Prepare arguments for F.cross_entropy
        kwargs = {'reduction': 'none'}
        if self.ignore_index is not None:
            kwargs['ignore_index'] = self.ignore_index

        # Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, **kwargs)

        # Compute the probabilities of the true classes
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class

        # Compute the focal loss
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha = self.alpha.to(inputs.device)
            at = alpha[targets]
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index is not None:
                valid_mask = (targets != self.ignore_index)
                loss = loss * valid_mask  # Exclude ignored elements
                total_loss = loss.sum()
                num_elements = valid_mask.sum()
            else:
                total_loss = loss.sum()
                num_elements = loss.numel()
            if num_elements > 0:
                return total_loss / num_elements
            else:
                return torch.tensor(0.0, device=inputs.device)
        else:
            return loss.sum()