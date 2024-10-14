#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from torch import einsum
import numpy as np
import torch


from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()  # Convert g to float

        num = 2 * einsum("bkwh,bkwh->", p, g)
        den = (p + g).sum() + 1e-10

        return 1 - num / den
    
class CombinedLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.dice_loss = DiceLoss(idk=self.idk)
        self.cross_entropy = CrossEntropy(idk=self.idk)
        self.weight_ce = kwargs.get('weight_ce', 1.0)   # Weight for Cross Entropy
        self.weight_dice = kwargs.get('weight_dice', 1.0)  # Weight for Dice Loss
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        loss_ce = self.cross_entropy(pred_softmax, weak_target)
        loss_dice = self.dice_loss(pred_softmax, weak_target)
        total_loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice
        return total_loss
    
class FocalLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.alpha = kwargs.get('alpha', None)  # Should be a tensor or list
        self.gamma = kwargs.get('gamma', 2.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        epsilon = 1e-10
        pred = pred.clamp(min=epsilon, max=1.0 - epsilon)

        # Compute log probabilities
        log_p = pred.log()

        # Compute focal loss
        ce_loss = - (target * log_p)  # Cross entropy
        p_t = pred * target + (1 - pred) * (1 - target)  # Probabilities of target classes
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                # Reshape alpha to match the dimensions of pred and target
                alpha = torch.tensor(self.alpha).to(pred.device).view(1, len(self.alpha), 1, 1)
            else:
                alpha = self.alpha
            # Broadcast alpha to match target and pred shape
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss
        loss = loss.sum() / (target.sum() + epsilon)

        return loss

    
class TverskyLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']  # indices of classes to supervise
        self.alpha = kwargs.get('alpha', None)  # alpha can be a list
        self.beta = kwargs.get('beta', None)    # beta can be a list
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        # Check the inputs
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        epsilon = 1e-10
        pred = pred.clamp(min=epsilon, max=1.0 - epsilon)

        # True positives, false positives, and false negatives
        TP = einsum("bkwh,bkwh->k", pred, target)  # True positives, summed per class
        FP = einsum("bkwh,bkwh->k", pred, 1 - target)  # False positives, summed per class
        FN = einsum("bkwh,bkwh->k", 1 - pred, target)  # False negatives, summed per class

        # Handle class-specific alpha and beta values
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha).to(pred.device).view(1, len(self.alpha), 1, 1)
        else:
            alpha = torch.tensor([0.5] * len(self.idk)).to(pred.device).view(1, len(self.idk), 1, 1)

        if self.beta is not None:
            beta = torch.tensor(self.beta).to(pred.device).view(1, len(self.beta), 1, 1)
        else:
            beta = torch.tensor([0.5] * len(self.idk)).to(pred.device).view(1, len(self.idk), 1, 1)

        # Tversky index per class
        Tversky_index = TP / (TP + alpha.squeeze() * FP + beta.squeeze() * FN + epsilon)

        # Tversky loss per class
        loss_per_class = 1 - Tversky_index

        # Average the loss across all classes to return a scalar
        loss = loss_per_class.mean()

        return loss

