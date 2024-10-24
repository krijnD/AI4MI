import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# Import your custom Focal Loss
from nnunetv2.training.loss.focal import FocalLoss

class nnUNetTrainerFocalLoss(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 75  # Set your desired number of epochs

    def _build_loss(self):
        # Initialize Focal Loss with appropriate parameters
        focal_loss = FocalLoss(
            gamma=2.0,
            alpha = None,  # Set class weights if needed (e.g., alpha=[0.25, 0.75])
            ignore_index=self.label_manager.ignore_label
        )

        # No non-linearity is applied since FocalLoss expects raw logits

        # Wrap the loss with DeepSupervisionWrapper if deep supervision is enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # Assign weights to each output
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # Do not use the lowest resolution output
            weights = weights / weights.sum()

            # Define a loss function that can be called by DeepSupervisionWrapper
            def loss_fn(outputs, targets):
                return focal_loss(outputs, targets)

            loss = DeepSupervisionWrapper(loss_fn, weights)
        else:
            loss = focal_loss

        return loss