import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1

# Import your custom MemoryEfficientTverskyLoss
from nnunetv2.training.loss.tversky import MemoryEfficientTverskyLoss

class nnUNetTrainerTverskyLoss(nnUNetTrainer):
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
        self.num_epochs = 75 # Set your desired number of epochs

    def _build_loss(self):
        # Initialize Tversky Loss with appropriate parameters
    #     loss = MemoryEfficientTverskyLoss(
    #         apply_nonlin=apply_nonlin,
    #         alpha=0.5,
    #         beta=0.5,
    #         batch_dice=self.configuration_manager.batch_dice,
    #         do_bg=do_bg,
    #         smooth=1e-5,
    #         ddp=self.is_ddp
    # )
        if self.label_manager.has_regions:
            loss = MemoryEfficientTverskyLoss(
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth=1e-5,
                ddp=self.is_ddp,
                apply_nonlin=torch.sigmoid
            )
        else:
            loss = MemoryEfficientTverskyLoss(
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=False,
                smooth=1e-5,
                ddp=self.is_ddp,
                apply_nonlin=softmax_helper_dim1
            )

        # Wrap the loss with DeepSupervisionWrapper if deep supervision is enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # Assign weights to each output
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # Do not use the lowest resolution output
            weights = weights / weights.sum()

            # Now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss