import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDiceLoss(nnUNetTrainer):
    def __init__(
    self,
    plans: dict,
    configuration: str,
    fold: int,
    dataset_json: dict,
    unpack_dataset: bool = True,
    device: torch.device = torch.device('cuda')
    ):
        # Call the base class __init__ with all required arguments
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Set num_epochs after the base class initialization
        self.num_epochs = 75

    def _build_loss(self):
        # loss = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
        #                             'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp},
        #                     apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)
        
        if self.label_manager.has_regions:
            loss = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
                                                'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                            apply_nonlin=torch.sigmoid)
        else:
            loss = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
                                                'do_bg': False, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                            apply_nonlin=softmax_helper_dim1)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceCELoss_noSmooth(nnUNetTrainer):
    def __init__(
    self,
    plans: dict,
    configuration: str,
    fold: int,
    dataset_json: dict,
    unpack_dataset: bool = True,
    device: torch.device = torch.device('cuda')
    ):
        # Call the base class __init__ with all required arguments
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Set num_epochs after the base class initialization
        self.num_epochs = 75
        
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

