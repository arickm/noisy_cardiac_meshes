"""
nnU-Net trainer with 750 epochs (ISBI 2027: same training length as models A and B).

nnU-Net only finds trainers inside its installed package: copy this file to
nnunetv2/training/nnUNetTrainer/variants/training_length/. The constructor signature differs between
nnU-Net versions (unpack_dataset was removed in 2.5), and nnU-Net reads the subclass signature to
store its init arguments, so the matching signature is chosen at import time.
"""

import inspect

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

NUM_EPOCHS = 750

if 'unpack_dataset' in inspect.signature(nnUNetTrainer.__init__).parameters:
    class nnUNetTrainer_750epochs(nnUNetTrainer):
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
            self.num_epochs = NUM_EPOCHS
else:
    class nnUNetTrainer_750epochs(nnUNetTrainer):
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = NUM_EPOCHS
