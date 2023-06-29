import torch
import torch.nn as nn

from segmentation_models_pytorch.losses import *
import segmentation_models_pytorch.losses as losses

class LossCombo(nn.Module):
    def __init__(self, alpha1, loss1, alpha2, loss2):
        super().__init__()
        self.loss1 = getattr(losses, loss1.name)(**loss1.args)
        self.loss2 = getattr(losses, loss2.name)(**loss2.args)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true):
        return self.alpha1 * self.loss1(y_pred, y_true) + self.alpha2 * self.loss2(y_pred, y_true)