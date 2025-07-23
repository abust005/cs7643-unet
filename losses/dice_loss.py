import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def diceCoefficient(predicted, target, n_classes=3, logits=True):

  if logits:
    predicted = F.softmax(predicted, dim=1)
    predicted = torch.argmax(predicted, dim=1)

  dice = 0

  for c in range(n_classes):
    pred_mask = torch.where(predicted==c, 1.0, 0.0)
    target_mask = torch.where(target==c, 1.0, 0.0)

    # sum over last two dimensions
    intersect = (pred_mask * target_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1))

    dice += (2 * intersect) / union

  dice = 1 - (dice / n_classes)

  return dice

class DiceLoss(nn.Module):
    def __init__(self, log_cosh=False, n_classes=3, device='cpu'):
        super().__init__()

        self.device = device

        if log_cosh:
           self.f = lambda x, y: torch.log(torch.cosh(diceCoefficient(x, y, n_classes)))
        else:
           self.f = lambda x, y: diceCoefficient(x, y, n_classes)

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = self.f(input, target)

        return loss
    