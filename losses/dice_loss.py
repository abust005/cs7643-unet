import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def diceCoefficient(predicted, target, n_classes=3):

  dice = 0

  for c in range(n_classes):
    target_mask = torch.where(target==c, 1.0, 0.0)
    pred_mask = torch.where(predicted==c, 1.0, 0.0)

    # sum over last two dimensions
    intersect = (pred_mask * target_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1))

    # add some small offset to avoid div by 0
    dice = dice + ((2 * intersect + 1e-5) / (union + 1e-5)) 

  dice = dice.mean() / n_classes

  return dice

class DiceLoss(nn.Module):
    def __init__(self, log_cosh=False, n_classes=3, device='cpu'):
        super().__init__()

        self.device = device
        self.n_classes = n_classes

        self.log_cosh = log_cosh

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """

        dice = torch.zeros(input.shape[0], requires_grad=True).to(device=input.device)
        pred_mask = F.softmax(input, dim=1)

        for c in range(self.n_classes):
          target_mask = torch.where(target==c, 1.0, 0.0)

          # sum over last two dimensions
          intersect = (pred_mask[:,c,:,:] * target_mask).sum(dim=(-2, -1))
          union = pred_mask[:,c,:,:].sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1))

          union[union <= 1e-7] = 1.0
          
          # add some small offset to avoid div by 0
          dice = dice + ((2 * intersect) / (union)) 

        loss = 1 - (dice.mean() / self.n_classes)

        if self.log_cosh:
          loss = torch.log(torch.cosh(loss))

        return loss

    