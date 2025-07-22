import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def reweight(cls_num_list, beta=0.9999):
    """
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """

    per_cls_weights = (1 - beta) / (1 - torch.pow(cls_num_list, beta)) 
    per_cls_weights /= torch.sum(per_cls_weights)
    per_cls_weights *= len(cls_num_list)

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0, device='cpu'):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.device = device
        self.weight = self.weight.to(device=device)
        self.softmax_fn = nn.Softmax2d()
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        ce = -self.ce_fn(input, target)
        pt_i = torch.exp(ce)
        loss = -torch.pow(1-pt_i, self.gamma) * ce

        return loss
