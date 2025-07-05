import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.softmax_fn = torch.nn.Softmax(dim=1)

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        # exp = torch.exp(input)
        # softmax = (exp / torch.broadcast_to(torch.sum(exp, dim=1, keepdim=True), exp.shape))
        
        softmax = self.softmax_fn(input)

        pt_i = softmax[torch.arange(len(target)), target]
        ce = -torch.log(pt_i)
        loss = self.weight[target] * torch.pow((1-pt_i), self.gamma) * ce

        loss = torch.sum(loss) / torch.sum(self.weight)

        return loss
