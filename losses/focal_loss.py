import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def debug(mask):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(mask[0, 0].cpu().numpy())
    axs[0, 1].imshow(mask[0, 1].cpu().numpy())
    axs[1, 0].imshow(mask[0, 2].cpu().numpy())
    axs[1, 1].imshow(mask[0, 3].cpu().numpy())

    plt.tight_layout()

    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)

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
        self.ce_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        ce = -self.ce_fn(input, target)
        pt_i = torch.exp(ce)
        loss = self.weight[torch.argmax(target, dim=1)] * -torch.pow(1-pt_i, self.gamma) * ce
        loss = torch.sum(loss) / (loss.shape[0] * loss.shape[1] * loss.shape[2])

        return loss
