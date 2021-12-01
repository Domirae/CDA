"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import List, Dict
import torch.nn as nn
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import cv2

__all__ = ['DomainDiscriminator']

def prob2entropy2(prob):
    # convert prob prediction maps to weighted self-information maps
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30))-torch.mul((1-prob), torch.log2((1-prob) + 1e-30))

def visualization(vis_num,step,epoch,feature,name):
    if step % vis_num  == 0: 
        plt.clf()
        cam  = feature[0][0].detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam_img = cam /np.max(cam)
        plt.imshow(cam,cmap='jet')
        wandb.log({name:wandb.Image(plt,caption="{}_{}".format(epoch,step))})


class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]

class LocalDomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self):
        super(LocalDomainDiscriminator, self).__init__(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1,padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1,padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1, stride=1,padding=0, bias=False),
            nn.Sigmoid()
        )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]




