###################################
# model/loss.py
###################################
# Description:
# * Loss function for neural transfer.

import torch.functional as F
import torch.nn as nn

class ContentLoss(nn.Module):
    """
    * Loss function for training
    neural transfer.
    """
    def __init__(self, target):
        """
        * Initialize loss function.
        """
        super(ContentLoss, self).__init__()
        self.__initialize(target)

    ###############
    # Interface Methods:
    ###############
    def forward(self, input):
        """
        * Forward pass image through
        loss function.
        """
        self.loss = F.mse_loss(input, self.target)
        return input

    ###############
    # Private Helpers:
    ###############
    def __initialize(self, target):
        """
        * Detach the target content
        to dynamically compute gradient.
        """
        self.target = target.detach()