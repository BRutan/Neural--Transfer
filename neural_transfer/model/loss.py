###################################
# model/loss.py
###################################
# Description:
# * Loss function for neural transfer.

from data.transform import gram_matrix
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
        super(ContentLoss, self).__init__()
        self.target = target.detach()

class StyleLoss(nn.Module):
    """
    * Loss function for "style" 
    between target image and style image.
    """
    def __init__(self, target_feature):
        """
        * Initialize loss function.
        """
        self.__initialize()

    def forward(self, input):
        """
        * Forward pass image through
        loss function.
        """
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    ###############
    # Private Helpers:
    ###############
    def __initialize(self, target_feature):
        """
        * Initialize object. 
        """
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
