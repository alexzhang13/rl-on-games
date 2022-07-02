import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN (nn.Module):
    """
    Base DQN class for running RL models.
    """

    def __init__(self):
        super.__init__()

    def reset(self):
        pass

    def forward(self, obs):
        pass


def TD_loss ():
    pass
