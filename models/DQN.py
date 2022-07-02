import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN (nn.Module):
    """
    Base DQN class for running RL models.
    """

    def __init__(self, num_frames, num_actions):
        super.__init__()
        self._num_frames = num_frames
        self._num_actions = num_actions

    def reset(self):
        pass

    def forward(self, obs):
        pass


def TD_loss ():
    pass
