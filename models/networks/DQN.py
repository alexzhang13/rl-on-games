import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN (nn.Module):
    """
    Base DQN class for running RL models. 
    Args:
        num_frames: number of frames stacked as input
    """

    def __init__(self, num_frames, num_actions):
        super().__init__()
        self._num_frames = num_frames
        self._num_actions = num_actions
        
        # mario: (4, 84, 84)
        self.net = nn.Sequential(
            nn.Conv2d(self._num_frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
			nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self._num_actions)
        )


    def forward(self, obs):
        actions = self.net(obs)
        return actions
