import torch.nn as nn
import torch.nn.functional as F
from gym import spaces


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.

    DQN(
        (conv1): Conv2d(210, 32, kernel_size=(8, 8), stride=(4, 4))
        (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (linear1): Linear(in_features=3136, out_features=512, bias=True)
        (linear2): Linear(in_features=512, out_features=6, bias=True)
    )

    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 7, 512)
        self.linear2 = nn.Linear(512, action_space.n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)  # flatten
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
