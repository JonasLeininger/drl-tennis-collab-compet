import numpy as np
import torch
import torch.nn as nn


class MADDPGCritic(nn.Module):

    def __init__(self, config, hidden_units=(512, 256)):
        super(MADDPGCritic, self).__init__()
        self.fc1 = nn.Linear(config.state_dim * 2, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + config.action_dim*2, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        self.reset_parameters()
        self.device = config.device
        self.to(self.device)

    def forward(self, state, actions):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.cat((x, actions), dim=1)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
