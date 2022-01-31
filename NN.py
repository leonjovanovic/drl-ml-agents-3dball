import torch
from torch import nn


class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape),
            nn.Tanh()
        )

    def forward(self, states):
        return self.model(states)


class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

    def forward(self, states, actions):
        return self.model(torch.cat((states, actions), 1))
