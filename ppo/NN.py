import torch
from torch import nn
from torch.distributions import Normal
import Config

torch.manual_seed(Config.seed)

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.actions_mean = nn.Sequential(
            nn.Linear(input_shape, 216),
            nn.ReLU(),
            nn.Linear(216, 216),
            nn.ReLU(),
            nn.Linear(216, output_shape),
            nn.Tanh()
        )

        self.actions_logstd = nn.Parameter(torch.zeros(output_shape))

    def forward(self, states, actions=None):
        actions_mean = self.actions_mean(states)
        actions_std = torch.exp(self.actions_logstd)
        normal_ds = Normal(actions_mean, actions_std)
        if actions is None:
            actions = normal_ds.sample()
        return actions, normal_ds.log_prob(actions)

class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 216),
            nn.ReLU(),
            nn.Linear(216, 216),
            nn.ReLU(),
            nn.Linear(216, 1)
        )

    def forward(self, states):
        return self.model(states)
