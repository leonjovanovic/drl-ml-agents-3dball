import torch

from NN import PolicyNN, CriticNN


class AgentControl:
    def __init__(self, state_shape, action_shape):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.moving_policy_nn = PolicyNN(state_shape, action_shape).to(self.device)

    def get_actions(self, states):
        return self.moving_policy_nn(torch.Tensor(states).to(self.device))
