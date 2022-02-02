import torch

import Config
from NN import PolicyNN, CriticNN


class AgentControl:
    def __init__(self, state_shape, action_shape):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(state_shape, action_shape).to(self.device)
        self.critic_nn = CriticNN(state_shape).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy_nn.parameters(), lr=Config.policy_lr, eps=Config.adam_eps)
        self.critic_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=Config.critic_lr, eps=Config.adam_eps)
        self.mse = torch.nn.MSELoss()

    def get_actions(self, state):
        actions, logprob = self.policy_nn(torch.Tensor(state).to(self.device))
        return actions.detach().cpu().numpy(), logprob.detach()

    def update_lr(self, n_step):
        frac = 1 - n_step / Config.total_steps
        self.policy_optim.param_groups[0]["lr"] = Config.policy_lr * frac
        self.critic_optim.param_groups[0]["lr"] = Config.critic_lr * frac

    def get_critic_value_d(self, states):
        return self.critic_nn(states).squeeze(-1).detach()

    def calculate_ratio(self, states, actions, logprob):
        _, new_logprob = self.policy_nn(states, actions)
        return torch.exp(torch.sum(new_logprob, dim=1) - torch.sum(logprob, dim=1).detach())

    def update_policy(self, ratio, advantages):
        # Normalize the advantages to reduce variance
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        clipped_ratio = torch.clip(ratio, 0.8, 1.2)
        policy_loss = torch.minimum(ratio * advantages_norm, clipped_ratio * advantages_norm)
        policy_loss = -torch.mean(policy_loss)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        # The gradients of the policy network are clipped so that the “global l2 norm” (i.e. the norm of the
        # concatenated gradients of all parameters) does not exceed 0.5
        torch.nn.utils.clip_grad_norm_(self.policy_nn.parameters(), Config.max_grad_norm)
        self.policy_optim.step()

        return policy_loss.detach().cpu().numpy()

    def update_critic(self, states, gt):
        estimated_value = self.critic_nn(states).squeeze(-1)
        critic_loss = self.mse(estimated_value, gt)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.detach().cpu().numpy()

    def get_policy_nn(self):
        return self.policy_nn
