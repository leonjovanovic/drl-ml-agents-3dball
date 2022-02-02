import torch
import Config
from NN import PolicyNN, CriticNN

class AgentControl:
    def __init__(self, state_shape, action_shape):
        self.action_shape = action_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.moving_policy_nn = PolicyNN(state_shape, action_shape).to(self.device)
        self.policy_optim = torch.optim.Adam(params=self.moving_policy_nn.parameters(), lr=Config.policy_lr,
                                             eps=Config.adam_eps)
        self.moving_critic_nn = CriticNN(state_shape + action_shape).to(self.device)
        self.critic_optim = torch.optim.Adam(params=self.moving_critic_nn.parameters(), lr=Config.critic_lr,
                                             eps=Config.adam_eps)
        self.target_policy_nn = PolicyNN(state_shape, action_shape).to(self.device)
        self.target_policy_nn.load_state_dict(self.moving_policy_nn.state_dict())
        self.target_critic_nn = CriticNN(state_shape + action_shape).to(self.device)
        self.target_critic_nn.load_state_dict(self.moving_critic_nn.state_dict())
        self.mse = torch.nn.MSELoss()

        self.noise_std = 0.1

    def get_actions(self, states, n_step):
        if n_step < Config.start_steps:
            return torch.rand(states.shape[0], 2) * 2 - 1
        else:
            actions = self.moving_policy_nn(torch.Tensor(states).to(self.device))
            noise = (self.noise_std ** 0.5) * torch.randn(self.action_shape).to(self.device)
            return torch.clip(actions + noise, -1, 1)

    def lr_decay(self, n_step):
        frac = 1 - n_step / Config.total_steps
        self.policy_optim.param_groups[0]["lr"] = frac * Config.policy_lr
        self.critic_optim.param_groups[0]["lr"] = frac * Config.critic_lr
        self.noise_std = self.noise_std * frac

    def update_critic(self, states, actions, rewards, new_states, dones):
        new_actions = self.target_policy_nn(new_states).detach()
        new_value = self.target_critic_nn(new_states, new_actions).squeeze(-1).detach()
        target = rewards + Config.gamma * new_value * (1 - dones)
        state_value = self.moving_critic_nn(states, actions).squeeze(-1)
        critic_loss = self.mse(state_value, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.detach().cpu().numpy()

    def update_policy(self, states):
        policy_actions = self.moving_policy_nn(states)
        critic_value = self.moving_critic_nn(states, policy_actions).squeeze(-1)
        # Used `-value` as we want to maximize the value given by the critic for our actions
        policy_loss = -torch.mean(critic_value)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return policy_loss.detach().cpu().numpy()

    def update_target_nns(self):
        # Update target networks by polyak averaging.
        with torch.no_grad():
            for mov, targ in zip(self.moving_critic_nn.parameters(), self.target_critic_nn.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                targ.data.mul_(Config.polyak)
                targ.data.add_((1 - Config.polyak) * mov.data)

            for mov, targ in zip(self.moving_policy_nn.parameters(), self.target_policy_nn.parameters()):
                targ.data.mul_(Config.polyak)
                targ.data.add_((1 - Config.polyak) * mov.data)

