import torch
import Config


class Buffer:
    def __init__(self, num_workers, state_shape, action_shape, episode_length):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer_index = 0
        self.episode_length = episode_length
        self.states = torch.zeros(Config.batch_size, state_shape).to(self.device)
        self.actions = torch.zeros(Config.batch_size, action_shape).to(self.device)
        self.logprob = torch.zeros(Config.batch_size, action_shape).to(self.device)
        self.rewards = torch.zeros(Config.batch_size).to(self.device)
        self.new_states = torch.zeros(Config.batch_size, state_shape).to(self.device)
        self.dones = torch.zeros(Config.batch_size).to(self.device)

        self.states_episode = torch.zeros(num_workers, self.episode_length, state_shape).to(self.device)
        self.actions_episode = torch.zeros(num_workers, self.episode_length, action_shape).to(self.device)
        self.logprob_episode = torch.zeros(num_workers, self.episode_length, action_shape).to(self.device)
        self.rewards_episode = torch.zeros(num_workers, self.episode_length).to(self.device)
        self.new_states_episode = torch.zeros(num_workers, self.episode_length, state_shape).to(self.device)
        self.dones_episode = torch.zeros(num_workers, self.episode_length).to(self.device)
        self.episode_step = torch.zeros(num_workers, dtype=torch.long).to(self.device)

        self.gt = torch.zeros(Config.batch_size + 1).to(self.device)
        self.advantages = torch.zeros(Config.batch_size + 1).to(self.device)
        self.full = False

    def add_old(self, decision_steps, actions, logprob):
        cnt = 0
        actionsTensor = torch.Tensor(actions).to(self.device)
        for obs, a_id in zip(decision_steps.obs[0], decision_steps.agent_id):
            self.states_episode[a_id, self.episode_step[a_id]] = torch.from_numpy(obs)
            self.actions_episode[a_id, self.episode_step[a_id]] = actionsTensor[cnt]
            self.logprob_episode[a_id, self.episode_step[a_id]] = logprob[cnt]
            cnt += 1

    def add(self, decision_steps, terminal_steps):
        for obs, a_id in zip(decision_steps.obs[0], decision_steps.agent_id):
            if decision_steps.reward[a_id] == 0:  # TERMINALNI JE KORAK, SKIPUJ OVO
                continue
            self.rewards_episode[a_id, self.episode_step[a_id]] = 0.1
            self.new_states_episode[a_id, self.episode_step[a_id]] = torch.from_numpy(obs)
            self.dones_episode[a_id, self.episode_step[a_id]] = 0
            self.episode_step[a_id] += 1

        for obs, a_id in zip(terminal_steps.obs[0], terminal_steps.agent_id):
            self.rewards_episode[a_id, self.episode_step[a_id]] = -1
            self.new_states_episode[a_id, self.episode_step[a_id]] = torch.from_numpy(obs)
            self.dones_episode[a_id, self.episode_step[a_id]] = 1
            self.episode_step[a_id] += 1

            if not self.full:
                last_index = min(self.buffer_index + self.episode_step[a_id], Config.batch_size)
                self.states[self.buffer_index: last_index] = self.states_episode[a_id, : last_index - self.buffer_index]
                self.actions[self.buffer_index: last_index] = self.actions_episode[a_id, : last_index - self.buffer_index]
                self.logprob[self.buffer_index: last_index] = self.logprob_episode[a_id, : last_index - self.buffer_index]
                self.rewards[self.buffer_index: last_index] = self.rewards_episode[a_id, : last_index - self.buffer_index]
                self.new_states[self.buffer_index: last_index] = self.new_states_episode[a_id, : last_index - self.buffer_index]
                self.dones[self.buffer_index: last_index] = self.dones_episode[a_id, : last_index - self.buffer_index]

                self.buffer_index = last_index % Config.batch_size
                if self.buffer_index == 0:
                    self.full = True
            self.episode_step[a_id] = 0

    def advantage(self, state_values, last_state_value):
        self.full = False
        gt = last_state_value
        for i in reversed(range(Config.batch_size)):
            gt = self.rewards[i] + Config.gamma * gt * (1 - self.dones[i])
            self.gt[i] = gt
            self.advantages[i] = gt - state_values[i]

    def gae_advantage(self, state_values, new_state_values):
        self.full = False
        self.gt[Config.batch_size] = new_state_values[-1]
        for i in reversed(range(Config.batch_size)):
            delta = self.rewards[i] + Config.gamma * new_state_values[i] * (1 - self.dones[i]) - state_values[i]
            self.advantages[i] = delta + Config.gae_lambda * Config.gamma * self.advantages[i+1] * (1 - self.dones[i])
            # For critic
            self.gt[i] = self.rewards[i] + Config.gamma * self.gt[i+1] * (1 - self.dones[i])

    def reset(self, full=False):
        if full:
            self.buffer_index = 0
        self.episode_step[self.episode_step != 0] = 0
