import torch
import numpy as np
import Config


class Buffer:
    def __init__(self, state_shape, action_shape, total_steps):
        self.buffer_size = Config.buffer_size
        self.buffer_index = -1
        self.initialized = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.states = torch.zeros(self.buffer_size, state_shape).to(device)
        self.actions = torch.zeros(self.buffer_size, action_shape).to(device)
        self.new_states = torch.zeros(self.buffer_size, state_shape).to(device)
        self.rewards = torch.zeros(self.buffer_size).to(device)
        self.dones = torch.zeros(self.buffer_size).to(device)

        self.old_state = torch.zeros(total_steps, state_shape).to(device)
        self.old_actions = torch.zeros(total_steps, action_shape).to(device)

    def add_old(self, decision_steps, actions):
        cnt = 0
        for obs, a_id in zip(decision_steps.obs[0], decision_steps.agent_id):
            self.old_state[a_id] = torch.from_numpy(obs)
            self.old_actions[a_id] = actions[cnt]
            cnt += 1

    def add(self, decision_steps, terminal_steps):
        for obs, a_id in zip(decision_steps.obs[0], decision_steps.agent_id):
            if decision_steps.reward[a_id] == 0:
                continue
            if self.buffer_index == Config.buffer_size - 1 and not self.initialized:
                self.initialized = True
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.states[self.buffer_index] = self.old_state[a_id]
            self.actions[self.buffer_index] = self.old_actions[a_id]
            self.new_states[self.buffer_index] = torch.from_numpy(obs)
            self.rewards[self.buffer_index] = 0.1
            self.dones[self.buffer_index] = 0

        for obs, a_id in zip(terminal_steps.obs[0], terminal_steps.agent_id):
            if self.buffer_index == Config.buffer_size - 1 and not self.initialized:
                self.initialized = True
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.states[self.buffer_index] = self.old_state[a_id]
            self.actions[self.buffer_index] = self.old_actions[a_id]
            self.new_states[self.buffer_index] = torch.from_numpy(obs)
            self.rewards[self.buffer_index] = -1
            self.dones[self.buffer_index] = 1

    def sample_indices(self, batch_size):
        indices = np.arange(min(self.buffer_size, self.buffer_index) if not self.initialized else self.buffer_size)
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        if len(indices) != 64:
            print(len(indices))
        return indices
