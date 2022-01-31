import torch

import Config


class Buffer:
    def __init__(self, state_shape, action_shape, total_steps):
        self.buffer_size = Config.buffer_size
        self.buffer_index = -1
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
        # ADD TO BUFFER
        # MI SMO OVDE NAPRAVILI KORAK I TREBA DA NADJEMO STARI STATE ZA TAJ AGENT ID
        # DA BI NOVI STATE, REWARD, ACTION DODALI U BUFFER
        for obs, a_id in zip(decision_steps.obs[0], decision_steps.agent_id):
            if decision_steps.reward[a_id] == 0:
                continue
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.states[self.buffer_index] = self.old_state[a_id]
            self.actions[self.buffer_index] = self.old_actions[a_id]
            self.new_states[self.buffer_index] = torch.from_numpy(obs)
            self.rewards[self.buffer_index] = 0.1
            self.dones[self.buffer_index] = 0

        cnt = 0
        for obs, a_id in zip(terminal_steps.obs[0], terminal_steps.agent_id):
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.states[self.buffer_index] = self.old_state[a_id]
            self.actions[self.buffer_index] = self.old_actions[a_id]
            self.new_states[self.buffer_index] = torch.from_numpy(obs)
            self.rewards[self.buffer_index] = -1
            self.dones[self.buffer_index] = 1
            cnt += 1
        print(self.buffer_index + 1)


