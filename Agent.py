import Config
from Buffer import Buffer
from AgentControl import AgentControl
from mlagents_envs.base_env import ActionTuple

class Agent:
    def __init__(self, state_shape, action_shape, num_steps):
        self.agent_control = AgentControl(state_shape, action_shape)
        self.buffer = Buffer(state_shape, action_shape, num_steps)

    def get_actions(self, decision_steps, n_step):
        actions = self.agent_control.get_actions(decision_steps.obs[0], n_step).detach()
        self.buffer.add_old(decision_steps, actions)
        self.agent_control.lr_decay(n_step + 1)
        return ActionTuple(continuous=actions.cpu().numpy())

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add(decision_steps, terminal_steps)
        print(self.buffer.buffer_index)

    def update(self):
        if self.buffer.buffer_index < Config.min_buffer_size:
            return
        # Sample random CONFIG BATCH SIZE minibatch
        indices = self.buffer.sample_indices(Config.batch_size)
        # Update Critic Moving NN using that minibatch
        critic_loss = self.agent_control.update_critic(self.buffer.states[indices], self.buffer.actions[indices], self.buffer.rewards[indices], self.buffer.new_states[indices], self.buffer.dones[indices])
        # Update Policy Moving NN using that minibatch
        policy_loss = self.agent_control.update_policy(self.buffer.states[indices])
        # Update Critic & Policy Target NN using Polyak averaging
        self.agent_control.update_target_nns()

    def record_data(self):
        pass

