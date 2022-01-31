from Buffer import Buffer
from AgentControl import AgentControl
from mlagents_envs.base_env import ActionTuple

class Agent:
    def __init__(self, state_shape, action_shape, num_steps):
        self.agent_control = AgentControl(state_shape, action_shape)
        self.buffer = Buffer(state_shape, action_shape, num_steps)

    def get_actions(self, decision_steps):
        actions = self.agent_control.get_actions(decision_steps.obs[0])
        self.buffer.add_old(decision_steps, actions)
        return ActionTuple(continuous=actions.detach().cpu().numpy())

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add(decision_steps, terminal_steps)

    def update(self):
        # Sample random CONFIG BATCH SIZE minibatch
        # Update Critic Moving NN using that minibatch
        # Update Policy Moving NN using that minibatch
        # Update Critic & Policy Target NN using Polyak averaging 

