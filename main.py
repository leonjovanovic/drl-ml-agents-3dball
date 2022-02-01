from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch

import Config
from Agent import Agent
from TestAgent import TestAgent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
# Start interacting with the environment.
env.reset()
behavior_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))
state_size = behavior_specs[0][0].shape[0]
action_size = behavior_specs[1].continuous_size
# NA POCETKU GLEDAMO KOME SVE TREBA AKCIJA----------------------
steps = list(env.get_steps(behavior_name))
decision_steps = steps[0]
terminal_steps = steps[1]
num_steps = len(decision_steps) + len(terminal_steps)
print(num_steps)
agent_ids = decision_steps.agent_id
print(behavior_name)
# CREATE ADDITIONAL CLASSES
agent = Agent(state_size, action_size, num_steps)
test_agent = TestAgent(env, behavior_name, state_size, action_size, num_steps)

# ULAZIMO U PETLJU
for n_step in range(Config.total_steps):
    #print(n_step)
    if agent.check_goal(n_step):
        if agent.test(test_agent):
            break
        decision_steps, terminal_steps = agent.get_steps(env, behavior_name)

    agent.calculate_ep_reward(decision_steps, terminal_steps)
    # NALAZIMO I SETUJEMO AKCIJU
    #actions = np.random.rand(decision_steps.reward.shape[0], action_size)*2-1
    actions = agent.get_actions(decision_steps, n_step)
    # UBACI STARI OBS I STARU AKCIJU
    env.set_actions(behavior_name, action=actions)
    # DO A STEP WITH GIVEN ACTIONS
    env.step()

    decision_steps, terminal_steps = agent.get_steps(env, behavior_name)
    # SEND IT TO THE MEMORY
    agent.add_to_buffer(decision_steps, terminal_steps)

    # UPDATE AGENT
    agent.update()

    agent.record_data(n_step)

env.close()



