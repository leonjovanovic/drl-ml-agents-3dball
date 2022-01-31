from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
from Agent import Agent

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
steps = list(env.get_steps("3DBall?team=0"))
decision_steps = steps[0]
terminal_steps = steps[1]
num_steps = len(decision_steps) + len(terminal_steps)
print(num_steps)
agent_ids = decision_steps.agent_id
print(behavior_name)
# CREATE ADDITIONAL CLASSES
agent = Agent(state_size, action_size, num_steps)

'''
print(k.obs[0][0])
print(k.reward)
print(k.agent_id)
print(k.group_id)
print(k.group_reward)
'''
# ULAZIMO U PETLJU
while True:
    # COLLECT STATE
    states = decision_steps.obs[0]
    old_decisions = decision_steps
    # NALAZIMO I SETUJEMO AKCIJU
    # REPLACE WITH NN
    #actions = np.random.rand(decision_steps.reward.shape[0], action_size)*2-1
    actions = agent.get_actions(decision_steps)
    # UBACI STARI OBS I STARU AKCIJU
    env.set_actions("3DBall?team=0", action=actions)
    # DO A STEP WITH GIVEN ACTIONS
    env.step()
    # RETREIVE REWARD, NEXT STATE AND DONE???
    steps = list(env.get_steps("3DBall?team=0"))
    decision_steps = steps[0]
    terminal_steps = steps[1]
    # SEND IT TO THE MEMORY
    agent.add_to_buffer(decision_steps, terminal_steps)

    # UPDATE AGENT
    agent.update()

    #break # IZBRISATI NA KRAJU

env.close()



