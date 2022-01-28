from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
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
states = torch.Tensor(decision_steps.obs[0]).to(device)
rewards = torch.Tensor(decision_steps.reward).to(device)
agent_ids = decision_steps.agent_id
print(rewards)
print(agent_ids)
'''in decision_steps:
    print(k.obs[0])
    print(k.reward)
    print(k.agent_id)
    print(k.group_id)
    print(k.group_reward)
    if not flag:
        print(k.action_mask)
        flag = True
    else:
        print(k.interrupted)
'''
# ULAZIMO U PETLJU
while True:
    # NALAZIMO AKCIJU
    print(np.random.rand(decision_steps.reward.shape[0], action_size)*2-1)
    actions = ActionTuple()
    actions.add_continuous(np.random.rand(decision_steps.reward.shape[0], action_size)*2-1)
    env.set_actions("3DBall?team=0", action=actions)
    env.step()

    steps = list(env.get_steps("3DBall?team=0"))
    decision_steps = steps[0]
    terminal_steps = steps[1]



