import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

import NN

MODEL_NAME = '3dBall_0_5.20.16.40.pt'

path = 'models/' + MODEL_NAME
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = UnityEnvironment(file_name='PATH_TO_FOLDER/UnityEnvironment.exe', seed=1, side_channels=[])
env.reset()
behavior_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))
state_shape = behavior_specs[0][0].shape[0]
action_shape = behavior_specs[1].continuous_size

nn = NN.PolicyNN(state_shape, action_shape).to(device)
nn.load_state_dict(torch.load(path))

steps = list(env.get_steps(behavior_name))
decision_steps, _ = steps[0], steps[1]

while True:
    actions, _ = nn(torch.Tensor(decision_steps.obs[0]).to(device))
    actionTuple = ActionTuple(continuous=actions.detach().cpu().numpy())
    env.set_actions(action=actionTuple, behavior_name=behavior_name)
    env.step()
    steps = list(env.get_steps(behavior_name))
    decision_steps = steps[0]
