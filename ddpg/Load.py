import torch.cuda
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import NN

MODEL_NAME = "3dBall1.18.25.19.pt"

path = 'models/' + MODEL_NAME

env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()
behaivor_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy_nn = NN.PolicyNN(behavior_specs[0][0].shape[0], behavior_specs[1].continuous_size).to(device)
policy_nn.load_state_dict(torch.load(path))

steps = list(env.get_steps(behaivor_name))
decision_steps = steps[0]

while True:
    actions = policy_nn(torch.Tensor(decision_steps.obs[0]).to(device)).detach().cpu().numpy()
    actionTuple = ActionTuple(continuous=actions)
    env.set_actions(action=actionTuple, behavior_name=behaivor_name)
    env.step()
    steps = list(env.get_steps(behaivor_name))
    decision_steps = steps[0]
    terminal_steps = steps[1]


#input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(6)]
#output_names = ["output1"]
#torch.onnx.export(policy_nn, torch.Tensor(decision_steps.obs[0]).to(device),MODEL_NAME+".onnx", input_names=input_names, output_names=output_names, verbose=True)
