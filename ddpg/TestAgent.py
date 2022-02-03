from collections import deque
import numpy as np
from mlagents_envs.base_env import ActionTuple
import Config
import NN
import torch

class TestAgent:
    def __init__(self, env, behavior_name, state_shape, action_shape, num_steps):
        self.env = env
        self.behavior_name = behavior_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = NN.PolicyNN(input_shape=state_shape, output_shape=action_shape).to(self.device)
        self.num_steps = num_steps
        self.return_queue = deque(maxlen=100)
        self.reward_agents = [0] * num_steps
        self.n_episode = 0

    def test(self, nn, writer, n_step):
        self.test_reset(nn, self.num_steps)
        steps_tuple = list(self.env.get_steps(self.behavior_name))
        decision_steps = steps_tuple[0]
        terminal_steps = steps_tuple[1]
        self.n_episode = 0
        # Create new enviroment and test it for 100 episodes using the model we trained
        print("Testing...")
        while self.n_episode < 100:
            # Get the action from Policy NN given the state
            actions = self.policy_nn(torch.Tensor(decision_steps.obs[0]).to(self.device)).detach().cpu().numpy()
            actionsTuple = ActionTuple(continuous=actions)
            self.env.set_actions(self.behavior_name, actionsTuple)
            self.env.step()
            steps_tuple = list(self.env.get_steps(self.behavior_name))
            decision_steps = steps_tuple[0]
            terminal_steps = steps_tuple[1]
            self.calculate_ep_reward(decision_steps, terminal_steps)
        print(self.return_queue)
        mean_return = np.mean(self.return_queue)
        self.env.reset()
        if writer is not None:
            writer.add_scalar('test100rew', mean_return, n_step)
        return self.check_goal(mean_return)

    def test_reset(self, nn, num_steps):
        self.return_queue = deque(maxlen=100)
        self.reward_agents = [0] * num_steps
        self.n_episode = 0
        self.policy_nn.load_state_dict(nn.state_dict())
        self.env.reset()

    def calculate_ep_reward(self, decision_steps, terminal_steps):
        for a_id in decision_steps.agent_id:
            self.reward_agents[a_id] += decision_steps.reward[a_id]
        cnt = 0
        for a_id in terminal_steps.agent_id:
            self.reward_agents[a_id] += terminal_steps.reward[cnt]
            self.return_queue.append(self.reward_agents[a_id])
            self.reward_agents[a_id] = 0
            self.n_episode += 1
            cnt += 1

    def check_goal(self, mean_return):
        if mean_return < 100:
            print("Goal NOT reached! Mean 100 test reward: " + str(np.round(mean_return, 2)))
            return False
        else:
            print("GOAL REACHED! Mean reward over 100 episodes is " + str(np.round(mean_return, 2)))
            # If we reached goal, save the model locally
            torch.save(self.policy_nn.state_dict(), 'models/3dBall' + Config.date_time + '.pt')
            return True
