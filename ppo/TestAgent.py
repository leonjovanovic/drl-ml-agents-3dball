from collections import deque

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple

import Config
from NN import PolicyNN


class TestAgent:
    def __init__(self, env, behavior_name, num_agents, state_shape, action_shape):
        self.env = env
        self.behavior_name = behavior_name
        self.num_agents = num_agents
        self.return_queue = deque(maxlen=100)
        self.reward_agents = [0] * self.num_agents
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(state_shape, action_shape).to(self.device)

    def test(self, policy_nn, writer, n_step):
        self.reset(policy_nn)
        print("Testing...")
        decision_steps, terminal_steps = self.get_steps()
        while self.n_episodes < Config.test_episodes:
            actions, _ = self.policy_nn(torch.Tensor(decision_steps.obs[0]).to(self.device))
            self.calculate_rewards(decision_steps, terminal_steps)
            self.env.set_actions(self.behavior_name, ActionTuple(continuous=actions.detach().cpu().numpy()))
            self.env.step()
            decision_steps, terminal_steps = self.get_steps()
        print(self.return_queue)
        mean_return = np.mean(self.return_queue)
        self.env.reset()
        if writer is not None:
            writer.add_scalar('test100rew', mean_return, n_step)
        return self.check_goal(mean_return)

    def reset(self, nn):
        self.policy_nn.load_state_dict(nn.state_dict())
        self.return_queue.clear()
        self.reward_agents = [0] * self.num_agents
        self.env.reset()
        self.n_episodes = 0

    def get_steps(self):
        steps = self.env.get_steps(self.behavior_name)
        return steps[0], steps[1]

    def calculate_rewards(self, decision_steps, terminal_steps):
        for a_id in decision_steps.agent_id:
            self.reward_agents[a_id] += decision_steps.reward[a_id]
        cnt = 0
        for a_id in terminal_steps.agent_id:
            self.reward_agents[a_id] += terminal_steps.reward[cnt]
            self.return_queue.append(self.reward_agents[a_id])
            self.reward_agents[a_id] = 0
            self.n_episodes += 1
            cnt += 1

    def check_goal(self, mean_return):
        if mean_return < 100:
            print("Goal NOT reached! Mean 100 test reward: " + str(np.round(mean_return, 2)))
            return False
        else:
            print("GOAL REACHED! Mean reward over 100 episodes is " + str(np.round(mean_return, 2)))
            # If we reached goal, save the model locally
            torch.save(self.policy_nn.state_dict(), 'models/3dBall_' + str(Config.seed) + "_" + Config.date_time + '.pt')
            return True
