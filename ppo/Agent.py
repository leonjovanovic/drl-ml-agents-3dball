import itertools
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter

import Config
from AgentControl import AgentControl
from Buffer import Buffer
from TestAgent import TestAgent


class Agent:
    # Role of Agent class is to coordinate between AgentControll where we do all calculations and Memory where we
    # store all of the data. Also the Agent will occasionally test the trained model.
    def __init__(self, env, behavior_name, num_agents, state_shape, action_shape, episode_length):
        self.agent_control = AgentControl(state_shape, action_shape)
        self.buffer = Buffer(num_agents, state_shape, action_shape, episode_length)
        self.test_agent = TestAgent(env, behavior_name, num_agents, state_shape, action_shape)
        self.writer = SummaryWriter(logdir="content/runs/" + str(Config.seed) + Config.writer_name) if Config.write else None
        self.policy_loss_mean = deque(maxlen=100)
        self.critic_loss_mean = deque(maxlen=100)
        self.return_queue = deque(maxlen=100)
        self.current_ep_rewards = []
        self.max_reward = -10
        self.num_agents = num_agents
        self.reward_agents = [0] * self.num_agents

    def get_actions(self, decision_steps):
        actions, logprob = self.agent_control.get_actions(decision_steps.obs[0])
        self.buffer.add_old(decision_steps, actions, logprob)
        return actions

    def update_lr(self, n_step):
        self.agent_control.update_lr(n_step)

    def calculate_ep_reward(self, decision_steps, terminal_steps):
        for a_id in decision_steps.agent_id:
            self.reward_agents[a_id] += decision_steps.reward[a_id]
        cnt = 0
        for a_id in terminal_steps.agent_id:
            self.reward_agents[a_id] += terminal_steps.reward[cnt]
            self.current_ep_rewards.append(self.reward_agents[a_id])
            self.return_queue.append(self.reward_agents[a_id])
            self.reward_agents[a_id] = 0
            cnt += 1

    @staticmethod
    def get_steps(env, behavior_name):
        steps = list(env.get_steps(behavior_name))
        return steps[0], steps[1]

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add(decision_steps, terminal_steps)

    def buffer_full(self):
        return self.buffer.full

    def calculate_advantage(self):
        # For basic advantage function we have to calculate future rewards we got from each state, where reward from
        # last state is estimation (since we only know rewards in steps we took, not after), discount them and
        # subtract from baseline which in this case will be estimated value of each state.
        # GAE advantage gives us to decide we want each state advantage to be calculated with
        # reward + estimate(next state) - estimate(state) which has low variance but high bias or with
        # reward + gamma*next_reward + ... + gamma^n * estimate(last next state) - estimate(state) which has high
        # variance but low bias. We can decide to calculate advantage with somethig between those two and Config.LAMBDA
        # will be hyperparameter for that
        state_values = self.agent_control.get_critic_value_d(self.buffer.states)
        if Config.gae:
            new_state_values = self.agent_control.get_critic_value_d(self.buffer.new_states)
            self.buffer.gae_advantage(state_values, new_state_values)
        else:
            last_state_value = self.agent_control.get_critic_value_d(self.buffer.new_states[-1])
            self.buffer.advantage(state_values, last_state_value)

    def update(self, indices):
        # Main PPO point is updating policy NN. This is done by calculating derivative of loss function and doing
        # gradient descent. First we have to calculate ratio. Second to find minimum between ratio*advantage and
        # clipped_ratio*advantage. Third to find mean of Config.MINIBATCH_SIZE losses.

        # To calculate ratio we need new and old action probability. We already have old when we fed states to
        # policy NN when we wanted to get action from it. We can get new action probabilities if we give same states
        # but also actions we got. With states NN can create Normal distribution and with action he will sample the same
        # part of distribution, but now with different probability because Normal distribution is different.
        ratio = self.agent_control.calculate_ratio(self.buffer.states[indices], self.buffer.actions[indices],
                                                   self.buffer.logprob[indices])
        policy_loss = self.agent_control.update_policy(ratio, self.buffer.advantages[indices])
        critic_loss = self.agent_control.update_critic(self.buffer.states[indices], self.buffer.gt[indices])

        self.policy_loss_mean.append(policy_loss)
        self.critic_loss_mean.append(critic_loss)

    def record_data(self, n_step):
        if len(self.current_ep_rewards) > 0:
            self.max_reward = np.maximum(self.max_reward, np.max(self.current_ep_rewards))
        print("St " + str(n_step) + "/" + str(Config.total_steps) + " Mean 100 policy loss: " + str(
            np.round(np.mean(self.policy_loss_mean), 4)) + " Mean 100 critic loss: " + str(
            np.round(np.mean(self.critic_loss_mean), 4)) + " Max reward: " + str(
            np.round(self.max_reward, 2)) + " Mean 100 reward: " + str(
            np.round(np.mean(self.return_queue), 2)) + " Last rewards: " + str(
            np.round(self.current_ep_rewards, 2)))

        if Config.write:
            self.writer.add_scalar('pg_loss', np.mean(self.policy_loss_mean), n_step)
            self.writer.add_scalar('vl_loss', np.mean(self.critic_loss_mean), n_step)
            self.writer.add_scalar('100rew', np.mean(self.return_queue), n_step)
            if len(self.current_ep_rewards) > 0:
                self.writer.add_scalar('rew', np.mean(self.current_ep_rewards), n_step)
        self.current_ep_rewards = []

    def check_test(self, n_step):
        # Agent will test the model every 50th iteration or if its performing well enough for past few episodes.
        if (n_step + 1) % 50 == 0 or (len(self.return_queue) >= 100 and np.mean(
                list(itertools.islice(self.return_queue, 90, 100))) >= 100):
            return True
        return False

    def test(self, n_step):
        # Test the model for 100 episodes. Since we pass environment to TestAgent (and not creating separate)
        # we need to reset some variables and return wether the goal is met.
        self.reward_agents = [0] * self.num_agents
        end = self.test_agent.test(self.agent_control.get_policy_nn(), self.writer, n_step)
        self.buffer.reset(full=False)
        return end
