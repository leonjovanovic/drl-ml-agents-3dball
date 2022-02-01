import itertools

import Config
from Buffer import Buffer
from AgentControl import AgentControl
from mlagents_envs.base_env import ActionTuple
from collections import deque
import numpy as np

class Agent:
    def __init__(self, state_shape, action_shape, num_steps):
        self.agent_control = AgentControl(state_shape, action_shape)
        self.buffer = Buffer(state_shape, action_shape, num_steps)
        self.num_steps = num_steps
        self.policy_loss_mean = deque(maxlen=100)
        self.critic_loss_mean = deque(maxlen=100)
        self.return_queue = deque(maxlen=100)
        self.reward_agents = [0] * num_steps
        self.current_ep_rewards = []
        self.max_reward = -10
        self.can_test_again = True

    def get_actions(self, decision_steps, n_step):
        actions = self.agent_control.get_actions(decision_steps.obs[0], n_step).detach()
        self.buffer.add_old(decision_steps, actions)
        self.agent_control.lr_decay(n_step + 1)
        return ActionTuple(continuous=actions.cpu().numpy())

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add(decision_steps, terminal_steps)

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
        # Add losses to deques
        self.policy_loss_mean.append(policy_loss)
        self.critic_loss_mean.append(critic_loss)

    def get_steps(self, env, behavior_name):
        steps = list(env.get_steps(behavior_name))
        if self.can_test_again is False and len(steps[1].agent_id) > 0:
            self.can_test_again = True
        return steps[0], steps[1]

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

    def record_data(self, n_step):
        if n_step % 100 != 0: # or self.buffer.buffer_index < Config.min_buffer_size:
            return
        if len(self.current_ep_rewards) > 0:
            self.max_reward = np.maximum(self.max_reward, np.max(self.current_ep_rewards))
        print("St " + str(n_step) + "/" + str(Config.total_steps) + " Mean 100 policy loss: " + str(
            np.round(np.mean(self.policy_loss_mean), 4)) + " Mean 100 critic loss: " + str(
            np.round(np.mean(self.critic_loss_mean), 4)) + " Max reward: " + str(
            np.round(self.max_reward, 2)) + " Mean 100 reward: " + str(
            np.round(np.mean(self.return_queue), 2)) + " Last rewards: " + str(
            np.round(self.current_ep_rewards, 2)))
        self.current_ep_rewards = []

    def check_goal(self, n_step):
        if (n_step + 1) % Config.test_every == 0 or (
                len(self.return_queue) >= 100 and
                np.mean(list(itertools.islice(self.return_queue, 75, 100))) >= 100 and
                self.can_test_again):
            return True
        return False

    def test(self, test_agent):
        self.reward_agents = [0] * self.num_steps
        self.can_test_again = False
        return test_agent.test(self.agent_control.moving_policy_nn)



