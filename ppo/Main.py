import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

import Config
from Agent import Agent

env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()
behavior_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))
state_shape = behavior_specs[0][0].shape[0]
action_shape = behavior_specs[1].continuous_size

steps = list(env.get_steps(behavior_name))
decision_steps = steps[0]
terminal_steps = steps[1]
num_agents = len(decision_steps.reward)

agent = Agent(num_agents, state_shape, action_shape)

for n_step in range(Config.total_steps):

    #if agent.check_test(n_step):
    #    agent.test(n_step)
    #    decision_steps, _ = agent.get_steps(env, behavior_name)

    agent.update_lr(n_step)

    while not agent.buffer_full():
        agent.calculate_ep_reward(decision_steps, terminal_steps)

        actions = agent.get_actions(decision_steps)
        env.set_actions(behavior_name, ActionTuple(continuous=actions))
        env.step()

        decision_steps, terminal_steps = agent.get_steps(env, behavior_name)

        agent.add_to_buffer(decision_steps, terminal_steps)

    agent.calculate_advantage()

    batch_indices = np.arange(Config.batch_size)
    for _ in range(Config.update_steps):
        np.random.shuffle(batch_indices)
        for mb in range(0, Config.batch_size, Config.minibatch_size):
            agent.update(batch_indices[mb: mb + Config.minibatch_size])

    agent.record_data(n_step)

env.close()
agent.writer.close()





