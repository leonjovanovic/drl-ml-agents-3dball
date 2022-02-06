import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

import Config
from Agent import Agent

for i in range(10):
    Config.seed = i
    print("RESETING " + str(Config.seed))
    env = UnityEnvironment(
        file_name='D:/Users/Leon Jovanovic/Documents/Computer Science/Unity Projects/ml-agents/Project/Build/UnityEnvironment.exe', seed=1,
        side_channels=[], no_graphics=True)
    env.reset()
    behavior_name = next(iter(env.behavior_specs.keys()))
    behavior_specs = next(iter(env.behavior_specs.values()))
    state_shape = behavior_specs[0][0].shape[0]
    action_shape = behavior_specs[1].continuous_size

    steps = list(env.get_steps(behavior_name))
    decision_steps = steps[0]
    terminal_steps = steps[1]
    num_agents = len(decision_steps.reward)

    agent = Agent(env, behavior_name, num_agents, state_shape, action_shape, 1001)

    for n_step in range(Config.total_steps):

        if agent.check_test(n_step):
            if agent.test(n_step):
                break
            decision_steps, terminal_steps = agent.get_steps(env, behavior_name)

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

    agent.writer.close()
    env.close()

# tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\drl-ml-agents-3dball\ppo\content\runs" --host=127.0.0.1
