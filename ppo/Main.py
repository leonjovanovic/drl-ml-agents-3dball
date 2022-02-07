import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

import Config
from Agent import Agent

# --------------------------------------------------- Initialization ---------------------------------------------------
env = UnityEnvironment(
    file_name='PATH_TO_FOLDER/UnityEnvironment.exe', seed=1,
    side_channels=[], no_graphics=True)
env.reset()
behavior_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))
state_shape = behavior_specs[0][0].shape[0]
action_shape = behavior_specs[1].continuous_size
# Get the agents which are requesting decision. In the first iteration every agent will request it.
steps = list(env.get_steps(behavior_name))
decision_steps = steps[0]
terminal_steps = steps[1]
num_agents = len(decision_steps.reward)
# Initialize the agent, which will handle training, printing, writing and testing
agent = Agent(env, behavior_name, num_agents, state_shape, action_shape, 1001)
# ------------------------------------------------------ Training ------------------------------------------------------
for n_step in range(Config.total_steps):
    # Check if the model is to be tested. If it is, test and retreive new required decisions because test reseted env.
    if agent.check_test(n_step):
        if agent.test(n_step):
            break
        decision_steps, terminal_steps = agent.get_steps(env, behavior_name)
    # Linear decay of the learning steps from initial value to zero.
    agent.update_lr(n_step)
    # Do a number of steps and save them in the buffer.
    while not agent.buffer_full():
        # For printing and writing to TensorBoard purposes, accumulate reward each step of an episode.
        agent.calculate_ep_reward(decision_steps, terminal_steps)
        # Get requested actions from policy neural network based on the states of those agents.
        # This function will also store state and action to the buffer to be paired with future new state and reward
        actions = agent.get_actions(decision_steps)
        env.set_actions(behavior_name, ActionTuple(continuous=actions))
        env.step()
        # Get new requested decisions or information about an agent finishing its episode.
        decision_steps, terminal_steps = agent.get_steps(env, behavior_name)
        # Pair new state, reward and done with previous state and action and store them into the main buffer.
        agent.add_to_buffer(decision_steps, terminal_steps)

    # Calculate advantage for policy NN loss
    agent.calculate_advantage()

    # Instead of shuffling whole memory, we will create indices and shuffle them after each update.
    batch_indices = np.arange(Config.batch_size)
    # We will use every collected step to update NNs Config.UPDATE_STEPS times
    for _ in range(Config.update_steps):
        np.random.shuffle(batch_indices)
        # Split the memory to mini-batches and use them to update NNs
        for mb in range(0, Config.batch_size, Config.minibatch_size):
            agent.update(batch_indices[mb: mb + Config.minibatch_size])

    # Record losses and rewards and print them to console and SummaryWriter for nice Tensorboard graphs
    agent.record_data(n_step)

agent.writer.close()
env.close()

# tensorboard --logdir="PATH_TO_FOLDER\ppo\content\runs" --host=127.0.0.1
