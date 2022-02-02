episode_length = 110
total_steps = 1000
update_steps = 10
batch_size = 2048
minibatch_size = 32

gae = True
gae_lambda = 0.95

seed = 1
policy_lr = 0.0003
critic_lr = 0.0004
max_grad_norm = 0.5
adam_eps = 1e-8

gamma = 0.99

write = False
