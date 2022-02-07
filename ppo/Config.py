import datetime

test_episodes = 100
total_steps = 350
update_steps = 5
batch_size = 2048
minibatch_size = 32

gae = True
gae_lambda = 0.85

seed = 0
policy_lr = 0.0003
critic_lr = 0.0004
max_grad_norm = 0.5
adam_eps = 1e-5

gamma = 0.99

write = True

now = datetime.datetime.now()
date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

writer_name = 'PPO_3dBall' + '_' + str(seed) + "_" + str(total_steps) + "_" + str(batch_size) + "_" + \
              str(minibatch_size) + "_" + str(update_steps) + "_" + "gae" + "_" + str(gamma) + "_" + \
              str(policy_lr)[-2:] + "_" + str(critic_lr)[-2:] + "_" + \
              str(adam_eps)[-2:] + "_" + date_time
