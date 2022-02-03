import datetime

total_steps = 100000
# koliko dugo idemo random steps, treba nekih 10k msm
start_steps = 3000
test_every = 5000

buffer_size = 20000
min_buffer_size = 1000
batch_size = 64

policy_lr = 0.0003
critic_lr = 0.0004
adam_eps = 1e-8

gamma = 0.99
polyak = 0.995

now = datetime.datetime.now()
date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

write = True
writer_name = '3dBall' + '_' + str(total_steps) + "_" + str(batch_size) + "_" + \
              str(start_steps) + "_" + str(gamma) + "_" + \
              str(policy_lr)[-2:] + "_" + str(critic_lr)[-2:] + "_" + \
              str(adam_eps)[-2:] + '_' + date_time
