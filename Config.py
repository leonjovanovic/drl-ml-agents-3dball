import datetime

total_steps = 100000
# koliko dugo idemo random steps, treba nekih 10k msm
start_steps = 3000
test_every = 5000


buffer_size = 20000
#1000
min_buffer_size = 1000
batch_size = 64

gamma = 0.99

policy_lr = 0.0003
critic_lr = 0.0004
adam_eps = 1e-8

polyak = 0.995

now = datetime.datetime.now()
date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
