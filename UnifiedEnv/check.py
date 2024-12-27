from stable_baselines3.common.env_checker import check_env
from unifiedenv import eth_env

max_hidden_block = 20
attacker_fraction = 0.5
follower_fraction = 0.5
relative_p = 0.54
dev = 0
random_interval = (0, 0.5)
frequency = 1
know_alpha = False
random_process = "iid"
is_random_process = False
steps = 10000

env = eth_env(
    max_hidden_block=max_hidden_block,
    attacker_fraction=attacker_fraction,
    follower_fraction=follower_fraction,
    relative_p=relative_p,
    dev=dev,
    random_interval=random_interval,
    frequency=frequency,
    know_alpha=know_alpha,
    random_process=random_process,
    is_random_process=is_random_process,
)

# It will check your custom environment and output additional warnings if needed
# check_env(env)

for i in range(steps):
    action = env.action_space.sample()
    print(action)