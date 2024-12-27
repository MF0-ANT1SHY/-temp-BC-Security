import warnings
warnings.filterwarnings('ignore')

import random
from statistics import mean
from utils import *
from uncle_maker_env import *

ALPHA = 0.40
TARGET_TIME = 12
TIMESTAMP_UPPER_BOUND = 72


class Random_Agent:
    def __init__(self):
        self.policy_dict = {k: 0 for k in range(1, 9)}
        self.policy_dict.update({k: random.choice(range(2)) for k in range(9, TIMESTAMP_UPPER_BOUND + 1)})

    def action(self, state):
        return self.policy_dict[state]

    def test_agent_policy(self, env, pi, n_episodes=100, max_steps=10000):
        results = []
        for _ in range(n_episodes):
            state = env.reset()
            steps = 0

            while True and steps < max_steps:
                action = pi(state[0])
                state, _, _, action = env.step(state, action)
                steps += 1
            results.append(env.reward_fraction)
        return np.sum(results) / len(results)


def test_probability(env, upper_bound, game_nums):
    """
    This method is decrypted
    """

    for i in range(9, upper_bound+1):
        print(f'State: {i}')

        attack_total_reward = 0
        for _ in range(game_nums):
            s, t = env.reset()
            _, r, _, _, _ = env.step(s, 1, t)
            attack_total_reward += r
        print(f'Action: Attack; reward probability: {attack_total_reward / game_nums}')

        honest_total_reward = 0
        for _ in range(game_nums):
            s, t = env.reset()
            _, r, _, _, _ = env.step(s, 0, t)
            honest_total_reward += r
        print(f'Action: Honest; reward probability: {honest_total_reward / game_nums}')

        print()

def check_ex(num):
    answer = []
    answer_or = []
    for _ in range(1000000):
        one = np.random.exponential(num)
        two = np.random.exponential(num)

        answer.append(abs(one - two))
        answer_or.append(one)
    print(mean(answer))
    print(mean(answer_or))


if __name__ == '__main__':
    env = UMEnv(
        attacker_fraction=ALPHA,
        target_time=TARGET_TIME,
        timestamp_upper_bound=TIMESTAMP_UPPER_BOUND,
        pre_defined_time_interval=8
    )

    # agent = Random_Agent()
    # print_policy(agent.action, TIMESTAMP_UPPER_BOUND)
    # agent.test_agent_policy(env, agent.action)

    # for _ in range(1):
    #     total_steps, game_nums, avg = 100000, 10, 0
    #     for _ in range(game_nums):
    #         s, t = env.reset()
    #         d = False
    #         for _ in range(total_steps):
    #             # a = 1 if d else 0
    #             a = 0
    #             s, r, d, _, t = env.step(s, a, t)
    #         avg += env.reward_fraction / game_nums
    #     print("UM,", avg)

    # save_chain(env.chain.blocks, file_path='output_blocks.txt')
    # plot_timestamp_distribution(env.chain.blocks, HONEST)
    # plot_timestamp_distribution(env.chain.blocks, ATTACK)
    # test_probability(env, TIMESTAMP_UPPER_BOUND, 100000)
    check_ex(12)