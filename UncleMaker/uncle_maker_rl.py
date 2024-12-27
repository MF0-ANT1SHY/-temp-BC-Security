import numpy as np
from uncle_maker_env import *
from utils import *

ALPHA = 0.40
TARGET_TIME = 12
TIMESTAMP_UPPER_BOUND = 72


def test_game(pi, max_steps=10000):
    results = []
    for _ in range(10):
        state, time = env.reset()
        steps = 0
        c = False

        while steps < max_steps:
            action = pi(state) if c else 0
            state, _, c, action, time = env.step(state, action, time)
            steps += 1
        results.append(env.reward_fraction)
    return np.sum(results) / len(results)


def select_action(state, Q, mode="both"):
    if mode == "explore":
        return np.random.randint(len(Q[state]))
    if mode == "exploit":
        return np.argmax(Q[state])
    if mode == "both":
        if np.random.random() > 0.5:
            return np.argmax(Q[state])
        else:
            return np.random.randint(len(Q[state]))


def play_game(env, Q, max_steps=1000):
    episode = []
    state, time = env.reset()
    can_attacked = False

    step = 0
    while True:
        action = select_action(state, Q, mode='both') if can_attacked else 0
        next_state, reward, can_attacked, action, time = env.step(state, action, time)
        experience = (state, action, reward)
        episode.append(experience)

        state, step = next_state, step + 1
        if step >= max_steps:
            break

    return np.array(episode, dtype=object)


def monte_carlo(env, episodes=100, test_policy_freq=100):
    nS, nA = TIMESTAMP_UPPER_BOUND + 1, 2
    Q = np.zeros((nS, nA), dtype=np.float64)
    pi = lambda s: np.argmax(Q[s])
    returns = {}

    print("Initial Policy:")
    print_policy(pi, TIMESTAMP_UPPER_BOUND)

    for i in range(episodes):
        episode = play_game(env, Q)
        visited = np.zeros((nS, nA), dtype=bool)

        for t, (state, action, _) in enumerate(episode):
            state_action = (state, action)
            if not visited[state][action]:
                visited[state][action] = True
                discount = np.array([0.9 ** i for i in range(len(episode[t:]))])
                reward = episode[t:, -1]
                G = np.sum(discount * reward)
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                # Q[state][action] = sum(returns[state_action]) / len(returns[state_action])
                Q[state][action] = Q[state][action] + 1 / len(returns[state_action]) * (G - Q[state][action])
        pi = lambda s: np.argmax(Q[s])

        if i % test_policy_freq == 0:
            print(f"Test episode {i} Reaches goal: ", test_game(pi))

    return pi, Q


if __name__ == '__main__':
    env = UMEnv(
        attacker_fraction=ALPHA,
        target_time=TARGET_TIME,
        timestamp_upper_bound=TIMESTAMP_UPPER_BOUND
    )

    policy_mc, Q = monte_carlo(env, episodes=1000)
    print("Final Policy:")
    print_policy(policy_mc, TIMESTAMP_UPPER_BOUND)

    results = []
    for _ in range(10):
        state, time = env.reset()
        steps = 0
        c = False

        while steps < 100000:
            action = policy_mc(state) if c else 0
            # action = 0
            state, _, c, action, time = env.step(state, action, time)
            steps += 1
        results.append(env.reward_fraction)
    print("UM agent reward: ", (np.sum(results) / len(results)))
    save_chain(env.chain.blocks, "block_chains.txt")
    plot_timestamp_distribution(env.chain.blocks, HONEST)
    plot_timestamp_distribution(env.chain.blocks, ATTACK)
