import os
import numpy as np
import matplotlib.pyplot as plt
from unifiedenv import eth_env
from unifiedenv import logger
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import random

# Create experiment parameters

FUZZING = True
PRINT_STATE = False
ALPHA_RANGE = np.arange(0.1, 0.51, 0.05)  # From 0.1 to 0.5 with step 0.05
TRIALS_PER_ALPHA = 1
RECORD_INTERVAL = 50  # Record data every 50 steps

STEPS_CONFIG = [{"name": "long", "steps": 10000}]


def calculate_error_metrics(results, alpha_range):
    """Calculate error metrics for each alpha value."""
    error_metrics = {}
    for alpha in alpha_range:
        data_array = np.array([data[0] for data in results[alpha]])
        mean_values = np.mean(data_array, axis=0)
        final_mean = mean_values[-1]  # Use the last value as the final result
        relative_error = abs(final_mean - alpha) / alpha * 100
        error_metrics[alpha] = relative_error
    # format the error metrics to 2 decimal places
    error_metrics = {alpha: round(error, 2) for alpha, error in error_metrics.items()}
    for alpha in sorted(error_metrics.keys()):
        logger.info(f"Alpha: {alpha:.2f}, Relative Error: {error_metrics[alpha]:.2f}%")
    return error_metrics


def run_env(alpha, steps, seed=None):
    """Initialize and run the environment with given parameters"""
    if seed is None:
        seed = int(time.time() * 1000000) % (2**32)
    np.random.seed(seed)

    env = eth_env(
        max_hidden_block=20,
        attacker_fraction=alpha,
        follower_fraction=0.5,
        dev=0.0,
        random_interval=(0, 0.5),
        know_alpha=False,
        relative_p=0.54,
    )
    return env.reset(), env


def simulate_random_actions(params):
    """Run simulation with random action sequences"""
    alpha, steps = params
    state, env = run_env(alpha, steps)
    trial_data = []
    reward_data = []
    accumulated_reward = 0

    for step in range(steps):
        action = env.action_space.sample()
        state = env._current_state
        previous_state = {
            "len_attacker_forking": state["len_attacker_forking"],
            "len_honest_forking": state["len_honest_forking"],
            "forking_status": state["forking_status"],
            "advantage": state["advantage"],
        }
        state, reward, _, _, _ = env.step(action)
        accumulated_reward += reward
        if PRINT_STATE:
            print(
                f"Step: {step}, from: {previous_state['len_attacker_forking'], previous_state['len_honest_forking'], previous_state['forking_status'], previous_state['advantage']}, Action: {env._explain_action(action)}, state: {state['len_attacker_forking'], state['len_honest_forking'], state['forking_status'], state['advantage']},  Reward: {reward}"
            )
        if (step + 1) % RECORD_INTERVAL == 0:
            attacker_fraction = env._attacker_block / (
                env._attacker_block + env._honest_block
            )
            trial_data.append(attacker_fraction)
            reward_data.append(accumulated_reward)

    return alpha, trial_data, reward_data


def simulate_fixed_actions(params):
    """Run simulation with fixed action sequences"""
    alpha, steps = params
    state, env = run_env(alpha, steps)
    trial_data = []
    reward_data = []
    accumulated_reward = 0

    for step in range(steps):
        # for action in (2, 3, 1):
        for action in ([0.4, 0.5], [0.5, 0.6], [-0.6, -1]):
            state = env._current_state
            previous_state = {
                "len_attacker_forking": state["len_attacker_forking"],
                "len_honest_forking": state["len_honest_forking"],
                "forking_status": state["forking_status"],
                "advantage": state["advantage"],
            }
            state, reward, _, _, _ = env.step(action)
            accumulated_reward += reward
            if PRINT_STATE:
                print(
                    f"Step: {step}, from: {previous_state['len_attacker_forking'], previous_state['len_honest_forking'], previous_state['forking_status'], previous_state['advantage']}, Action: {env._explain_action(action)}, state: {state['len_attacker_forking'], state['len_honest_forking'], state['forking_status'], state['advantage']},  Reward: {reward}"
                )
        if (step + 1) % RECORD_INTERVAL == 0:
            attacker_fraction = env._attacker_block / (
                env._attacker_block + env._honest_block
            )
            trial_data.append(attacker_fraction)
            reward_data.append(accumulated_reward)

    env.reset()

    return alpha, trial_data, reward_data


def run_parallel_simulation(steps_config, fuzzing=False):
    """Run parallel simulation with either random or fixed actions"""
    steps = steps_config["steps"]
    params_list = [
        (alpha, steps) for alpha in ALPHA_RANGE for _ in range(TRIALS_PER_ALPHA)
    ]

    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Starting parallel simulation with {num_processes} processes")

    with Pool(processes=num_processes) as pool:
        simulation_func = simulate_random_actions if fuzzing else simulate_fixed_actions
        results_list = list(
            tqdm(
                pool.imap(simulation_func, params_list),
                total=len(params_list),
                desc=f"Simulating {steps} steps",
            )
        )

    results = {alpha: [] for alpha in ALPHA_RANGE}
    for alpha, trial_data, reward_data in results_list:
        results[alpha].append((trial_data, reward_data))

    return results


def plot_results(results, steps_config):
    """Plot the results of the simulation"""
    if not os.path.exists(".plot"):
        os.makedirs(".plot")

    STEPS_PER_GAME = steps_config["steps"]
    x_points = np.arange(RECORD_INTERVAL, STEPS_PER_GAME + 1, RECORD_INTERVAL)

    # Plot mining power distribution
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 14})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ALPHA_RANGE)))

    for alpha, color in zip(ALPHA_RANGE, colors):
        data_array = np.array([data[0] for data in results[alpha]])
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)

        plt.plot(x_points, mean_values, label=f"α={alpha:.2f}", color=color)
        plt.fill_between(
            x_points,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.2,
        )
        plt.axhline(y=alpha, color=color, linestyle="--", alpha=0.5)

    plt.xlabel("Simulation Steps")
    plt.ylabel("Relative Block Fraction")
    plt.title(f"Mining Power Distribution Over Time\n({STEPS_PER_GAME} steps)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(
        f'.plot/alpha_convergence_{steps_config["name"]}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot accumulated rewards
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 14})

    for alpha, color in zip(ALPHA_RANGE, colors):
        data_array = np.array([data[1] for data in results[alpha]])
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)

        plt.plot(x_points, mean_values, label=f"α={alpha:.2f}", color=color)
        plt.fill_between(
            x_points,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Simulation Steps")
    plt.ylabel("Accumulated Reward")
    plt.title(f"Accumulated Rewards Over Time\n({STEPS_PER_GAME} steps)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(
        f'.plot/rewards_{steps_config["name"]}.png', dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(f"Simulation complete for {STEPS_PER_GAME} steps")


if __name__ == "__main__":
    for config in STEPS_CONFIG:
        logger.info(f"Starting simulation for {config['steps']} steps")
        results = run_parallel_simulation(config, fuzzing=FUZZING)
        plot_results(results, config)
        error_metrics = calculate_error_metrics(results, ALPHA_RANGE)
        logger.info(f"Error metrics: {error_metrics}")
