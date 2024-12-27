import matplotlib.pyplot as plt
from collections import Counter

HONEST = "honest"
ATTACK = "attack"

def print_policy(pi, ns, n_cols=9):
    arrays = {k: v for k, v in enumerate(('H', 'A'))}
    for s in range(1, ns + 1):
        a = pi(s)
        print(f"| {str(s).zfill(2)} {arrays[a].rjust(6)}", end=" ")
        if s % n_cols == 0:
            print("|")


def save_chain(blocks, file_path=None):
    output = ""

    # Collect output
    for block in blocks:
        if not block.replace:
            output += (f"        [||||||||]\n"
                       f"        [||||||||]\n"
                       f"        [||||||||]\n"
                       f"        [||||||||]\n")
        else:

            output += (f"        [||||||||]     \\\\\\\\\\\\\\\\\\\\\\\\\\\ \n"
                       f"        [||||||||]       \\\\\\\\\\\\\\\\\\\\\\\\\\\ \n"
                       f"        [||||||||]          \\\\\\\\\\\\\\\\\\\\\\\\\\\ \n"
                       f"        [||||||||]            \\\\\\\\\\\\\\\\\\\\\\\\\\\ \n")
        output += str(block)

    # Write to file if file_path is provided
    if file_path:
        with open(file_path, 'w') as file:
            file.write(output)


def plot_timestamp_distribution(blocks, target_identity):
    filtered_blocks = [block for block in blocks if block.identity == target_identity]
    time_intervals = [block.interval for block in filtered_blocks]
    time_intervals_counts = Counter(time_intervals)

    plt.figure(figsize=(10, 6))
    plt.bar(time_intervals_counts.keys(), time_intervals_counts.values(), color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.title(f'Timestamp Distribution for Identity: {target_identity}')
    plt.savefig(f'Timestamp_Distribution_for_Identity_{target_identity}.png')


if __name__ == '__main__':
    from uncle_maker import UMEnv

    ALPHA = 0.40
    TARGET_TIME = 12
    TIMESTAMP_UPPER_BOUND = 72
    env = UMEnv(
        attacker_fraction=ALPHA,
        target_time=TARGET_TIME,
        timestamp_upper_bound=TIMESTAMP_UPPER_BOUND
    )

    s, t = env.reset()
    d = False
    for _ in range(100000):
        # a = 1 if d else 0
        a = 0
        s, r, d, _, t = env.step(s, a, t)
    print(env.reward_fraction)
    plot_timestamp_distribution(env.chain.blocks, target_identity=HONEST)
    plot_timestamp_distribution(env.chain.blocks, target_identity=ATTACK)
    save_chain(env.chain.blocks, file_path='output_blocks.txt')
