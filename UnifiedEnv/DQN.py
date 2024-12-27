from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from unifiedenv import eth_env
import argparse

env_name = "eth_env"
# read from args
parser = argparse.ArgumentParser()
parser.add_argument("-rl", type=str, default="dqn", help="RL algorithm name")
args = parser.parse_args()
RL_name = args.rl
max_hidden_block = 20
attacker_fraction = 0.4
follower_fraction = 0.5
dev = 0.0
random_interval = (0, 0.5)
know_alpha = False
relative_p = 0.54
frequency = 1
random_process = "iid"
is_random_process = False
visualize = False
mask_argument = True
discrete = True
# RL parameters
progress_bar = True
trainning_timesteps = int(10e8)


def initialize_env():
    env = eth_env(
        max_hidden_block=max_hidden_block,
        attacker_fraction=attacker_fraction,
        follower_fraction=follower_fraction,
        dev=dev,
        random_interval=random_interval,
        know_alpha=know_alpha,
        relative_p=relative_p,
        frequency=frequency,
        random_process=random_process,
        is_random_process=is_random_process,
        visualize=visualize,
        mask_argument=mask_argument,
    )
    env.reset()
    return env


class RecordFractionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RecordFractionCallback, self).__init__(verbose)
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.rollout_count += 1
        self.logger.record("rollout/round", self.rollout_count)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                print("TensorBoard logging active.")
                return
        print("WARNING: TensorBoard logging not active.")

    def _on_step(self) -> bool:
        """
        This method will be called by the EventCallback after every step.
        For child callback (of an EventCallback), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        fraction = self.training_env.envs[0].unwrapped.reward_fraction
        self.logger.record("train/fraction", fraction)
        return True


fraction_callback = RecordFractionCallback()

env = initialize_env()

if RL_name == "a2c":
    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./{RL_name}_{env_name}_tensorboard/",
    ).learn(
        total_timesteps=trainning_timesteps,
        progress_bar=progress_bar,
        callback=fraction_callback,
    )
elif RL_name == "ppo":
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./{RL_name}_{env_name}_tensorboard/",
    ).learn(
        total_timesteps=trainning_timesteps,
        progress_bar=progress_bar,
        callback=fraction_callback,
    )
elif RL_name == "dqn":
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./{RL_name}_{env_name}_tensorboard/",
    ).learn(
        total_timesteps=trainning_timesteps,
        progress_bar=progress_bar,
        callback=fraction_callback,
    )

model.save(f"{RL_name}_{env_name}")

del model

if RL_name == "a2c":
    model = A2C.load(f"{RL_name}_{env_name}")
elif RL_name == "ppo":
    model = PPO.load(f"{RL_name}_{env_name}")
elif RL_name == "dqn":
    model = DQN.load(f"{RL_name}_{env_name}")

env = initialize_env()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
