import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import argparse

from environment import SM_env_with_stale

class PrioritizedReplayBuffer:
    def __init__(self, 
                 max_samples=10000,
                 alpha=0.6,
                 beta0=0.1,
                 beta_rate=0.999,
                 eps=1e-6):
        self.max_samples = max_samples
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray)
        self.n_entries = 0
        self.next_index = 0
        self.alpha = alpha
        self.beta = beta0
        self.beta0 = beta0
        self.beta_rate = beta_rate
        self.eps = eps

    def update(self, index, priorities):
        self.memory[index, 1] = priorities

    def store(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[:self.n_entries, 1].max()
        self.memory[self.next_index, 1] = priority
        self.memory[self.next_index, 0] = np.array(sample, dtype=object)
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index = (self.next_index + 1) % self.max_samples

    def sample(self, batch_size):
        self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        entries = self.memory[:self.n_entries]

        priorities = entries[:, 1] + self.eps
        scaled_priorities = priorities ** self.alpha
        probs = np.array(scaled_priorities / np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs) ** -self.beta
        normalized_weights = weights / weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])

        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, 0]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, samples_stacks


class QNet_Duel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.layer_norm = nn.LayerNorm(hidden_size).to(device)
        self.linear2 = nn.Linear(hidden_size, output_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        hidden = F.relu(self.layer_norm(self.linear1(x)))
        advantage = self.linear2(hidden)
        value = self.linear3(hidden)
        value = value.expand_as(advantage)
        qvalue = value + advantage - advantage.mean(-1, keepdim=True).expand_as(advantage)
        return qvalue


class QTrainer:
    def __init__(self, lr, gamma, input_dim, hidden_dim, output_dim, device):
        self.gamma = gamma
        self.device = device
        self.model = QNet_Duel(input_dim, hidden_dim, output_dim, device).to(self.device)
        self.target_model = QNet_Duel(input_dim, hidden_dim, output_dim, device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss().to(self.device)
        self.copy_model()

    def copy_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, experiences):
        state, action, reward, next_state, done = experiences

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.long).to(self.device)

        Q_value = self.model(state).gather(-1, action)

        Q_value_next_index = self.model(next_state).detach().max(-1)[1].to(self.device)
        Q_value_next_index = torch.unsqueeze(Q_value_next_index, -1)
        Q_value_next_target = self.target_model(next_state).detach()
        Q_value_next = Q_value_next_target.gather(-1, Q_value_next_index)

        target = (reward + self.gamma * Q_value_next * (1 - done)).to(self.device)
        td_error = Q_value - target

        loss = self.criterion(Q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        priorities = np.abs(td_error.detach().cpu().numpy())
        return priorities


class Agent:
    def __init__(self, state_space, action_space, hidden_dim, max_explore, gamma, max_memory, lr, device):
        self.max_explore = max_explore
        self.PRB = PrioritizedReplayBuffer(max_samples=max_memory)
        self.nS = state_space
        self.nA = action_space
        self.step = 0
        self.n_game = 0
        self.device = device
        self.trainer = QTrainer(lr, gamma, self.nS, hidden_dim, self.nA, self.device)

    def remember(self, state, action, reward, next_state, done):
        self.PRB.store((state, action, reward, next_state, done))

    def train_long_memory(self, batch_size):
        if self.PRB.n_entries > batch_size:
            ids, _, experiences = self.PRB.sample(batch_size)
            priorities = self.trainer.train_step(experiences)
            self.PRB.update(ids, priorities)

    def get_action(self, state, n_game):
        # self.trainer.model.eval()
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.trainer.model(state).detach().cpu().numpy().squeeze()

        epsilon = self.max_explore - n_game
        if random.randint(0, self.max_explore) < epsilon:
            final_move = np.random.randint(len(prediction))
        else:
            final_move = prediction.argmax()
        return final_move


class Train:
    
    def __init__(self, env_hyperparams, agent_hyperparams, train_hyperparams):
        self.create_env(env_hyperparams)
        self.create_agent(agent_hyperparams)
        self.max_game = train_hyperparams["max_game"]
        self.max_step = train_hyperparams["max_step"]
        self.batch_size = train_hyperparams["batch_size"]

    def create_env(self, hyperparams):
        # self.env = eth_env(max_hidden_block  =   hyperparams["max_hidden_block"],
        #               attacker_fraction      =   hyperparams["attacker_fraction"],
        #               follower_fraction      =   hyperparams["follower_fraction"],
        #               dev                    =   hyperparams["dev"],
        #               random_interval        =   hyperparams["random_interval"],
        #               know_alpha             =   hyperparams["know_alpha"],
        #               relative_p             =   hyperparams["relative_p"]
        # )

        # change the environment to btc, keep the same to the squirRL
        self.env = env = SM_env_with_stale(max_hidden_block = hyperparams["max_hidden_block"], attacker_fraction = hyperparams["attacker_fraction"], follower_fraction = hyperparams["follower_fraction"], rule = "longest", stale_rate=0.0, dev = 0.0, know_alpha = hyperparams["know_alpha"], random_interval=hyperparams["random_interval"], random_process = "iid", frequency=6)

    def create_agent(self, hyperparams):
        self.agent = Agent(state_space  =   hyperparams["state_space"],
                      action_space      =   hyperparams["action_space"],
                      hidden_dim        =   hyperparams["hidden_dim"],
                      max_explore       =   hyperparams["max_explore"],
                      gamma             =   hyperparams["gamma"],
                      max_memory        =   hyperparams["max_memory"],
                      lr                =   hyperparams["lr"],
                      device            =   hyperparams["device"]
        )

    def start_training(self):
        env, agent = self.env, self.agent
        results = []
        state_new = env.reset()
        total_step = 0
        while agent.n_game <= self.max_game:
            state_old = state_new
            action = agent.get_action(state_old, agent.n_game)
            state_new, reward, done, _ = env.step(state_new, action)
            agent.remember(state_old, action, reward, state_new, done)
            agent.train_long_memory(batch_size=self.batch_size)
            agent.step += 1
            total_step += 1

            if total_step % 10 == 0:
                agent.trainer.copy_model()

            if done or agent.step > self.max_step:
                results.append(env.reward_fraction)
                state_new = env.reset()
                agent.step = 0
                agent.n_game += 1

                print("Running episode {}, Average reward reaches {:.2f}%".format(agent.n_game, np.mean(results[-100:]) * 100))
        # print("Training Finished")
        print(f"The final evaluated reward: {results[-1] * 100:.2f}%")

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

    # Add argument parser
    parser = argparse.ArgumentParser(description='RL training with customizable environment parameters')
    parser.add_argument('--max_hidden_block', type=int, default=20, help='Maximum length of main chain or private chain')
    parser.add_argument('--attacker_fraction', type=float, default=0.4, help='Attacker fraction (alpha)')
    parser.add_argument('--follower_fraction', type=float, default=0.5, help='Follower fraction (gamma)')
    parser.add_argument('--dev', type=float, default=0, help='Development parameter')
    parser.add_argument('--random_interval_min', type=float, default=0, help='Minimum value for random interval')
    parser.add_argument('--random_interval_max', type=float, default=0.5, help='Maximum value for random interval')
    parser.add_argument('--know_alpha', type=bool, default=True, help='Knowledge of alpha parameter')
    parser.add_argument('--relative_p', type=float, default=0.54, help='Relative probability parameter')
    
    args = parser.parse_args()

    # Agent Hyperparameters
    agent_hyperparams = {
        "load_module": False,                   # Whether to load an existing model
        "RL_load_path": f'./',                  # Specify the model loading path
        "save_path": f'./',                     # Specify the model save path
        "save_interval": 500,                   # Specify the model saving interval
        # agent module
        "state_space": 5,                      # Should be equal to the length of state vector
        "action_space": 4,                      # Equal to the size of the agent's action space
        "hidden_dim": 32,                       # Agent neural network hidden layer size
        "max_explore": 1000,                    # Epsilon alternatives
        "gamma": 0.9,                           # Discount factor, how much does agent value future reward
        "lr": 0.0005,                           # Learning Rate
        # hardware
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # buffer
        "max_memory": 50000,                    # Experience buffer size
        "alpha0": 0.6,
        "beta0": 0.1,
        "beta_rate": 0.999,
        "eps": 1e-6
    }

    # Environment Hyperparameters
    env_hyperparams = {
        "max_hidden_block": args.max_hidden_block,
        "attacker_fraction": args.attacker_fraction,
        "follower_fraction": args.follower_fraction,
        "dev": args.dev,
        "random_interval": (args.random_interval_min, args.random_interval_max),
        "know_alpha": args.know_alpha,
        "relative_p": args.relative_p,
    }

    # Training Hyperparameters
    train_hyperparams = {
        "max_game": 100,                       # Total training times
        "max_step": 10000,                        # Single training step length
        "batch_size": 256,                      # The size of the experience set taken from the buffer
    }

    train_instance = Train(env_hyperparams, agent_hyperparams, train_hyperparams)
    train_instance.start_training()
