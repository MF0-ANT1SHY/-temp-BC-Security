# BC-Security

A research project implementing Reinforcement Learning agents for studying Selfish Mining attacks in Proof of Work (PoW) blockchains.

## Project Structure

BC-Security/

├── UnifiedEnv/ # Main environment and RL agent implementations

├── UncleMaker/ # Uncle Maker attack reproduction

├── test/ # Testing and evaluation submissions

├── CoinData/ # Data collection and processing

└── Comparison/ # (deprecated)

## Setup and Installation

1. Navigate to the UnifiedEnv directory:

```bash
cd UnifiedEnv
```

2. Install the required dependencies:

```bash
conda create --name <env> --file requirements.txt
conda activate <env>
```

3. Install additional dependencies:

```bash
pip install -r requirements_2.txt
```

## Usage

The project implements MDP environments following the gymnasium interface in `UnifiedEnv/unifiedenv.py`. Training is performed using StableBaselines3.

To start training an agent, use:

```bash
python A2C.py -rl <algorithm>
```

Available algorithms:

- `a2c`: Advantage Actor-Critic
- `ppo`: Proximal Policy Optimization
- `dqn`: Deep Q-Network

## Environment

The environment implementation follows the gymnasium standard interface:

- State space: Blockchain state representation
- Action space: Available mining actions
- Reward function: Mining rewards and penalties (following SquirRL paper)

## Development

- Environment implementations are located in `UnifiedEnv/unifiedenv.py`
- Training scripts use StableBaselines3 for RL algorithms
- Data collection utilities are available in `CoinData/`

## Contributing

pending
