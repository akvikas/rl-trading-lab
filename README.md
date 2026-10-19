# RL Trading Lab

An educational, dependency-free Python sandbox for experimenting with tabular Q-learning on long-only trading decisions.

> This project is for learning and experimentation, not investment advice. It uses synthetic price data and does not model every real-world trading risk.

## Overview

RL Trading Lab models a small trading environment, trains a Q-learning agent on a reproducible synthetic price series, and compares the learned policy with a buy-and-hold baseline. It was built to make the reinforcement-learning loop easy to inspect without external data sources or machine-learning dependencies.

## Features

- Generates reproducible geometric-Brownian-motion price paths
- Models a long-only, one-share portfolio with configurable transaction costs
- Uses a compact state space: price trend and current position
- Supports `hold`, `buy`, and `sell` actions
- Trains with epsilon-greedy tabular Q-learning
- Reports learned-policy and buy-and-hold portfolio values
- Includes unit tests for core environment and agent behaviour

## Tech Stack

- Python 3.10+
- Standard library only
- `unittest`

## Architecture

```text
Synthetic price path
        ↓
TradingEnvironment (state, portfolio rules, reward)
        ↓
QLearningAgent (epsilon-greedy action selection and Q-value updates)
        ↓
Training routine
        ↓
Policy result compared with buy-and-hold
```

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/akvikas/rl-trading-lab.git
cd rl-trading-lab
python -m pip install -e .
```

No environment variables or external services are required.

## Running the Project

```bash
python -m rl_trading_lab.training --episodes 1000
```

Useful options:

```bash
python -m rl_trading_lab.training --episodes 500 --seed 7
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project Structure

```text
src/rl_trading_lab/
├── agent.py          # Tabular Q-learning agent
├── environment.py    # Synthetic market and portfolio simulator
├── training.py       # Command-line training routine
└── __init__.py
tests/
└── test_environment.py
pyproject.toml
```

## Interview Highlights

- Designed a deterministic synthetic market environment so experiments can be reproduced with a seed.
- Represented observations using a discrete trend bucket and position flag, keeping the Q-table small and inspectable.
- Implemented epsilon-greedy exploration and the standard temporal-difference Q-learning update.
- Applied transaction costs to buy and sell operations and used the change in portfolio value as the reward.
- Added unit tests for position transitions, terminal episodes, and Q-value updates.

## Future Improvements

- Add configurable market parameters and training hyperparameters to the CLI.
- Add evaluation across multiple random seeds.
- Visualize price paths, equity curves, and training outcomes.
- Document the simulator’s simplifying assumptions in more detail.

## Author

[Vikas AK](https://github.com/akvikas)
