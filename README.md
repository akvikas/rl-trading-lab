# RL Trading Lab

A small, dependency-free reinforcement-learning sandbox for experimenting with long-only trading decisions. It generates a synthetic price series, trains a tabular Q-learning agent, and compares the learned policy with buy-and-hold.

This is an educational simulator, not investment advice. It intentionally avoids live market connections and does not account for all real-world trading risks.

## Quick start

Requires Python 3.10 or later.

```bash
python -m rl_trading_lab.training --episodes 1000
```

Or, without installing the package:

```bash
PYTHONPATH=src python -m rl_trading_lab.training --episodes 1000
```

## How it works

- The environment produces a reproducible geometric-Brownian-motion price path.
- State is a three-way return bucket (`down`, `flat`, `up`) plus the current position (`cash`, `long`).
- Actions are `hold`, `buy`, and `sell`.
- Rewards are the change in portfolio value after transaction costs.
- Q-learning trains an action-value table with epsilon-greedy exploration.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Project layout

```
src/rl_trading_lab/environment.py  # Market simulator and trading rules
src/rl_trading_lab/agent.py        # Q-learning agent
src/rl_trading_lab/training.py     # CLI training entry point
tests/                             # Unit tests
```
