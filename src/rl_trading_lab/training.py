"""Command-line training routine."""

from __future__ import annotations

import argparse

from .agent import QLearningAgent
from .environment import TradingEnvironment


def train(episodes: int, seed: int) -> tuple[QLearningAgent, TradingEnvironment]:
    environment = TradingEnvironment(seed=seed)
    agent = QLearningAgent(seed=seed)
    for episode in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    return agent, environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning trading agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    agent, environment = train(args.episodes, args.seed)
    state = environment.reset()
    done = False
    while not done:
        state, _, done, _ = environment.step(agent.act(state, explore=False))
    result = environment.result()
    print(f"Trained episodes: {args.episodes}")
    print(f"Policy portfolio: ${result.portfolio_value:,.2f}")
    print(f"Buy-and-hold:     ${result.buy_and_hold_value:,.2f}")


if __name__ == "__main__":
    main()
