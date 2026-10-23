"""Training and held-out evaluation routines for the trading sandbox."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Sequence

from .agent import QLearningAgent
from .environment import MarketRegime, TradingEnvironment


@dataclass(frozen=True)
class EvaluationResult:
    """One greedy-policy result on a previously unseen price path."""

    seed: int
    policy_value: float
    buy_and_hold_value: float
    policy_return: float
    buy_and_hold_return: float


def train(episodes: int, seed: int, regime: MarketRegime = "gbm") -> tuple[QLearningAgent, TradingEnvironment]:
    """Train on reproducible paths generated from one training-seed stream."""
    environment = TradingEnvironment(seed=seed, regime=regime)
    agent = QLearningAgent(seed=seed)
    for _ in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    return agent, environment


def evaluate_policy(
    agent: QLearningAgent, seed: int, regime: MarketRegime = "gbm"
) -> tuple[EvaluationResult, list[float], list[float], list[float]]:
    """Run a frozen, greedy policy on a held-out path and keep its curves."""
    environment = TradingEnvironment(seed=seed, regime=regime)
    state = environment.reset()
    prices = [environment.prices[0]]
    policy_values = [environment.value]
    buy_and_hold_shares = int(
        environment.initial_cash // (environment.prices[0] * (1 + environment.transaction_cost))
    )
    buy_and_hold_cash = (
        environment.initial_cash
        - buy_and_hold_shares * environment.prices[0] * (1 + environment.transaction_cost)
    )
    buy_and_hold_values = [environment.initial_cash]
    done = False
    while not done:
        state, _, done, _ = environment.step(agent.act(state, explore=False))
        prices.append(environment.prices[environment.index])
        policy_values.append(environment.value)
        buy_and_hold_values.append(buy_and_hold_cash + buy_and_hold_shares * environment.prices[environment.index])
    result = environment.result()
    evaluation = EvaluationResult(
        seed=seed,
        policy_value=result.portfolio_value,
        buy_and_hold_value=result.buy_and_hold_value,
        policy_return=result.portfolio_value / environment.initial_cash - 1,
        buy_and_hold_return=result.buy_and_hold_value / environment.initial_cash - 1,
    )
    return evaluation, prices, policy_values, buy_and_hold_values


def evaluate_held_out(
    episodes: int, training_seed: int, test_seeds: Sequence[int], regime: MarketRegime = "gbm"
) -> dict[str, object]:
    """Train once, then aggregate results across held-out evaluation seeds."""
    if not test_seeds:
        raise ValueError("test_seeds must not be empty")
    agent, _ = train(episodes=episodes, seed=training_seed, regime=regime)
    results = [evaluate_policy(agent, seed, regime)[0] for seed in test_seeds]
    excess_returns = [item.policy_return - item.buy_and_hold_return for item in results]
    return {
        "training_seed": training_seed,
        "test_seeds": list(test_seeds),
        "regime": regime,
        "episodes": episodes,
        "policy_return_mean": mean(item.policy_return for item in results),
        "policy_return_std": pstdev(item.policy_return for item in results),
        "buy_and_hold_return_mean": mean(item.buy_and_hold_return for item in results),
        "buy_and_hold_return_std": pstdev(item.buy_and_hold_return for item in results),
        "excess_return_mean": mean(excess_returns),
        "excess_return_std": pstdev(excess_returns),
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a tabular Q-learning trading agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--regime", choices=("gbm", "trend", "mean_reverting"), default="gbm")
    args = parser.parse_args()
    agent, _ = train(args.episodes, args.seed, args.regime)
    result, _, _, _ = evaluate_policy(agent, args.seed + 10_000, args.regime)
    print(f"Trained episodes: {args.episodes}")
    print(f"Held-out policy portfolio: ${result.policy_value:,.2f}")
    print(f"Held-out buy-and-hold:     ${result.buy_and_hold_value:,.2f}")


if __name__ == "__main__":
    main()
