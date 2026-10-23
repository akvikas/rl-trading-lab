"""Out-of-sample benchmark comparison across multiple seeds.

Compares:
1. Learned Tabular Q-Learning Policy (greedy evaluation)
2. Passive Buy-and-Hold Baseline
3. Uniform Random Policy

Ensures zero data leakage by evaluating strictly on unseen out-of-sample seeds.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from random import Random

from .agent import QLearningAgent
from .environment import ACTIONS, BUY, HOLD, SELL, MarketRegime, TradingEnvironment
from .training import train


@dataclass(frozen=True)
class PolicyStats:
    policy_name: str
    returns: list[float]  # Percentage returns (e.g., +5.2 for 5.2%)
    final_values: list[float]
    trades: list[int]

    @property
    def mean_return(self) -> float:
        return statistics.mean(self.returns)

    @property
    def std_return(self) -> float:
        return statistics.stdev(self.returns) if len(self.returns) > 1 else 0.0

    @property
    def median_return(self) -> float:
        return statistics.median(self.returns)

    @property
    def min_return(self) -> float:
        return min(self.returns)

    @property
    def max_return(self) -> float:
        return max(self.returns)

    @property
    def avg_trades(self) -> float:
        return statistics.mean(self.trades)


def evaluate_policy_on_env(
    env: TradingEnvironment,
    agent: QLearningAgent | None = None,
    mode: str = "q_policy",
    seed: int = 0,
) -> tuple[float, float, int]:
    """Evaluates a policy on an environment instance.

    Returns:
        (percentage_return, final_portfolio_value, trade_count)
    """
    state = env.reset()
    done = False
    trade_count = 0
    rng = Random(seed)

    while not done:
        if mode == "q_policy":
            assert agent is not None
            action = agent.act(state, explore=False)
        elif mode == "buy_and_hold":
            # Buy immediately at step 0 if not holding, then hold
            action = BUY if env.shares == 0 else HOLD
        elif mode == "random":
            action = rng.choice(ACTIONS)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        prev_shares = env.shares
        state, _, done, info = env.step(action)
        if env.shares != prev_shares:
            trade_count += 1

    final_val = env.value
    pct_return = (final_val - env.initial_cash) / env.initial_cash * 100.0
    return pct_return, final_val, trade_count


def run_benchmark(
    train_episodes: int = 500,
    train_seed: int = 42,
    num_eval_seeds: int = 50,
    eval_seed_start: int = 1000,
    steps: int = 252,
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
    regime: MarketRegime = "gbm",
) -> dict[str, PolicyStats]:
    """Runs training on a dedicated seed, then benchmarks out-of-sample on unseen seeds."""
    print(f"[*] Training Q-learning agent ({train_episodes} episodes, seed={train_seed}, regime={regime})...")
    agent, _ = train(episodes=train_episodes, seed=train_seed, regime=regime)

    q_returns, q_values, q_trades = [], [], []
    bnh_returns, bnh_values, bnh_trades = [], [], []
    rnd_returns, rnd_values, rnd_trades = [], [], []

    print(f"[*] Benchmarking across {num_eval_seeds} out-of-sample seeds (seed {eval_seed_start} to {eval_seed_start + num_eval_seeds - 1})...")

    for i in range(num_eval_seeds):
        test_seed = eval_seed_start + i

        # 1. Learned Q-Learning policy
        env_q = TradingEnvironment(
            steps=steps, initial_cash=initial_cash, transaction_cost=transaction_cost, seed=test_seed, regime=regime
        )
        r_q, v_q, t_q = evaluate_policy_on_env(env_q, agent=agent, mode="q_policy")
        q_returns.append(r_q)
        q_values.append(v_q)
        q_trades.append(t_q)

        # 2. Buy-and-Hold baseline
        env_bnh = TradingEnvironment(
            steps=steps, initial_cash=initial_cash, transaction_cost=transaction_cost, seed=test_seed, regime=regime
        )
        r_bnh, v_bnh, t_bnh = evaluate_policy_on_env(env_bnh, mode="buy_and_hold")
        bnh_returns.append(r_bnh)
        bnh_values.append(v_bnh)
        bnh_trades.append(t_bnh)

        # 3. Uniform Random policy
        env_rnd = TradingEnvironment(
            steps=steps, initial_cash=initial_cash, transaction_cost=transaction_cost, seed=test_seed, regime=regime
        )
        r_rnd, v_rnd, t_rnd = evaluate_policy_on_env(env_rnd, mode="random", seed=test_seed)
        rnd_returns.append(r_rnd)
        rnd_values.append(v_rnd)
        rnd_trades.append(t_rnd)

    return {
        "Q-Learning (Trained)": PolicyStats("Q-Learning (Trained)", q_returns, q_values, q_trades),
        "Buy-and-Hold (Passive)": PolicyStats("Buy-and-Hold (Passive)", bnh_returns, bnh_values, bnh_trades),
        "Random Policy (Uniform)": PolicyStats("Random Policy (Uniform)", rnd_returns, rnd_values, rnd_trades),
    }


def format_markdown_report(results: dict[str, PolicyStats], num_seeds: int, regime: str = "gbm") -> str:
    q = results["Q-Learning (Trained)"]
    bnh = results["Buy-and-Hold (Passive)"]
    rnd = results["Random Policy (Uniform)"]

    wins_vs_bnh = sum(1 for q_r, b_r in zip(q.returns, bnh.returns) if q_r >= b_r)
    wins_vs_rnd = sum(1 for q_r, r_r in zip(q.returns, rnd.returns) if q_r >= r_r)
    win_rate_bnh = (wins_vs_bnh / num_seeds) * 100.0
    win_rate_rnd = (wins_vs_rnd / num_seeds) * 100.0

    lines = [
        f"### Benchmark Evaluation Report (Regime: `{regime}`)",
        "",
        f"**Evaluation Methodology:** Out-of-sample evaluation across **{num_seeds} reproducible synthetic market seeds** (seeds 1000–{1000 + num_seeds - 1}) generated via regime `{regime}` (252 trading days/steps). All strategies incur a **0.10% (10 bps) transaction fee** per trade. Zero data leakage: training was isolated strictly to training seed 42.",
        "",
        "| Strategy | Mean Return | Std Dev (Risk) | Median Return | Min / Max Return | Avg Trades |",
        "|---|---|---|---|---|---|",
        f"| **Q-Learning Policy (Trained)** | **{q.mean_return:+.2f}%** | ±{q.std_return:.2f}% | {q.median_return:+.2f}% | [{q.min_return:+.2f}%, {q.max_return:+.2f}%] | {q.avg_trades:.1f} |",
        f"| **Buy-and-Hold Baseline** | {bnh.mean_return:+.2f}% | ±{bnh.std_return:.2f}% | {bnh.median_return:+.2f}% | [{bnh.min_return:+.2f}%, {bnh.max_return:+.2f}%] | {bnh.avg_trades:.1f} |",
        f"| **Random Policy Baseline** | {rnd.mean_return:+.2f}% | ±{rnd.std_return:.2f}% | {rnd.median_return:+.2f}% | [{rnd.min_return:+.2f}%, {rnd.max_return:+.2f}%] | {rnd.avg_trades:.1f} |",
        "",
        f"- **Win Rate vs Buy-and-Hold:** {win_rate_bnh:.1f}% ({wins_vs_bnh}/{num_seeds} episodes)",
        f"- **Win Rate vs Random Policy:** {win_rate_rnd:.1f}% ({wins_vs_rnd}/{num_seeds} episodes)",
        "",
        "> **Key Engineering Takeaway & Limitations:**",
        "> - **Synthetic Data vs Real Markets:** The simulator uses synthetic price models with constant parameters. It does not account for order book liquidity, slippage, or macroeconomic regime shifts.",
        "> - **Transaction Cost Sensitivity:** Frequent churn rapidly destroys returns due to compounding commission drag, highlighting the value of transaction-aware reward formulations.",
        "> - **Educational Goal:** This benchmark verifies model learning, out-of-sample evaluation discipline, and pipeline correctness—it is not financial advice or proof of real-world profitability.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Q-Learning policy vs baselines across seeds.")
    parser.add_argument("--train-episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed for training environment")
    parser.add_argument("--eval-seeds", type=int, default=50, help="Number of test seeds to evaluate")
    parser.add_argument("--eval-seed-start", type=int, default=1000, help="Starting test seed index")
    parser.add_argument("--regime", choices=("gbm", "trend", "mean_reverting"), default="gbm", help="Market regime")
    args = parser.parse_args()

    results = run_benchmark(
        train_episodes=args.train_episodes,
        train_seed=args.train_seed,
        num_eval_seeds=args.eval_seeds,
        eval_seed_start=args.eval_seed_start,
        regime=args.regime,
    )
    report = format_markdown_report(results, args.eval_seeds, regime=args.regime)
    print("\n" + report)


if __name__ == "__main__":
    main()
