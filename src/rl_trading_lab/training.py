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


@dataclass(frozen=True)
class TradeRecord:
    """Individual trade execution record for transparent audit ledger."""

    trade_id: int
    step: int
    action: str  # "BUY" or "SELL"
    price: float
    shares: int
    cost_or_proceed: float
    fee: float
    realized_pnl: float | None
    realized_pnl_pct: float | None
    cash_after: float
    portfolio_value: float
    trend_state: str  # "Downtrend", "Neutral", "Uptrend"


@dataclass(frozen=True)
class DetailedPolicyRun:
    """Comprehensive policy evaluation run with trade-by-trade analytics."""

    evaluation: EvaluationResult
    prices: list[float]
    policy_values: list[float]
    buy_and_hold_values: list[float]
    trades: list[TradeRecord]
    buy_steps: list[int]
    buy_prices: list[float]
    sell_steps: list[int]
    sell_prices: list[float]
    total_trades: int
    closed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_realized_profit: float
    gross_profits: float
    gross_losses: float
    profit_factor: float
    max_drawdown: float
    total_fees_paid: float
    excess_return: float


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


def evaluate_policy_detailed(
    agent: QLearningAgent, seed: int, regime: MarketRegime = "gbm"
) -> DetailedPolicyRun:
    """Run frozen greedy policy with full trade-by-trade tracking and execution audit."""
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

    trades: list[TradeRecord] = []
    buy_steps: list[int] = []
    buy_prices: list[float] = []
    sell_steps: list[int] = []
    sell_prices: list[float] = []

    active_position_cost = 0.0
    total_fees_paid = 0.0

    done = False
    trend_labels = {0: "Downtrend", 1: "Neutral", 2: "Uptrend"}

    while not done:
        curr_step = environment.index
        curr_price = environment.prices[curr_step]
        curr_trend = trend_labels.get(state[0], "Neutral")
        prev_shares = environment.shares

        action = agent.act(state, explore=False)
        next_state, _, done, _ = environment.step(action)

        new_shares = environment.shares
        prices.append(environment.prices[environment.index])
        policy_values.append(environment.value)
        buy_and_hold_values.append(buy_and_hold_cash + buy_and_hold_shares * environment.prices[environment.index])

        # Trade detection
        if prev_shares == 0 and new_shares > 0:
            # BUY executed
            fee = new_shares * curr_price * environment.transaction_cost
            cost = new_shares * curr_price * (1 + environment.transaction_cost)
            total_fees_paid += fee
            active_position_cost = cost

            buy_steps.append(curr_step)
            buy_prices.append(curr_price)

            trades.append(
                TradeRecord(
                    trade_id=len(trades) + 1,
                    step=curr_step,
                    action="BUY",
                    price=curr_price,
                    shares=new_shares,
                    cost_or_proceed=cost,
                    fee=fee,
                    realized_pnl=None,
                    realized_pnl_pct=None,
                    cash_after=environment.cash,
                    portfolio_value=environment.value,
                    trend_state=curr_trend,
                )
            )

        elif prev_shares > 0 and new_shares == 0:
            # SELL executed
            fee = prev_shares * curr_price * environment.transaction_cost
            proceed = prev_shares * curr_price * (1 - environment.transaction_cost)
            total_fees_paid += fee
            realized_pnl = proceed - active_position_cost
            realized_pnl_pct = (realized_pnl / active_position_cost * 100.0) if active_position_cost > 0 else 0.0

            sell_steps.append(curr_step)
            sell_prices.append(curr_price)

            trades.append(
                TradeRecord(
                    trade_id=len(trades) + 1,
                    step=curr_step,
                    action="SELL",
                    price=curr_price,
                    shares=prev_shares,
                    cost_or_proceed=proceed,
                    fee=fee,
                    realized_pnl=realized_pnl,
                    realized_pnl_pct=realized_pnl_pct,
                    cash_after=environment.cash,
                    portfolio_value=environment.value,
                    trend_state=curr_trend,
                )
            )
            active_position_cost = 0.0

        state = next_state

    result = environment.result()
    evaluation = EvaluationResult(
        seed=seed,
        policy_value=result.portfolio_value,
        buy_and_hold_value=result.buy_and_hold_value,
        policy_return=result.portfolio_value / environment.initial_cash - 1,
        buy_and_hold_return=result.buy_and_hold_value / environment.initial_cash - 1,
    )

    # Compute trade-by-trade analytics
    closed_trades = [t for t in trades if t.action == "SELL" and t.realized_pnl is not None]
    winning_trades = sum(1 for t in closed_trades if (t.realized_pnl or 0) > 0)
    losing_trades = sum(1 for t in closed_trades if (t.realized_pnl or 0) <= 0)
    win_rate = (winning_trades / len(closed_trades) * 100.0) if closed_trades else 0.0

    gross_profits = sum(t.realized_pnl for t in closed_trades if (t.realized_pnl or 0) > 0)
    gross_losses = abs(sum(t.realized_pnl for t in closed_trades if (t.realized_pnl or 0) < 0))
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (99.0 if gross_profits > 0 else 1.0)
    total_realized_profit = sum((t.realized_pnl or 0) for t in closed_trades)

    # Max Drawdown
    peak = policy_values[0]
    max_dd = 0.0
    for v in policy_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

    excess_return = evaluation.policy_return - evaluation.buy_and_hold_return

    return DetailedPolicyRun(
        evaluation=evaluation,
        prices=prices,
        policy_values=policy_values,
        buy_and_hold_values=buy_and_hold_values,
        trades=trades,
        buy_steps=buy_steps,
        buy_prices=buy_prices,
        sell_steps=sell_steps,
        sell_prices=sell_prices,
        total_trades=len(trades),
        closed_trades=len(closed_trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_realized_profit=total_realized_profit,
        gross_profits=gross_profits,
        gross_losses=gross_losses,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        total_fees_paid=total_fees_paid,
        excess_return=excess_return,
    )


def evaluate_policy(
    agent: QLearningAgent, seed: int, regime: MarketRegime = "gbm"
) -> tuple[EvaluationResult, list[float], list[float], list[float]]:
    """Run a frozen, greedy policy on a held-out path and keep its curves."""
    detailed = evaluate_policy_detailed(agent, seed, regime)
    return detailed.evaluation, detailed.prices, detailed.policy_values, detailed.buy_and_hold_values


def evaluate_held_out(
    episodes: int, training_seed: int, test_seeds: Sequence[int], regime: MarketRegime = "gbm"
) -> dict[str, object]:
    """Train once, then aggregate results across held-out evaluation seeds."""
    if not test_seeds:
        raise ValueError("test_seeds must not be empty")
    agent, _ = train(episodes=episodes, seed=training_seed, regime=regime)
    detailed_runs = [evaluate_policy_detailed(agent, seed, regime) for seed in test_seeds]
    results = [item.evaluation for item in detailed_runs]
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
        "detailed_runs": detailed_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a tabular Q-learning trading agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--regime", choices=("gbm", "trend", "mean_reverting"), default="gbm")
    args = parser.parse_args()
    agent, _ = train(args.episodes, args.seed, args.regime)
    detailed = evaluate_policy_detailed(agent, args.seed + 10_000, args.regime)
    print(f"Trained episodes:          {args.episodes}")
    print(f"Held-out policy portfolio: ${detailed.evaluation.policy_value:,.2f}")
    print(f"Held-out buy-and-hold:     ${detailed.evaluation.buy_and_hold_value:,.2f}")
    print(f"Total trades executed:     {detailed.total_trades}")
    print(f"Win rate:                  {detailed.win_rate:.1f}% ({detailed.winning_trades}W / {detailed.losing_trades}L)")
    print(f"Total realized profit:     ${detailed.total_realized_profit:+,.2f}")
    print(f"Total fees incurred:       ${detailed.total_fees_paid:,.2f}")
    print(f"Max drawdown:              {detailed.max_drawdown:.2f}%")


if __name__ == "__main__":
    main()
