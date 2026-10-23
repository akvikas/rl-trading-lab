"""A deliberately small, deterministic trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from random import Random
from typing import Literal


HOLD, BUY, SELL = range(3)
ACTIONS = (HOLD, BUY, SELL)
MarketRegime = Literal["gbm", "trend", "mean_reverting"]


@dataclass(frozen=True)
class EpisodeResult:
    portfolio_value: float
    buy_and_hold_value: float
    steps: int


class TradingEnvironment:
    """Long-only, fully-invested simulator with discrete observations.

    A reset creates a synthetic price path, keeping experiments reproducible
    when the same seed is supplied.
    """

    def __init__(self, steps: int = 252, initial_cash: float = 10_000.0,
                 transaction_cost: float = 0.001, seed: int | None = 7,
                 regime: MarketRegime = "gbm") -> None:
        if steps < 2:
            raise ValueError("steps must be at least 2")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if regime not in {"gbm", "trend", "mean_reverting"}:
            raise ValueError(f"unknown market regime: {regime}")
        self.steps = steps
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.regime = regime
        self._rng = Random(seed)
        self.prices: list[float] = []
        self.reset()

    def reset(self) -> tuple[int, int]:
        self.prices = self._make_prices()
        self.index = 0
        self.cash = self.initial_cash
        self.shares = 0
        self._previous_value = self.initial_cash
        return self.state

    def _make_prices(self) -> list[float]:
        price = 100.0
        prices = [price]
        daily_drift, daily_volatility = 0.00025, 0.012
        previous_return = 0.0
        for _ in range(self.steps - 1):
            shock = self._rng.gauss(0, 1)
            # GBM has no predictable next-day component. The other regimes
            # deliberately make the observed return informative about the next
            # one, which lets this small tabular state learn a real signal.
            autocorrelation = {"gbm": 0.0, "trend": 0.35, "mean_reverting": -0.35}[self.regime]
            log_return = (
                daily_drift - daily_volatility ** 2 / 2
                + autocorrelation * previous_return
                + daily_volatility * shock
            )
            price *= exp(log_return)
            prices.append(round(price, 4))
            previous_return = log_return
        return prices

    @property
    def value(self) -> float:
        return self.cash + self.shares * self.prices[self.index]

    @property
    def state(self) -> tuple[int, int]:
        if self.index == 0:
            trend = 1
        else:
            change = self.prices[self.index] / self.prices[self.index - 1] - 1
            trend = 0 if change < -0.002 else 2 if change > 0.002 else 1
        return trend, int(self.shares > 0)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict[str, float]]:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if self.index >= self.steps - 1:
            raise RuntimeError("episode is finished; call reset()")

        price = self.prices[self.index]
        if action == BUY and self.shares == 0:
            cost_per_share = price * (1 + self.transaction_cost)
            quantity = int(self.cash // cost_per_share)
            if quantity:
                self.cash -= quantity * cost_per_share
                self.shares = quantity
        elif action == SELL and self.shares:
            self.cash += self.shares * price * (1 - self.transaction_cost)
            self.shares = 0

        self.index += 1
        value = self.value
        reward = value - self._previous_value
        self._previous_value = value
        done = self.index == self.steps - 1
        return self.state, reward, done, {"portfolio_value": value}

    def result(self) -> EpisodeResult:
        buy_and_hold_shares = int(self.initial_cash // (self.prices[0] * (1 + self.transaction_cost)))
        buy_and_hold_cash = self.initial_cash - buy_and_hold_shares * self.prices[0] * (1 + self.transaction_cost)
        return EpisodeResult(
            portfolio_value=self.value,
            buy_and_hold_value=buy_and_hold_cash + buy_and_hold_shares * self.prices[-1],
            steps=self.index,
        )
