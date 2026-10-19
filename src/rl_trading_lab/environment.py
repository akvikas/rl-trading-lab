"""A deliberately small, deterministic trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from random import Random


HOLD, BUY, SELL = range(3)
ACTIONS = (HOLD, BUY, SELL)


@dataclass(frozen=True)
class EpisodeResult:
    portfolio_value: float
    buy_and_hold_value: float
    steps: int


class TradingEnvironment:
    """Long-only one-share simulator with discrete observations.

    A reset creates a synthetic price path, keeping experiments reproducible
    when the same seed is supplied.
    """

    def __init__(self, steps: int = 252, initial_cash: float = 10_000.0,
                 transaction_cost: float = 0.001, seed: int | None = 7) -> None:
        if steps < 2:
            raise ValueError("steps must be at least 2")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.steps = steps
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
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
        for _ in range(self.steps - 1):
            shock = self._rng.gauss(0, 1)
            price *= exp((daily_drift - daily_volatility ** 2 / 2) + daily_volatility * shock)
            prices.append(round(price, 4))
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
            total = price * (1 + self.transaction_cost)
            if self.cash >= total:
                self.cash -= total
                self.shares = 1
        elif action == SELL and self.shares:
            self.cash += price * (1 - self.transaction_cost)
            self.shares = 0

        self.index += 1
        value = self.value
        reward = value - self._previous_value
        self._previous_value = value
        done = self.index == self.steps - 1
        return self.state, reward, done, {"portfolio_value": value}

    def result(self) -> EpisodeResult:
        return EpisodeResult(
            portfolio_value=self.value,
            buy_and_hold_value=self.initial_cash / self.prices[0] * self.prices[-1],
            steps=self.index,
        )
