"""Tabular Q-learning implementation."""

from __future__ import annotations

from collections import defaultdict
from random import Random

from .environment import ACTIONS


class QLearningAgent:
    def __init__(self, learning_rate: float = 0.15, discount: float = 0.95,
                 epsilon: float = 0.2, seed: int | None = 7) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self._rng = Random(seed)
        self.q: defaultdict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))

    def act(self, state: tuple[int, int], explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        values = self.q[state]
        return max(ACTIONS, key=lambda action: values[action])

    def learn(self, state: tuple[int, int], action: int, reward: float,
              next_state: tuple[int, int], done: bool) -> None:
        future = 0.0 if done else max(self.q[next_state])
        estimate = self.q[state][action]
        target = reward + self.discount * future
        self.q[state][action] += self.learning_rate * (target - estimate)
