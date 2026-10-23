import math
import unittest

from rl_trading_lab.environment import (
    ACTIONS,
    BUY,
    HOLD,
    SELL,
    EpisodeResult,
    TradingEnvironment,
)


class TradingEnvironmentTests(unittest.TestCase):
    def test_init_validation(self):
        with self.assertRaises(ValueError):
            TradingEnvironment(steps=1)
        with self.assertRaises(ValueError):
            TradingEnvironment(initial_cash=0.0)
        with self.assertRaises(ValueError):
            TradingEnvironment(initial_cash=-500.0)

    def test_reset_reinitialization(self):
        env = TradingEnvironment(steps=10, initial_cash=5000.0, seed=42)
        env.step(BUY)
        self.assertEqual(env.shares, 1)
        self.assertNotEqual(env.cash, 5000.0)

        initial_state = env.reset()
        self.assertEqual(initial_state, (1, 0))
        self.assertEqual(env.shares, 0)
        self.assertEqual(env.cash, 5000.0)
        self.assertEqual(env.index, 0)
        self.assertEqual(len(env.prices), 10)

    def test_seeded_reproducibility(self):
        env1 = TradingEnvironment(steps=20, seed=123)
        prices1 = list(env1.prices)
        env2 = TradingEnvironment(steps=20, seed=123)
        prices2 = list(env2.prices)
        self.assertEqual(prices1, prices2)

        env3 = TradingEnvironment(steps=20, seed=999)
        self.assertNotEqual(prices1, env3.prices)

    def test_buy_then_sell_changes_position_and_applies_transaction_cost(self):
        fee = 0.01  # 1%
        env = TradingEnvironment(steps=5, initial_cash=1000.0, transaction_cost=fee, seed=1)
        env.reset()
        p0 = env.prices[0]

        # Step 0: BUY
        state, reward, done, info = env.step(BUY)
        self.assertEqual(env.shares, 1)
        expected_cash_after_buy = 1000.0 - (p0 * (1.0 + fee))
        self.assertAlmostEqual(env.cash, expected_cash_after_buy, places=4)

        # Step 1: SELL
        p1 = env.prices[1]
        state, reward, done, info = env.step(SELL)
        self.assertEqual(env.shares, 0)
        expected_cash_after_sell = expected_cash_after_buy + (p1 * (1.0 - fee))
        self.assertAlmostEqual(env.cash, expected_cash_after_sell, places=4)

    def test_insufficient_cash_prevents_buy(self):
        # Initial cash less than initial price of 100
        env = TradingEnvironment(steps=4, initial_cash=50.0, seed=1)
        env.reset()
        env.step(BUY)
        # Should not buy because cash ($50) < price (~$100)
        self.assertEqual(env.shares, 0)
        self.assertEqual(env.cash, 50.0)

    def test_invalid_action_raises_value_error(self):
        env = TradingEnvironment(steps=4, seed=1)
        with self.assertRaises(ValueError):
            env.step(-1)
        with self.assertRaises(ValueError):
            env.step(99)

    def test_finished_episode_cannot_advance(self):
        env = TradingEnvironment(steps=3, seed=1)
        # steps=3 -> 2 steps advance from index 0 to 2
        env.step(HOLD)
        _, _, done, _ = env.step(HOLD)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            env.step(HOLD)

    def test_reward_matches_delta_portfolio_value(self):
        env = TradingEnvironment(steps=5, seed=10)
        env.reset()
        v0 = env.value
        _, reward, _, info = env.step(BUY)
        v1 = info["portfolio_value"]
        self.assertAlmostEqual(reward, v1 - v0, places=6)

    def test_state_trend_encoding(self):
        env = TradingEnvironment(steps=10, seed=5)
        trend, has_share = env.state
        self.assertIn(trend, (0, 1, 2))
        self.assertEqual(has_share, 0)

    def test_result_metrics(self):
        env = TradingEnvironment(steps=5, initial_cash=10000.0, seed=1)
        while env.index < env.steps - 1:
            env.step(HOLD)
        res = env.result()
        self.assertIsInstance(res, EpisodeResult)
        self.assertAlmostEqual(res.portfolio_value, 10000.0, places=2)
        expected_bnh = 10000.0 / env.prices[0] * env.prices[-1]
        self.assertAlmostEqual(res.buy_and_hold_value, expected_bnh, places=2)
        self.assertEqual(res.steps, 4)


if __name__ == "__main__":
    unittest.main()
