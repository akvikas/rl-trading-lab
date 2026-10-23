import math
import unittest

from rl_trading_lab.environment import TradingEnvironment
from rl_trading_lab.training import evaluate_policy_detailed, train


class EndToEndSmokeTests(unittest.TestCase):
    def test_training_and_evaluation_smoke(self):
        """End-to-end smoke test: trains for 5 episodes, verifies numerical sanity."""
        train_seed = 42
        agent, env = train(episodes=5, seed=train_seed)

        # Verify Q-table contains trained values
        self.assertGreater(len(agent.q), 0)
        for state, actions in agent.q.items():
            for action_q in actions:
                self.assertTrue(math.isfinite(action_q))

        # Evaluate on an unseen test seed (out-of-sample)
        test_env = TradingEnvironment(steps=50, initial_cash=10_000.0, seed=100)
        state = test_env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.act(state, explore=False)
            self.assertIn(action, (0, 1, 2))
            next_state, reward, done, info = test_env.step(action)

            # Assert reward is finite
            self.assertTrue(math.isfinite(reward), f"Reward is not finite: {reward}")

            # Assert portfolio value is strictly positive and finite
            portfolio_val = info["portfolio_value"]
            self.assertTrue(math.isfinite(portfolio_val))
            self.assertGreater(portfolio_val, 0.0)

            state = next_state
            step_count += 1

        self.assertEqual(step_count, 49)
        result = test_env.result()
        self.assertTrue(math.isfinite(result.portfolio_value))
        self.assertTrue(math.isfinite(result.buy_and_hold_value))
        self.assertGreater(result.portfolio_value, 0.0)
        self.assertGreater(result.buy_and_hold_value, 0.0)

    def test_detailed_policy_run_ledger(self):
        """Verifies that detailed evaluation extracts valid trade ledger and profit metrics."""
        agent, _ = train(episodes=15, seed=7, regime="mean_reverting")
        detailed = evaluate_policy_detailed(agent, seed=1007, regime="mean_reverting")

        self.assertGreater(len(detailed.prices), 0)
        self.assertEqual(len(detailed.prices), len(detailed.policy_values))
        self.assertIsInstance(detailed.total_trades, int)
        self.assertGreaterEqual(detailed.win_rate, 0.0)
        self.assertLessEqual(detailed.win_rate, 100.0)
        self.assertTrue(math.isfinite(detailed.max_drawdown))
        self.assertTrue(math.isfinite(detailed.total_fees_paid))
        self.assertGreaterEqual(detailed.total_fees_paid, 0.0)

        # Verify trade records have valid types and values
        for trade in detailed.trades:
            self.assertIn(trade.action, ("BUY", "SELL"))
            self.assertGreater(trade.price, 0.0)
            self.assertGreater(trade.shares, 0)
            self.assertTrue(math.isfinite(trade.fee))
            if trade.realized_pnl is not None:
                self.assertTrue(math.isfinite(trade.realized_pnl))


if __name__ == "__main__":
    unittest.main()
