import unittest

from rl_trading_lab.agent import QLearningAgent
from rl_trading_lab.environment import BUY, HOLD, SELL, TradingEnvironment


class TradingEnvironmentTests(unittest.TestCase):
    def test_buy_then_sell_changes_position(self):
        environment = TradingEnvironment(steps=4, seed=1)
        environment.reset()
        environment.step(BUY)
        self.assertEqual(environment.shares, 1)
        environment.step(SELL)
        self.assertEqual(environment.shares, 0)

    def test_finished_episode_cannot_advance(self):
        environment = TradingEnvironment(steps=2, seed=1)
        environment.step(HOLD)
        with self.assertRaises(RuntimeError):
            environment.step(HOLD)

    def test_agent_updates_q_value(self):
        agent = QLearningAgent(seed=1)
        agent.learn((1, 0), BUY, 10.0, (2, 1), False)
        self.assertGreater(agent.q[(1, 0)][BUY], 0)


if __name__ == "__main__":
    unittest.main()
