import unittest

from rl_trading_lab.agent import QLearningAgent
from rl_trading_lab.environment import ACTIONS, BUY, HOLD, SELL


class QLearningAgentTests(unittest.TestCase):
    def test_default_q_values_zero(self):
        agent = QLearningAgent(seed=42)
        state = (1, 0)
        self.assertEqual(agent.q[state], [0.0, 0.0, 0.0])

    def test_act_greedy_deterministic(self):
        agent = QLearningAgent(seed=42)
        state = (2, 0)
        agent.q[state] = [0.1, 0.9, 0.3]  # BUY is best
        action = agent.act(state, explore=False)
        self.assertEqual(action, BUY)

    def test_bellman_update_calculation(self):
        lr = 0.2
        gamma = 0.9
        agent = QLearningAgent(learning_rate=lr, discount=gamma, seed=42)
        s1 = (1, 0)
        s2 = (2, 1)

        # Populate next state Q-values
        agent.q[s2] = [1.0, 5.0, 2.0]  # max is 5.0
        # Learn from transition with reward 10.0
        agent.learn(state=s1, action=BUY, reward=10.0, next_state=s2, done=False)

        # Expected: Q(s1, BUY) = 0 + 0.2 * (10.0 + 0.9 * 5.0 - 0) = 0.2 * 14.5 = 2.9
        self.assertAlmostEqual(agent.q[s1][BUY], 2.9, places=6)

    def test_terminal_state_ignores_future_q(self):
        lr = 0.5
        gamma = 0.9
        agent = QLearningAgent(learning_rate=lr, discount=gamma, seed=42)
        s1 = (0, 1)
        s2 = (0, 0)
        agent.q[s2] = [100.0, 200.0, 300.0]  # should be ignored since done=True

        agent.learn(state=s1, action=SELL, reward=20.0, next_state=s2, done=True)
        # Expected: Q(s1, SELL) = 0 + 0.5 * (20.0 + 0 - 0) = 10.0
        self.assertAlmostEqual(agent.q[s1][SELL], 10.0, places=6)

    def test_reproducible_agent_exploration(self):
        agent1 = QLearningAgent(epsilon=1.0, seed=123)
        actions1 = [agent1.act((1, 0), explore=True) for _ in range(20)]

        agent2 = QLearningAgent(epsilon=1.0, seed=123)
        actions2 = [agent2.act((1, 0), explore=True) for _ in range(20)]

        self.assertEqual(actions1, actions2)


if __name__ == "__main__":
    unittest.main()
