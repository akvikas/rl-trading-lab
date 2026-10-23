import unittest

from rl_trading_lab.benchmark import format_markdown_report, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_run_benchmark_fast(self):
        results = run_benchmark(
            train_episodes=5,
            train_seed=42,
            num_eval_seeds=3,
            eval_seed_start=2000,
            steps=20,
            regime="mean_reverting",
        )
        self.assertIn("Q-Learning (Trained)", results)
        self.assertIn("Buy-and-Hold (Passive)", results)
        self.assertIn("Random Policy (Uniform)", results)

        report = format_markdown_report(results, num_seeds=3, regime="mean_reverting")
        self.assertIn("Benchmark Evaluation Report", report)
        self.assertIn("Q-Learning Policy", report)
        self.assertIn("Buy-and-Hold", report)


if __name__ == "__main__":
    unittest.main()
