import unittest

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline
from conclave.stages.deliberation import _bounded_timeout_seconds, _effective_min_time_left_seconds


class DeliberationTests(unittest.TestCase):
    def test_critic_agrees_parsing(self):
        pipeline = ConclavePipeline(get_config())
        self.assertTrue(pipeline._critic_agrees("Verdict: AGREE"))
        self.assertFalse(pipeline._critic_agrees("Verdict: DISAGREE"))
        self.assertTrue(pipeline._critic_agrees("Verdict\nAGREE"))
        self.assertFalse(pipeline._critic_agrees("Verdict - Disagree"))

    def test_bounded_timeout_respects_remaining_budget(self):
        self.assertEqual(_bounded_timeout_seconds(300, 11.8), 11)
        self.assertEqual(_bounded_timeout_seconds(300, 0.4), 1)
        self.assertEqual(_bounded_timeout_seconds(20, None), 20)

    def test_effective_min_time_left_scales_down_for_short_runs(self):
        self.assertEqual(_effective_min_time_left_seconds(150, 180), 45.0)
        self.assertEqual(_effective_min_time_left_seconds(150, 900), 150.0)
        self.assertEqual(_effective_min_time_left_seconds(0, 180), 0.0)


if __name__ == "__main__":
    unittest.main()
