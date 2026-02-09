import unittest

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline


class DeliberationTests(unittest.TestCase):
    def test_critic_agrees_parsing(self):
        pipeline = ConclavePipeline(get_config())
        self.assertTrue(pipeline._critic_agrees("Verdict: AGREE"))
        self.assertFalse(pipeline._critic_agrees("Verdict: DISAGREE"))
        self.assertTrue(pipeline._critic_agrees("Verdict\nAGREE"))
        self.assertFalse(pipeline._critic_agrees("Verdict - Disagree"))


if __name__ == "__main__":
    unittest.main()
