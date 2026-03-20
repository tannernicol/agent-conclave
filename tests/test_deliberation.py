import unittest
from pathlib import Path
import tempfile

from conclave.config import Config, get_config
from conclave.pipeline import ConclavePipeline
from conclave.stages.deliberation import (
    _bounded_timeout_seconds,
    _effective_min_time_left_seconds,
    _filter_open_disagreements,
    deliberate,
)


class DeliberationTests(unittest.TestCase):
    def test_critic_agrees_parsing(self):
        pipeline = ConclavePipeline(get_config())
        self.assertTrue(pipeline._critic_agrees("Verdict: AGREE"))
        self.assertFalse(pipeline._critic_agrees("Verdict: DISAGREE"))
        self.assertTrue(pipeline._critic_agrees("Verdict\nAGREE"))
        self.assertFalse(pipeline._critic_agrees("Verdict - Disagree"))
        self.assertTrue(
            pipeline._critic_agrees(
                "All prior disagreements are resolved.\n\nDisagreements: None remaining."
            )
        )

    def test_bounded_timeout_respects_remaining_budget(self):
        self.assertEqual(_bounded_timeout_seconds(300, 11.8), 11)
        self.assertEqual(_bounded_timeout_seconds(300, 0.4), 1)
        self.assertEqual(_bounded_timeout_seconds(20, None), 20)

    def test_effective_min_time_left_scales_down_for_short_runs(self):
        self.assertEqual(_effective_min_time_left_seconds(150, 180), 45.0)
        self.assertEqual(_effective_min_time_left_seconds(150, 900), 150.0)
        self.assertEqual(_effective_min_time_left_seconds(0, 180), 0.0)

    def test_filter_open_disagreements_ignores_resolved_markers(self):
        items = [
            "**Portability gap** -> **RESOLVED.**",
            "None remaining.",
            "Need rollback coverage before ship.",
        ]
        self.assertEqual(_filter_open_disagreements(items), ["Need rollback coverage before ship."])

    def test_panel_escalation_reopens_round_after_critic_agrees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ConclavePipeline(Config({
                "data_dir": str(Path(tmpdir)),
                "deliberation": {
                    "max_rounds": 2,
                    "require_agreement": True,
                    "panel": {
                        "enabled": True,
                        "review_on_agreement": True,
                        "max_rounds": 2,
                        "require_all": False,
                    },
                },
            }))
            pipeline._model_label = lambda model_id: model_id  # type: ignore[method-assign]

            route = {
                "domain": "general",
                "plan": {"reasoner": "cli:codex", "critic": "cli:claude"},
                "panel_models": ["cli:gemini"],
            }
            counters = {"reasoner": 0, "critic": 0, "critic_panel": 0}
            prompts: dict[tuple[str, int], str] = {}
            responses = {
                ("reasoner", 1): "Draft 1",
                ("critic", 1): "Disagreements:\nVerdict:\nAGREE",
                ("critic_panel", 1): "Disagreements:\n- Missing rollback plan\nVerdict:\nDISAGREE",
                ("reasoner", 2): "Draft 2",
                ("critic", 2): "Disagreements:\nVerdict:\nAGREE",
                ("critic_panel", 2): "Disagreements:\nVerdict:\nAGREE",
            }

            def _fake_call(model_id, prompt, role=None, timeout_seconds=None):
                counters[role] += 1
                key = (role, counters[role])
                prompts[key] = prompt
                return responses[key]

            pipeline._call_model = _fake_call  # type: ignore[method-assign]

            result = deliberate(pipeline, "Should we ship this?", {"output_type": None}, route)

            self.assertEqual(counters["critic"], 2)
            self.assertEqual(counters["critic_panel"], 2)
            self.assertFalse(result["rounds"][0]["agreement"])
            self.assertTrue(result["agreement"])
            self.assertEqual(result["disagreements"], [])
            self.assertIn("Missing rollback plan", result["all_disagreements"])
            self.assertIn("Missing rollback plan", prompts[("reasoner", 2)])

    def test_high_importance_forces_panel_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ConclavePipeline(Config({
                "data_dir": str(Path(tmpdir)),
                "deliberation": {
                    "max_rounds": 2,
                    "require_agreement": True,
                    "panel": {
                        "enabled": False,
                        "max_rounds": 2,
                        "require_all": False,
                    },
                },
            }))
            pipeline._run_meta = {"importance": "high"}
            pipeline._model_label = lambda model_id: model_id  # type: ignore[method-assign]

            route = {
                "domain": "general",
                "plan": {"reasoner": "cli:codex", "critic": "cli:claude"},
                "panel_models": ["cli:gemini"],
            }
            counters = {"reasoner": 0, "critic": 0, "critic_panel": 0}
            responses = {
                ("reasoner", 1): "Draft 1",
                ("critic", 1): "Disagreements:\nVerdict:\nAGREE",
                ("critic_panel", 1): "Disagreements:\n- Missing rollback plan\nVerdict:\nDISAGREE",
                ("reasoner", 2): "Draft 2",
                ("critic", 2): "Disagreements:\nVerdict:\nAGREE",
                ("critic_panel", 2): "Disagreements:\nVerdict:\nAGREE",
            }

            def _fake_call(model_id, prompt, role=None, timeout_seconds=None):
                counters[role] += 1
                return responses[(role, counters[role])]

            pipeline._call_model = _fake_call  # type: ignore[method-assign]

            result = deliberate(pipeline, "Important launch decision", {"output_type": None}, route)

            self.assertTrue(result["agreement"])
            self.assertEqual(counters["critic_panel"], 2)
            self.assertEqual(len(result["panel"]), 2)


if __name__ == "__main__":
    unittest.main()
