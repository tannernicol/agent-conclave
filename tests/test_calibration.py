"""Tests for self-organizing role calibration in the Planner."""
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from conclave.models.planner import (
    Planner,
    ModelCalibration,
    CALIBRATION_DOMAINS,
    CALIBRATION_PROMPT,
    DOMAIN_TO_CALIBRATION,
)


class TestModelCalibration(unittest.TestCase):
    def test_score_for_domain_normalizes(self):
        cal = ModelCalibration(
            model_id="test",
            domain_scores={"security": 9.0, "general": 5.0, "code_review": 2.0},
        )
        self.assertAlmostEqual(cal.score_for_domain("security"), 0.9)
        self.assertAlmostEqual(cal.score_for_domain("general"), 0.5)
        self.assertAlmostEqual(cal.score_for_domain("code_review"), 0.2)

    def test_score_for_unknown_domain_uses_general(self):
        cal = ModelCalibration(
            model_id="test",
            domain_scores={"general": 7.0},
        )
        self.assertAlmostEqual(cal.score_for_domain("unknown_xyz"), 0.7)

    def test_score_for_none_domain_uses_general(self):
        cal = ModelCalibration(
            model_id="test",
            domain_scores={"general": 6.0},
        )
        self.assertAlmostEqual(cal.score_for_domain(None), 0.6)


class TestPlannerCalibration(unittest.TestCase):
    def _make_planner(self) -> Planner:
        return Planner(
            weights={"latency": 0.35, "reliability": 0.25, "cost": 0.2, "affinity": 0.2},
            role_affinity={},
        )

    def test_parse_calibration_response_valid_json(self):
        planner = self._make_planner()
        response = '{"security": 9, "code_review": 7, "research": 4, "creative": 3, "general": 6}'
        cal = planner.parse_calibration_response("cli:claude", response)

        self.assertEqual(cal.model_id, "cli:claude")
        self.assertEqual(cal.domain_scores["security"], 9.0)
        self.assertTrue(cal.calibrated_at)

    def test_parse_calibration_response_markdown_wrapped(self):
        planner = self._make_planner()
        response = '```json\n{"security": 8, "code_review": 9, "research": 5, "creative": 4, "general": 6}\n```'
        cal = planner.parse_calibration_response("cli:codex", response)

        self.assertEqual(cal.domain_scores["code_review"], 9.0)

    def test_parse_calibration_response_invalid_defaults_to_5(self):
        planner = self._make_planner()
        cal = planner.parse_calibration_response("cli:claude", "I can't rate myself")

        for domain in CALIBRATION_DOMAINS:
            self.assertEqual(cal.domain_scores[domain], 5.0)

    def test_parse_calibration_clamps_values(self):
        planner = self._make_planner()
        response = '{"security": 15, "code_review": -3, "research": 5, "creative": 5, "general": 5}'
        cal = planner.parse_calibration_response("test", response)

        self.assertEqual(cal.domain_scores["security"], 10.0)
        self.assertEqual(cal.domain_scores["code_review"], 1.0)

    def test_save_and_load_calibration(self):
        planner = self._make_planner()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cal.json"
            calibrations = {
                "cli:claude": ModelCalibration(
                    model_id="cli:claude",
                    domain_scores={"security": 9.0, "general": 7.0},
                    calibrated_at=datetime.now(timezone.utc).isoformat(),
                )
            }
            planner.save_calibration(calibrations, cache_path)
            loaded = planner.load_calibration(cache_path)

            self.assertIn("cli:claude", loaded)
            self.assertEqual(loaded["cli:claude"].domain_scores["security"], 9.0)

    def test_load_calibration_skips_stale(self):
        planner = self._make_planner()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cal.json"
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            data = {
                "cli:claude": {
                    "domain_scores": {"security": 9.0},
                    "calibrated_at": old_time,
                }
            }
            cache_path.write_text(json.dumps(data))
            loaded = planner.load_calibration(cache_path)

            self.assertEqual(loaded, {})

    def test_load_calibration_keeps_fresh(self):
        planner = self._make_planner()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cal.json"
            fresh_time = datetime.now(timezone.utc).isoformat()
            data = {
                "cli:claude": {
                    "domain_scores": {"security": 8.0, "general": 6.0},
                    "calibrated_at": fresh_time,
                }
            }
            cache_path.write_text(json.dumps(data))
            loaded = planner.load_calibration(cache_path)

            self.assertIn("cli:claude", loaded)

    def test_get_domain_score_no_calibration(self):
        planner = self._make_planner()
        score = planner.get_domain_score("cli:claude", "security")
        self.assertEqual(score, 0.5)

    def test_get_domain_score_with_calibration(self):
        planner = self._make_planner()
        planner._calibration_cache["cli:claude"] = ModelCalibration(
            model_id="cli:claude",
            domain_scores={"security": 9.0, "general": 5.0},
        )
        score = planner.get_domain_score("cli:claude", "security")
        self.assertAlmostEqual(score, 0.9)

    def test_scoring_with_domain_and_calibration(self):
        planner = self._make_planner()
        planner._calibration_cache["model-a"] = ModelCalibration(
            model_id="model-a",
            domain_scores={"security": 9.0, "general": 5.0},
        )
        planner._calibration_cache["model-b"] = ModelCalibration(
            model_id="model-b",
            domain_scores={"security": 3.0, "general": 5.0},
        )
        card_a = {
            "id": "model-a",
            "capabilities": {"text_reasoning": True, "json_reliability": "medium"},
            "perf_baseline": {"p50_latency_ms": 2000},
            "cost": {"usd_per_1m_input_tokens": 0},
        }
        card_b = {
            "id": "model-b",
            "capabilities": {"text_reasoning": True, "json_reliability": "medium"},
            "perf_baseline": {"p50_latency_ms": 2000},
            "cost": {"usd_per_1m_input_tokens": 0},
        }

        score_a, _ = planner._score_with_details("reasoner", card_a, domain="security")
        score_b, _ = planner._score_with_details("reasoner", card_b, domain="security")

        # model-a has higher security score, should score higher for security domain
        self.assertGreater(score_a, score_b)

    def test_outcome_learning_updates_scores(self):
        planner = self._make_planner()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cal.json"
            planner._calibration_cache["winner"] = ModelCalibration(
                model_id="winner",
                domain_scores={"security": 7.0},
                calibrated_at=datetime.now(timezone.utc).isoformat(),
            )
            planner._calibration_cache["loser"] = ModelCalibration(
                model_id="loser",
                domain_scores={"security": 7.0},
                calibrated_at=datetime.now(timezone.utc).isoformat(),
            )

            planner.update_calibration_from_outcome(
                winner_model_id="winner",
                other_model_ids=["loser"],
                domain="security",
                cache_path=cache_path,
            )

            self.assertAlmostEqual(planner._calibration_cache["winner"].domain_scores["security"], 7.1)
            self.assertAlmostEqual(planner._calibration_cache["loser"].domain_scores["security"], 6.95)

    def test_choose_models_for_roles_with_domain(self):
        """Ensure the domain parameter is accepted by choose_models_for_roles."""
        planner = self._make_planner()
        registry = [{
            "id": "cli:claude",
            "capabilities": {"text_reasoning": True, "json_reliability": "high"},
            "perf_baseline": {"p50_latency_ms": 2000},
            "cost": {"usd_per_1m_input_tokens": 0},
        }]
        assignments = planner.choose_models_for_roles(
            roles=["reasoner"],
            registry=registry,
            domain="security",
        )
        self.assertIn("reasoner", assignments)


class TestDomainToCalibrationMapping(unittest.TestCase):
    def test_security_maps_to_security(self):
        self.assertEqual(DOMAIN_TO_CALIBRATION["security"], "security")

    def test_code_review_maps_to_code_review(self):
        self.assertEqual(DOMAIN_TO_CALIBRATION["code_review"], "code_review")

    def test_calibration_prompt_exists(self):
        self.assertIn("security", CALIBRATION_PROMPT)
        self.assertIn("code_review", CALIBRATION_PROMPT)
        self.assertIn("JSON", CALIBRATION_PROMPT)


if __name__ == "__main__":
    unittest.main()
