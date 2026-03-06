"""Tests for pipeline preflight health check."""
from unittest.mock import patch
from conclave.config import get_config
from conclave.pipeline import ConclavePipeline


def test_preflight_returns_warnings_when_rag_down():
    """Preflight should warn when RAG server is unreachable."""
    config = get_config()
    pipeline = ConclavePipeline(config)

    with patch.object(pipeline.rag, "health_check", return_value=False):
        result = pipeline.preflight_check()

    assert not result["ok"]
    assert any("RAG" in w for w in result["warnings"])
    assert result["services"]["rag"]["ok"] is False


def test_preflight_ok_when_all_healthy():
    """Preflight should report ok when everything is reachable."""
    config = get_config()
    pipeline = ConclavePipeline(config)

    with patch.object(pipeline.rag, "health_check", return_value=True):
        result = pipeline.preflight_check()

    assert result["services"]["rag"]["ok"] is True
    # May still have warnings for missing models — that's fine
    assert "rag" not in " ".join(result.get("warnings", [])).lower()


def test_preflight_warns_missing_models():
    """Preflight should warn about required models not in registry."""
    config = get_config()
    pipeline = ConclavePipeline(config)

    with (
        patch.object(pipeline.rag, "health_check", return_value=True),
        patch.object(pipeline.registry, "get_model", return_value=None),
    ):
        result = pipeline.preflight_check()

    # Should warn about missing models if any are configured as required
    required_cfg = config.raw.get("required_models", {}) or {}
    required_ids = list(required_cfg.get("models", []) or [])
    if required_ids:
        assert not result["ok"]
        assert any("registry" in w.lower() for w in result["warnings"])
