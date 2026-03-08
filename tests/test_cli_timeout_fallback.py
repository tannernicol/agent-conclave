from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from conclave.cli import _execute_run
from conclave.config import Config
from conclave.pipeline import ConclavePipeline, PipelineResult, RequiredModelError


class _FakeStore:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    def create_run(self, query: str, meta: dict | None = None) -> str:
        run_id = "run-123"
        self.runs[run_id] = {"id": run_id, "query": query, "meta": meta or {}}
        return run_id

    def get_run(self, run_id: str) -> dict | None:
        return self.runs.get(run_id)

    def fail_run(self, run_id: str, error: str) -> None:
        run = self.runs.setdefault(run_id, {"id": run_id})
        run["status"] = "failed"
        run["error"] = error


class _FakePipeline:
    def __init__(self) -> None:
        self.config = SimpleNamespace(raw={})
        self.store = _FakeStore()
        self.seen_timeout = None

    def run(self, query: str, collections=None, meta=None, run_id=None) -> PipelineResult:
        self.seen_timeout = self.config.raw["pipeline"]["run_timeout_seconds"]
        return PipelineResult(
            run_id=run_id or "run-123",
            consensus={"answer": "ok", "insufficient_evidence": False},
            artifacts={},
        )


def test_execute_run_uses_pipeline_timeout_override():
    pipeline = _FakePipeline()
    args = Namespace(progress=False, max_seconds=17)

    result = _execute_run(pipeline, "question", None, None, args)

    assert result.run_id == "run-123"
    assert pipeline.seen_timeout == 17


def test_call_cli_model_clamps_timeout_to_remaining_time(tmp_path: Path):
    pipeline = ConclavePipeline(Config({
        "data_dir": str(tmp_path),
        "pipeline": {"run_timeout_seconds": 30},
    }))
    pipeline._run_start_time = time.time() - 29.4
    pipeline.registry.get_model = lambda model_id: {  # type: ignore[method-assign]
        "command": ["codex"],
        "timeout_seconds": 120,
    }
    captured: dict[str, int] = {}

    def _fake_run(**kwargs):
        captured["timeout_seconds"] = kwargs["timeout_seconds"]
        return SimpleNamespace(text="ok", duration_ms=1.0, ok=True, error=None, stderr=None)

    pipeline.cli.run = _fake_run  # type: ignore[method-assign]

    result = pipeline._call_cli_model("cli:codex", "hello")

    assert result.ok is True
    assert captured["timeout_seconds"] == 1


def test_summarize_falls_back_to_reasoner_when_summarizer_fails(tmp_path: Path):
    pipeline = ConclavePipeline(Config({
        "data_dir": str(tmp_path),
        "pipeline": {"run_timeout_seconds": 120},
    }))
    pipeline._run_start_time = time.time()

    def _raise(*args, **kwargs):
        raise RequiredModelError("cli:claude failed: timeout")

    pipeline._call_model = _raise  # type: ignore[method-assign]
    deliberation = {
        "agreement": True,
        "reasoner": "**Decision**\n\nSubmit it as Medium.\n\n**Rationale**\n\n- Authorization constraint is unsigned.\n",
        "critic": "",
        "disagreements": [],
    }
    route = {
        "plan": {"summarizer": "cli:claude"},
        "plan_details": {"summarizer": {"id": "cli:claude", "label": "Claude"}},
    }

    consensus = pipeline._summarize(
        "Should this be submitted?",
        {},
        deliberation,
        route,
        {"evidence_count": 1, "pdf_ratio": 0.0, "off_domain_ratio": 0.0, "max_signal_score": 1.0},
    )

    assert consensus["fallback_used"] is True
    assert "timeout" in consensus["fallback_reason"]
    assert "Submit it as Medium." in consensus["answer"]
    assert "## Runtime note" in consensus["answer"]
