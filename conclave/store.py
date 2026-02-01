"""Persistent store for Conclave runs and consensus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import time
import uuid


@dataclass
class DecisionStore:
    data_dir: Path

    def _runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def _latest_path(self) -> Path:
        return self.data_dir / "latest.json"

    def create_run(self, query: str, meta: Dict[str, Any] | None = None) -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = self._runs_dir() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "running",
            "query": query,
            "meta": meta or {},
            "events": [],
        }
        self._write_run(run_id, payload)
        return run_id

    def append_event(self, run_id: str, event: Dict[str, Any]) -> None:
        run = self.get_run(run_id)
        if not run:
            return
        run.setdefault("events", []).append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **event,
        })
        self._write_run(run_id, run)

    def finalize_run(self, run_id: str, consensus: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
        run = self.get_run(run_id)
        if not run:
            return
        run["status"] = "complete"
        run["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        run["consensus"] = consensus
        run["artifacts"] = artifacts
        self._write_run(run_id, run)
        self._latest_path().parent.mkdir(parents=True, exist_ok=True)
        self._latest_path().write_text(json.dumps(run, indent=2))

    def fail_run(self, run_id: str, error: str) -> None:
        run = self.get_run(run_id)
        if not run:
            return
        run["status"] = "failed"
        run["error"] = error
        run["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_run(run_id, run)

    def get_run(self, run_id: str) -> Dict[str, Any] | None:
        path = self._runs_dir() / run_id / "run.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def latest(self) -> Dict[str, Any] | None:
        if not self._latest_path().exists():
            return None
        try:
            return json.loads(self._latest_path().read_text())
        except Exception:
            return None

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        runs = []
        if not self._runs_dir().exists():
            return runs
        for run_dir in sorted(self._runs_dir().iterdir(), reverse=True)[:limit]:
            path = run_dir / "run.json"
            if not path.exists():
                continue
            try:
                runs.append(json.loads(path.read_text()))
            except Exception:
                continue
        return runs

    def _write_run(self, run_id: str, payload: Dict[str, Any]) -> None:
        run_dir = self._runs_dir() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "run.json"
        path.write_text(json.dumps(payload, indent=2))
