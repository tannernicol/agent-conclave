"""Persistent store for Conclave runs and consensus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import json
import time
import uuid
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None


@dataclass
class DecisionStore:
    data_dir: Path

    def _runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self._runs_dir() / run_id

    def _latest_path(self) -> Path:
        return self.data_dir / "latest.json"

    def _prompt_latest_dir(self) -> Path:
        return self.data_dir / "prompts" / "latest"

    def _prompt_latest_path(self, prompt_id: str) -> Path:
        return self._prompt_latest_dir() / f"{prompt_id}.json"

    def create_run(self, query: str, meta: Dict[str, Any] | None = None) -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = self._runs_dir() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "running",
            "query": query,
            "meta": meta or {},
            "events": [],
        }
        self._write_run(run_id, payload)
        return run_id

    def append_event(self, run_id: str, event: Dict[str, Any]) -> None:
        def _update(run: Dict[str, Any]) -> Dict[str, Any]:
            run.setdefault("events", []).append({
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                **event,
            })
            return run
        self._locked_update(run_id, _update)

    def update_meta(self, run_id: str, meta: Dict[str, Any]) -> None:
        def _update(run: Dict[str, Any]) -> Dict[str, Any]:
            current = run.get("meta", {}) or {}
            current.update(meta)
            run["meta"] = current
            return run
        self._locked_update(run_id, _update)

    def finalize_run(self, run_id: str, consensus: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
        def _update(run: Dict[str, Any]) -> Dict[str, Any]:
            run["status"] = "complete"
            run["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            run["consensus"] = consensus
            run["artifacts"] = artifacts
            return run
        run = self._locked_update(run_id, _update)
        if not run:
            return
        self._latest_path().parent.mkdir(parents=True, exist_ok=True)
        self._latest_path().write_text(json.dumps(run, indent=2))
        prompt_id = (run.get("meta") or {}).get("prompt_id")
        if prompt_id:
            self._prompt_latest_dir().mkdir(parents=True, exist_ok=True)
            self._prompt_latest_path(str(prompt_id)).write_text(json.dumps(run, indent=2))

    def fail_run(self, run_id: str, error: str) -> None:
        def _update(run: Dict[str, Any]) -> Dict[str, Any]:
            run["status"] = "failed"
            run["error"] = error
            run["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            return run
        self._locked_update(run_id, _update)

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

    def latest_for_prompt(self, prompt_id: str) -> Dict[str, Any] | None:
        path = self._prompt_latest_path(prompt_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
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

    def _locked_update(self, run_id: str, updater) -> Dict[str, Any] | None:
        run_dir = self._runs_dir() / run_id
        path = run_dir / "run.json"
        if not path.exists():
            return None
        if fcntl is None:
            run = self.get_run(run_id)
            if not run:
                return None
            updated = updater(run)
            self._write_run(run_id, updated)
            return updated
        run_dir.mkdir(parents=True, exist_ok=True)
        with path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                data = handle.read()
                if not data.strip():
                    return None
                run = json.loads(data)
                updated = updater(run)
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(updated, indent=2))
                return updated
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
