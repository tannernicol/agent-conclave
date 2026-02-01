"""Structured audit logging for Conclave runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import time


@dataclass
class AuditLog:
    path: Path

    def log(self, event: str, data: Dict[str, Any] | None = None) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "data": data or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
