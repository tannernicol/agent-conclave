"""Model registry for capability cards + telemetry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    registry_path: Path
    benchmarks_path: Path
    health_path: Path
    cards: Dict[str, Dict[str, Any]]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelRegistry":
        registry_path = Path(config.get("registry_path", "/home/tanner/.conclave/models/registry.json"))
        benchmarks_path = Path(config.get("benchmarks_path", "/home/tanner/.conclave/models/benchmarks.jsonl"))
        health_path = Path(config.get("health_path", "/home/tanner/.conclave/models/health.json"))
        cards = {card["id"]: card for card in config.get("cards", [])}
        instance = cls(registry_path, benchmarks_path, health_path, cards)
        instance._load_registry()
        return instance

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            data = json.loads(self.registry_path.read_text())
            for model_id, stored in data.get("models", {}).items():
                if model_id in self.cards:
                    merged = dict(self.cards[model_id])
                    merged.update(stored)
                    self.cards[model_id] = merged
                else:
                    self.cards[model_id] = stored
        except Exception:
            logger.warning("Failed to load registry data", exc_info=True)
            return

    def save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": self.cards,
        }
        self.registry_path.write_text(json.dumps(payload, indent=2))

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self.cards.values())

    def get_model(self, model_id: str) -> Dict[str, Any] | None:
        return self.cards.get(model_id)

    def update_metrics(self, model_id: str, observation: Dict[str, Any]) -> None:
        if model_id not in self.cards:
            return
        entry = self.cards[model_id].setdefault("metrics", {})
        entry.update(observation)
        self._append_benchmark(model_id, observation)
        self._update_health(model_id, observation)
        self.save()

    def _append_benchmark(self, model_id: str, observation: Dict[str, Any]) -> None:
        self.benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_id": model_id,
            **observation,
        }
        with self.benchmarks_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _update_health(self, model_id: str, observation: Dict[str, Any]) -> None:
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(self.health_path.read_text())
        except Exception:
            existing = {"models": {}}
        existing.setdefault("models", {})[model_id] = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **observation,
        }
        self.health_path.write_text(json.dumps(existing, indent=2))
