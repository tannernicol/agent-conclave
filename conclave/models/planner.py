"""Role planner: assigns models to roles deterministically."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


ROLE_REQUIREMENTS = {
    "router": {"text_reasoning": True},
    "explorer": {"text_reasoning": True},
    "reasoner": {"text_reasoning": True},
    "critic": {"text_reasoning": True, "json_reliability": "medium"},
    "builder": {"code_generation": True},
    "summarizer": {"text_reasoning": True},
}


@dataclass
class Planner:
    weights: Dict[str, float]
    role_affinity: Dict[str, Dict[str, float]]
    prefer_local: bool = True
    prefer_best: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Planner":
        return cls(
            weights=config.get("weights", {}),
            role_affinity=config.get("role_affinity", {}),
            prefer_local=bool(config.get("prefer_local", True)),
            prefer_best=bool(config.get("prefer_best", False)),
        )

    def choose_models_for_roles(
        self,
        roles: list[str],
        registry: list[Dict[str, Any]],
        role_constraints: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, str]:
        assignments: Dict[str, str] = {}
        constraints = role_constraints or {}
        for role in roles:
            best_model = None
            best_score = -1.0
            for card in registry:
                if not self._satisfies(role, card, constraints.get(role, {})):
                    continue
                score = self._score(role, card)
                if score > best_score:
                    best_score = score
                    best_model = card.get("id")
            if best_model:
                assignments[role] = best_model
        return assignments

    def _satisfies(self, role: str, card: Dict[str, Any], constraint: Dict[str, Any]) -> bool:
        caps = card.get("capabilities", {})
        reqs = ROLE_REQUIREMENTS.get(role, {})
        for key, value in reqs.items():
            if key not in caps:
                return False
            if isinstance(value, str):
                if str(caps.get(key)).lower() not in {value, "high"}:
                    return False
            else:
                if caps.get(key) is not value:
                    return False
        for key, value in constraint.items():
            if key not in caps:
                return False
            if isinstance(value, str):
                if str(caps.get(key)).lower() not in {value, "high"}:
                    return False
            else:
                if caps.get(key) is not value:
                    return False
        return True

    def _score(self, role: str, card: Dict[str, Any]) -> float:
        weights = self.weights or {"latency": 0.35, "reliability": 0.25, "cost": 0.2, "affinity": 0.2}
        metrics = card.get("metrics", {})
        baseline = card.get("perf_baseline", {})
        latency_ms = metrics.get("p50_latency_ms") or baseline.get("p50_latency_ms") or 2000
        error_rate = metrics.get("error_rate", 0.0)
        timeout_rate = metrics.get("timeout_rate", 0.0)
        reliability = max(0.0, 1.0 - (error_rate + timeout_rate))
        cost = card.get("cost", {}).get("usd_per_1m_input_tokens", 0.0)
        latency_score = 1.0 / (1.0 + (latency_ms / 1000.0))
        cost_score = 1.0 / (1.0 + cost)
        affinity_score = self._affinity(role, card)

        score = (
            weights.get("latency", 0.35) * latency_score
            + weights.get("reliability", 0.25) * reliability
            + weights.get("cost", 0.2) * cost_score
            + weights.get("affinity", 0.2) * affinity_score
        )

        if self.prefer_local and card.get("kind") == "local":
            score *= 1.05
        if self.prefer_best and card.get("capabilities", {}).get("json_reliability") == "high":
            score *= 1.05
        return score

    def _affinity(self, role: str, card: Dict[str, Any]) -> float:
        caps = card.get("capabilities", {})
        affinity = self.role_affinity.get(role, {})
        reasoning = 1.0 if caps.get("text_reasoning") else 0.0
        json_rel = caps.get("json_reliability")
        json_score = 0.3
        if json_rel == "high":
            json_score = 1.0
        elif json_rel == "medium":
            json_score = 0.7
        speed_score = 1.0
        baseline = card.get("perf_baseline", {})
        latency_ms = baseline.get("p50_latency_ms", 2000)
        if latency_ms > 4000:
            speed_score = 0.4
        elif latency_ms > 2500:
            speed_score = 0.6
        elif latency_ms > 1500:
            speed_score = 0.8

        score = (
            affinity.get("reasoning", 0.4) * reasoning
            + affinity.get("json", 0.3) * json_score
            + affinity.get("speed", 0.3) * speed_score
        )
        return score
