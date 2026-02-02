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
    role_overrides: Dict[str, str] | None = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Planner":
        return cls(
            weights=config.get("weights", {}),
            role_affinity=config.get("role_affinity", {}),
            prefer_local=bool(config.get("prefer_local", True)),
            prefer_best=bool(config.get("prefer_best", False)),
            role_overrides=config.get("role_overrides", {}),
        )

    def choose_models_for_roles(
        self,
        roles: list[str],
        registry: list[Dict[str, Any]],
        role_constraints: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, str]:
        plan = self.plan_with_rationale(roles, registry, role_constraints=role_constraints)
        return plan["assignments"]

    def plan_with_rationale(
        self,
        roles: list[str],
        registry: list[Dict[str, Any]],
        role_constraints: Dict[str, Dict[str, Any]] | None = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        assignments: Dict[str, str] = {}
        rationale: Dict[str, Any] = {}
        constraints = role_constraints or {}
        registry_map = {card.get("id"): card for card in registry if card.get("id")}
        for role in roles:
            candidates = []
            best_model = None
            best_score = -1.0
            override = None
            if self.role_overrides:
                override = self.role_overrides.get(role)
            if override:
                card = registry_map.get(override)
                if card:
                    ok, reason = self._check_requirements(role, card, constraints.get(role, {}))
                    candidates.append({
                        "id": card.get("id"),
                        "eligible": ok,
                        "score": None,
                        "details": {"override": True, "reason": reason or "ok"},
                    })
                    if ok:
                        assignments[role] = card.get("id")
                        rationale[role] = {
                            "selected": card.get("id"),
                            "weights": self.weights,
                            "preferences": {
                                "prefer_local": self.prefer_local,
                                "prefer_best": self.prefer_best,
                            },
                            "candidates": candidates,
                        }
                        continue
                else:
                    candidates.append({
                        "id": override,
                        "eligible": False,
                        "reason": "override missing from registry",
                    })
            for card in registry:
                ok, reason = self._check_requirements(role, card, constraints.get(role, {}))
                if not ok:
                    candidates.append({
                        "id": card.get("id"),
                        "eligible": False,
                        "reason": reason,
                    })
                    continue
                score, details = self._score_with_details(role, card)
                candidates.append({
                    "id": card.get("id"),
                    "eligible": True,
                    "score": score,
                    "details": details,
                })
                if score > best_score:
                    best_score = score
                    best_model = card.get("id")
            if best_model:
                assignments[role] = best_model
            # keep top scores first
            eligible = [c for c in candidates if c.get("eligible")]
            eligible.sort(key=lambda x: x.get("score", 0), reverse=True)
            ineligible = [c for c in candidates if not c.get("eligible")]
            rationale[role] = {
                "selected": best_model,
                "weights": self.weights,
                "preferences": {
                    "prefer_local": self.prefer_local,
                    "prefer_best": self.prefer_best,
                },
                "candidates": eligible[:top_n] + ineligible[:top_n],
            }
        return {"assignments": assignments, "rationale": rationale}

    def _check_requirements(self, role: str, card: Dict[str, Any], constraint: Dict[str, Any]) -> tuple[bool, str]:
        caps = card.get("capabilities", {})
        reqs = ROLE_REQUIREMENTS.get(role, {})
        for key, value in reqs.items():
            if key not in caps:
                return False, f"missing capability: {key}"
            if isinstance(value, str):
                if str(caps.get(key)).lower() not in {value, "high"}:
                    return False, f"capability {key} below {value}"
            else:
                if caps.get(key) != value:
                    return False, f"capability {key} != {value}"
        for key, value in constraint.items():
            if key not in caps:
                return False, f"constraint missing: {key}"
            if isinstance(value, str):
                if str(caps.get(key)).lower() not in {value, "high"}:
                    return False, f"constraint {key} below {value}"
            else:
                if caps.get(key) != value:
                    return False, f"constraint {key} != {value}"
        return True, ""

    def _score(self, role: str, card: Dict[str, Any]) -> float:
        score, _ = self._score_with_details(role, card)
        return score

    def _score_with_details(self, role: str, card: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
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

        multiplier = 1.0
        if self.prefer_local and card.get("kind") == "local":
            multiplier *= 1.05
        if self.prefer_best and card.get("capabilities", {}).get("json_reliability") == "high":
            multiplier *= 1.05
        score *= multiplier

        details = {
            "latency_ms": latency_ms,
            "latency_score": round(latency_score, 4),
            "reliability": round(reliability, 4),
            "cost": cost,
            "cost_score": round(cost_score, 4),
            "affinity_score": round(affinity_score, 4),
            "multiplier": round(multiplier, 4),
            "final_score": round(score, 4),
        }
        return score, details

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
