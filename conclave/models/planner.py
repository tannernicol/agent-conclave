"""Role planner: assigns models to roles with domain-aware self-organization."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

logger = logging.getLogger(__name__)

ROLE_REQUIREMENTS = {
    "router": {"text_reasoning": True},
    "explorer": {"text_reasoning": True},
    "reasoner": {"text_reasoning": True},
    "creator": {"text_reasoning": True},
    "critic": {"text_reasoning": True, "json_reliability": "medium"},
    "reviewer": {"text_reasoning": True, "json_reliability": "medium"},
    "builder": {"code_generation": True},
    "summarizer": {"text_reasoning": True},
}

CALIBRATION_DOMAINS = [
    "security",
    "code_review",
    "research",
    "creative",
    "general",
]

CALIBRATION_PROMPT = """Rate your expertise from 1-10 in each domain:
- security: Cybersecurity, vulnerability analysis, secure coding
- code_review: Code quality, architecture, debugging
- research: Analysis, comparison, literature review
- creative: Writing, brainstorming, content creation
- general: Broad reasoning, planning, problem solving

Reply as JSON only: {"security": N, "code_review": N, "research": N, "creative": N, "general": N}"""

# Map routing domains to calibration domains
DOMAIN_TO_CALIBRATION = {
    "security": "security",
    "code_review": "code_review",
    "research": "research",
    "creative": "creative",
    "general": "general",
}

CALIBRATION_CACHE_DAYS = 7


@dataclass
class ModelCalibration:
    model_id: str
    domain_scores: Dict[str, float]
    calibrated_at: str = ""

    def score_for_domain(self, domain: str | None) -> float:
        """Get calibration score for a routing domain, normalized to 0-1."""
        cal_domain = DOMAIN_TO_CALIBRATION.get(domain or "", "general")
        raw = self.domain_scores.get(cal_domain, 5.0)
        return max(0.0, min(1.0, raw / 10.0))


@dataclass
class Planner:
    weights: Dict[str, float]
    role_affinity: Dict[str, Dict[str, float]]
    prefer_local: bool = True
    prefer_best: bool = False
    role_overrides: Dict[str, str] | None = None
    self_organize: bool = False
    allow_overrides: bool = True
    budget_config: Dict[str, Any] | None = None
    _calibration_cache: Dict[str, ModelCalibration] = field(default_factory=dict)

    def with_overrides(self, role_overrides: Dict[str, str] | None) -> "Planner":
        return Planner(
            weights=self.weights,
            role_affinity=self.role_affinity,
            prefer_local=self.prefer_local,
            prefer_best=self.prefer_best,
            role_overrides=role_overrides or {},
            self_organize=self.self_organize,
            allow_overrides=self.allow_overrides,
            budget_config=self.budget_config,
            _calibration_cache=self._calibration_cache,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Planner":
        self_org = config.get("self_organize", {}) or {}
        budget_cfg = self_org.get("budget") if isinstance(self_org, dict) else {}
        return cls(
            weights=config.get("weights", {}),
            role_affinity=config.get("role_affinity", {}),
            prefer_local=bool(config.get("prefer_local", True)),
            prefer_best=bool(config.get("prefer_best", False)),
            role_overrides=config.get("role_overrides", {}),
            self_organize=bool(self_org.get("enabled", False)),
            allow_overrides=bool(self_org.get("allow_overrides", True)),
            budget_config=budget_cfg if isinstance(budget_cfg, dict) else {},
        )

    def choose_models_for_roles(
        self,
        roles: list[str],
        registry: list[Dict[str, Any]],
        role_constraints: Dict[str, Dict[str, Any]] | None = None,
        domain: str | None = None,
    ) -> Dict[str, str]:
        plan = self.plan_with_rationale(roles, registry, role_constraints=role_constraints, domain=domain)
        return plan["assignments"]

    def plan_with_rationale(
        self,
        roles: list[str],
        registry: list[Dict[str, Any]],
        role_constraints: Dict[str, Dict[str, Any]] | None = None,
        budget_context: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        noise: float = 0.0,
        rng: random.Random | None = None,
        domain: str | None = None,
    ) -> Dict[str, Any]:
        assignments: Dict[str, str] = {}
        rationale: Dict[str, Any] = {}
        constraints = role_constraints or {}
        registry_map = {card.get("id"): card for card in registry if card.get("id")}
        effective_weights = self._effective_weights(budget_context)
        for role in roles:
            candidates = []
            best_model = None
            best_score = -1.0
            override = None
            if self.role_overrides and (not self.self_organize or self.allow_overrides):
                override = self.role_overrides.get(role)
            if override:
                card = registry_map.get(override)
                if card:
                    allow_unhealthy = True  # explicit overrides bypass health checks
                    ok, reason = self._check_requirements(
                        role,
                        card,
                        constraints.get(role, {}),
                        allow_unhealthy=allow_unhealthy,
                    )
                    candidates.append({
                        "id": card.get("id"),
                        "eligible": ok,
                        "score": None,
                        "details": {
                            "override": True,
                            "reason": reason or "ok",
                            "allow_unhealthy": allow_unhealthy,
                        },
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
                score, details = self._score_with_details(
                    role,
                    card,
                    effective_weights,
                    budget_context,
                    noise=noise,
                    rng=rng,
                    domain=domain,
                )
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
                "weights": effective_weights,
                "preferences": {
                    "prefer_local": self.prefer_local,
                    "prefer_best": self.prefer_best,
                    "self_organize": self.self_organize,
                },
                "budget": budget_context or self.budget_config or {},
                "candidates": eligible[:top_n] + ineligible[:top_n],
            }
        return {"assignments": assignments, "rationale": rationale}

    # --- Calibration system ---

    def load_calibration(self, cache_path: Path | None = None) -> Dict[str, ModelCalibration]:
        """Load cached model calibration data."""
        path = cache_path or (Path.home() / ".conclave" / "model_calibration.json")
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                return {}
            # Check freshness
            result: Dict[str, ModelCalibration] = {}
            for model_id, entry in data.items():
                calibrated_at = entry.get("calibrated_at", "")
                try:
                    cal_time = datetime.fromisoformat(calibrated_at)
                    age_days = (datetime.now(timezone.utc) - cal_time).days
                    if age_days > CALIBRATION_CACHE_DAYS:
                        continue
                except (ValueError, TypeError):
                    continue
                result[model_id] = ModelCalibration(
                    model_id=model_id,
                    domain_scores=entry.get("domain_scores", {}),
                    calibrated_at=calibrated_at,
                )
            return result
        except Exception as e:
            logger.debug(f"Failed to load calibration cache: {e}")
            return {}

    def save_calibration(
        self,
        calibrations: Dict[str, ModelCalibration],
        cache_path: Path | None = None,
    ) -> None:
        """Save calibration data to cache."""
        path = cache_path or (Path.home() / ".conclave" / "model_calibration.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for model_id, cal in calibrations.items():
            data[model_id] = {
                "domain_scores": cal.domain_scores,
                "calibrated_at": cal.calibrated_at,
            }
        path.write_text(json.dumps(data, indent=2))

    def parse_calibration_response(self, model_id: str, text: str) -> ModelCalibration:
        """Parse a calibration response from a model."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            clean = text.strip()
            if "```" in clean:
                start = clean.index("```")
                end = clean.rindex("```")
                inner = clean[start:end]
                # Remove the opening ``` and optional language tag
                inner = inner.split("\n", 1)[1] if "\n" in inner else inner[3:]
                clean = inner.strip()
            if clean.startswith("{"):
                scores = json.loads(clean)
            else:
                # Try to find JSON in the text
                import re
                match = re.search(r'\{[^{}]+\}', text)
                if match:
                    scores = json.loads(match.group())
                else:
                    scores = {}
        except (json.JSONDecodeError, ValueError):
            scores = {}

        # Validate and normalize scores
        domain_scores: Dict[str, float] = {}
        for domain in CALIBRATION_DOMAINS:
            raw = scores.get(domain, 5.0)
            try:
                val = float(raw)
                domain_scores[domain] = max(1.0, min(10.0, val))
            except (ValueError, TypeError):
                domain_scores[domain] = 5.0

        return ModelCalibration(
            model_id=model_id,
            domain_scores=domain_scores,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_domain_score(self, model_id: str, domain: str | None) -> float:
        """Get a model's calibration score for a domain, normalized to 0-1."""
        cal = self._calibration_cache.get(model_id)
        if not cal:
            return 0.5  # Neutral default
        return cal.score_for_domain(domain)

    def update_calibration_from_outcome(
        self,
        winner_model_id: str,
        other_model_ids: List[str],
        domain: str | None,
        cache_path: Path | None = None,
    ) -> None:
        """Boost winner's domain score, reduce others slightly."""
        cal_domain = DOMAIN_TO_CALIBRATION.get(domain or "", "general")

        if winner_model_id in self._calibration_cache:
            cal = self._calibration_cache[winner_model_id]
            current = cal.domain_scores.get(cal_domain, 5.0)
            cal.domain_scores[cal_domain] = min(10.0, current + 0.1)

        for model_id in other_model_ids:
            if model_id == winner_model_id:
                continue
            if model_id in self._calibration_cache:
                cal = self._calibration_cache[model_id]
                current = cal.domain_scores.get(cal_domain, 5.0)
                cal.domain_scores[cal_domain] = max(1.0, current - 0.05)

        self.save_calibration(self._calibration_cache, cache_path)

    # --- Scoring ---

    def _check_requirements(
        self,
        role: str,
        card: Dict[str, Any],
        constraint: Dict[str, Any],
        allow_unhealthy: bool = False,
    ) -> tuple[bool, str]:
        metrics = card.get("metrics", {})
        if metrics.get("ok") is False and not allow_unhealthy:
            return False, "model unhealthy"
        caps = card.get("capabilities_override") or card.get("capabilities", {})
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

    def _score(self, role: str, card: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
        score, _ = self._score_with_details(role, card, weights or self.weights, None)
        return score

    def _score_with_details(
        self,
        role: str,
        card: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        budget_context: Optional[Dict[str, Any]] = None,
        noise: float = 0.0,
        rng: random.Random | None = None,
        domain: str | None = None,
    ) -> tuple[float, Dict[str, Any]]:
        weights = weights or {"latency": 0.35, "reliability": 0.25, "cost": 0.2, "affinity": 0.2}
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

        # Domain calibration boost
        domain_score = self.get_domain_score(card.get("id", ""), domain)
        has_calibration = bool(self._calibration_cache)

        if has_calibration and domain:
            # With calibration: 0.3*affinity + 0.2*latency + 0.2*reliability + 0.1*cost + 0.2*domain
            score = (
                weights.get("affinity", 0.2) * 0.75 * affinity_score
                + weights.get("latency", 0.35) * 0.7 * latency_score
                + weights.get("reliability", 0.25) * reliability
                + weights.get("cost", 0.2) * cost_score
                + 0.2 * domain_score
            )
        else:
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

        jitter = 0.0
        if noise and noise > 0:
            rng = rng or random
            jitter = rng.uniform(-noise, noise)
            score = max(0.0, score + jitter)

        details = {
            "latency_ms": latency_ms,
            "latency_score": round(latency_score, 4),
            "reliability": round(reliability, 4),
            "cost": cost,
            "cost_score": round(cost_score, 4),
            "affinity_score": round(affinity_score, 4),
            "domain_score": round(domain_score, 4),
            "multiplier": round(multiplier, 4),
            "final_score": round(score, 4),
            "budget": self._budget_details(budget_context),
            "noise": round(jitter, 4),
        }
        return score, details

    def _effective_weights(self, budget_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        base = dict(self.weights or {"latency": 0.35, "reliability": 0.25, "cost": 0.2, "affinity": 0.2})
        budget = budget_context or self.budget_config or {}
        if not budget or not bool(budget.get("enabled", False)):
            return base
        total = float(budget.get("total_tokens", 0) or 0)
        remaining = float(budget.get("remaining_tokens", total) if total else 0)
        if total <= 0:
            return base
        ratio = max(0.0, min(1.0, remaining / total))
        pressure = 1.0 - ratio
        boost = float(budget.get("cost_weight_boost", 0.35))
        base["cost"] = base.get("cost", 0.2) + (pressure * boost)
        # renormalize weights
        total_weight = sum(base.values()) or 1.0
        for key in list(base.keys()):
            base[key] = base[key] / total_weight
        return base

    def _budget_details(self, budget_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not budget_context:
            return {}
        return {
            "enabled": bool(budget_context.get("enabled", False)),
            "total_tokens": budget_context.get("total_tokens"),
            "remaining_tokens": budget_context.get("remaining_tokens"),
        }

    def _affinity(self, role: str, card: Dict[str, Any]) -> float:
        caps = card.get("capabilities_override") or card.get("capabilities", {})
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
