"""Core decision pipeline for Conclave."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import fnmatch
import mimetypes
import os
import base64
import json
import logging
import math
import random
import re
import shutil
import subprocess
import time
import httpx

from conclave.config import Config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner
from conclave.models.ollama import OllamaClient
from conclave.models.cli import CliClient
from conclave.rag import RagClient, NasIndex
from conclave.store import DecisionStore
from conclave.audit import AuditLog
from conclave.mcp import load_mcp_servers
from conclave.mcp_bridge import MCPBridge
from conclave.domains import get_domain_instructions
from conclave.verification import OnDemandFetcher


class RunTimeoutError(Exception):
    """Raised when a pipeline run exceeds its timeout."""
    pass


class RequiredModelError(Exception):
    """Raised when a required model fails during a run."""
    pass


class InsufficientModelsError(Exception):
    """Raised when fewer than 2 distinct intelligent models participated."""
    pass


@dataclass
class PipelineResult:
    run_id: str
    consensus: Dict[str, Any]
    artifacts: Dict[str, Any]


class ConclavePipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.registry = ModelRegistry.from_config(config.models)
        self.planner = Planner.from_config(config.planner)
        self.ollama = OllamaClient()
        self.cli = CliClient()
        self.rag = RagClient(config.rag.get("base_url", "http://localhost:8091"))
        self.index = NasIndex(
            data_dir=config.data_dir,
            allowlist=config.index.get("allowlist", []),
            exclude_patterns=config.index.get("exclude_patterns", []),
            max_file_mb=int(config.index.get("max_file_mb", 2)),
        )
        self.store = DecisionStore(config.data_dir)
        self.mcp = MCPBridge(config_path=config.mcp_config_path)
        self.verifier = OnDemandFetcher(config, self.mcp, self.rag)
        self._audit: AuditLog | None = None
        self._run_id: str | None = None
        self._run_start_time: float = 0.0
        self._context_char_limit: int | None = None
        self._calibration_cache: Dict[str, Dict[str, Any]] = {}
        self._calibration_cache_time: float = 0.0
        self._calibration_cache_ttl: float = 300.0  # 5 minutes
        self._last_model_results: Dict[str, Dict[str, Any]] = {}
        self._run_models_used: set[str] = set()
        self._token_budget_total: float | None = None
        self._token_budget_used: float = 0.0
        self._budget_context: Dict[str, Any] | None = None
        self._agent_sync_cache: Dict[tuple[str, int, bool], str] = {}
        self._agent_sync_meta: Dict[str, Any] = {}
        self._token_usage_by_model: Dict[str, Dict[str, float]] = {}
        self._image_usage: Dict[str, int] = {}
        self._vision_usage: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)

    def _check_timeout(self, phase: str = "unknown") -> None:
        """Check if the current run has exceeded its timeout."""
        if self._run_start_time <= 0:
            return
        elapsed = time.time() - self._run_start_time
        timeout = self.config.run_timeout_seconds
        if elapsed > timeout:
            raise RunTimeoutError(f"Run exceeded {timeout}s timeout during {phase} phase (elapsed: {elapsed:.1f}s)")

    def _time_remaining(self) -> float | None:
        if self._run_start_time <= 0:
            return None
        return float(self.config.run_timeout_seconds) - (time.time() - self._run_start_time)

    def _init_token_budget(self, meta: Optional[Dict[str, Any]]) -> None:
        cfg = (self.config.raw.get("planner", {}) or {}).get("self_organize", {}) or {}
        budget_cfg = cfg.get("budget") if isinstance(cfg, dict) else {}
        if not isinstance(budget_cfg, dict) or not budget_cfg.get("enabled", False):
            self._token_budget_total = None
            self._token_budget_used = 0.0
            self._budget_context = None
            return
        total = float(budget_cfg.get("total_tokens", 0) or 0)
        remaining = None
        if meta:
            if meta.get("token_budget_total") is not None:
                try:
                    total = float(meta.get("token_budget_total") or total)
                except Exception:
                    pass
            if meta.get("token_budget_remaining") is not None:
                try:
                    remaining = float(meta.get("token_budget_remaining"))
                except Exception:
                    remaining = None
            if remaining is None and meta.get("token_budget_used") is not None:
                try:
                    used = float(meta.get("token_budget_used"))
                    remaining = max(0.0, total - used)
                except Exception:
                    remaining = None
        if remaining is None:
            remaining = total if total > 0 else 0.0
        self._token_budget_total = total if total > 0 else None
        self._token_budget_used = max(0.0, (total - remaining)) if total else 0.0
        self._budget_context = {
            "enabled": True,
            "total_tokens": total,
            "remaining_tokens": remaining,
            "cost_weight_boost": float(budget_cfg.get("cost_weight_boost", 0.35) or 0.35),
        }

    def _consume_tokens(self, model_id: str, prompt: str, output: str) -> None:
        prompt_tokens = float(len(prompt)) / 4.0
        output_tokens = float(len(output)) / 4.0
        tokens = prompt_tokens + output_tokens
        usage = self._token_usage_by_model.setdefault(model_id, {"input_tokens": 0.0, "output_tokens": 0.0})
        usage["input_tokens"] += prompt_tokens
        usage["output_tokens"] += output_tokens
        if tokens <= 0:
            return
        if self._token_budget_total is None:
            return
        self._token_budget_used += tokens
        remaining = max(0.0, float(self._token_budget_total) - self._token_budget_used)
        if self._budget_context:
            self._budget_context["remaining_tokens"] = remaining

    def run(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> PipelineResult:
        run_id = run_id or self.store.create_run(query, meta=meta)
        audit = AuditLog(self.store.run_dir(run_id) / "audit.jsonl")
        self._audit = audit
        self._run_id = run_id
        self._run_start_time = time.time()
        self._run_models_used = set()
        self._context_char_limit = None
        self._agent_sync_cache = {}
        self._agent_sync_meta = {}
        self._token_usage_by_model = {}
        self._image_usage = {}
        self._vision_usage = {}
        if meta and meta.get("context_char_limit"):
            try:
                self._context_char_limit = int(meta.get("context_char_limit"))
            except Exception:
                self._context_char_limit = None
        if meta:
            try:
                self.store.update_meta(run_id, meta)
            except Exception:
                pass
        audit.log("mcp.available", {"servers": list(load_mcp_servers().keys())})
        audit.log("run.start", {"query": query, "meta": meta or {}, "timeout": self.config.run_timeout_seconds})
        try:
            self.store.append_event(run_id, {"phase": "preflight", "status": "start"})
            self._calibrate_models(run_id)
            audit.log("preflight.complete")
            self._check_timeout("preflight")

            self.store.append_event(run_id, {"phase": "requirements", "status": "start"})
            requirements = self._check_required_models()
            if not requirements.get("ok", True):
                consensus = self._requirements_failed_answer(query, requirements)
                cost_estimate = self._estimate_run_cost()
                previous = self._latest_for_meta(meta)
                artifacts = {
                    "requirements": requirements,
                    "reconcile": {
                        "previous_run_id": previous.get("id") if previous else None,
                        "changed": True,
                    },
                    "cost_estimate": cost_estimate,
                }
                self.store.finalize_run(run_id, consensus, artifacts)
                self.store.append_event(run_id, {"phase": "requirements", "status": "failed", "details": requirements})
                audit.log("requirements.failed", requirements)
                return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)
            self.store.append_event(run_id, {"phase": "requirements", "status": "ok", "details": requirements})
            audit.log("requirements.ok", requirements)
            self._check_timeout("requirements")

            self.store.append_event(run_id, {"phase": "route", "status": "start"})
            self._init_token_budget(meta)
            agent_set = None
            if meta and meta.get("agent_set"):
                agent_set = self._resolve_agent_set(str(meta.get("agent_set")))
            role_overrides = None
            if meta and isinstance(meta.get("role_overrides"), dict):
                role_overrides = meta.get("role_overrides")
            domain_override = meta.get("domain") if meta else None
            route = self._route_query(
                query,
                collections,
                budget_context=self._budget_context,
                agent_set=agent_set,
                role_overrides=role_overrides,
                domain_override=domain_override,
            )
            plan_details = self._plan_details(route.get("plan", {}))
            route["plan_details"] = plan_details
            panel_models = list(route.get("panel_models") or [])
            deliberation_cfg = self.config.raw.get("deliberation", {}) or {}
            panel_cfg = deliberation_cfg.get("panel", {}) if isinstance(deliberation_cfg, dict) else {}
            filter_panel = bool(panel_cfg.get("filter_unavailable", True))
            if panel_models and filter_panel:
                required_cfg = self.config.raw.get("required_models", {}) or {}
                ping_prompt = str(panel_cfg.get("ping_prompt") or required_cfg.get("ping_prompt") or "Return only: OK")
                ping_timeout = panel_cfg.get("ping_timeout_seconds")
                try:
                    ping_timeout = int(ping_timeout) if ping_timeout is not None else None
                except Exception:
                    ping_timeout = None
                panel_health = self._check_model_list(panel_models, ping_prompt, timeout_seconds=ping_timeout)
                required_set = set(self._required_model_ids())
                filtered = [mid for mid in panel_models if mid in panel_health.get("available", []) or mid in required_set]
                if filtered != panel_models:
                    route["panel_models"] = filtered
                route["panel_health"] = panel_health
                if self._run_id:
                    self.store.append_event(self._run_id, {
                        "phase": "panel",
                        "status": "filtered" if filtered != panel_models else "ok",
                        "details": panel_health,
                    })
            route["panel_details"] = self._panel_details(route.get("panel_models") or [])
            self.store.append_event(run_id, {"phase": "route", "status": "done", "route": route, "models": plan_details})
            audit.log("route.decided", route)
            self._check_timeout("route")

            required_cfg = self.config.raw.get("required_models", {}) or {}
            if required_cfg.get("enabled", False):
                required_models = list(required_cfg.get("models") or [])
                if required_models:
                    assigned = set((route.get("plan") or {}).values())
                    assigned.update(route.get("panel_models") or [])
                    missing_in_plan = [mid for mid in required_models if mid not in assigned]
                    if missing_in_plan:
                        requirements = {
                            "ok": False,
                            "required": required_models,
                            "missing_in_plan": missing_in_plan,
                        }
                        consensus = self._requirements_failed_answer(query, requirements)
                        cost_estimate = self._estimate_run_cost()
                        previous = self._latest_for_meta(meta)
                        artifacts = {
                            "requirements": requirements,
                            "route": route,
                            "reconcile": {
                                "previous_run_id": previous.get("id") if previous else None,
                                "changed": True,
                            },
                            "cost_estimate": cost_estimate,
                        }
                        self.store.finalize_run(run_id, consensus, artifacts)
                        self.store.append_event(run_id, {"phase": "requirements", "status": "failed", "details": requirements})
                        audit.log("requirements.failed", requirements)
                        return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)

            self.store.append_event(run_id, {"phase": "retrieve", "status": "start"})
            context = self._retrieve_context(query, route, meta)
            self._check_timeout("retrieve")
            stats = context.get("stats", {})
            self.store.append_event(run_id, {"phase": "retrieve", "status": "done", "context": {"rag": len(context["rag"]), "nas": len(context["nas"]), "evidence": stats.get("evidence_count", 0)}})
            audit.log("retrieve.complete", {
                "rag_count": len(context["rag"]),
                "nas_count": len(context["nas"]),
                "evidence_count": stats.get("evidence_count", 0),
                "pdf_ratio": stats.get("pdf_ratio", 0),
                "max_signal_score": stats.get("max_signal_score", 0),
                "input_path": stats.get("input_path"),
                "rag_errors": stats.get("rag_errors", []),
                "source_errors": stats.get("source_errors", []),
                "rag_samples": context["rag"][:3],
                "nas_samples": context["nas"][:3],
                "source_samples": context.get("sources", [])[:2],
            })
            if meta and meta.get("output_type"):
                context["output_type"] = str(meta.get("output_type"))
            output_meta = self._output_meta(context.get("output_type"), route)
            if output_meta:
                context["output"] = output_meta
            quality = self._evaluate_quality(context)
            audit.log("quality.check", quality)
            self.store.append_event(run_id, {"phase": "quality", **quality})
            issues = quality.get("issues", [])
            fail_on_rag_errors = bool(self.config.quality.get("fail_on_rag_errors", False))
            block_on_insufficient = bool(self.config.quality.get("block_on_insufficient", False))
            should_block = block_on_insufficient and (quality.get("insufficient") or (fail_on_rag_errors and "rag_errors" in issues))
            if should_block and bool(self.config.quality.get("strict", True)):
                consensus = self._insufficient_evidence_answer(query, quality)
                output_path, output_artifacts = self._write_output_file(run_id, consensus, context.get("output_type"), context)
                cost_estimate = self._estimate_run_cost()
                previous = self._latest_for_meta(meta)
                artifacts = {
                    "route": route,
                    "context": context,
                    "deliberation": {},
                    "quality": quality,
                    "reconcile": {
                        "previous_run_id": previous.get("id") if previous else None,
                        "changed": True,
                    },
                    "cost_estimate": cost_estimate,
                }
                if output_path:
                    artifacts["output_path"] = output_path
                if output_artifacts:
                    artifacts["output_artifacts"] = output_artifacts
                self.store.finalize_run(run_id, consensus, artifacts)
                audit.log("settlement.complete", {
                    "consensus": consensus,
                    "reconcile": artifacts["reconcile"],
                    "quality": quality,
                    "note": "insufficient_evidence",
                })
                return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)

            self._check_timeout("quality")
            self.store.append_event(run_id, {"phase": "deliberate", "status": "start"})
            anneal = self._anneal_consensus(
                query=query,
                context=context,
                route=route,
                collections=collections,
                output_type=context.get("output_type"),
                agent_set=agent_set,
                role_overrides=role_overrides,
            )
            deliberation = anneal.get("deliberation") or {}
            route = anneal.get("route") or route
            anneal_artifacts = anneal.get("annealing") if isinstance(anneal, dict) else None
            self._apply_output_meta(context, context.get("output_type"), route)
            self.store.append_event(run_id, {"phase": "deliberate", "status": "done", "agreement": deliberation.get("agreement")})
            audit.log("deliberate.complete", {
                "reasoner": deliberation.get("reasoner", ""),
                "critic": deliberation.get("critic", ""),
                "disagreements": deliberation.get("disagreements", []),
                "rounds": deliberation.get("rounds", []),
                "agreement": deliberation.get("agreement"),
            })
            self._check_timeout("deliberate")

            deliberation = self._resolve_consensus(query, context, deliberation, route)

            # HR-1: Minimum 2 intelligent models gate
            min_intelligent = int(self.config.raw.get("deliberation", {}).get("min_intelligent_models", 2))
            intelligent_providers = self._count_intelligent_providers()
            if len(intelligent_providers) < min_intelligent:
                raise InsufficientModelsError(
                    f"Only {intelligent_providers or set()} participated. "
                    f"Consensus requires >= {min_intelligent} of: claude, codex, gemini."
                )

            fail_on_no_agreement = bool(self.config.raw.get("deliberation", {}).get("fail_on_no_agreement", False))
            if fail_on_no_agreement and not deliberation.get("agreement", False):
                consensus = self._insufficient_evidence_answer(query, quality, note="no_agreement")
                output_path, output_artifacts = self._write_output_file(run_id, consensus, context.get("output_type"), context)
                cost_estimate = self._estimate_run_cost()
                previous = self._latest_for_meta(meta)
                reconcile = {
                    "previous_run_id": previous.get("id") if previous else None,
                    "changed": True,
                }
                artifacts = {
                    "route": route,
                    "context": context,
                    "deliberation": deliberation,
                    "quality": quality,
                    "reconcile": reconcile,
                    "cost_estimate": cost_estimate,
                }
                if anneal_artifacts:
                    artifacts["annealing"] = anneal_artifacts
                if output_path:
                    artifacts["output_path"] = output_path
                if output_artifacts:
                    artifacts["output_artifacts"] = output_artifacts
                self.store.finalize_run(run_id, consensus, artifacts)
                audit.log("settlement.complete", {
                    "consensus": consensus,
                    "reconcile": artifacts["reconcile"],
                    "quality": quality,
                    "note": "no_agreement",
                })
                return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)

            consensus = self._summarize(query, context, deliberation, route, quality)
            output_path, output_artifacts = self._write_output_file(run_id, consensus, context.get("output_type"), context)
            cost_estimate = self._estimate_run_cost()
            previous = self._latest_for_meta(meta)
            prev_answer = str(previous.get("consensus", {}).get("answer", "")) if previous else ""
            curr_answer = str(consensus.get("answer", ""))
            reconcile = {
                "previous_run_id": previous.get("id") if previous else None,
                "changed": bool(previous and prev_answer != curr_answer),
                "similarity_score": round(self._text_similarity(prev_answer, curr_answer), 4) if prev_answer and curr_answer else None,
            }
            artifacts = {
                "route": route,
                "context": context,
                "deliberation": deliberation,
                "quality": quality,
                "reconcile": reconcile,
                "cost_estimate": cost_estimate,
            }
            if anneal_artifacts:
                artifacts["annealing"] = anneal_artifacts
            if output_path:
                artifacts["output_path"] = output_path
            if output_artifacts:
                artifacts["output_artifacts"] = output_artifacts
            self.store.finalize_run(run_id, consensus, artifacts)
            audit.log("settlement.complete", {
                "consensus": consensus,
                "reconcile": reconcile,
            })

            # Log to memory MCP for cross-session learning
            self._log_to_memory(query, route, consensus, quality)

            # Post decision to agent-sync bus so other agents know what Conclave decided
            self._post_to_agent_sync(query, consensus, route)

            return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)
        except RequiredModelError as exc:
            requirements = {
                "ok": False,
                "required": self._required_model_ids(),
                "failed": [{"id": "required-model", "reason": str(exc)}],
            }
            consensus = self._requirements_failed_answer(query, requirements)
            output_type = meta.get("output_type") if meta else None
            output_path, output_artifacts = self._write_output_file(run_id, consensus, output_type)
            cost_estimate = self._estimate_run_cost()
            previous = self._latest_for_meta(meta)
            artifacts = {
                "requirements": requirements,
                "reconcile": {
                    "previous_run_id": previous.get("id") if previous else None,
                    "changed": True,
                },
                "cost_estimate": cost_estimate,
            }
            if output_path:
                artifacts["output_path"] = output_path
            if output_artifacts:
                artifacts["output_artifacts"] = output_artifacts
            self.store.finalize_run(run_id, consensus, artifacts)
            self.store.append_event(run_id, {"phase": "requirements", "status": "failed", "details": requirements})
            audit.log("requirements.failed", requirements)
            return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)
        except RunTimeoutError as exc:
            error_msg = f"run interrupted by timeout"
            self.store.fail_run(run_id, error_msg)
            audit.log("run.timeout", {"error": str(exc), "elapsed": time.time() - self._run_start_time})
            raise
        except Exception as exc:
            self.store.fail_run(run_id, str(exc))
            audit.log("run.failed", {"error": str(exc)})
            raise
        finally:
            self._audit = None
            self._run_id = None
            self._run_start_time = 0.0
            self._context_char_limit = None
            # Close MCP connections
            try:
                self.mcp.close_all()
            except Exception:
                pass

    def _calibrate_models(self, run_id: str) -> None:
        calibration = self.config.calibration
        if not calibration.get("enabled", True):
            return

        # Check cache - skip calibration if recent enough
        now = time.time()
        if self._calibration_cache and (now - self._calibration_cache_time) < self._calibration_cache_ttl:
            # Apply cached metrics
            for model_id, observation in self._calibration_cache.items():
                try:
                    self.registry.update_metrics(model_id, observation)
                except Exception:
                    pass
            self.store.append_event(run_id, {"phase": "calibration", "status": "cached", "models": len(self._calibration_cache)})
            return

        # Get role-overridden models to skip (they're always used regardless)
        role_overrides = set((self.config.planner.get("role_overrides") or {}).values())

        providers = calibration.get("providers") or ["ollama"]
        max_seconds = float(calibration.get("max_seconds", 20))
        prompt = calibration.get("ping_prompt", "Return only: OK")
        self_report_cfg = self.config.raw.get("self_report", {}) or {}
        self_report_enabled = bool(self_report_cfg.get("enabled", False))
        self_report_providers = self_report_cfg.get("providers") or providers
        self_report_max_seconds = float(self_report_cfg.get("max_seconds", 30))
        self_report_start = time.perf_counter()
        start = time.perf_counter()
        new_cache: Dict[str, Dict[str, Any]] = {}

        for card in self.registry.list_models():
            if time.perf_counter() - start > max_seconds:
                break
            model_id = card.get("id", "")

            # Skip role-overridden models - they're always used regardless of calibration
            if model_id in role_overrides:
                self.store.append_event(run_id, {"phase": "calibration", "model": model_id, "skipped": "role_override"})
                continue

            provider = str(card.get("provider") or model_id.split(":", 1)[0])
            if provider not in providers:
                continue
            if model_id.startswith("ollama:"):
                model_name = model_id.split(":", 1)[1]
                result = self.ollama.generate(model_name, prompt, temperature=0)
            elif model_id.startswith("cli:"):
                result = self._call_cli_model(model_id, prompt, role="calibration")
            else:
                continue
            observation = {
                "p50_latency_ms": round(result.duration_ms, 2),
                "ok": result.ok,
                "error_rate": 0.0 if result.ok else 1.0,
                "timeout_rate": 0.0,
            }
            new_cache[model_id] = observation
            try:
                self.registry.update_metrics(model_id, observation)
            except Exception:
                pass
            self.store.append_event(run_id, {"phase": "calibration", "model": model_id, "ok": result.ok})

            if self_report_enabled:
                if time.perf_counter() - self_report_start > self_report_max_seconds:
                    continue
                if provider not in self_report_providers:
                    continue
                if not self._should_self_report(card, self_report_cfg):
                    continue
                report = self._self_report_capabilities(model_id, card, self_report_cfg)
                if report and self._run_id:
                    self.store.append_event(self._run_id, {
                        "phase": "self_report",
                        "model": model_id,
                        "ok": True,
                        "updated": True,
                        "fields": sorted(report.keys()),
                    })
                elif self._run_id:
                    self.store.append_event(self._run_id, {
                        "phase": "self_report",
                        "model": model_id,
                        "ok": False,
                    })

        # Update cache
        self._calibration_cache = new_cache
        self._calibration_cache_time = now

    def _route_query(
        self,
        query: str,
        collections: Optional[List[str]],
        budget_context: Optional[Dict[str, Any]] = None,
        plan_noise: float = 0.0,
        rng: random.Random | None = None,
        agent_set: Optional[Dict[str, Any]] = None,
        role_overrides: Optional[Dict[str, str]] = None,
        domain_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        q = query.lower()

        # 1. Caller-specified domain takes absolute priority
        if domain_override:
            domain = domain_override
        else:
            # 2. Match against configurable keyword_map from config
            domain = "general"
            routing_cfg = self.config.raw.get("routing", {}) or {}
            keyword_map = routing_cfg.get("domain_keywords", {}) or {}
            priority = routing_cfg.get("domain_priority") or list(keyword_map.keys())
            for key in priority:
                keywords = [str(item).lower() for item in (keyword_map.get(key) or [])]
                if not keywords:
                    continue
                if any(word in q for word in keywords):
                    domain = key
                    break

        needs_tax = any(word in q for word in [
            "tax", "irs", "1099", "basis", "deduction", "section",
            "schedule f", "passive", "material participation", "hobby loss",
        ])
        base = collections or self.config.rag.get("domain_collections", {}).get(domain) or self.config.rag.get("default_collections", [])
        required_collections: List[str] = []
        if needs_tax:
            tax_collections = self.config.rag.get("domain_collections", {}).get("tax", [])
            for item in tax_collections:
                if item not in base:
                    base.append(item)
            required_collections = list(tax_collections)
        allowlist = self.config.rag.get("domain_allowlist", {}).get(domain, [])
        selected, catalog = self._expand_collections(domain, base, explicit=bool(collections), allowlist=allowlist)
        if allowlist and (self.config.rag.get("enforce_allowlist", True) or not collections):
            filtered = [item for item in selected if item in allowlist]
            if filtered:
                selected = filtered
        roles = list(agent_set.get("roles") if isinstance(agent_set, dict) and agent_set.get("roles") else ["router", "reasoner", "critic", "summarizer"])
        eligible_models = None
        if isinstance(agent_set, dict) and agent_set.get("eligible_models"):
            eligible_models = [str(mid) for mid in agent_set.get("eligible_models") if str(mid)]
        registry = self.registry.list_models()
        if eligible_models:
            registry = [card for card in registry if card.get("id") in eligible_models]
        base_overrides = dict(self.planner.role_overrides or {})
        if role_overrides:
            base_overrides.update({str(k): str(v) for k, v in role_overrides.items() if str(k) and str(v)})
        planner = self.planner.with_overrides(base_overrides) if base_overrides else self.planner
        plan_with_rationale = planner.plan_with_rationale(
            roles,
            registry,
            budget_context=budget_context,
            noise=plan_noise,
            rng=rng,
        )
        if isinstance(agent_set, dict) and agent_set.get("panel_models"):
            panel_models = [str(mid) for mid in agent_set.get("panel_models") if self.registry.get_model(str(mid))]
        else:
            panel_models = self._resolve_panel_models(plan_with_rationale.get("assignments", {}))
        panel_require_all = agent_set.get("panel_require_all") if isinstance(agent_set, dict) else None
        panel_min_ratio = agent_set.get("panel_min_ratio") if isinstance(agent_set, dict) else None
        return {
            "domain": domain,
            "collections": selected,
            "required_collections": required_collections,
            "allowlist": allowlist,
            "roles": roles,
            "plan": plan_with_rationale["assignments"],
            "rationale": plan_with_rationale["rationale"],
            "panel_models": panel_models,
            "panel_require_all": panel_require_all,
            "panel_min_ratio": panel_min_ratio,
            "rag_catalog": catalog,
            "role_overrides": base_overrides,
        }

    def _resolve_panel_models(self, plan: Dict[str, Any]) -> List[str]:
        deliberation_cfg = self.config.raw.get("deliberation", {}) or {}
        panel_cfg = deliberation_cfg.get("panel", {}) if isinstance(deliberation_cfg, dict) else {}
        model_ids = list(panel_cfg.get("model_ids") or [])
        if not model_ids:
            required_cfg = self.config.raw.get("required_models", {}) or {}
            model_ids = list(required_cfg.get("models") or [])
        include_plan_models = bool(panel_cfg.get("include_plan_models", True))
        if include_plan_models:
            for mid in (plan or {}).values():
                if mid and mid not in model_ids:
                    model_ids.append(mid)
        # Keep only models that exist in registry
        filtered = []
        for mid in model_ids:
            if self.registry.get_model(mid):
                filtered.append(mid)
        return filtered

    def _should_self_report(self, card: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
        ttl_seconds = float(cfg.get("ttl_seconds", 86400))
        last = None
        meta = card.get("self_report") or {}
        if isinstance(meta, dict):
            last = meta.get("updated_at")
        if isinstance(last, (int, float)):
            if time.time() - float(last) < ttl_seconds:
                return False
        model_id = card.get("id", "")
        ok, _ = self._model_available(model_id)
        return ok

    def _self_report_capabilities(
        self,
        model_id: str,
        card: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        base_caps = card.get("capabilities", {}) or {}
        prompt = cfg.get("prompt") or (
            "You are calibrating your own capability card for routing decisions.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            "  \"capabilities\": {\n"
            "    \"text_reasoning\": true|false,\n"
            "    \"code_generation\": \"low\"|\"medium\"|\"high\",\n"
            "    \"code_review\": \"low\"|\"medium\"|\"high\",\n"
            "    \"tool_use\": true|false,\n"
            "    \"image_understanding\": \"none\"|\"limited\"|\"full\",\n"
            "    \"json_reliability\": \"low\"|\"medium\"|\"high\"\n"
            "  },\n"
            "  \"strengths\": [\"...\"],\n"
            "  \"weaknesses\": [\"...\"],\n"
            "  \"confidence\": \"low\"|\"medium\"|\"high\"\n"
            "}\n"
        )
        out = self._call_model(model_id, prompt, role="self_report")
        parsed = self._parse_json_payload(out)
        if not parsed:
            return None
        caps = parsed.get("capabilities") if isinstance(parsed, dict) else None
        if not isinstance(caps, dict):
            caps = parsed if isinstance(parsed, dict) else {}
        override = self._normalize_capabilities(caps, base_caps)
        meta = {
            "updated_at": time.time(),
            "strengths": parsed.get("strengths") if isinstance(parsed, dict) else None,
            "weaknesses": parsed.get("weaknesses") if isinstance(parsed, dict) else None,
            "confidence": parsed.get("confidence") if isinstance(parsed, dict) else None,
        }
        try:
            self.registry.update_capabilities_override(model_id, override, meta)
        except Exception:
            return None
        return override

    def _parse_json_payload(self, text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        blob = text[start:end + 1]
        try:
            return json.loads(blob)
        except Exception:
            return None

    def _normalize_capabilities(self, caps: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
        def _bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                val = value.strip().lower()
                if val in {"true", "yes", "y", "1"}:
                    return True
                if val in {"false", "no", "n", "0"}:
                    return False
            return default

        def _level(value: Any, default: str) -> str:
            if isinstance(value, str):
                val = value.strip().lower()
                if val in {"low", "medium", "high"}:
                    return val
            if isinstance(value, (int, float)):
                if value < 0.34:
                    return "low"
                if value < 0.67:
                    return "medium"
                return "high"
            return default

        def _image(value: Any, default: str) -> str:
            if isinstance(value, str):
                val = value.strip().lower()
                if val in {"none", "limited", "full"}:
                    return val
            return default

        override = dict(base)
        if "text_reasoning" in caps:
            override["text_reasoning"] = _bool(caps.get("text_reasoning"), bool(base.get("text_reasoning", True)))
        if "tool_use" in caps:
            override["tool_use"] = _bool(caps.get("tool_use"), bool(base.get("tool_use", False)))
        if "code_generation" in caps:
            override["code_generation"] = _level(caps.get("code_generation"), str(base.get("code_generation", "low")))
        if "code_review" in caps:
            override["code_review"] = _level(caps.get("code_review"), str(base.get("code_review", "low")))
        if "json_reliability" in caps:
            override["json_reliability"] = _level(caps.get("json_reliability"), str(base.get("json_reliability", "medium")))
        if "image_understanding" in caps:
            override["image_understanding"] = _image(caps.get("image_understanding"), str(base.get("image_understanding", "none")))
        return override

    def _retrieve_context(self, query: str, route: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rag_results: List[Dict[str, Any]] = []
        rag_cfg = self.config.rag
        max_per_collection = int(rag_cfg.get("max_results_per_collection", 8))
        prefer_non_pdf = bool(rag_cfg.get("prefer_non_pdf", False))
        semantic = rag_cfg.get("semantic")
        disable_domains = set(rag_cfg.get("disable_domains", []) or [])
        if route.get("domain") not in disable_domains:
            for coll in route.get("collections", []):
                rag_results.extend(self.rag.search(query, collection=coll, limit=max_per_collection, semantic=semantic))
        rag_results = self._filter_rag_results(rag_results)
        allowlist = route.get("allowlist") or []
        if allowlist:
            rag_results = [item for item in rag_results if item.get("collection") in allowlist]
        nas_results = []
        file_results = self.rag.search_files(query, limit=10)
        if self.config.index.get("enabled", True):
            auto_build = bool(self.config.index.get("auto_build", False))
            if self.index.db_path.exists() or auto_build:
                self._maybe_refresh_index()
                nas_results = self.index.search(query, limit=10)
        combined_files = file_results + nas_results
        if prefer_non_pdf:
            rag_results.sort(key=lambda x: str(x.get("path") or x.get("name") or "").lower().endswith(".pdf"))
        user_inputs = self._load_user_input(meta)
        if not user_inputs and query:
            user_inputs = [{
                "title": "prompt",
                "snippet": query,
                "full_text": query,
                "collection": "user-input",
                "source": "user",
            }]
        source_items: List[Dict[str, Any]] = []
        source_errors: List[Dict[str, Any]] = []
        mcp_results: Dict[str, Any] = {}
        domain = route.get("domain")
        input_artifacts: List[Dict[str, Any]] = []
        instructions = user_inputs[0].get("full_text") if user_inputs else ""
        artifact_paths = self._extract_artifact_paths(str(instructions))
        if artifact_paths:
            input_artifacts = self._summarize_artifacts(artifact_paths)
            if input_artifacts:
                source_items.extend(input_artifacts)

        output_type = meta.get("output_type") if meta else None
        image_paths = [item.get("path") for item in input_artifacts if item.get("kind") == "image" and item.get("path")]
        if image_paths and self._output_requires(output_type, "image_understanding"):
            vision_prompt = (
                "Summarize the attached photos for design decisions. "
                "Note lighting, dominant materials, current cabinet color/finish, wall/trim colors, "
                "flooring, counters, and any constraints that affect cabinet color choices."
            )
            vision_summary, provider = self._vision_summary(vision_prompt, image_paths)
            if vision_summary:
                if provider:
                    self._record_vision_usage(provider, len(image_paths))
                context["vision_summary"] = vision_summary
                source_items.append({
                    "path": f"{provider}://vision/summary",
                    "title": f"Vision Summary ({provider})",
                    "snippet": vision_summary[:1600],
                    "collection": f"vision-{provider}",
                    "source": "vision",
                })

        previous = self._latest_for_meta(meta)
        previous_run = None
        if previous and previous.get("consensus", {}).get("answer"):
            previous_run = {
                "id": previous.get("id"),
                "created_at": previous.get("created_at"),
                "agreement": (previous.get("artifacts") or {}).get("deliberation", {}).get("agreement"),
                "answer": str(previous.get("consensus", {}).get("answer", ""))[:2400],
            }

        on_demand = self.verifier.fetch(domain or "general", query)
        if on_demand.items:
            source_items.extend(on_demand.items)
        if on_demand.errors:
            source_errors.extend(on_demand.errors)
        if self._audit and (on_demand.items or on_demand.errors):
            self._audit.log("sources.on_demand", {
                "domain": domain,
                "items": len(on_demand.items),
                "errors": on_demand.errors,
            })

        evidence_limit = None
        if meta and meta.get("evidence_limit"):
            try:
                evidence_limit = int(meta.get("evidence_limit"))
            except Exception:
                evidence_limit = None
        required_collections = route.get("required_collections", [])
        evidence, stats = self._select_evidence(
            rag_results,
            combined_files,
            limit=evidence_limit or 12,
            preferred_collections=route.get("collections", []),
            required_collections=required_collections,
            domain=route.get("domain"),
            domain_paths=self.config.quality.get("domain_paths", {}),
            collection_reliability=self.config.rag.get("collection_reliability", {}),
            user_items=user_inputs,
            source_items=source_items,
        )
        rag_errors = self.rag.drain_errors()
        if rag_errors:
            stats["rag_errors"] = rag_errors
        if source_errors:
            stats["source_errors"] = source_errors
        if on_demand.items:
            stats["on_demand_count"] = len(on_demand.items)
        if on_demand.errors:
            stats["on_demand_errors"] = on_demand.errors
        if user_inputs:
            stats["input_path"] = user_inputs[0].get("path")
        return {
            "rag": rag_results,
            "nas": combined_files,
            "sources": source_items,
            "evidence": evidence,
            "stats": stats,
            "user_inputs": user_inputs,
            "input_artifacts": input_artifacts,
            "previous_run": previous_run,
            "agent_sync": self._agent_sync_summary(),
        }

    def _extract_focus_queries(self, instructions: str) -> list[str]:
        if not instructions:
            return []
        import re
        from pathlib import Path
        queries: list[str] = []
        paths = re.findall(r"(?:programs|crates)/[A-Za-z0-9_./-]+\.rs", instructions)
        for path in paths:
            queries.append(path)
            queries.append(Path(path).name)
        questions = re.findall(r"\\*\\*Question:\\*\\*\\s*(.+)", instructions)
        for q in questions:
            queries.append(q.strip())
        artifact_paths = re.findall(r"^-\\s*(\\S+)$", instructions, re.MULTILINE)
        for item in artifact_paths:
            if item.startswith("http"):
                continue
            queries.append(item)
            queries.append(Path(item).name)
        constants = re.findall(r"([A-Z_]{3,})\\s*=", instructions)
        queries.extend(constants)
        unique = []
        seen = set()
        for item in queries:
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:12]

    def _latest_for_meta(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any] | None:
        """Find the most recent prior run that is relevant to the current query.

        Only returns a previous run if:
        1. Same prompt_id (explicit re-run of the same prompt), OR
        2. Same domain AND meaningful query overlap (>30% word overlap).

        This prevents unrelated prior runs (e.g. data structures) from
        contaminating the context of a new query (e.g. resume review).
        """
        if meta and meta.get("prompt_id"):
            latest = self.store.latest_for_prompt(str(meta.get("prompt_id")))
            if latest:
                return latest
        # Don't blindly return the global latest — it may be completely unrelated.
        # Only return it if the caller provided a prompt_id that matched above.
        return None

    def _extract_file_paths(self, instructions: str) -> list[str]:
        if not instructions:
            return []
        import re
        paths = re.findall(r"(?:programs|crates)/[A-Za-z0-9_./-]+\.rs", instructions)
        unique = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return unique

    def _is_health_prompt(self, prompt: str) -> bool:
        trimmed = prompt.strip().lower()
        return trimmed in {
            "return only: ok",
            "return only ok",
            "reply with just the word 'ok'",
            "reply with just the word \"ok\"",
        }

    def _agent_sync_config(self) -> Dict[str, Any]:
        cfg = self.config.raw.get("agent_sync", {}) or {}
        mode = str(cfg.get("mode", "bus")).strip().lower()
        include_base = mode not in {"bus", "bus-only"}
        try:
            max_cloud = int(cfg.get("max_chars_cloud", 1200))
        except Exception:
            max_cloud = 1200
        try:
            max_local = int(cfg.get("max_chars_local", 800))
        except Exception:
            max_local = 800
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "mode": mode,
            "include_base": include_base,
            "max_chars_cloud": max_cloud,
            "max_chars_local": max_local,
        }

    def _get_agent_sync_context(self, target_class: str) -> str:
        cfg = self._agent_sync_config()
        if not cfg.get("enabled", False):
            return ""
        include_base = bool(cfg.get("include_base", False))
        max_chars = int(cfg.get("max_chars_cloud" if target_class == "cloud" else "max_chars_local", 1200))
        key = (target_class, max_chars, include_base)
        if key in self._agent_sync_cache:
            return self._agent_sync_cache[key]
        if not shutil.which("context-injector"):
            self._agent_sync_cache[key] = ""
            return ""
        cmd = ["context-injector", "--for", target_class, "--max-chars", str(max_chars)]
        if not include_base:
            cmd.append("--no-base")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                self._agent_sync_cache[key] = ""
                return ""
            context = (result.stdout or "").strip()
            self._agent_sync_cache[key] = context
            contexts = self._agent_sync_meta.setdefault("contexts", {})
            contexts[target_class] = {
                "chars": len(context),
                "max_chars": max_chars,
                "include_base": include_base,
            }
            self._agent_sync_meta.update({
                "enabled": True,
                "mode": cfg.get("mode", "bus"),
            })
            return context
        except Exception:
            self._agent_sync_cache[key] = ""
            return ""

    def _agent_sync_summary(self) -> Dict[str, Any]:
        cfg = self._agent_sync_config()
        if not cfg.get("enabled", False):
            return {"enabled": False}
        summary = {
            "enabled": True,
            "mode": cfg.get("mode", "bus"),
        }
        contexts = self._agent_sync_meta.get("contexts")
        if contexts:
            summary["contexts"] = contexts
        return summary

    def _post_to_agent_sync(self, query: str, consensus: Dict[str, Any], route: Dict[str, Any]) -> None:
        """Post consensus decision to agent-sync bus so other agents know what Conclave decided."""
        cfg = self._agent_sync_config()
        if not cfg.get("enabled", False) or not cfg.get("post_decisions", True):
            return
        agent_sync_path = Path.home() / "bin" / "agent-sync"
        if not agent_sync_path.exists():
            return
        try:
            domain = route.get("domain", "general")
            confidence = consensus.get("confidence", "medium")
            answer = consensus.get("answer", "")[:800]
            insufficient = consensus.get("insufficient_evidence", False)

            if insufficient:
                subject = f"Conclave: {domain} - insufficient evidence"
                body = f"Query: {query[:100]}\nResult: Need more evidence"
            else:
                subject = f"Conclave: {domain} decision ({confidence})"
                body = f"Query: {query[:100]}\n\nDecision:\n{answer}"

            subprocess.run(
                [
                    str(agent_sync_path), "post",
                    "--from", "conclave",
                    "--to", "all",
                    "--type", "decision",
                    "--subject", subject,
                    "--body", body,
                    "--auto-inject", "true",
                    "--model-class", "both",
                    "--ttl-hours", "12",
                ],
                capture_output=True,
                timeout=5,
            )
        except Exception as e:
            self.logger.debug(f"Agent sync post failed: {e}")

    def _apply_agent_sync(self, prompt: str, model_id: str, role: Optional[str]) -> str:
        if role in {"requirements", "calibration", "self_report"}:
            return prompt
        if self._is_health_prompt(prompt):
            return prompt
        cfg = self._agent_sync_config()
        if not cfg.get("enabled", False):
            return prompt
        target_class = "local"
        if model_id.startswith("cli:"):
            target_class = "cloud"
            card = self.registry.get_model(model_id) or {}
            command = " ".join(card.get("command") or [])
            if "with-context" in command:
                return prompt
        context = self._get_agent_sync_context(target_class)
        if not context:
            return prompt
        return f"{context}\n\n# Task\n\n{prompt}"

    def _is_ok_response(self, text: str) -> bool:
        if not text:
            return False
        for line in text.splitlines():
            if line.strip().upper().rstrip(".!") == "OK":
                return True
        return False

    def _looks_like_agents_missing(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        if "agents.md" not in lower:
            return False
        return any(token in lower for token in (
            "couldn't find",
            "could not find",
            "does not exist",
            "doesn’t exist",
            "not found",
            "missing",
        ))

    def _check_required_models(self) -> Dict[str, Any]:
        cfg = self.config.raw.get("required_models", {}) or {}
        enabled = bool(cfg.get("enabled", False))
        required = list(cfg.get("models") or [])
        prompt = str(cfg.get("ping_prompt", "Return only: OK"))
        if not enabled or not required:
            return {"ok": True, "required": required, "available": [], "missing": [], "failed": []}

        missing: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        available: List[str] = []

        for model_id in required:
            card = self.registry.get_model(model_id)
            if not card:
                missing.append({"id": model_id, "reason": "model not in registry"})
                continue
            ok, reason = self._model_available(model_id, allow_unhealthy=True)
            if not ok:
                missing.append({"id": model_id, "reason": reason})
                continue
            if model_id.startswith("ollama:"):
                model_name = model_id.split(":", 1)[1]
                result = self.ollama.generate(model_name, prompt, temperature=0)
                ok_response = self._is_ok_response(result.text or "")
                observation = {
                    "p50_latency_ms": round(result.duration_ms, 2),
                    "ok": result.ok and ok_response,
                    "error_rate": 0.0 if (result.ok and ok_response) else 1.0,
                    "timeout_rate": 0.0,
                }
                try:
                    self.registry.update_metrics(model_id, observation)
                except Exception:
                    pass
                if not result.ok:
                    failed.append({"id": model_id, "reason": result.error or "unknown error"})
                elif not ok_response:
                    failed.append({"id": model_id, "reason": "health check returned unexpected output"})
                else:
                    available.append(model_id)
                continue
            if model_id.startswith("cli:"):
                result = self._call_cli_model(model_id, prompt, role="requirements")
                ok_response = self._is_ok_response(result.text or "")
                interrupted = result.error in {"exit -15", "exit -9"}
                observation = {
                    "p50_latency_ms": round(result.duration_ms, 2),
                    "ok": result.ok and ok_response,
                    "error_rate": 0.0 if (result.ok and ok_response) else 1.0,
                    "timeout_rate": 1.0 if result.error == "timeout" else 0.0,
                }
                try:
                    self.registry.update_metrics(model_id, observation)
                except Exception:
                    pass
                if not result.ok:
                    failed.append({
                        "id": model_id,
                        "reason": result.error or "unknown error",
                        "stderr": result.stderr,
                        "output": result.text or None,
                        "interrupted": interrupted,
                    })
                elif not ok_response:
                    if result.ok and self._looks_like_agents_missing(result.text or ""):
                        available.append(model_id)
                        continue
                    failed.append({
                        "id": model_id,
                        "reason": "health check returned unexpected output",
                        "stderr": result.stderr,
                        "output": result.text or None,
                    })
                else:
                    available.append(model_id)
                continue
            missing.append({"id": model_id, "reason": "unsupported provider"})

        ok = not missing and not failed
        return {
            "ok": ok,
            "required": required,
            "available": available,
            "missing": missing,
            "failed": failed,
        }

    def _check_model_list(self, model_ids: List[str], prompt: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Check availability for a specific list of models without enforcing requirements."""
        missing: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        available: List[str] = []

        for model_id in model_ids:
            card = self.registry.get_model(model_id)
            if not card:
                missing.append({"id": model_id, "reason": "model not in registry"})
                continue
            ok, reason = self._model_available(model_id, allow_unhealthy=True)
            if not ok:
                missing.append({"id": model_id, "reason": reason})
                continue
            if model_id.startswith("ollama:"):
                model_name = model_id.split(":", 1)[1]
                result = self.ollama.generate(model_name, prompt, temperature=0)
                ok_response = self._is_ok_response(result.text or "")
                observation = {
                    "p50_latency_ms": round(result.duration_ms, 2),
                    "ok": result.ok and ok_response,
                    "error_rate": 0.0 if (result.ok and ok_response) else 1.0,
                    "timeout_rate": 0.0,
                }
                try:
                    self.registry.update_metrics(model_id, observation)
                except Exception:
                    pass
                if not result.ok:
                    failed.append({"id": model_id, "reason": result.error or "unknown error"})
                elif not ok_response:
                    failed.append({"id": model_id, "reason": "health check returned unexpected output"})
                else:
                    available.append(model_id)
                continue
            if model_id.startswith("cli:"):
                result = self._call_cli_model(model_id, prompt, role="requirements", timeout_seconds=timeout_seconds)
                ok_response = self._is_ok_response(result.text or "")
                observation = {
                    "p50_latency_ms": round(result.duration_ms, 2),
                    "ok": result.ok and ok_response,
                    "error_rate": 0.0 if (result.ok and ok_response) else 1.0,
                    "timeout_rate": 1.0 if result.error == "timeout" else 0.0,
                }
                try:
                    self.registry.update_metrics(model_id, observation)
                except Exception:
                    pass
                if not result.ok:
                    stderr = (result.stderr or "").lower()
                    quota_hint = any(token in stderr for token in ("quota", "rate limit", "capacity", "exhausted", "429"))
                    if model_id == "cli:gemini" and (result.error == "timeout" or quota_hint):
                        skipped.append({"id": model_id, "reason": result.error or "quota", "stderr": result.stderr})
                        continue
                    failed.append({"id": model_id, "reason": result.error or "unknown error", "stderr": result.stderr})
                elif not ok_response:
                    failed.append({"id": model_id, "reason": "health check returned unexpected output", "stderr": result.stderr})
                else:
                    available.append(model_id)
                continue
            missing.append({"id": model_id, "reason": "unsupported provider"})

        ok = not missing and not failed
        return {
            "ok": ok,
            "models": model_ids,
            "available": available,
            "missing": missing,
            "failed": failed,
            "skipped": skipped,
        }

    def _required_model_ids(self) -> List[str]:
        cfg = self.config.raw.get("required_models", {}) or {}
        if not cfg.get("enabled", False):
            return []
        return list(cfg.get("models") or [])

    def _requirements_failed_answer(self, query: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        required = requirements.get("required") or []
        missing = requirements.get("missing") or []
        failed = requirements.get("failed") or []
        missing_in_plan = requirements.get("missing_in_plan") or []
        interrupted = any(
            str(item.get("reason", "")).startswith("exit -15")
            or str(item.get("reason", "")).startswith("exit -9")
            or item.get("interrupted")
            for item in failed
        )
        lines = [
            "### Run interrupted during model check" if interrupted else "### Run cancelled: required model unavailable",
            "",
            f"**Query:** {query}",
            "",
            "**Required models:**",
        ]
        if required:
            for model_id in required:
                lines.append(f"- {model_id}")
        else:
            lines.append("- (none configured)")
        lines.append("")
        if missing:
            lines.append("**Missing models:**")
            for item in missing:
                lines.append(f"- {item.get('id')}: {item.get('reason')}")
            lines.append("")
        if failed:
            lines.append("**Failed checks:**")
            for item in failed:
                reason = item.get("reason") or "unknown error"
                detail = item.get("output") or item.get("stderr") or ""
                detail = detail.strip().splitlines()[0] if detail else ""
                if detail:
                    lines.append(f"- {item.get('id')}: {reason}. Details: {detail}")
                else:
                    lines.append(f"- {item.get('id')}: {reason}")
            lines.append("")
        if missing_in_plan:
            lines.append("**Not assigned to any role:**")
            for model_id in missing_in_plan:
                lines.append(f"- {model_id}")
            lines.append("")
        if interrupted:
            lines.extend([
                "**How to fix:**",
                "- Wait for Conclave to finish restarting, then re-run.",
                "- If this repeats without a restart, check the model CLI for stability.",
                "",
                "**Confidence Level: Low**",
            ])
        else:
            lines.extend([
                "**How to fix:**",
                "- Ensure the CLI is installed and on PATH.",
                "- Verify you are logged in (e.g., `claude`, `codex`, `gemini`).",
                "- Update the model command path in `~/.config/conclave/config.yaml` if needed.",
                "",
                "**Confidence Level: Low**",
            ])
        answer = "\n".join(lines)
        return {
            "answer": answer,
            "confidence": "low",
            "confidence_model": "low",
            "confidence_auto": "low",
            "pope": "### Run interrupted during model check" if interrupted else "### Run cancelled: required model unavailable",
            "fallback_used": True,
            "insufficient_evidence": False,
            "requirements_failed": True,
            "interrupted": interrupted,
        }

    def _extract_artifact_paths(self, instructions: str) -> list[str]:
        if not instructions:
            return []
        import re
        # Match absolute paths after "- " and allow any non-whitespace chars
        paths = re.findall(r"^-\s*(/\S+)$", instructions, re.MULTILINE)
        unique = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return unique

    def _summarize_artifacts(self, artifact_paths: List[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for raw in artifact_paths[:12]:
            path = Path(raw)
            if not path.exists():
                continue
            ext = path.suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}:
                kind = "image"
            elif ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                kind = "video"
            elif ext in {".pdf", ".docx", ".txt", ".md", ".rtf"}:
                kind = "document"
            else:
                kind = "file"
            mime = mimetypes.guess_type(str(path))[0] or ""
            size_mb = 0.0
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = 0.0
            size_label = f"{size_mb:.1f} MB" if size_mb else "size unknown"
            snippet = f"User artifact ({kind}): {path.name} ({ext or 'file'}, {size_label}). Attached for reference."
            items.append({
                "path": str(path),
                "title": path.name,
                "snippet": snippet,
                "collection": "user-artifact",
                "source": "user",
                "mime": mime,
                "kind": kind,
            })
        return items

    def _filter_rag_results(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cfg = self.config.rag
        exclude_collections = set(cfg.get("exclude_collections", []) or [])
        exclude_patterns = cfg.get("exclude_path_patterns", []) or []
        min_score = float(cfg.get("min_score", 0.0))
        min_snippet_len = int(cfg.get("min_snippet_len", 40))
        min_filename_snippet = int(cfg.get("min_filename_snippet", 120))
        filtered = []
        for item in items:
            collection = item.get("collection")
            if collection and collection in exclude_collections:
                continue
            path = item.get("path") or item.get("name") or ""
            if path:
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(path, pattern):
                        break
                else:
                    pass
                if any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns):
                    continue
            try:
                score_val = float(item.get("score", 0.0))
            except Exception:
                score_val = 0.0
            if score_val < min_score:
                continue
            snippet = (item.get("snippet") or item.get("match_line") or "").strip()
            match_type = str(item.get("match_type", "")).lower()
            if match_type == "filename" and len(snippet) < min_filename_snippet:
                continue
            if len(snippet) < min_snippet_len:
                continue
            filtered.append(item)
        return filtered

    def _read_file_excerpt(self, path: Path, max_lines: int = 120, max_chars: int = 2000) -> str:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return ""
        excerpt = []
        total = 0
        for idx, line in enumerate(lines[:max_lines], start=1):
            entry = f"{idx}: {line}"
            excerpt.append(entry)
            total += len(entry)
            if total > max_chars:
                break
        return "\n".join(excerpt)

    def _deliberate(self, query: str, context: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        reasoner_model = plan.get("creator") or plan.get("reasoner") or next(iter(plan.values()), None)
        critic_model = plan.get("critic") or plan.get("reviewer") or reasoner_model
        config = self.config.raw.get("deliberation", {})
        max_rounds = int(config.get("max_rounds", 3))
        require_agreement = bool(config.get("require_agreement", True))
        stop_on_repeat = bool(config.get("stop_on_repeat_disagreements", False))
        stability_rounds = int(config.get("stability_rounds", 0))
        max_draft_chars = int(config.get("max_draft_chars", 4000))
        max_feedback_chars = int(config.get("max_feedback_chars", 4000))
        max_disagreements = int(config.get("max_disagreements_per_review", 8))
        model_timeout = config.get("model_timeout_seconds")
        panel_cfg = config.get("panel", {}) if isinstance(config, dict) else {}
        panel_enabled = bool(panel_cfg.get("enabled", False))
        panel_models = list(route.get("panel_models") or [])
        panel_require_all = bool(panel_cfg.get("require_all", True))
        panel_min_ratio = panel_cfg.get("min_agree_ratio") if isinstance(panel_cfg, dict) else None
        panel_timeout = panel_cfg.get("timeout_seconds") if isinstance(panel_cfg, dict) else None
        panel_max_rounds = panel_cfg.get("max_rounds") if isinstance(panel_cfg, dict) else None
        priority_models = self._priority_models(config)
        priority_require_all = bool(config.get("priority_require_all", True))
        priority_min_ratio = config.get("priority_min_ratio")
        if route.get("panel_require_all") is not None:
            panel_require_all = bool(route.get("panel_require_all"))
        if route.get("panel_min_ratio") is not None:
            try:
                panel_min_ratio = float(route.get("panel_min_ratio"))
            except Exception:
                pass
        min_time_left = float(config.get("min_time_left_seconds", 0) or 0)
        context_blob = self._format_context(context)
        runtime_blob = self._format_runtime_context(route, context)
        instructions = self._user_instructions(context)
        previous_run = context.get("previous_run") or {}
        prev_answer = previous_run.get("answer") or ""
        domain = route.get("domain")
        domain_hints = get_domain_instructions(domain)
        domain_instructions = domain_hints.deliberation_hint
        output_instructions = self._output_instructions(context.get("output_type"))
        output_meta = context.get("output") or {}
        missing_caps = output_meta.get("missing") or []
        tool_guard = (
            "Do not use tools, run shell commands, or attempt network access. "
            "Respond directly with your analysis."
        )
        rounds = []
        panel_rounds: List[Dict[str, Any]] = []
        diversity_checks: List[Dict[str, Any]] = []
        diversity_calls = 0
        reasoner_out = ""
        critic_out = ""
        agreement = False
        required_set = set(self._required_model_ids())
        stable_signature = ""
        stable_count = 0
        stop_reason: str | None = None
        previous_disagreements: List[str] = []
        final_disagreements: List[str] = []

        for round_idx in range(1, max_rounds + 1):
            remaining = self._time_remaining()
            if min_time_left and (remaining or 0) < min_time_left:
                stop_reason = "timeout_guard"
                break
            budget = None
            if remaining is not None:
                budget = max(0.0, remaining - float(min_time_left))
            per_call_timeout = None
            if budget is not None:
                per_call_timeout = int(max(20, min(float(model_timeout or budget), budget))) if budget > 0 else None
            elif model_timeout:
                try:
                    per_call_timeout = int(model_timeout)
                except Exception:
                    per_call_timeout = None
            analysis_prompt = (
                "You are the reasoner. Provide a careful analysis and propose a decision.\n"
                "Be specific and prescriptive. If evidence is weak, proceed with assumptions and mark confidence low.\n"
                "Do not refuse or defer just because data is missing.\n"
                "Negotiate toward consensus: address each disagreement explicitly and show what changed.\n"
                "Separate product recommendation from Conclave mechanics confirmation; confirmation failures are constraints, not blockers.\n\n"
                f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nContext:\n{context_blob}\n"
            )
            if round_idx > 1:
                draft_excerpt = self._truncate_text(reasoner_out, max_draft_chars)
                critic_excerpt = self._truncate_text(critic_out, max_feedback_chars)
                unresolved = previous_disagreements[:max_disagreements]
                unresolved_blob = ""
                if unresolved:
                    unresolved_blob = "\nUnresolved disagreements (must address each):\n"
                    for item in unresolved:
                        unresolved_blob += f"- {item}\n"
                analysis_prompt = (
                    "You are the reasoner. Revise the decision to address the critic's feedback.\n\n"
                    f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nContext:\n{context_blob}\n\n"
                    f"{unresolved_blob}\n"
                    f"Previous draft:\n{draft_excerpt}\n\n"
                    f"Critic feedback:\n{critic_excerpt}\n"
                )
                analysis_prompt += (
                    "\nInclude a 'Resolution log' section that lists each prior disagreement and marks each as RESOLVED or ACCEPTED (with tradeoff).\n"
                    "If a disagreement is about runtime confirmation, explicitly mark it as ACCEPTED/FAILED with a remediation step.\n"
                )
            if domain_instructions or instructions or output_instructions or tool_guard:
                analysis_prompt += f"\n{domain_instructions}\n"
                analysis_prompt += f"\n{tool_guard}\n"
                if output_instructions:
                    analysis_prompt += f"\n{output_instructions}\n"
                if missing_caps:
                    analysis_prompt += (
                        "\nCapability gaps detected: "
                        f"{', '.join(missing_caps)}. Provide a best-effort answer, "
                        "note the limitation, and include a concrete plan for generating the missing artifacts if tools are later enabled.\n"
                    )
                if instructions:
                    analysis_prompt += f"Instructions from input:\n{instructions}\n"
            if prev_answer:
                analysis_prompt += (
                    "\nPrevious consensus (Version N-1):\n"
                    f"{self._truncate_text(prev_answer, 1200)}\n"
                    "Reconcile with the prior result. Include a short 'Reconciliation' section noting what changed and why. "
                    "If no material change, say 'No change' and explain stability.\n"
                )
            reasoner_out = self._call_model(reasoner_model, analysis_prompt, role="reasoner", timeout_seconds=per_call_timeout)

            critic_prompt = (
                "You are the critic. Challenge the reasoning, list disagreements and gaps, and suggest fixes.\n"
                "Do not reject the task as out-of-scope; focus on improving the draft.\n"
                "Do not introduce brand new disagreements in later rounds unless they are critical.\n"
                "Negotiate toward consensus: if the draft clearly acknowledges an inherent limitation and proposes a concrete verification step, treat that item as resolved.\n"
                "Evaluate product recommendation separately from Conclave mechanics confirmation; if confirmation items are marked Failed/Unverified with remediation, do not treat as blocking.\n"
                "If remaining issues are minor or non-blocking, respond with AGREE and list up to 3 minor follow-ups.\n"
                "List at most 3 disagreements; consolidate overlapping items.\n"
                "Return sections:\nDisagreements:\n- ...\nGaps:\n- ...\nVerdict:\nAGREE or DISAGREE\n\n"
                f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nReasoner draft:\n{reasoner_out}\n"
            )
            if round_idx > 1 and previous_disagreements:
                critic_prompt += "\nPrevious disagreements (resolve if addressed; only list unresolved):\n"
                for item in previous_disagreements[:max_disagreements]:
                    critic_prompt += f"- {item}\n"
                critic_prompt += "\nIf all prior disagreements are resolved, respond with AGREE.\n"
            if domain_instructions or instructions or output_instructions or tool_guard:
                critic_prompt += f"\n{domain_instructions}\n"
                critic_prompt += f"\n{tool_guard}\n"
                if output_instructions:
                    critic_prompt += f"\n{output_instructions}\n"
                if missing_caps:
                    critic_prompt += (
                        "\nCapability gaps detected: "
                        f"{', '.join(missing_caps)}. Ensure the draft acknowledges the limitation and provides a fallback.\n"
                    )
                if instructions:
                    critic_prompt += f"Instructions from input:\n{instructions}\n"
            if prev_answer:
                critic_prompt += (
                    "\nPrevious consensus (Version N-1):\n"
                    f"{self._truncate_text(prev_answer, 800)}\n"
                    "Ensure the draft reconciles with prior results and clearly justifies any changes.\n"
                )
            use_panel = panel_enabled and panel_models
            if panel_max_rounds is not None:
                try:
                    use_panel = use_panel and (round_idx <= int(panel_max_rounds))
                except Exception:
                    pass
            if use_panel:
                panel_reviews = []
                next_panel_models = []
                for model_id in panel_models:
                    effective_panel_timeout = panel_timeout
                    if per_call_timeout is not None:
                        if effective_panel_timeout is None:
                            effective_panel_timeout = per_call_timeout
                        else:
                            effective_panel_timeout = min(int(effective_panel_timeout), int(per_call_timeout))
                    review = self._call_model(model_id, critic_prompt, role="critic_panel", timeout_seconds=effective_panel_timeout)
                    meta = self._last_model_results.get(model_id, {})
                    ok = meta.get("ok", True)
                    error = meta.get("error")
                    stderr = meta.get("stderr")
                    skip_optional = False
                    if not ok and model_id == "cli:gemini":
                        stderr_lower = (stderr or "").lower()
                        quota_hint = any(token in stderr_lower for token in ("quota", "rate limit", "capacity", "exhausted", "429"))
                        if error == "timeout" or quota_hint:
                            skip_optional = True
                    verdict = self._critic_agrees(review) if ok else False
                    verdict_label = "agree" if verdict else ("skipped" if skip_optional else ("error" if not ok else "disagree"))
                    panel_reviews.append({
                        "model_id": model_id,
                        "label": self._model_label(model_id),
                        "verdict": verdict_label,
                        "ok": ok,
                        "error": error,
                        "stderr": stderr,
                        "skipped": skip_optional,
                        "disagreements": self._extract_disagreements(review) if ok else ([] if skip_optional else ([f"model failed: {error}"] if error else [])),
                        "text": review,
                    })
                    if ok or model_id in required_set:
                        next_panel_models.append(model_id)
                if next_panel_models != panel_models:
                    panel_models = next_panel_models
                agreement = self._panel_agreement(
                    panel_reviews,
                    require_all=panel_require_all,
                    min_ratio=None,
                    priority_models=priority_models,
                    priority_require_all=priority_require_all,
                    priority_min_ratio=priority_min_ratio,
                )
                if panel_min_ratio is not None and not panel_require_all:
                    try:
                        agreement = self._panel_agreement(
                            panel_reviews,
                            require_all=False,
                            min_ratio=float(panel_min_ratio),
                            priority_models=priority_models,
                            priority_require_all=priority_require_all,
                            priority_min_ratio=priority_min_ratio,
                        )
                    except Exception:
                        agreement = self._panel_agreement(
                            panel_reviews,
                            require_all=panel_require_all,
                            min_ratio=None,
                            priority_models=priority_models,
                            priority_require_all=priority_require_all,
                            priority_min_ratio=priority_min_ratio,
                        )
                critic_out = self._format_panel_feedback(panel_reviews, max_disagreements=max_disagreements)
                filtered_disagreements = self._aggregate_panel_disagreements(
                    panel_reviews,
                    priority_models=priority_models,
                )
                # Calculate weighted agreement ratio for transparency
                weighted_agrees = sum(
                    self._model_confidence_weight(r.get("model_id", ""))
                    for r in panel_reviews if r.get("verdict") == "agree" and r.get("ok", True)
                )
                total_weight = sum(
                    self._model_confidence_weight(r.get("model_id", ""))
                    for r in panel_reviews if r.get("ok", True)
                )
                weighted_ratio = round(weighted_agrees / total_weight, 3) if total_weight > 0 else 0.0

                round_entry = {
                    "round": round_idx,
                    "agreement": agreement,
                    "disagreements": filtered_disagreements,
                    "weighted_ratio": weighted_ratio,
                }
                panel_rounds.append({
                    "round": round_idx,
                    "agreement": agreement,
                    "weighted_ratio": weighted_ratio,
                    "reviews": panel_reviews,
                })
                rounds.append(round_entry)
            else:
                critic_out = self._call_model(critic_model, critic_prompt, role="critic", timeout_seconds=per_call_timeout)
                agreement = self._critic_agrees(critic_out)
                round_entry = {
                    "round": round_idx,
                    "agreement": agreement,
                    "disagreements": self._extract_disagreements(critic_out),
                }
                rounds.append(round_entry)
                diversity_entry = self._maybe_run_diversity_check(
                    query=query,
                    context_blob=context_blob,
                    reasoner_out=reasoner_out,
                    critic_out=critic_out,
                    round_idx=round_idx,
                    max_rounds=max_rounds,
                    agreement=agreement,
                    domain_instructions=domain_instructions,
                    output_instructions=output_instructions,
                    instructions=instructions,
                    calls_so_far=diversity_calls,
                )
                if diversity_entry:
                    diversity_checks.append(diversity_entry)
                    diversity_calls += 1
            if self._run_id:
                self.store.append_event(self._run_id, {"phase": "deliberate", **round_entry})
            previous_disagreements = list(round_entry.get("disagreements") or [])
            final_disagreements = list(round_entry.get("disagreements") or [])
            signature = self._disagreement_signature(round_entry.get("disagreements") or [])
            if stop_on_repeat and stability_rounds > 0 and signature:
                if signature == stable_signature:
                    stable_count += 1
                else:
                    stable_signature = signature
                    stable_count = 1
                if stable_count >= stability_rounds and not agreement:
                    round_entry["stopped_reason"] = "stable_disagreements"
                    stop_reason = "stable_disagreements"
                    break
            if agreement or not require_agreement:
                break

        result = {
            "reasoner": reasoner_out,
            "critic": critic_out,
            "disagreements": final_disagreements,
            "rounds": rounds,
            "agreement": agreement,
            "stopped_reason": stop_reason,
            "panel": panel_rounds,
            "panel_models": panel_models,
            "diversity": diversity_checks,
        }
        # Add quality score to deliberation result
        result["quality_score"] = self._deliberation_score(result)
        return result

    def _deliberation_score(self, deliberation: Dict[str, Any]) -> float:
        """
        Calculate deliberation quality score (0-1) based on:
        - Agreement status (40% weight)
        - Weighted panel agreement ratio (25% weight)
        - Disagreement count (20% weight)
        - Convergence/revision distance (10% weight)
        - Stop reason penalty (5% weight)
        """
        disagreements = deliberation.get("disagreements") or []
        agreement = bool(deliberation.get("agreement"))
        stopped_reason = deliberation.get("stopped_reason")
        panel_rounds = deliberation.get("panel") or []

        # 1. Base agreement score (0.4 weight)
        agreement_score = 1.0 if agreement else 0.0

        # 2. Weighted panel ratio if available (0.25 weight)
        panel_ratio = 0.5  # Default if no panel data
        if panel_rounds:
            latest = panel_rounds[-1]
            panel_ratio = float(latest.get("weighted_ratio", 0.5))

        # 3. Disagreement penalty (0.2 weight) - softer curve
        max_disagreements = 8
        disagreement_score = max(0.0, 1.0 - (len(disagreements) / max_disagreements))

        # 4. Convergence score based on rounds needed (0.1 weight)
        # Faster convergence = higher score
        rounds = deliberation.get("rounds") or []
        max_rounds = 5
        convergence_score = 1.0
        if len(rounds) > 1:
            # Reward quick convergence, penalize needing many rounds
            convergence_score = max(0.3, 1.0 - ((len(rounds) - 1) / max_rounds))

        # 5. Stop reason penalty (0.05 weight)
        stop_penalty = 0.0
        if stopped_reason:
            # Early stop reasons hurt score
            stop_penalty = 0.5 if stopped_reason in ("timeout_guard", "max_rounds") else 0.2

        # Weighted combination
        score = (
            agreement_score * 0.40 +
            panel_ratio * 0.25 +
            disagreement_score * 0.20 +
            convergence_score * 0.10 +
            (1.0 - stop_penalty) * 0.05
        )

        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Jaccard similarity on word trigrams."""
        if not a or not b:
            return 0.0
        words_a = a.lower().split()
        words_b = b.lower().split()
        if len(words_a) < 3 or len(words_b) < 3:
            set_a = set(words_a)
            set_b = set(words_b)
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)
        trigrams_a = set(tuple(words_a[i:i+3]) for i in range(len(words_a) - 2))
        trigrams_b = set(tuple(words_b[i:i+3]) for i in range(len(words_b) - 2))
        if not trigrams_a or not trigrams_b:
            return 0.0
        return len(trigrams_a & trigrams_b) / len(trigrams_a | trigrams_b)

    def _anneal_consensus(
        self,
        query: str,
        context: Dict[str, Any],
        route: Dict[str, Any],
        collections: Optional[List[str]],
        output_type: Optional[str],
        agent_set: Optional[Dict[str, Any]] = None,
        role_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config.raw.get("annealing", {}) or {}
        if not cfg.get("enabled", False):
            deliberation = self._deliberate(query, context, route)
            return {
                "deliberation": deliberation,
                "route": route,
                "annealing": None,
            }

        max_iterations = int(cfg.get("max_iterations", 3))
        stable_rounds = int(cfg.get("stable_rounds", 2))
        schedule = str(cfg.get("schedule", "linear")).lower()
        temp_start = float(cfg.get("temperature_start", 1.2))
        temp_end = float(cfg.get("temperature_end", 0.3))
        noise_start = float(cfg.get("noise_start", 0.18))
        noise_end = float(cfg.get("noise_end", 0.05))
        accept_min = float(cfg.get("accept_worse_min_prob", 0.05))
        shuffle_panel = bool(cfg.get("shuffle_panel", True))
        seed = cfg.get("seed")
        rng = random.Random(int(seed)) if seed is not None else random
        perturb_prompt = bool(cfg.get("perturb_prompt", False))
        perturbation_phrases = list(cfg.get("perturbation_phrases", [
            "Consider contrarian viewpoints.",
            "Focus on practical constraints.",
            "Prioritize cost-effectiveness.",
            "Weight long-term outcomes heavily.",
            "Consider what could go wrong.",
        ]))
        content_convergence = bool(cfg.get("content_convergence", False))
        similarity_threshold = float(cfg.get("similarity_threshold", 0.85))

        iterations: List[Dict[str, Any]] = []
        required = set(self._required_model_ids())

        def ensure_required(candidate: Dict[str, Any]) -> None:
            if not required:
                return
            panel = list(candidate.get("panel_models") or [])
            plan_models = list((candidate.get("plan") or {}).values())
            missing = [mid for mid in required if mid not in panel and mid not in plan_models]
            if missing:
                panel.extend(missing)
                candidate["panel_models"] = panel

        def finalize_route(candidate: Dict[str, Any]) -> None:
            candidate["plan_details"] = self._plan_details(candidate.get("plan", {}))
            candidate["panel_details"] = self._panel_details(candidate.get("panel_models") or [])

        current_route = dict(route)
        ensure_required(current_route)
        finalize_route(current_route)
        self._apply_output_meta(context, output_type, current_route)
        current_deliberation = self._deliberate(query, context, current_route)
        current_score = self._deliberation_score(current_deliberation)
        best_route = current_route
        best_deliberation = current_deliberation
        best_score = current_score
        no_improve = 0

        iterations.append({
            "iteration": 1,
            "accepted": True,
            "score": round(current_score, 4),
            "agreement": bool(current_deliberation.get("agreement")),
            "temperature": round(self._anneal_value(temp_start, temp_end, 1, max_iterations, schedule), 4),
            "noise": round(self._anneal_value(noise_start, noise_end, 1, max_iterations, schedule), 4),
            "route": {
                "plan": current_route.get("plan"),
                "panel_models": current_route.get("panel_models"),
            },
        })

        content_stable_count = 0
        best_answer = str((best_deliberation.get("rounds") or [{}])[-1].get("reasoner", "")) if best_deliberation.get("rounds") else ""

        for idx in range(2, max_iterations + 1):
            min_time_left = float(cfg.get("min_time_left_seconds", 0) or 0)
            if min_time_left and (self._time_remaining() or 0) < min_time_left:
                break
            temperature = self._anneal_value(temp_start, temp_end, idx, max_iterations, schedule)
            noise = self._anneal_value(noise_start, noise_end, idx, max_iterations, schedule)

            # Prompt perturbation: slightly reframe the query each iteration
            perturbed_query = query
            perturbation_used = None
            if perturb_prompt and perturbation_phrases:
                perturbation_used = rng.choice(perturbation_phrases) if hasattr(rng, 'choice') else random.choice(perturbation_phrases)
                perturbed_query = f"{perturbation_used}\n\n{query}"

            candidate_route = self._route_query(
                perturbed_query,
                collections,
                budget_context=self._budget_context,
                plan_noise=noise,
                rng=rng,
                agent_set=agent_set,
                role_overrides=role_overrides,
            )
            ensure_required(candidate_route)
            if shuffle_panel and candidate_route.get("panel_models"):
                panel = list(candidate_route.get("panel_models") or [])
                rng.shuffle(panel)
                candidate_route["panel_models"] = panel
            finalize_route(candidate_route)
            self._apply_output_meta(context, output_type, candidate_route)
            candidate_deliberation = self._deliberate(perturbed_query, context, candidate_route)
            candidate_score = self._deliberation_score(candidate_deliberation)

            # Content convergence detection
            similarity = 0.0
            candidate_answer = str((candidate_deliberation.get("rounds") or [{}])[-1].get("reasoner", "")) if candidate_deliberation.get("rounds") else ""
            if content_convergence and best_answer and candidate_answer:
                similarity = self._text_similarity(best_answer, candidate_answer)
                if similarity >= similarity_threshold:
                    content_stable_count += 1
                else:
                    content_stable_count = 0

            delta = candidate_score - current_score
            accepted = False
            prob = 0.0
            if delta >= 0:
                accepted = True
            else:
                temp = max(0.0001, float(temperature))
                prob = math.exp(delta / temp)
                prob = max(prob, accept_min)
                accepted = rng.random() <= prob

            if accepted:
                current_route = candidate_route
                current_deliberation = candidate_deliberation
                current_score = candidate_score

            if candidate_score > best_score:
                best_score = candidate_score
                best_route = candidate_route
                best_deliberation = candidate_deliberation
                best_answer = candidate_answer
                no_improve = 0
            else:
                no_improve += 1

            iteration_data: Dict[str, Any] = {
                "iteration": idx,
                "accepted": accepted,
                "score": round(candidate_score, 4),
                "agreement": bool(candidate_deliberation.get("agreement")),
                "temperature": round(temperature, 4),
                "noise": round(noise, 4),
                "accept_prob": round(prob, 4) if delta < 0 else None,
                "route": {
                    "plan": candidate_route.get("plan"),
                    "panel_models": candidate_route.get("panel_models"),
                },
            }
            if content_convergence:
                iteration_data["similarity"] = round(similarity, 4)
            if perturbation_used:
                iteration_data["perturbation"] = perturbation_used
            iterations.append(iteration_data)

            if stable_rounds and no_improve >= stable_rounds:
                break

            # Early convergence: content is stable across iterations
            if content_convergence and content_stable_count >= stable_rounds:
                break

        if self._run_id:
            self.store.append_event(self._run_id, {
                "phase": "annealing",
                "status": "done",
                "iterations": len(iterations),
                "best_score": round(best_score, 4),
            })

        annealing = {
            "enabled": True,
            "iterations": iterations,
            "best_score": round(best_score, 4),
            "selected_iteration": max(iterations, key=lambda item: item.get("score", -1)).get("iteration") if iterations else 1,
            "schedule": {
                "temperature_start": temp_start,
                "temperature_end": temp_end,
                "noise_start": noise_start,
                "noise_end": noise_end,
                "schedule": schedule,
            },
        }
        return {
            "deliberation": best_deliberation,
            "route": best_route,
            "annealing": annealing,
        }

    def _resolve_consensus(self, query: str, context: Dict[str, Any], deliberation: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        if deliberation.get("agreement", False):
            return deliberation
        cfg = self.config.raw.get("deliberation", {}) or {}
        priority_models = self._priority_models(cfg)
        if priority_models:
            priority_agreement = self._priority_agreement_from_deliberation(deliberation, priority_models, cfg)
            if priority_agreement is False:
                return deliberation
        resolver_cfg = cfg.get("resolver", {}) if isinstance(cfg, dict) else {}
        if not resolver_cfg.get("enabled", False):
            return deliberation
        plan = route.get("plan", {})
        model_id = resolver_cfg.get("model") or plan.get("summarizer") or plan.get("reasoner") or plan.get("critic")
        if not model_id:
            return deliberation
        disagreements = []
        rounds = deliberation.get("rounds") or []
        if rounds:
            disagreements = rounds[-1].get("disagreements") or []
        if not disagreements:
            disagreements = deliberation.get("disagreements") or []
        disagreements = disagreements[: int(resolver_cfg.get("max_disagreements", 6))]
        context_blob = self._format_context(context)
        runtime_blob = self._format_runtime_context(route, context)
        instructions = self._user_instructions(context)
        resolver_prompt = (
            "You are the consensus resolver. Decide if remaining disagreements are material enough to block consensus.\n"
            "If the draft explicitly:\n"
            "- marks unconfirmable runtime requirements as Failed/Unverified AND states what evidence would confirm them, and\n"
            "- provides an actionable plan for any remaining gaps (e.g., iteration plan, target profile, tax assumptions),\n"
            "then you should respond with AGREE even if confirmation items are Failed.\n"
            "Only respond with DISAGREE if there are material decision flaws not addressed by the draft.\n"
            "Return:\nVerdict: AGREE or DISAGREE\nReason: <1-3 sentences>\n\n"
            f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nContext:\n{context_blob}\n\n"
            f"Reasoner draft:\n{deliberation.get('reasoner','')}\n\n"
            f"Critic notes:\n{deliberation.get('critic','')}\n\n"
            f"Open disagreements:\n" + "\n".join(f"- {d}" for d in disagreements)
        )
        if instructions:
            resolver_prompt += f"\n\nInstructions from input:\n{instructions}\n"
        verdict_text = self._call_model(model_id, resolver_prompt, role="resolver")
        agreed = self._critic_agrees(verdict_text)
        if agreed:
            deliberation = dict(deliberation)
            deliberation["agreement"] = True
            deliberation["agreement_override"] = "resolver"
            deliberation["agreement_reason"] = verdict_text.strip()
        return deliberation

    def _model_confidence_weight(self, model_id: str) -> float:
        """Get confidence weight for a model based on json_reliability capability."""
        card = self.registry.get_model(model_id) or {}
        reliability = card.get("json_reliability", "medium")
        # Weight mapping: high=1.0, medium=0.7, low=0.4
        weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
        return weights.get(str(reliability).lower(), 0.7)

    def _panel_agreement(
        self,
        reviews: List[Dict[str, Any]],
        require_all: bool = True,
        min_ratio: float | None = None,
        priority_models: List[str] | None = None,
        priority_require_all: bool = True,
        priority_min_ratio: float | None = None,
        weighted: bool = True,
    ) -> bool:
        if not reviews:
            return False
        if priority_models:
            priority_reviews = [
                r for r in reviews
                if r.get("model_id") in priority_models and r.get("ok", True)
            ]
            if priority_reviews:
                if weighted:
                    # Weighted priority voting by model confidence
                    weighted_agrees = sum(
                        self._model_confidence_weight(r.get("model_id", ""))
                        for r in priority_reviews if r.get("verdict") == "agree"
                    )
                    total_weight = sum(
                        self._model_confidence_weight(r.get("model_id", ""))
                        for r in priority_reviews
                    )
                    if total_weight > 0:
                        ratio = weighted_agrees / total_weight
                        if priority_min_ratio is not None:
                            return ratio >= float(priority_min_ratio)
                        return ratio >= 0.6  # Default 60% weighted agreement
                agrees = sum(1 for r in priority_reviews if r.get("verdict") == "agree")
                total = len(priority_reviews)
                if priority_require_all or priority_min_ratio is None:
                    return agrees == total
                return (agrees / total) >= float(priority_min_ratio)
        eligible = [r for r in reviews if r.get("ok", True)]
        if not eligible:
            return False

        # Use weighted voting if enabled (default: True)
        if weighted:
            weighted_agrees = sum(
                self._model_confidence_weight(r.get("model_id", ""))
                for r in eligible if r.get("verdict") == "agree"
            )
            total_weight = sum(
                self._model_confidence_weight(r.get("model_id", ""))
                for r in eligible
            )
            if total_weight > 0:
                ratio = weighted_agrees / total_weight
                if min_ratio is not None:
                    return ratio >= float(min_ratio)
                return ratio >= 0.6  # Default 60% weighted agreement

        # Fallback to unweighted voting
        verdicts = [r.get("verdict") == "agree" for r in eligible]
        if require_all:
            return all(verdicts)
        if min_ratio is not None:
            return (sum(1 for v in verdicts if v) / max(1, len(verdicts))) >= float(min_ratio)
        return sum(1 for v in verdicts if v) >= max(1, len(verdicts) // 2 + 1)

    def _aggregate_panel_disagreements(
        self,
        reviews: List[Dict[str, Any]],
        priority_models: List[str] | None = None,
    ) -> List[str]:
        disagreements: List[str] = []
        for review in reviews:
            if priority_models and review.get("model_id") not in priority_models:
                continue
            if review.get("verdict") == "agree":
                continue
            for item in review.get("disagreements") or []:
                if item not in disagreements:
                    disagreements.append(item)
        return disagreements

    def _priority_models(self, cfg: Dict[str, Any] | None = None) -> List[str]:
        config = cfg if isinstance(cfg, dict) else (self.config.raw.get("deliberation", {}) or {})
        models = config.get("priority_models") or []
        return [str(model_id) for model_id in models if model_id]

    def _priority_agreement_from_deliberation(
        self,
        deliberation: Dict[str, Any],
        priority_models: List[str],
        cfg: Dict[str, Any],
    ) -> bool | None:
        if not priority_models:
            return None
        panel_rounds = deliberation.get("panel") or []
        if not panel_rounds:
            return None
        latest_reviews = panel_rounds[-1].get("reviews") or []
        if not latest_reviews:
            return None
        priority_reviews = [
            r for r in latest_reviews
            if r.get("model_id") in priority_models and r.get("ok", True)
        ]
        if not priority_reviews:
            return None
        priority_require_all = bool(cfg.get("priority_require_all", True))
        priority_min_ratio = cfg.get("priority_min_ratio")
        agrees = sum(1 for r in priority_reviews if r.get("verdict") == "agree")
        total = len(priority_reviews)
        if priority_require_all or priority_min_ratio is None:
            return agrees == total
        return (agrees / total) >= float(priority_min_ratio)

    def _format_panel_feedback(self, reviews: List[Dict[str, Any]], max_disagreements: int = 8) -> str:
        lines = ["Panel review summary:"]
        for review in reviews:
            label = review.get("label") or review.get("model_id") or "model"
            verdict = review.get("verdict", "disagree")
            lines.append(f"\n[{label}] Verdict: {verdict.upper()}")
            disagreements = review.get("disagreements") or []
            if disagreements:
                lines.append("Disagreements:")
                for item in disagreements[:max_disagreements]:
                    lines.append(f"- {item}")
        return "\n".join(lines)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated]..."

    def _extract_fenced_block(self, text: str, lang: str) -> str:
        if not text or not lang:
            return ""
        pattern = re.compile(rf"```{re.escape(lang)}\\s*([\\s\\S]*?)```", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return ""
        return match.group(1).strip()

    def _render_palette_svg(self, colors: List[str]) -> str:
        width = max(320, 120 * len(colors))
        height = 120
        swatch_width = width / max(len(colors), 1)
        rects = []
        labels = []
        for idx, color in enumerate(colors):
            x = idx * swatch_width
            rects.append(
                f'<rect x="{x:.2f}" y="0" width="{swatch_width:.2f}" height="{height}" fill="{color}"/>'
            )
            labels.append(
                f'<text x="{x + 12:.2f}" y="{height - 12}" font-size="14" fill="#ffffff" font-family="sans-serif">{color}</text>'
            )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            + "".join(rects)
            + "".join(labels)
            + "</svg>"
        )

    def _artifact_descriptor(self, name: str) -> Dict[str, Any]:
        ext = Path(name).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}:
            kind = "image"
        elif ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            kind = "video"
        elif ext in {".html", ".css"}:
            kind = "web"
        else:
            kind = "document"
        mime = mimetypes.guess_type(name)[0] or ""
        return {"name": name, "kind": kind, "mime": mime}

    def _generate_output_artifacts(
        self,
        run_id: str,
        consensus: Dict[str, Any],
        output_type: Optional[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        key = self._normalize_output_type(output_type)
        answer = (consensus or {}).get("answer") or ""
        if not key or not answer:
            return []
        run_dir = self.store.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: List[Dict[str, Any]] = []

        def _write(name: str, content: str) -> None:
            path = run_dir / name
            path.write_text(content)
            artifacts.append(self._artifact_descriptor(name))

        input_artifacts = (context or {}).get("input_artifacts") or []
        image_paths = [item.get("path") for item in input_artifacts if item.get("kind") == "image" and item.get("path")]
        vision_summary = (context or {}).get("vision_summary")

        if key == "resume":
            _write("resume.md", answer)
        elif key == "document":
            _write("document.md", answer)
        elif key == "video_brief":
            _write("video_brief.md", answer)
        elif key == "webpage_redesign":
            html_block = self._extract_fenced_block(answer, "html")
            css_block = self._extract_fenced_block(answer, "css")
            if html_block:
                _write("index.html", html_block)
            if css_block:
                _write("styles.css", css_block)
            if not html_block and not css_block:
                _write("webpage_redesign.md", answer)
        elif key == "web_prompt_pack":
            _write("web_prompt_pack.md", answer)
        elif key == "image_palette":
            hex_colors = re.findall(r"#[0-9a-fA-F]{3,6}", answer)
            unique = []
            seen = set()
            for color in hex_colors:
                norm = color.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                unique.append(color)
            top_colors = unique[:3]
            _write("palette.md", answer)
            if top_colors:
                svg = self._render_palette_svg(top_colors)
                _write("palette.svg", svg)
            if top_colors and (image_paths or vision_summary):
                for idx, color in enumerate(top_colors, start=1):
                    prompt = (
                        "Using the provided kitchen photos, generate a high-fidelity photorealistic image of the same kitchen "
                        f"with the cabinets painted {color}. Preserve layout, materials, lighting, and camera angle. "
                        "Do not change counters, floors, appliances, or decor."
                    )
                    images, provider = self._generate_images(prompt, image_paths, vision_summary=vision_summary)
                    if not images:
                        continue
                    try:
                        data = base64.b64decode(images[0])
                    except Exception:
                        continue
                    name = f"cabinet-color-{idx}.png"
                    (run_dir / name).write_bytes(data)
                    if provider:
                        self._record_image_usage(provider, 1)
                    meta = self._artifact_descriptor(name)
                    if provider:
                        meta["provider"] = provider
                    artifacts.append(meta)
        return artifacts

    def _write_output_file(
        self,
        run_id: str,
        consensus: Dict[str, Any],
        output_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str | None, List[Dict[str, Any]]]:
        answer = (consensus or {}).get("answer")
        if not answer:
            return None, []
        try:
            run_dir = self.store.run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "output.md"
            path.write_text(str(answer))
            artifacts: List[Dict[str, Any]] = []
            if output_type:
                artifacts.append(self._artifact_descriptor("output.md"))
            artifacts.extend(self._generate_output_artifacts(run_id, consensus, output_type, context))
            return str(path), artifacts
        except Exception:
            return None, []

    def _disagreement_signature(self, disagreements: List[str]) -> str:
        cleaned = []
        for item in disagreements:
            if not item:
                continue
            norm = " ".join(str(item).strip().lower().split())
            if norm:
                cleaned.append(norm)
        if not cleaned:
            return ""
        return "|".join(sorted(set(cleaned)))

    def _critic_agrees(self, critic: str) -> bool:
        lower = critic.lower()
        lines = [line.strip() for line in critic.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            lower_line = line.lower()
            if "verdict" not in lower_line:
                continue
            verdict = ""
            match = re.search(r"verdict\s*[:\-–—]\s*(.+)", line, re.IGNORECASE)
            if match:
                verdict = match.group(1).strip().lower()
            else:
                inline = re.search(r"verdict\s+(.*)", line, re.IGNORECASE)
                if inline:
                    verdict = inline.group(1).strip().lower()
            if verdict in {"verdict", "verdict."}:
                verdict = ""
            if not verdict:
                for next_line in lines[idx + 1:]:
                    verdict = next_line.strip().lower()
                    if verdict:
                        break
            if verdict:
                cleaned = re.sub(r"[^a-z]+", " ", verdict).strip()
                if "disagree" in cleaned or cleaned in {"no", "reject"}:
                    return False
                if cleaned.startswith("agree") or cleaned in {"yes", "accept", "approved"}:
                    return True
        for line in lines:
            lower_line = line.lower()
            if lower_line in {"agree", "agreed"}:
                return True
            if lower_line in {"disagree", "disagreed"}:
                return False
        return "verdict" in lower and "agree" in lower and "disagree" not in lower

    def _summarize(self, query: str, context: Dict[str, Any], deliberation: Dict[str, Any], route: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        summarizer_model = plan.get("summarizer") or plan.get("reasoner")
        context_blob = self._format_context(context)
        instructions = self._user_instructions(context)
        domain = route.get("domain")
        output_instructions = self._output_instructions(context.get("output_type"))
        evidence_hint = (
            f"Evidence count: {quality.get('evidence_count', 0)}, "
            f"pdf_ratio: {quality.get('pdf_ratio', 0):.2f}, "
            f"off_domain_ratio: {quality.get('off_domain_ratio', 0):.2f}, "
            f"signal: {quality.get('max_signal_score', 0):.2f}"
        )
        domain_hints = get_domain_instructions(domain)
        domain_instructions = domain_hints.summarizer_hint or domain_hints.deliberation_hint
        panel_health = route.get("panel_health") or {}
        lower_query = query.lower()
        strict_model_requirement = ("all models" in lower_query or "gemini" in lower_query)
        allow_optional_waiver = not strict_model_requirement
        panel_rounds = deliberation.get("panel") or []
        panel_models = route.get("panel_models") or []
        review_ok = set()
        review_seen = set()
        for panel_round in panel_rounds:
            for review in panel_round.get("reviews") or []:
                model_id = review.get("model_id")
                if not model_id:
                    continue
                review_seen.add(model_id)
                if review.get("ok") is True:
                    review_ok.add(model_id)
        skipped_optional = [
            entry.get("id")
            for entry in (panel_health.get("skipped") or [])
            if isinstance(entry, dict) and entry.get("id")
        ]
        failed_panel = [
            entry.get("id")
            for entry in (panel_health.get("failed") or [])
            if isinstance(entry, dict) and entry.get("id")
        ]
        for panel_round in panel_rounds:
            for review in panel_round.get("reviews") or []:
                if review.get("skipped"):
                    skipped_optional.append(review.get("model_id"))
                elif review.get("ok") is False:
                    failed_panel.append(review.get("model_id"))
        missing_reviews = [mid for mid in panel_models if mid and mid not in review_seen]
        if missing_reviews:
            skipped_optional.extend(missing_reviews)
        skipped_optional = [item for item in skipped_optional if item]
        if skipped_optional:
            skipped_optional = sorted(set(skipped_optional))
        failed_panel = [item for item in failed_panel if item]
        if failed_panel:
            failed_panel = sorted(set(failed_panel))
        if review_ok:
            skipped_optional = [item for item in skipped_optional if item not in review_ok]
            failed_panel = [item for item in failed_panel if item not in review_ok]

        def _sanitize_optional_notes(text: str) -> str:
            if not allow_optional_waiver or not skipped_optional or not text:
                return text
            lines = []
            for line in text.splitlines():
                lower = line.lower()
                if (
                    "gemini" in lower
                    or "all models" in lower
                    or "panel skipped" in lower
                    or "quota" in lower
                    or "model exclusion" in lower
                    or "rate limit" in lower
                    or "exhausted" in lower
                ):
                    continue
                lines.append(line)
            return "\n".join(lines).strip()

        diversity_notes = deliberation.get("diversity") or []
        diversity_blob = ""
        if diversity_notes:
            entries = []
            for item in diversity_notes:
                model_id = item.get("model_id") or item.get("model") or "diversity"
                notes = _sanitize_optional_notes((item.get("notes") or "").strip())
                if not notes:
                    continue
                entries.append(f"[{model_id}]\n{notes}")
            if entries:
                diversity_blob = "\n\n".join(entries)
        panel_blob = ""
        agreement = bool(deliberation.get("agreement", False))
        panel_rounds = panel_rounds or []
        round_count = len(deliberation.get("rounds") or [])
        if panel_rounds:
            latest_panel = panel_rounds[-1].get("reviews") or []
            entries = []
            for item in latest_panel:
                label = item.get("label") or item.get("model_id") or "panel"
                verdict = item.get("verdict", "disagree")
                notes = _sanitize_optional_notes((item.get("text") or "").strip())
                if not notes:
                    continue
                entries.append(f"[{label} | {verdict}]\n{notes}")
            if entries:
                panel_blob = "\n\n".join(entries)
        runtime_blob = self._format_runtime_context(route, context)
        if failed_panel:
            runtime_blob += f"\nPanel failed: {', '.join(failed_panel)}"
        if skipped_optional:
            runtime_blob += f"\nPanel skipped: {', '.join(skipped_optional)}"
        no_agreement_note = ""
        if not agreement:
            no_agreement_note = (
                "Consensus was not reached. Still provide a best-effort Outcome section and split it into:\n"
                "- Agreed Foundation: what the models converged on (facts, assumptions, constraints).\n"
                "- Recommended Direction: the most defensible path given the agreement.\n"
                "- Open Disagreements: summarize the top unresolved points (max 3) and why they matter.\n"
                "Do not frame this as a failure; treat optional model unavailability as a note, not a blocker.\n"
            )
        optional_note = ""
        if allow_optional_waiver and skipped_optional:
            optional_note = f"Optional panel models skipped: {', '.join(skipped_optional)} (ignore for confirmation).\n"
        reasoner_notes = _sanitize_optional_notes((deliberation.get("reasoner") or "").strip())
        critic_notes = _sanitize_optional_notes((deliberation.get("critic") or "").strip())
        waiver_instruction = ""
        if allow_optional_waiver:
            waiver_instruction = " If optional models were skipped, do not mark 'all models used' as failed; mark as waived/ignored."
        summary_prompt = (
            "You are the summarizer. Produce a consensus answer with bullet points."
            " Be prescriptive and actionable. Do not mention model limitations."
            " If evidence is weak, proceed with best-effort assumptions and mark confidence low instead of refusing."
            " Include an Evidence section listing the top sources (file paths or collection names)."
            " Include Risks/Uncertainties and Follow-ups. Include a confidence level (low/medium/high)."
            " Do not include a heading that says \"Final Consensus Answer\"."
            " If asked to confirm Conclave mechanics, include a Confirmation Checklist table with status and Runtime evidence."
            " Do not claim consensus if agreement is false."
            f"{waiver_instruction}\n\n"
            f"{no_agreement_note}"
            f"{optional_note}"
            f"Question: {query}\n\nRuntime:\n{runtime_blob}\n\nContext:\n{context_blob}\n\n"
            f"Deliberation: agreement={agreement}, rounds={round_count}\n"
            f"Evidence quality: {evidence_hint}\n\n"
            f"{domain_instructions}\n"
            f"{output_instructions}\n"
            f"{'Instructions from input:\\n' + instructions + '\\n' if instructions else ''}"
            f"Reasoner notes:\n{reasoner_notes}\n\n"
            f"Critic notes:\n{critic_notes}\n"
        )
        if panel_blob:
            summary_prompt += f"\nPanel notes:\n{panel_blob}\n"
        if diversity_blob:
            summary_prompt += f"\nDiversity check notes:\n{diversity_blob}\n"
        summary = self._call_model(summarizer_model, summary_prompt, role="summarizer")
        fallback_used = False
        if not summary.strip():
            summary = self._fallback_summary(query, deliberation)
            fallback_used = True
        model_conf = self._extract_confidence(summary)
        auto_conf = self._auto_confidence(quality)
        final_conf = self._merge_confidence(model_conf, auto_conf)
        # Build models_used from route plan
        models_used = {}
        plan_details = route.get("plan_details", {})
        for role, info in plan_details.items():
            if isinstance(info, dict) and info.get("id"):
                models_used[role] = {
                    "id": info["id"],
                    "label": info.get("label", info["id"]),
                }
            elif isinstance(info, str):
                models_used[role] = {"id": info, "label": self._model_label(info)}
        panel_models = route.get("panel_models") or []
        if panel_models:
            models_used["panel"] = [
                {
                    "id": model_id,
                    "label": self._model_label(model_id),
                }
                for model_id in panel_models
            ]

        return {
            "answer": summary.strip(),
            "confidence": final_conf,
            "confidence_model": model_conf,
            "confidence_auto": auto_conf,
            "pope": summary.strip().splitlines()[0] if summary.strip() else "",
            "fallback_used": fallback_used,
            "insufficient_evidence": False,
            "models_used": models_used,
        }

    def _call_model(self, model_id: Optional[str], prompt: str, role: Optional[str] = None, timeout_seconds: Optional[int] = None) -> str:
        if not model_id:
            return ""
        required_models = set(self._required_model_ids())
        is_required = model_id in required_models
        model_label = self._model_label(model_id)
        prompt_to_send = self._apply_agent_sync(prompt, str(model_id), role)
        if model_id.startswith("ollama:"):
            model = model_id.split(":", 1)[1]
            result = self.ollama.generate(model, prompt_to_send, temperature=0.2)
            self._record_model_observation(model_id, result)
            self._consume_tokens(model_id, prompt_to_send, result.text or "")
            if result.ok:
                self._run_models_used.add(model_id)
            self._last_model_results[model_id] = {
                "ok": result.ok,
                "error": result.error,
                "stderr": None,
            }
            audit = self._audit
            run_id = self._run_id
            payload = {
                "role": role,
                "model_id": model_id,
                "model_label": model_label,
                "ok": result.ok,
                "duration_ms": round(result.duration_ms, 2),
                "error": result.error,
            }
            if audit:
                audit.log("model.call", payload)
            if run_id:
                self.store.append_event(run_id, {"phase": "model", **payload})
            if is_required and not result.ok:
                raise RequiredModelError(f"{model_id} failed: {result.error or 'unknown error'}")
            return result.text
        if model_id.startswith("cli:") or model_id.startswith("gemini-api:"):
            result = self._call_cli_model(model_id, prompt_to_send, role=role, timeout_seconds=timeout_seconds) if model_id.startswith("cli:") else self._call_gemini_api(model_id, prompt_to_send, timeout_seconds=timeout_seconds)
            self._record_model_observation(model_id, result)
            self._consume_tokens(model_id, prompt_to_send, result.text or "")
            if result.ok:
                self._run_models_used.add(model_id)
            self._last_model_results[model_id] = {
                "ok": result.ok,
                "error": result.error,
                "stderr": getattr(result, 'stderr', None),
            }
            audit = self._audit
            run_id = self._run_id
            cli_label = self._parse_cli_model_label(result.stderr or "")
            if cli_label:
                model_label = cli_label
            payload = {
                "role": role,
                "model_id": model_id,
                "model_label": model_label,
                "ok": result.ok,
                "duration_ms": round(result.duration_ms, 2),
                "error": result.error,
                "stderr": result.stderr,
            }
            if audit:
                audit.log("model.call", payload)
            if run_id:
                self.store.append_event(run_id, {"phase": "model", **payload})
            if is_required and not result.ok:
                raise RequiredModelError(f"{model_id} failed: {result.error or 'unknown error'}")
            if not result.ok:
                fallback = (self.registry.get_model(model_id) or {}).get("fallback_model")
                if fallback and fallback != model_id:
                    if audit:
                        audit.log("model.fallback", {"role": role, "from": model_id, "to": fallback, "error": result.error})
                    if run_id:
                        self.store.append_event(run_id, {"phase": "model", "role": role, "model_id": model_id, "fallback_model": fallback, "error": result.error})
                    return self._call_model(fallback, prompt, role=role)
            return result.text
        return ""

    def _count_intelligent_providers(self) -> set[str]:
        """Normalize model IDs to provider level and filter to intelligent providers."""
        intelligent = {"claude", "codex", "gemini"}
        providers: set[str] = set()
        for model_id in self._run_models_used:
            if model_id.startswith("cli:claude"):
                providers.add("claude")
            elif model_id.startswith("cli:codex"):
                providers.add("codex")
            elif model_id.startswith("cli:gemini") or model_id.startswith("gemini-api:"):
                providers.add("gemini")
            # ollama models don't count as intelligent
        return providers & intelligent

    def _call_gemini_api(self, model_id: str, prompt: str, timeout_seconds: Optional[int] = None) -> Any:
        """Call Gemini via native API. Returns a result object compatible with _call_model."""
        from conclave.models.gemini import GeminiClient, GeminiResult
        if not hasattr(self, '_gemini_client'):
            self._gemini_client = GeminiClient()
        model_name = model_id.split(":", 1)[1] if ":" in model_id else "2.5-flash"
        timeout = timeout_seconds or 120
        return self._gemini_client.generate(prompt=prompt, model=model_name, timeout=timeout)

    def _call_cli_model(self, model_id: str, prompt: str, role: Optional[str] = None, timeout_seconds: Optional[int] = None) -> Any:
        card = self.registry.get_model(model_id) or {}
        command = card.get("command") or []
        prompt_mode = card.get("prompt_mode", "arg")
        stdin_flag = card.get("stdin_flag")
        if timeout_seconds is None:
            timeout_seconds = int(card.get("timeout_seconds", 90))
        env = card.get("env") or {}
        cwd = card.get("cwd")
        result = self.cli.run(
            command=list(command),
            prompt=prompt,
            prompt_mode=str(prompt_mode),
            stdin_flag=str(stdin_flag) if stdin_flag else None,
            timeout_seconds=timeout_seconds,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        return result

    def _model_label(self, model_id: str) -> str:
        card = self.registry.get_model(model_id) or {}
        return str(card.get("model_label") or card.get("model_name") or model_id)

    def _parse_cli_model_label(self, stderr: str) -> str | None:
        import re
        match = re.search(r"model:\\s*([A-Za-z0-9._:-]+)", stderr)
        if match:
            return match.group(1)
        return None

    def _plan_details(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        for role, model_id in (plan or {}).items():
            if not model_id:
                continue
            details[role] = {
                "id": model_id,
                "label": self._model_label(model_id),
            }
        return details

    def _panel_details(self, panel_models: List[str]) -> List[Dict[str, Any]]:
        details = []
        for model_id in panel_models or []:
            details.append({
                "id": model_id,
                "label": self._model_label(model_id),
            })
        return details

    def _record_model_observation(self, model_id: str, result: Any) -> None:
        card = self.registry.get_model(model_id) or {}
        prev_metrics = card.get("metrics", {})
        prev_error = float(prev_metrics.get("error_rate", 0.0))
        prev_timeout = float(prev_metrics.get("timeout_rate", 0.0))
        decay = 0.8
        error_rate = prev_error * decay
        timeout_rate = prev_timeout * decay
        if not result.ok:
            error_rate = min(1.0, error_rate + 0.2)
            if result.error and "timeout" in result.error.lower():
                timeout_rate = min(1.0, timeout_rate + 0.2)
        observation = {
            "p50_latency_ms": round(result.duration_ms, 2),
            "ok": result.ok,
            "error_rate": round(error_rate, 4),
            "timeout_rate": round(timeout_rate, 4),
        }
        try:
            self.registry.update_metrics(model_id, observation)
        except Exception:
            pass

    def _format_context(self, context: Dict[str, Any]) -> str:
        lines = []
        evidence = context.get("evidence") or []
        if evidence:
            for item in evidence[:12]:
                path = item.get("path")
                line = item.get("line")
                title = item.get("title") or (f"{path}:{line}" if path and line else path)
                source = item.get("source", "context").upper()
                meta = f"{item.get('collection', '')}".strip()
                if meta:
                    meta = f" ({meta})"
                lines.append(
                    f"- [{source}] {title}{meta}: {item.get('snippet') or ''} [signal={item.get('signal_score', 0):.2f}]"
                )
            blob = "\n".join(lines)
            if self._context_char_limit:
                return blob[: self._context_char_limit]
            return blob
        for item in context.get("rag", [])[:12]:
            title = item.get("title") or item.get("name") or item.get("path")
            lines.append(f"- [RAG] {title}: {item.get('snippet') or item.get('match_line') or ''}")
        for item in context.get("nas", [])[:12]:
            lines.append(f"- [NAS] {item.get('title')}: {item.get('snippet', '')}")
        blob = "\n".join(lines)
        if self._context_char_limit:
            return blob[: self._context_char_limit]
        return blob

    def _format_runtime_context(self, route: Dict[str, Any], context: Dict[str, Any]) -> str:
        lines: List[str] = []
        required = self._required_model_ids()
        if required:
            lines.append(f"Required models: {', '.join(required)}")
        plan = route.get("plan") or {}
        if plan:
            lines.append("Plan:")
            for role, model_id in plan.items():
                lines.append(f"- {role}: {model_id}")
        panel_models = route.get("panel_models") or []
        if panel_models:
            lines.append(f"Panel models: {', '.join(panel_models)}")
        panel_health = route.get("panel_health") or {}
        if panel_health:
            available = panel_health.get("available") or []
            failed = panel_health.get("failed") or []
            skipped = panel_health.get("skipped") or []
            if available:
                lines.append(f"Panel available: {', '.join(available)}")
            if failed:
                failed_ids = [entry.get("id") for entry in failed if isinstance(entry, dict)]
                if failed_ids:
                    lines.append(f"Panel failed: {', '.join(failed_ids)}")
            if skipped:
                skipped_ids = [entry.get("id") for entry in skipped if isinstance(entry, dict)]
                if skipped_ids:
                    lines.append(f"Panel skipped: {', '.join(skipped_ids)}")
        previous = context.get("previous_run") or {}
        if previous:
            lines.append(f"Previous run: {previous.get('id')} ({previous.get('created_at')})")
            if previous.get("agreement") is not None:
                lines.append(f"Previous agreement: {previous.get('agreement')}")
            prev_answer = previous.get("answer") or ""
            if prev_answer:
                lines.append("Previous consensus excerpt:")
                lines.append(prev_answer[:800])
        output_meta = context.get("output") or {}
        if output_meta:
            output_type = output_meta.get("type")
            if output_type:
                lines.append(f"Output type: {output_type}")
            required_caps = output_meta.get("requires") or []
            optional_caps = output_meta.get("optional") or []
            available_caps = output_meta.get("available") or []
            missing_caps = output_meta.get("missing") or []
            if required_caps:
                lines.append(f"Output requires: {', '.join(required_caps)}")
            if optional_caps:
                lines.append(f"Output optional: {', '.join(optional_caps)}")
            if available_caps:
                lines.append(f"Output available: {', '.join(available_caps)}")
            if missing_caps:
                lines.append(f"Output missing: {', '.join(missing_caps)}")
        planner_cfg = self.config.raw.get("planner", {}) or {}
        self_org = planner_cfg.get("self_organize") or {}
        if isinstance(self_org, dict):
            lines.append(f"Planner self_organize: {'enabled' if self_org.get('enabled', False) else 'disabled'}")
            budget_cfg = self_org.get("budget") if isinstance(self_org, dict) else {}
            if isinstance(budget_cfg, dict) and budget_cfg.get("enabled", False):
                lines.append(f"Planner budget: enabled (total_tokens={budget_cfg.get('total_tokens')})")
        role_overrides = planner_cfg.get("role_overrides") if isinstance(planner_cfg, dict) else None
        if isinstance(role_overrides, dict) and role_overrides:
            overrides = ", ".join(f"{k}->{v}" for k, v in role_overrides.items())
            lines.append(f"Role overrides: {overrides}")
        agent_sync = (context.get("agent_sync") or {}) if isinstance(context, dict) else {}
        if agent_sync:
            if agent_sync.get("enabled") is not None:
                lines.append(f"Agent sync: {'enabled' if agent_sync.get('enabled') else 'disabled'}")
            if agent_sync.get("mode"):
                lines.append(f"Agent sync mode: {agent_sync.get('mode')}")
        return "\n".join(lines)

    def _normalize_output_type(self, output_type: Optional[str]) -> str:
        if not output_type:
            return ""
        key = str(output_type).strip().lower()
        key = re.sub(r"\s+", "_", key)
        aliases = {
            "image": "image_palette",
            "image_prompt": "image_brief",
            "image_brief": "image_brief",
            "visual": "image_palette",
            "palette": "image_palette",
            "color_palette": "image_palette",
            "video": "video_brief",
            "video_brief": "video_brief",
            "resume": "resume",
            "cv": "resume",
            "doc": "document",
            "document": "document",
            "web_prompt": "web_prompt_pack",
            "web_prompt_pack": "web_prompt_pack",
            "prompt_pack": "web_prompt_pack",
            "chatgpt_pack": "web_prompt_pack",
            "chatgpt_prompt_pack": "web_prompt_pack",
            "webpage": "webpage_redesign",
            "web": "webpage_redesign",
            "website": "webpage_redesign",
            "webpage_redesign": "webpage_redesign",
            "site_redesign": "webpage_redesign",
            "build": "build_spec",
            "build_plan": "plan",
            "build_spec": "build_spec",
            "spec": "build_spec",
            "plan": "plan",
            "checklist": "checklist",
            "decision": "decision",
            "report": "report",
            "3d": "model_3d_brief",
            "3d_model": "model_3d_brief",
            "model": "model_3d_brief",
        }
        return aliases.get(key, key)

    def _resolve_agent_set(self, agent_set_id: str) -> Dict[str, Any] | None:
        if not agent_set_id:
            return None
        cfg = self.config.raw.get("agent_sets", {}) or {}
        if isinstance(cfg, dict):
            entry = cfg.get(agent_set_id)
            if isinstance(entry, dict):
                resolved = dict(entry)
                resolved["id"] = agent_set_id
                return resolved
        return None

    def _output_registry(self) -> Dict[str, Any]:
        outputs_cfg = self.config.raw.get("outputs", {}) or {}
        types = outputs_cfg.get("types")
        if isinstance(types, dict) and types:
            return types
        return {
            "report": {"label": "Report", "requires": [], "artifacts": ["output.md"]},
            "decision": {"label": "Decision", "requires": [], "artifacts": ["output.md"]},
            "plan": {"label": "Plan", "requires": [], "artifacts": ["output.md"]},
            "checklist": {"label": "Checklist", "requires": [], "artifacts": ["output.md"]},
            "build_spec": {"label": "Build Spec", "requires": [], "artifacts": ["output.md"]},
            "resume": {"label": "Resume", "requires": [], "artifacts": ["resume.md"]},
            "document": {"label": "Document", "requires": [], "artifacts": ["document.md"]},
            "webpage_redesign": {"label": "Webpage Redesign", "requires": [], "artifacts": ["index.html", "styles.css"]},
            "web_prompt_pack": {"label": "Web Prompt Pack", "requires": [], "artifacts": ["web_prompt_pack.md"]},
            "image_palette": {"label": "Image Palette", "requires": ["image_understanding"], "optional": ["image_generation"], "artifacts": ["palette.svg", "palette.md"]},
            "image_brief": {"label": "Image Brief", "requires": [], "optional": ["image_generation"], "artifacts": ["output.md"]},
            "video_brief": {"label": "Video Brief", "requires": [], "optional": ["video_generation"], "artifacts": ["video_brief.md"]},
            "model_3d_brief": {"label": "3D Model Brief", "requires": [], "artifacts": ["output.md"]},
        }

    def output_types(self) -> List[Dict[str, Any]]:
        registry = self._output_registry()
        outputs: List[Dict[str, Any]] = []
        for key, entry in registry.items():
            outputs.append({
                "id": key,
                "label": entry.get("label") or key,
                "requires": list(entry.get("requires") or []),
                "optional": list(entry.get("optional") or []),
                "artifacts": list(entry.get("artifacts") or []),
            })
        outputs.sort(key=lambda item: item.get("label", item.get("id", "")))
        return outputs

    def _openai_config(self) -> Dict[str, Any] | None:
        tools_cfg = self.config.raw.get("tools", {}) or {}
        openai_cfg = tools_cfg.get("openai", {}) if isinstance(tools_cfg, dict) else {}
        if not isinstance(openai_cfg, dict) or not openai_cfg.get("enabled", False):
            return None
        api_key = openai_cfg.get("api_key")
        if not api_key:
            env_key = str(openai_cfg.get("api_key_env") or "OPENAI_API_KEY")
            api_key = os.getenv(env_key)
        if not api_key:
            return None
        return {
            "api_key": api_key,
            "api_base": str(openai_cfg.get("api_base") or "https://api.openai.com/v1"),
            "responses_model": str(openai_cfg.get("responses_model") or "gpt-5"),
            "image_model": str(openai_cfg.get("image_model") or "gpt-image-1"),
            "image_quality": str(openai_cfg.get("image_quality") or "high"),
            "image_size": str(openai_cfg.get("image_size") or "auto"),
            "image_background": str(openai_cfg.get("image_background") or "auto"),
            "input_fidelity": str(openai_cfg.get("input_fidelity") or "high"),
            "timeout_seconds": int(openai_cfg.get("timeout_seconds") or 90),
        }

    def _openai_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        cfg = self._openai_config()
        if not cfg:
            return None
        url = cfg["api_base"].rstrip("/") + endpoint
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=cfg["timeout_seconds"])
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if self._audit:
                self._audit.log("openai.error", {"endpoint": endpoint, "error": str(exc)})
            return None

    def _extract_response_text(self, payload: Dict[str, Any] | None) -> str:
        if not payload:
            return ""
        text = payload.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        outputs = payload.get("output") or []
        parts = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if content.get("type") in {"output_text", "text"}:
                        parts.append(content.get("text") or "")
            elif item.get("type") == "output_text":
                parts.append(item.get("text") or "")
        return "\n".join([p for p in parts if p]).strip()

    def _extract_response_images(self, payload: Dict[str, Any] | None) -> List[str]:
        images: List[str] = []
        if not payload:
            return images
        outputs = payload.get("output") or []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_generation_call":
                result = item.get("result") or item.get("b64_json")
                if isinstance(result, str) and result.strip():
                    images.append(result.strip())
        return images

    def _build_openai_image_inputs(self, paths: List[str]) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        for raw in paths[:6]:
            path = Path(raw)
            if not path.exists():
                continue
            try:
                data = path.read_bytes()
            except Exception:
                continue
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            b64 = base64.b64encode(data).decode("utf-8")
            contents.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{b64}",
            })
        return contents

    def _openai_vision_summary(self, prompt: str, image_paths: List[str]) -> str:
        cfg = self._openai_config()
        if not cfg or not image_paths:
            return ""
        contents = [{"type": "input_text", "text": prompt}]
        contents.extend(self._build_openai_image_inputs(image_paths))
        payload = {
            "model": cfg["responses_model"],
            "input": [
                {
                    "role": "user",
                    "content": contents,
                }
            ],
        }
        resp = self._openai_post("/responses", payload)
        return self._extract_response_text(resp)

    def _openai_generate_image(self, prompt: str, image_paths: List[str]) -> List[str]:
        cfg = self._openai_config()
        if not cfg:
            return []
        contents = [{"type": "input_text", "text": prompt}]
        if image_paths:
            contents.extend(self._build_openai_image_inputs(image_paths))
        payload = {
            "model": cfg["responses_model"],
            "input": [
                {
                    "role": "user",
                    "content": contents,
                }
            ],
            "tools": [
                {
                    "type": "image_generation",
                    "quality": cfg["image_quality"],
                    "size": cfg["image_size"],
                    "background": cfg["image_background"],
                    "input_fidelity": cfg["input_fidelity"],
                }
            ],
            "tool_choice": {"type": "image_generation"},
        }
        resp = self._openai_post("/responses", payload)
        return self._extract_response_images(resp)

    def _gemini_config(self) -> Dict[str, Any] | None:
        tools_cfg = self.config.raw.get("tools", {}) or {}
        gemini_cfg = tools_cfg.get("gemini", {}) if isinstance(tools_cfg, dict) else {}
        if not isinstance(gemini_cfg, dict) or not gemini_cfg.get("enabled", False):
            return None
        api_key = gemini_cfg.get("api_key")
        if not api_key:
            env_key = str(gemini_cfg.get("api_key_env") or "GEMINI_API_KEY")
            api_key = os.getenv(env_key)
        if not api_key:
            return None
        return {
            "api_key": api_key,
            "api_base": str(gemini_cfg.get("api_base") or "https://generativelanguage.googleapis.com/v1beta"),
            "vision_model": str(gemini_cfg.get("vision_model") or "gemini-2.0-flash"),
            "image_model": str(gemini_cfg.get("image_model") or "gemini-2.5-flash-image"),
            "image_generation": bool(gemini_cfg.get("image_generation", True)),
            "image_understanding": bool(gemini_cfg.get("image_understanding", True)),
            "allow_image_input": bool(gemini_cfg.get("allow_image_input", False)),
            "timeout_seconds": int(gemini_cfg.get("timeout_seconds") or 90),
            "response_modalities": list(gemini_cfg.get("response_modalities") or ["IMAGE"]),
            "media_resolution": gemini_cfg.get("media_resolution"),
        }

    def _gemini_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        cfg = self._gemini_config()
        if not cfg:
            return None
        url = cfg["api_base"].rstrip("/") + endpoint
        headers = {
            "x-goog-api-key": cfg["api_key"],
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=cfg["timeout_seconds"])
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if self._audit:
                self._audit.log("gemini.error", {"endpoint": endpoint, "error": str(exc)})
            return None

    def _build_gemini_parts(self, prompt: str, image_paths: List[str], allow_images: bool) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        if prompt:
            parts.append({"text": prompt})
        if not allow_images:
            return parts
        for raw in image_paths[:6]:
            path = Path(raw)
            if not path.exists():
                continue
            try:
                data = path.read_bytes()
            except Exception:
                continue
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            b64 = base64.b64encode(data).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": b64,
                }
            })
        return parts

    def _gemini_generate_content(
        self,
        model: str,
        parts: List[Dict[str, Any]],
        response_modalities: Optional[List[str]] = None,
        media_resolution: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }
        generation_cfg: Dict[str, Any] = {}
        if response_modalities:
            generation_cfg["responseModalities"] = response_modalities
        if media_resolution:
            generation_cfg["mediaResolution"] = media_resolution
        if generation_cfg:
            payload["generationConfig"] = generation_cfg
        model_path = model
        if not model_path.startswith("models/"):
            model_path = f"models/{model_path}"
        return self._gemini_post(f"/{model_path}:generateContent", payload)

    def _extract_gemini_text(self, payload: Dict[str, Any] | None) -> str:
        if not payload:
            return ""
        texts: List[str] = []
        for cand in payload.get("candidates") or []:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if text:
                    texts.append(str(text))
        return "\n".join(texts).strip()

    def _extract_gemini_images(self, payload: Dict[str, Any] | None) -> List[str]:
        images: List[str] = []
        if not payload:
            return images
        for cand in payload.get("candidates") or []:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                inline = part.get("inline_data") or part.get("inlineData") or {}
                data = inline.get("data")
                if data:
                    images.append(str(data))
        return images

    def _gemini_vision_summary(self, prompt: str, image_paths: List[str]) -> str:
        cfg = self._gemini_config()
        if not cfg or not image_paths or not cfg.get("image_understanding"):
            return ""
        parts = self._build_gemini_parts(prompt, image_paths, allow_images=True)
        resp = self._gemini_generate_content(cfg["vision_model"], parts, media_resolution=cfg.get("media_resolution"))
        return self._extract_gemini_text(resp)

    def _gemini_generate_image(
        self,
        prompt: str,
        image_paths: List[str],
        vision_summary: Optional[str] = None,
    ) -> List[str]:
        cfg = self._gemini_config()
        if not cfg or not cfg.get("image_generation"):
            return []
        final_prompt = prompt
        if vision_summary:
            final_prompt = f"{prompt}\n\nReference description:\n{vision_summary}"
        parts = self._build_gemini_parts(final_prompt, image_paths, allow_images=cfg.get("allow_image_input", False))
        resp = self._gemini_generate_content(
            cfg["image_model"],
            parts,
            response_modalities=cfg.get("response_modalities") or ["IMAGE"],
            media_resolution=cfg.get("media_resolution"),
        )
        return self._extract_gemini_images(resp)

    def _image_preference(self) -> List[str]:
        tools_cfg = self.config.raw.get("tools", {}) or {}
        prefs = tools_cfg.get("image_preference") or []
        if not isinstance(prefs, list) or not prefs:
            prefs = ["openai", "gemini", "local"]
        return [str(item).strip().lower() for item in prefs if str(item).strip()]

    def _provider_supports(self, provider: str, capability: str) -> bool:
        provider = str(provider or "").lower()
        if provider == "openai":
            cfg = self._openai_config()
            return bool(cfg) and capability in {"image_generation", "image_understanding"}
        if provider == "gemini":
            cfg = self._gemini_config()
            if not cfg:
                return False
            if capability == "image_generation":
                return bool(cfg.get("image_generation", True))
            if capability == "image_understanding":
                return bool(cfg.get("image_understanding", True))
            return False
        return False

    def _vision_summary(self, prompt: str, image_paths: List[str]) -> tuple[str, str]:
        for provider in self._image_preference():
            if not self._provider_supports(provider, "image_understanding"):
                continue
            if provider == "openai":
                summary = self._openai_vision_summary(prompt, image_paths)
            elif provider == "gemini":
                summary = self._gemini_vision_summary(prompt, image_paths)
            else:
                summary = ""
            if summary:
                return summary, provider
        return "", ""

    def _generate_images(
        self,
        prompt: str,
        image_paths: List[str],
        vision_summary: Optional[str] = None,
    ) -> tuple[List[str], str]:
        for provider in self._image_preference():
            if not self._provider_supports(provider, "image_generation"):
                continue
            if provider == "openai":
                images = self._openai_generate_image(prompt, image_paths)
            elif provider == "gemini":
                images = self._gemini_generate_image(prompt, image_paths, vision_summary=vision_summary)
            else:
                images = []
            if images:
                return images, provider
        return [], ""

    def _record_image_usage(self, provider: str, count: int) -> None:
        if not provider or count <= 0:
            return
        self._image_usage[provider] = self._image_usage.get(provider, 0) + int(count)

    def _record_vision_usage(self, provider: str, count: int) -> None:
        if not provider or count <= 0:
            return
        self._vision_usage[provider] = self._vision_usage.get(provider, 0) + int(count)

    def _estimate_run_cost(self) -> Dict[str, Any]:
        estimate = {
            "total_usd": 0.0,
            "models": [],
            "images": [],
            "vision": [],
            "assumptions": {
                "chars_per_token": 4,
                "note": "Token counts are estimated from character length.",
            },
        }
        total = 0.0
        for model_id, usage in sorted(self._token_usage_by_model.items()):
            card = self.registry.get_model(model_id) or {}
            cost_cfg = card.get("cost", {}) or {}
            in_rate = float(cost_cfg.get("usd_per_1m_input_tokens", 0.0) or 0.0)
            out_rate = float(cost_cfg.get("usd_per_1m_output_tokens", 0.0) or 0.0)
            input_tokens = float(usage.get("input_tokens", 0.0))
            output_tokens = float(usage.get("output_tokens", 0.0))
            cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0
            total += cost
            estimate["models"].append({
                "id": model_id,
                "label": self._model_label(model_id),
                "input_tokens": round(input_tokens, 2),
                "output_tokens": round(output_tokens, 2),
                "usd_per_1m_input_tokens": in_rate,
                "usd_per_1m_output_tokens": out_rate,
                "cost_usd": round(cost, 6),
            })

        tools_cfg = self.config.raw.get("tools", {}) or {}
        costs_cfg = tools_cfg.get("costs", {}) if isinstance(tools_cfg, dict) else {}
        image_costs = costs_cfg.get("image_generation", {}) if isinstance(costs_cfg, dict) else {}
        vision_costs = costs_cfg.get("image_understanding", {}) if isinstance(costs_cfg, dict) else {}

        for provider, count in sorted(self._image_usage.items()):
            per_image = float((image_costs or {}).get(provider, 0.0) or 0.0)
            cost = count * per_image
            total += cost
            estimate["images"].append({
                "provider": provider,
                "count": count,
                "usd_per_image": per_image,
                "cost_usd": round(cost, 6),
            })

        for provider, count in sorted(self._vision_usage.items()):
            per_image = float((vision_costs or {}).get(provider, 0.0) or 0.0)
            cost = count * per_image
            total += cost
            estimate["vision"].append({
                "provider": provider,
                "count": count,
                "usd_per_image": per_image,
                "cost_usd": round(cost, 6),
            })

        estimate["total_usd"] = round(total, 6)
        return estimate

    def _available_capabilities(self, route: Dict[str, Any]) -> List[str]:
        capabilities: set[str] = set()
        tools_cfg = self.config.raw.get("tools", {}) or {}
        tool_caps = tools_cfg.get("capabilities", {}) or {}
        if isinstance(tool_caps, dict):
            for name, enabled in tool_caps.items():
                if bool(enabled):
                    capabilities.add(str(name))
        openai_cfg = self._openai_config()
        if openai_cfg:
            capabilities.add("image_generation")
            capabilities.add("image_understanding")
        gemini_cfg = self._gemini_config()
        if gemini_cfg:
            if gemini_cfg.get("image_generation", True):
                capabilities.add("image_generation")
            if gemini_cfg.get("image_understanding", True):
                capabilities.add("image_understanding")
        model_ids = list((route.get("plan") or {}).values()) + list(route.get("panel_models") or [])
        for model_id in model_ids:
            card = self.registry.get_model(str(model_id)) or {}
            caps = dict(card.get("capabilities", {}) or {})
            override = card.get("capabilities_override") or {}
            if isinstance(override, dict):
                caps.update(override)
            if caps.get("image_generation"):
                capabilities.add("image_generation")
            if str(caps.get("image_understanding", "")).lower() in {"limited", "full"}:
                capabilities.add("image_understanding")
            if caps.get("tool_use"):
                capabilities.add("tool_use")
        return sorted(capabilities)

    def _output_meta(self, output_type: Optional[str], route: Dict[str, Any]) -> Dict[str, Any]:
        key = self._normalize_output_type(output_type)
        if not key:
            return {}
        registry = self._output_registry()
        entry = registry.get(key, {}) if isinstance(registry, dict) else {}
        requires = list(entry.get("requires") or [])
        optional = list(entry.get("optional") or [])
        available = self._available_capabilities(route)
        missing = [req for req in requires if req not in available]
        return {
            "type": key,
            "label": entry.get("label") or key,
            "requires": requires,
            "optional": optional,
            "available": available,
            "missing": missing,
            "artifacts": list(entry.get("artifacts") or []),
        }

    def _apply_output_meta(self, context: Dict[str, Any], output_type: Optional[str], route: Dict[str, Any]) -> None:
        if not context or not output_type:
            return
        context["output_type"] = output_type
        output_meta = self._output_meta(output_type, route)
        if output_meta:
            context["output"] = output_meta

    def _output_requires(self, output_type: Optional[str], capability: str) -> bool:
        if not output_type:
            return False
        key = self._normalize_output_type(output_type)
        registry = self._output_registry()
        entry = registry.get(key, {}) if isinstance(registry, dict) else {}
        requires = entry.get("requires") or []
        return capability in requires

    def _output_instructions(self, output_type: Optional[str]) -> str:
        if not output_type:
            return ""
        key = self._normalize_output_type(output_type)
        if key == "image_brief":
            return (
                "Output type: IMAGE BRIEF.\n"
                "Return:\n"
                "- Image prompt (single paragraph)\n"
                "- Negative prompt\n"
                "- Composition & camera\n"
                "- Lighting\n"
                "- Palette\n"
                "- Aspect ratio & resolution\n"
                "- 2 variant ideas\n"
            )
        if key == "image_palette":
            return (
                "Output type: IMAGE PALETTE.\n"
                "Return:\n"
                "- Top 3 color choices with HEX codes\n"
                "- Finish recommendation (e.g., matte, satin)\n"
                "- Why each color works (lighting, materials, style)\n"
                "- Risks / tradeoffs\n"
                "- Suggested test steps (sample swatches, lighting checks)\n"
            )
        if key == "model_3d_brief":
            return (
                "Output type: 3D MODEL SPEC.\n"
                "Return:\n"
                "- Target format (GLB/FBX)\n"
                "- Units & scale\n"
                "- Dimensions (approx)\n"
                "- Parts list / hierarchy\n"
                "- Materials/shaders\n"
                "- Deliverable file structure\n"
            )
        if key == "build_spec":
            return (
                "Output type: BUILD SPEC.\n"
                "Return:\n"
                "- Objective\n"
                "- Functional requirements\n"
                "- Non-functional requirements\n"
                "- Interfaces / APIs\n"
                "- Data model\n"
                "- Acceptance criteria\n"
                "- Milestones\n"
            )
        if key == "resume":
            return (
                "Output type: RESUME.\n"
                "Return a clean, ATS-friendly resume in Markdown with sections:\n"
                "- Summary\n"
                "- Experience (bullet points with impact metrics)\n"
                "- Skills\n"
                "- Education\n"
                "- Projects (optional)\n"
            )
        if key == "document":
            return (
                "Output type: DOCUMENT.\n"
                "Return a structured document with headings, short paragraphs, and bullets as needed.\n"
            )
        if key == "webpage_redesign":
            return (
                "Output type: WEBPAGE REDESIGN.\n"
                "Return:\n"
                "- Design goals\n"
                "- Information architecture (sections)\n"
                "- Updated copy (if needed)\n"
                "- HTML in a fenced ```html``` block\n"
                "- CSS in a fenced ```css``` block\n"
                "- Responsive considerations\n"
            )
        if key == "web_prompt_pack":
            return (
                "Output type: WEB PROMPT PACK.\n"
                "Return a ChatGPT web-ready prompt pack with sections:\n"
                "- Primary Prompt (copy/paste ready)\n"
                "- What to Attach (files/photos + notes)\n"
                "- Follow-up Prompts (3-5 short refinement prompts)\n"
                "- Constraints & Guardrails\n"
                "- Success Checklist\n"
                "If the task involves images, include an image prompt and a negative prompt.\n"
            )
        if key == "decision":
            return (
                "Output type: DECISION.\n"
                "Return:\n"
                "- Decision\n"
                "- Rationale\n"
                "- Risks\n"
                "- Next steps\n"
            )
        if key == "checklist":
            return (
                "Output type: CHECKLIST.\n"
                "Return a checklist using '- [ ]' items grouped by phase.\n"
            )
        if key == "plan":
            return (
                "Output type: PLAN.\n"
                "Return:\n"
                "- Goals\n"
                "- Phases with deliverables\n"
                "- Dependencies\n"
                "- Risks\n"
            )
        if key == "video_brief":
            return (
                "Output type: VIDEO BRIEF.\n"
                "Return:\n"
                "- Concept & objective\n"
                "- Target length and format\n"
                "- Script or narration outline\n"
                "- Shot list / storyboard beats\n"
                "- Visual style references\n"
                "- Audio / music notes\n"
                "- Required assets\n"
            )
        return (
            "Output type: REPORT.\n"
            "Return a concise report with Summary, Recommendations, Risks, and Follow-ups.\n"
        )

    def _extract_confidence(self, summary: str) -> str:
        lower = summary.lower()
        if "confidence: high" in lower:
            return "high"
        if "confidence: low" in lower:
            return "low"
        if "confidence" in lower:
            return "medium"
        return "medium"

    def _merge_confidence(self, model: str, auto: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        return auto if order.get(auto, 1) < order.get(model, 1) else model

    def _auto_confidence(self, quality: Dict[str, Any]) -> str:
        issues = set(quality.get("issues", []))
        if quality.get("insufficient") or "rag_errors" in issues:
            return "low"
        pdf_ratio = float(quality.get("pdf_ratio", 0))
        off_domain_ratio = float(quality.get("off_domain_ratio", 0))
        evidence_count = int(quality.get("evidence_count", 0))
        max_signal = float(quality.get("max_signal_score", 0))
        avg_signal = float(quality.get("avg_signal_score", 0))
        if "pdf_heavy" in issues or "off_domain" in issues:
            return "medium"
        high_min = int(self.config.quality.get("high_evidence_min", 6))
        if avg_signal >= float(self.config.quality.get("high_signal_threshold", 1.5)) and evidence_count >= high_min and pdf_ratio < 0.5:
            return "high"
        if off_domain_ratio > 0.5:
            return "low"
        if evidence_count < int(self.config.quality.get("min_evidence", 2)) or max_signal < float(self.config.quality.get("low_signal_threshold", 0.5)):
            return "low"
        return "medium"

    def _fallback_summary(self, query: str, deliberation: Dict[str, Any]) -> str:
        disagreements = deliberation.get("disagreements", [])
        critic = deliberation.get("critic", "")
        reasoner = deliberation.get("reasoner", "")
        diversity_notes = deliberation.get("diversity") or []
        agreement = bool(deliberation.get("agreement", False))
        lines = [
            "**Consensus:**",
            "",
            f"- **Query**: {query}",
            "",
        ]
        if agreement:
            lines.extend([
                "**Reasoner Draft:**",
                reasoner.strip() or "No reasoner output.",
                "",
                "**Critic Disagreements:**",
            ])
        else:
            lines.extend([
                "**Agreed Foundation:**",
                reasoner.strip() or "No reasoner output.",
                "",
                "**Open Disagreements:**",
            ])
        if disagreements:
            for item in disagreements[:5]:
                lines.append(f"- {item}")
        elif critic.strip():
            lines.append(critic.strip()[:800])
        else:
            lines.append("No critic output.")
        if diversity_notes:
            lines.append("")
            lines.append("**Diversity Check:**")
            for item in diversity_notes[:3]:
                model_id = item.get("model_id") or item.get("model") or "diversity"
                notes = (item.get("notes") or "").strip()
                if notes:
                    lines.append(f"- [{model_id}] {notes[:400]}")
        lines.append("")
        lines.append("**Confidence Level**: Low")
        return "\n".join(lines)

    def _maybe_run_diversity_check(
        self,
        query: str,
        context_blob: str,
        reasoner_out: str,
        critic_out: str,
        round_idx: int,
        max_rounds: int,
        agreement: bool,
        domain_instructions: str,
        output_instructions: str,
        instructions: str,
        calls_so_far: int,
    ) -> Dict[str, Any] | None:
        cfg = self.config.raw.get("diversity_check", {}) or {}
        if not cfg.get("enabled", False):
            return None

        max_calls = int(cfg.get("max_calls", 1))
        if calls_so_far >= max_calls:
            return None

        min_round = int(cfg.get("min_round", 1))
        max_round = int(cfg.get("max_round", max_rounds))
        if round_idx < min_round or round_idx > max_round:
            return None

        trigger = str(cfg.get("trigger", "disagreement_or_random")).lower()
        if trigger == "disagreement" and agreement:
            return None

        anneal_cfg = cfg.get("annealing", {}) or {}
        start_prob = float(anneal_cfg.get("start_prob", cfg.get("probability", 0.0)))
        end_prob = float(anneal_cfg.get("end_prob", start_prob))
        schedule = str(anneal_cfg.get("schedule", "linear")).lower()
        prob = self._anneal_value(start_prob, end_prob, round_idx, max_rounds, schedule)
        if not agreement:
            prob *= float(cfg.get("disagreement_boost", 1.0))
        prob = min(1.0, max(0.0, prob))

        rng = self._diversity_rng(cfg)
        if trigger == "always":
            should_run = True
        elif prob <= 0:
            should_run = False
        else:
            should_run = rng.random() <= prob

        if not should_run:
            if self._run_id:
                self.store.append_event(self._run_id, {
                    "phase": "diversity_check",
                    "status": "skipped",
                    "round": round_idx,
                    "probability": round(prob, 4),
                    "trigger": trigger,
                    "agreement": agreement,
                })
            return None

        model_ids = list(cfg.get("model_ids") or [])
        if not model_ids:
            if self._run_id:
                self.store.append_event(self._run_id, {
                    "phase": "diversity_check",
                    "status": "skipped",
                    "round": round_idx,
                    "reason": "no model_ids configured",
                })
            return None

        temp_start = float(anneal_cfg.get("temperature_start", 1.0))
        temp_end = float(anneal_cfg.get("temperature_end", temp_start))
        temperature = self._anneal_value(temp_start, temp_end, round_idx, max_rounds, schedule)
        role = str(cfg.get("role", "critic"))
        model_id, candidates = self._select_diversity_model(model_ids, role, temperature, rng=rng)
        if not model_id:
            if self._run_id:
                self.store.append_event(self._run_id, {
                    "phase": "diversity_check",
                    "status": "skipped",
                    "round": round_idx,
                    "reason": "no eligible models",
                    "candidates": candidates[:6],
                })
            return None

        prompt = (
            "You are the diversity check. Provide alternative perspectives, highlight blind spots, and note risks.\n"
            "Keep it short and actionable. Return sections:\n"
            "Dissent:\n- ...\n"
            "Risks:\n- ...\n"
            "If wrong:\n- ...\n"
            "Confidence: low|medium|high\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context_blob}\n\n"
            f"Reasoner draft:\n{reasoner_out}\n\n"
            f"Critic feedback:\n{critic_out}\n"
        )
        if domain_instructions or instructions or output_instructions:
            prompt += f"\n{domain_instructions}\n"
            if output_instructions:
                prompt += f"\n{output_instructions}\n"
            if instructions:
                prompt += f"Instructions from input:\n{instructions}\n"

        diversity_out = self._call_model(model_id, prompt, role="diversity")
        meta = self._last_model_results.get(model_id, {})
        if not meta.get("ok", True) and model_id == "cli:gemini":
            stderr_lower = (meta.get("stderr") or "").lower()
            quota_hint = any(token in stderr_lower for token in ("quota", "rate limit", "capacity", "exhausted", "429"))
            if meta.get("error") == "timeout" or quota_hint:
                if self._run_id:
                    self.store.append_event(self._run_id, {
                        "phase": "diversity_check",
                        "status": "skipped",
                        "round": round_idx,
                        "model_id": model_id,
                        "reason": meta.get("error") or "quota",
                    })
                return None
        entry = {
            "round": round_idx,
            "model_id": model_id,
            "notes": diversity_out.strip(),
            "probability": round(prob, 4),
            "temperature": round(temperature, 4),
        }
        if self._run_id:
            self.store.append_event(self._run_id, {
                "phase": "diversity_check",
                "status": "done",
                "round": round_idx,
                "model_id": model_id,
                "probability": round(prob, 4),
                "temperature": round(temperature, 4),
                "candidates": candidates[:6],
            })
        return entry

    def _diversity_rng(self, cfg: Dict[str, Any]) -> random.Random:
        seed = cfg.get("seed")
        if seed is None:
            return random
        try:
            return random.Random(int(seed))
        except Exception:
            return random

    def _anneal_value(self, start: float, end: float, step: int, total: int, schedule: str) -> float:
        if total <= 1:
            return float(start)
        t = max(0.0, min(1.0, (step - 1) / (total - 1)))
        if schedule == "exp":
            if start <= 0 or end <= 0:
                return float(end)
            return float(start * ((end / start) ** t))
        return float(start + (end - start) * t)

    def _select_diversity_model(
        self,
        model_ids: List[str],
        role: str,
        temperature: float,
        rng: random.Random | None = None,
    ) -> tuple[str | None, List[Dict[str, Any]]]:
        candidates: List[Dict[str, Any]] = []
        eligible: List[Dict[str, Any]] = []
        for model_id in model_ids:
            ok, reason = self._model_available(model_id)
            if not ok:
                candidates.append({"id": model_id, "eligible": False, "reason": reason})
                continue
            card = self.registry.get_model(model_id)
            if not card:
                candidates.append({"id": model_id, "eligible": False, "reason": "missing card"})
                continue
            allow_unhealthy = bool(card.get("fallback_model"))
            ok, reason = self.planner._check_requirements(role, card, {}, allow_unhealthy=allow_unhealthy)
            if not ok:
                candidates.append({"id": model_id, "eligible": False, "reason": reason})
                continue
            score, details = self.planner._score_with_details(role, card)
            entry = {"id": model_id, "eligible": True, "score": score, "details": details}
            candidates.append(entry)
            eligible.append(entry)

        if not eligible:
            return None, candidates

        eligible.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        if len(eligible) == 1 or temperature <= 0:
            return eligible[0]["id"], candidates

        temp = max(0.05, float(temperature))
        scores = [item.get("score", 0.0) for item in eligible]
        max_score = max(scores) if scores else 0.0
        weights = []
        for score in scores:
            weight = math.exp((score - max_score) / temp)
            weights.append(weight)
        total = sum(weights)
        if total <= 0:
            return eligible[0]["id"], candidates
        rng = rng or random
        roll = rng.random() * total
        upto = 0.0
        for item, weight in zip(eligible, weights):
            upto += weight
            if roll <= upto:
                return item["id"], candidates
        return eligible[0]["id"], candidates

    def _model_available(self, model_id: str, allow_unhealthy: bool = False) -> tuple[bool, str]:
        card = self.registry.get_model(model_id)
        if not card:
            return False, "model not in registry"
        metrics = card.get("metrics", {})
        if metrics.get("ok") is False and not allow_unhealthy:
            return False, "model unhealthy"
        if card.get("provider") == "cli":
            command = list(card.get("command") or [])
            if not command:
                return False, "missing command"
            cmd0 = str(command[0])
            if "/" in cmd0:
                path = Path(cmd0).expanduser()
                if not path.exists():
                    return False, "command not found"
            else:
                if shutil.which(cmd0) is None:
                    return False, "command not found"
        return True, ""

    def _insufficient_evidence_answer(self, query: str, quality: Dict[str, Any], note: str | None = None) -> Dict[str, Any]:
        details = []
        min_evidence = int(self.config.quality.get("min_evidence", 2))
        details.append(f"- Evidence count: {quality.get('evidence_count', 0)} (min {min_evidence})")
        details.append(f"- Max signal score: {quality.get('max_signal_score', 0):.2f}")
        details.append(f"- PDF ratio: {quality.get('pdf_ratio', 0):.2f}")
        issues = quality.get("issues", []) or []
        if issues:
            details.append(f"- Issues: {', '.join(issues)}")
        required_count = quality.get("required_collection_count")
        required_hits = quality.get("required_collection_hits")
        if required_count:
            details.append(f"- Required collection hits: {required_hits or 0}/{required_count}")
        if quality.get("rag_errors"):
            details.append(f"- Retrieval errors: {len(quality.get('rag_errors', []))} (RAG server)")
        if quality.get("source_errors"):
            details.append(f"- Source fetch errors: {len(quality.get('source_errors', []))} (on-demand sources)")
        if note:
            details.append(f"- Note: {note}")
        answer = "\n".join([
            "### Insufficient Evidence for High-Fidelity Answer",
            "",
            f"**Query:** {query}",
            "",
            "**Why this is insufficient:**",
            *details,
            "",
            "**Recommended next steps:**",
            "- Specify the target collection explicitly.",
            "- Run `conclave index` if indexed context is missing.",
            "- Add/clean high-signal sources (markdown or structured notes).",
            "- Check `RAG server` availability if retrieval failed.",
            "",
            "**Confidence Level: Low**",
        ])
        return {
            "answer": answer,
            "confidence": "low",
            "confidence_model": "low",
            "confidence_auto": "low",
            "pope": "### Insufficient Evidence for High-Fidelity Answer",
            "fallback_used": True,
            "insufficient_evidence": True,
        }

    def _extract_disagreements(self, critic: str) -> list[str]:
        lines = []
        capture = False
        import re
        numbered = re.compile(r"^\d+[\).]")
        for line in critic.splitlines():
            stripped = line.strip()
            lower = stripped.lower().lstrip("# ").strip()
            if "disagreements" in lower:
                capture = True
                continue
            if capture:
                if "gaps" in lower:
                    break
                if stripped.startswith("-"):
                    item = stripped.lstrip("- ").strip()
                    if item:
                        lines.append(item)
                elif numbered.match(stripped):
                    item = numbered.sub("", stripped).strip()
                    if item:
                        lines.append(item)
        # De-duplicate while preserving order
        seen = set()
        unique = []
        for item in lines:
            norm = " ".join(item.lower().split())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            unique.append(item)
        return unique

    def _load_user_input(self, meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not meta:
            return []
        path_value = meta.get("input_path")
        if not path_value:
            return []
        path = Path(str(path_value))
        if not path.exists():
            return []
        try:
            content = path.read_text(errors="ignore")
        except Exception:
            return []
        snippet = content.strip().splitlines()
        snippet_text = " ".join(snippet[:8])[:400]
        return [{
            "path": str(path),
            "title": path.stem,
            "snippet": snippet_text,
            "full_text": content[:5000],
            "collection": "user-input",
            "base_dir": str(path.parent),
        }]

    def _user_instructions(self, context: Dict[str, Any]) -> str:
        inputs = context.get("user_inputs") or []
        for item in inputs:
            full_text = item.get("full_text")
            if full_text:
                return str(full_text).strip()
        return ""

    def _score_item(
        self,
        item: Dict[str, Any],
        source: str,
        preferred_collections: List[str] | None = None,
        domain: str | None = None,
        domain_paths: Dict[str, List[str]] | None = None,
        collection_reliability: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        path = item.get("path") or item.get("file_path") or item.get("name")
        title = item.get("title") or item.get("name") or (Path(path).name if path else "unknown")
        snippet = item.get("snippet") or item.get("match_line") or ""
        match_type = str(item.get("match_type", "")).lower()
        collection = item.get("collection") or item.get("source")
        line = item.get("line")
        if line is None and isinstance(snippet, str):
            import re
            match = re.match(r"\\s*(\\d+):", snippet)
            if match:
                try:
                    line = int(match.group(1))
                except Exception:
                    line = None
        score_val = 0.0
        try:
            score_val = float(item.get("score", 0.0))
        except Exception:
            score_val = 0.0
        ext = Path(path).suffix.lower() if path else ""
        if source == "user":
            signal = 2.0
        elif source == "source":
            signal = 1.6
        else:
            signal = 1.0 + min(score_val, 1.0) * 0.4
        if ext == ".pdf":
            signal -= 0.3
        elif ext in {".md", ".txt", ".json", ".yaml", ".yml", ".toml"}:
            signal += 0.3
        reliability = None
        if collection_reliability and collection:
            reliability = str(collection_reliability.get(collection, "")).lower()
        if reliability == "high":
            signal += 0.2
        elif reliability == "low":
            signal -= 0.2
        snippet_len = len(snippet.strip())
        if match_type == "filename" and source != "user":
            signal -= 0.5
        if snippet_len < 40 and source != "user":
            signal -= 0.2
        if snippet_len < 20 and source != "user":
            signal -= 0.2
        on_domain = False
        domain_known = False
        if preferred_collections:
            if collection:
                domain_known = True
                if collection in preferred_collections:
                    on_domain = True
        if domain and domain_paths and path:
            patterns = domain_paths.get(domain, [])
            if patterns:
                domain_known = True
                for pattern in patterns:
                    if fnmatch.fnmatch(path, pattern):
                        on_domain = True
                        break
        if source == "user" or source == "source":
            domain_known = True
            on_domain = True
        if preferred_collections and domain_known and domain != "general":
            if on_domain:
                signal += 0.2
            else:
                signal -= 0.2
        mtime = None
        if path:
            try:
                stat = Path(path).stat()
                mtime = stat.st_mtime
                age_days = (time.time() - mtime) / 86400
                if age_days < 30:
                    signal += 0.3
                elif age_days < 180:
                    signal += 0.1
                elif age_days > 730:
                    signal -= 0.2
            except Exception:
                pass
        if signal < 0:
            signal = 0.0
        return {
            "source": source,
            "path": path,
            "title": title,
            "snippet": snippet,
            "collection": collection,
            "extension": ext,
            "signal_score": round(signal, 3),
            "mtime": mtime,
            "on_domain": on_domain if domain_known else None,
            "line": line,
            "match_type": match_type,
            "snippet_len": snippet_len,
        }

    def _select_evidence(
        self,
        rag: List[Dict[str, Any]],
        nas: List[Dict[str, Any]],
        limit: int = 12,
        preferred_collections: List[str] | None = None,
        required_collections: List[str] | None = None,
        domain: str | None = None,
        domain_paths: Dict[str, List[str]] | None = None,
        collection_reliability: Dict[str, str] | None = None,
        user_items: List[Dict[str, Any]] | None = None,
        source_items: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        items = []
        user_input_present = False
        for item in rag:
            items.append(self._score_item(item, "rag", preferred_collections, domain, domain_paths, collection_reliability))
        for item in nas:
            enriched = self._maybe_attach_line(item)
            items.append(self._score_item(enriched, "nas", preferred_collections, domain, domain_paths, collection_reliability))
        if user_items:
            for item in user_items:
                user_input_present = True
                items.append(self._score_item(item, "user", preferred_collections, domain, domain_paths, collection_reliability))
        if source_items:
            items.extend([
                self._score_item(item, "source", preferred_collections, domain, domain_paths, collection_reliability) for item in source_items
            ])
        items.sort(key=lambda x: x.get("signal_score", 0), reverse=True)
        selected: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (item.get("path") or item.get("title") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                break
        pdf_count = sum(1 for item in selected if item.get("extension") == ".pdf")
        high_signal_threshold = float(self.config.quality.get("high_signal_threshold", 1.5))
        strong_count = 0
        content_count = 0
        non_user_count = 0
        required_hits = 0
        required_set = set([c for c in (required_collections or []) if c])
        for item in selected:
            if item.get("source") != "user":
                non_user_count += 1
            snippet_len = int(item.get("snippet_len") or 0)
            match_type = item.get("match_type") or ""
            if snippet_len >= 80 and match_type != "filename":
                content_count += 1
            if item.get("source") != "user" and snippet_len >= 80 and match_type != "filename":
                if float(item.get("signal_score", 0)) >= high_signal_threshold:
                    strong_count += 1
            if required_set:
                collection = item.get("collection")
                if collection in required_set:
                    required_hits += 1
        off_domain = 0
        on_domain = 0
        domain_known = 0
        for item in selected:
            if item.get("on_domain") is None:
                continue
            domain_known += 1
            if item.get("on_domain"):
                on_domain += 1
            else:
                off_domain += 1
        stats = {
            "evidence_count": len(selected),
            "total_candidates": len(items),
            "pdf_ratio": (pdf_count / len(selected)) if selected else 0.0,
            "max_signal_score": max([item.get("signal_score", 0) for item in selected], default=0),
            "avg_signal_score": round(sum(item.get("signal_score", 0) for item in selected) / len(selected), 3) if selected else 0.0,
            "off_domain_ratio": (off_domain / domain_known) if domain_known else 0.0,
            "domain_known": domain_known,
            "on_domain": on_domain,
            "off_domain": off_domain,
            "strong_evidence_count": strong_count,
            "content_evidence_count": content_count,
            "non_user_evidence_count": non_user_count,
            "required_collection_hits": required_hits,
            "required_collection_count": len(required_set),
            "user_input_present": user_input_present,
            "domain": domain or "general",
        }
        return selected, stats

    def _maybe_attach_line(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if item.get("line"):
            return item
        path = item.get("path") or item.get("file_path")
        snippet = item.get("snippet") or item.get("match_line")
        if not path or not snippet:
            return item
        line = self._find_line_number(path, snippet)
        if line:
            item = dict(item)
            item["line"] = line
        return item

    def _find_line_number(self, path: str, snippet: str) -> int | None:
        import re
        try:
            text = re.sub(r"<[^>]+>", "", snippet)
            tokens = re.findall(r"[A-Za-z0-9_]+", text)
            if not tokens:
                return None
            phrase = " ".join(tokens[:4]).strip()
            needle = phrase.lower()
            with open(path, "r", errors="ignore") as handle:
                for idx, line in enumerate(handle, start=1):
                    if needle and needle in line.lower():
                        return idx
            for token in tokens:
                if len(token) < 6:
                    continue
                with open(path, "r", errors="ignore") as handle:
                    for idx, line in enumerate(handle, start=1):
                        if token in line:
                            return idx
        except Exception:
            return None
        return None

    def _evaluate_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stats = context.get("stats", {})
        min_evidence = int(self.config.quality.get("min_evidence", 2))
        pdf_ratio_limit = float(self.config.quality.get("pdf_ratio_limit", 0.7))
        off_domain_limit = float(self.config.quality.get("off_domain_ratio_limit", 0.5))
        low_signal_threshold = float(self.config.quality.get("low_signal_threshold", 0.5))
        high_signal_threshold = float(self.config.quality.get("high_signal_threshold", 1.0))
        min_strong = int(self.config.quality.get("min_strong_evidence", 1))
        min_content = int(self.config.quality.get("min_content_evidence", 1))
        min_non_user = int(self.config.quality.get("min_non_user_evidence", 1))
        min_required = int(self.config.quality.get("min_required_collection_hits", 0))
        evidence_count = int(stats.get("evidence_count", 0))
        max_signal = float(stats.get("max_signal_score", 0))
        avg_signal = float(stats.get("avg_signal_score", 0))
        pdf_ratio = float(stats.get("pdf_ratio", 0))
        off_domain_ratio = float(stats.get("off_domain_ratio", 0))
        domain_known = int(stats.get("domain_known", 0))
        strong_count = int(stats.get("strong_evidence_count", 0))
        content_count = int(stats.get("content_evidence_count", 0))
        non_user_count = int(stats.get("non_user_evidence_count", 0))
        required_hits = int(stats.get("required_collection_hits", 0))
        required_count = int(stats.get("required_collection_count", 0))
        domain = stats.get("domain") or "general"
        user_input_present = bool(stats.get("user_input_present"))

        # Apply domain risk multiplier to base thresholds for high-stakes domains
        risk_multipliers = self.config.quality.get("domain_risk_multipliers", {}) or {}
        risk_multiplier = float(risk_multipliers.get(domain, 1.0))
        if risk_multiplier > 1.0:
            min_evidence = int(math.ceil(min_evidence * risk_multiplier))
            min_strong = int(math.ceil(max(min_strong, 1) * risk_multiplier))
            min_content = int(math.ceil(max(min_content, 1) * risk_multiplier))
            min_non_user = int(math.ceil(max(min_non_user, 1) * risk_multiplier))
            # Also increase signal threshold for high-risk domains
            low_signal_threshold = min(low_signal_threshold * risk_multiplier, high_signal_threshold)

        overrides = (self.config.quality.get("domain_overrides", {}) or {}).get(domain, {})
        if overrides:
            min_evidence = int(overrides.get("min_evidence", min_evidence))
            min_strong = int(overrides.get("min_strong_evidence", min_strong))
            min_content = int(overrides.get("min_content_evidence", min_content))
            min_non_user = int(overrides.get("min_non_user_evidence", min_non_user))
            min_required = int(overrides.get("min_required_collection_hits", min_required))
        allow_user_only = bool(overrides.get("allow_user_only", False)) if overrides else False
        user_only_ok = allow_user_only and user_input_present
        issues = []
        if evidence_count < min_evidence and not user_only_ok:
            issues.append("insufficient_evidence")
        if max_signal < low_signal_threshold and not user_only_ok:
            issues.append("low_signal")
        if (strong_count < min_strong or content_count < min_content or non_user_count < min_non_user) and not user_only_ok:
            issues.append("low_relevance")
        effective_required = min_required
        if required_count:
            effective_required = max(min_required, 1)
        if required_count and required_hits < effective_required:
            issues.append("missing_required_evidence")
        if pdf_ratio > pdf_ratio_limit:
            issues.append("pdf_heavy")
        if domain_known and off_domain_ratio > off_domain_limit:
            issues.append("off_domain")
        if stats.get("rag_errors"):
            issues.append("rag_errors")
        if stats.get("source_errors"):
            issues.append("source_errors")
        # Calculate signal quality score (0-1) based on evidence strength
        signal_quality = 0.0
        if evidence_count > 0:
            # Base score from signal strength
            signal_quality = min(1.0, avg_signal / high_signal_threshold) * 0.4
            # Boost for meeting evidence threshold
            signal_quality += min(1.0, evidence_count / max(min_evidence, 1)) * 0.3
            # Boost for strong evidence
            signal_quality += min(1.0, strong_count / max(min_strong, 1)) * 0.2
            # Penalty for high PDF ratio or off-domain
            if pdf_ratio > pdf_ratio_limit:
                signal_quality *= 0.8
            if off_domain_ratio > off_domain_limit:
                signal_quality *= 0.8

        return {
            "evidence_count": evidence_count,
            "min_evidence": min_evidence,
            "max_signal_score": max_signal,
            "avg_signal_score": avg_signal,
            "signal_quality": round(signal_quality, 3),
            "risk_multiplier": risk_multiplier,
            "pdf_ratio": pdf_ratio,
            "off_domain_ratio": off_domain_ratio,
            "required_collection_hits": stats.get("required_collection_hits", 0),
            "required_collection_count": stats.get("required_collection_count", 0),
            "rag_errors": stats.get("rag_errors", []),
            "source_errors": stats.get("source_errors", []),
            "issues": issues,
            "insufficient": bool(issues and ("insufficient_evidence" in issues or "low_signal" in issues or "low_relevance" in issues)),
        }

    def _expand_collections(
        self,
        domain: str,
        base: List[str],
        explicit: bool = False,
        allowlist: List[str] | None = None,
    ) -> tuple[List[str], Dict[str, Any]]:
        rag_cfg = self.config.rag
        use_server = bool(rag_cfg.get("use_server_collections", True))
        skip_empty = bool(rag_cfg.get("skip_empty_collections", True))
        patterns = rag_cfg.get("dynamic_patterns", {}).get(domain, [])
        catalog = {"server": [], "selected": []}
        if not use_server or (explicit and rag_cfg.get("trust_explicit_collections", True)):
            return list(dict.fromkeys(base)), catalog
        server = self.rag.collections()
        catalog["server"] = server
        available = {}
        for item in server:
            name = item.get("name")
            if not name:
                continue
            if skip_empty and not item.get("exists"):
                continue
            if skip_empty and item.get("file_count", 0) == 0:
                continue
            if allowlist and name not in allowlist:
                continue
            available[name] = item
        selected = list(dict.fromkeys([c for c in base if c in available or not skip_empty]))
        if patterns and not explicit:
            for name, meta in available.items():
                blob = f"{name} {meta.get('description','')}".lower()
                if any(pat in blob for pat in patterns):
                    if name not in selected:
                        selected.append(name)
        catalog["selected"] = selected
        return selected, catalog

    def _maybe_refresh_index(self) -> None:
        db_path = self.index.db_path
        cfg = self.config.index
        auto_build = bool(cfg.get("auto_build", False))
        auto_refresh_days = cfg.get("auto_refresh_days", 0)
        if not db_path.exists():
            if auto_build:
                self.index.index()
            return
        if auto_refresh_days:
            age_seconds = time.time() - db_path.stat().st_mtime
            if age_seconds > auto_refresh_days * 24 * 3600:
                self.index.index()

    def _log_to_memory(self, query: str, route: Dict[str, Any], consensus: Dict[str, Any], quality: Dict[str, Any]) -> None:
        """Log run outcome to memory MCP for cross-session learning."""
        try:
            domain = route.get("domain", "general")
            confidence = consensus.get("confidence", "medium")
            insufficient = consensus.get("insufficient_evidence", False)

            # Build learning summary
            if insufficient:
                learning = f"Query '{query[:80]}' failed with insufficient evidence in {domain} domain"
                importance = "low"
            elif confidence == "high":
                learning = f"High-confidence answer for '{query[:80]}' in {domain} domain"
                importance = "medium"
            else:
                learning = f"Answered '{query[:80]}' in {domain} domain with {confidence} confidence"
                importance = "low"

            # Include quality metrics as context
            context = f"evidence={quality.get('evidence_count', 0)}, signal={quality.get('max_signal_score', 0):.2f}"
            if quality.get("issues"):
                context += f", issues={','.join(quality.get('issues', []))}"

            # Log to memory MCP (async, don't wait for result)
            self.mcp.memory_learn(
                category=domain if domain != "general" else "conclave",
                learning=learning,
                context=context,
                importance=importance,
            )

            # Log action to today's thread
            result = "insufficient" if insufficient else f"{confidence} confidence"
            self.mcp.memory_log_action(
                action=f"conclave: {domain} query",
                result=result,
            )
        except Exception as e:
            self.logger.debug(f"Memory logging failed: {e}")
