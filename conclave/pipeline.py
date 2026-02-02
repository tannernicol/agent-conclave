"""Core decision pipeline for Conclave."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import logging
import fnmatch

from conclave.config import Config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner
from conclave.models.ollama import OllamaClient
from conclave.models.cli import CliClient
from conclave.rag import RagClient, NasIndex
from conclave.store import DecisionStore
from conclave.audit import AuditLog
from conclave.mcp import load_mcp_servers
from conclave.sources import HealthDashboardClient, MoneyClient


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
        sources_cfg = config.sources
        self.health_source = HealthDashboardClient(
            base_url=sources_cfg.get("health_dashboard_url", "http://127.0.0.1:8094"),
            pages=list(sources_cfg.get("health_pages", ["/my-health.html", "/recommendations.html"])),
        )
        self.money_source = MoneyClient(
            base_url=sources_cfg.get("money_api_url", "http://127.0.0.1:8000"),
            endpoints=list(sources_cfg.get("money_endpoints", ["/api/summary", "/api/networth"])),
        )
        self.index = NasIndex(
            data_dir=config.data_dir,
            allowlist=config.index.get("allowlist", []),
            exclude_patterns=config.index.get("exclude_patterns", []),
            max_file_mb=int(config.index.get("max_file_mb", 2)),
        )
        self.store = DecisionStore(config.data_dir)
        self._audit: AuditLog | None = None
        self._run_id: str | None = None
        self._context_char_limit: int | None = None
        self.logger = logging.getLogger(__name__)

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
        self._context_char_limit = None
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
        audit.log("run.start", {"query": query, "meta": meta or {}})
        try:
            self.store.append_event(run_id, {"phase": "preflight", "status": "start"})
            self._calibrate_models(run_id)
            audit.log("preflight.complete")

            self.store.append_event(run_id, {"phase": "route", "status": "start"})
            route = self._route_query(query, collections)
            self.store.append_event(run_id, {"phase": "route", "status": "done", "route": route, "models": route.get("plan", {})})
            audit.log("route.decided", route)

            self.store.append_event(run_id, {"phase": "retrieve", "status": "start"})
            context = self._retrieve_context(query, route, meta)
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
            quality = self._evaluate_quality(context)
            audit.log("quality.check", quality)
            self.store.append_event(run_id, {"phase": "quality", **quality})
            if quality.get("insufficient") and bool(self.config.quality.get("strict", True)):
                consensus = self._insufficient_evidence_answer(query, quality)
                artifacts = {
                    "route": route,
                    "context": context,
                    "deliberation": {},
                    "quality": quality,
                    "reconcile": {
                        "previous_run_id": self._latest_for_meta(meta),
                        "changed": True,
                    },
                }
                self.store.finalize_run(run_id, consensus, artifacts)
                audit.log("settlement.complete", {
                    "consensus": consensus,
                    "reconcile": artifacts["reconcile"],
                    "quality": quality,
                    "note": "insufficient_evidence",
                })
                return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)

            self.store.append_event(run_id, {"phase": "deliberate", "status": "start"})
            deliberation = self._deliberate(query, context, route)
            self.store.append_event(run_id, {"phase": "deliberate", "status": "done", "agreement": deliberation.get("agreement")})
            audit.log("deliberate.complete", {
                "reasoner": deliberation.get("reasoner", ""),
                "critic": deliberation.get("critic", ""),
                "disagreements": deliberation.get("disagreements", []),
                "rounds": deliberation.get("rounds", []),
                "agreement": deliberation.get("agreement"),
            })

            consensus = self._summarize(query, context, deliberation, route, quality)
            if route.get("domain") == "bounty":
                valid, note = self._validate_bounty_output(consensus.get("answer", ""))
                if not valid:
                    issues = quality.get("issues", [])
                    issues.append("bounty_format")
                    quality["issues"] = issues
                    quality["insufficient"] = True
                    if bool(self.config.quality.get("strict", True)):
                        consensus = self._insufficient_evidence_answer(query, quality, note=note)
                        artifacts = {
                            "route": route,
                            "context": context,
                            "deliberation": deliberation,
                            "quality": quality,
                            "reconcile": {
                                "previous_run_id": self._latest_for_meta(meta),
                                "changed": True,
                            },
                        }
                        self.store.finalize_run(run_id, consensus, artifacts)
                        audit.log("settlement.complete", {
                            "consensus": consensus,
                            "reconcile": artifacts["reconcile"],
                            "quality": quality,
                            "note": f"bounty_output_invalid: {note}",
                        })
                        return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)
            previous = self._latest_for_meta(meta)
            reconcile = {
                "previous_run_id": previous.get("id") if previous else None,
                "changed": bool(previous and previous.get("consensus", {}).get("answer") != consensus.get("answer")),
            }
            artifacts = {
                "route": route,
                "context": context,
                "deliberation": deliberation,
                "quality": quality,
                "reconcile": reconcile,
            }
            self.store.finalize_run(run_id, consensus, artifacts)
            audit.log("settlement.complete", {
                "consensus": consensus,
                "reconcile": reconcile,
            })
            return PipelineResult(run_id=run_id, consensus=consensus, artifacts=artifacts)
        except Exception as exc:
            self.store.fail_run(run_id, str(exc))
            audit.log("run.failed", {"error": str(exc)})
            raise
        finally:
            self._audit = None
            self._run_id = None
            self._context_char_limit = None

    def _calibrate_models(self, run_id: str) -> None:
        calibration = self.config.calibration
        if not calibration.get("enabled", True):
            return
        providers = calibration.get("providers") or ["ollama"]
        max_seconds = float(calibration.get("max_seconds", 20))
        prompt = calibration.get("ping_prompt", "Return only: OK")
        start = time.perf_counter()
        for card in self.registry.list_models():
            if time.perf_counter() - start > max_seconds:
                break
            model_id = card.get("id", "")
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
            try:
                self.registry.update_metrics(model_id, observation)
            except Exception:
                pass
            self.store.append_event(run_id, {"phase": "calibration", "model": model_id, "ok": result.ok})

    def _route_query(self, query: str, collections: Optional[List[str]]) -> Dict[str, Any]:
        q = query.lower()
        domain = "general"
        needs_tax = False
        tax_keywords = ["tax", "irs", "1099", "w-2", "w2", "basis", "deduction", "section", "schedule f", "passive", "material participation", "hobby loss"]
        if any(word in q for word in ["tax", "irs", "1099", "w-2", "w2", "basis", "deduction"]):
            domain = "tax"
            needs_tax = True
        if any(word in q for word in ["health", "lab", "blood", "apoe", "genetic", "medication", "vitamin", "supplement", "multivitamin"]):
            domain = "health"
        if any(word in q for word in ["money", "portfolio", "allocation", "asset mix", "rebalance", "invest"]):
            domain = "money"
        if any(word in q for word in ["farm", "farmland", "orchard", "agriculture", "acreage", "ranch", "livestock", "soil"]):
            domain = "agriculture"
        if any(word in q for word in tax_keywords):
            needs_tax = True
        if any(word in q for word in ["bounty", "vuln", "exploit", "smart contract", "immunefi"]):
            domain = "bounty"
        base = collections or self.config.rag.get("domain_collections", {}).get(domain) or self.config.rag.get("default_collections", [])
        required_collections: List[str] = []
        if needs_tax:
            tax_collections = self.config.rag.get("domain_collections", {}).get("tax", [])
            for item in tax_collections:
                if item not in base:
                    base.append(item)
            required_collections = list(tax_collections)
        selected, catalog = self._expand_collections(domain, base, explicit=bool(collections))
        roles = ["router", "reasoner", "critic", "summarizer"]
        plan_with_rationale = self.planner.plan_with_rationale(roles, self.registry.list_models())
        return {
            "domain": domain,
            "collections": selected,
            "required_collections": required_collections,
            "roles": roles,
            "plan": plan_with_rationale["assignments"],
            "rationale": plan_with_rationale["rationale"],
            "rag_catalog": catalog,
        }

    def _retrieve_context(self, query: str, route: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rag_results: List[Dict[str, Any]] = []
        rag_cfg = self.config.rag
        max_per_collection = int(rag_cfg.get("max_results_per_collection", 8))
        prefer_non_pdf = bool(rag_cfg.get("prefer_non_pdf", False))
        semantic = rag_cfg.get("semantic")
        for coll in route.get("collections", []):
            rag_results.extend(self.rag.search(query, collection=coll, limit=max_per_collection, semantic=semantic))
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
        source_items: List[Dict[str, Any]] = []
        source_errors: List[Dict[str, Any]] = []
        if route.get("domain") == "health":
            source_items.extend(self.health_source.fetch())
            source_errors.extend(self.health_source.drain_errors())
        if route.get("domain") == "money":
            source_items.extend(self.money_source.fetch())
            source_errors.extend(self.money_source.drain_errors())
        if route.get("domain") == "bounty" and user_inputs:
            instructions = user_inputs[0].get("full_text") or ""
            extra_queries = self._extract_focus_queries(str(instructions))
            for query_term in extra_queries:
                rag_results.extend(self.rag.search(query_term, limit=3))
                file_results.extend(self.rag.search_files(query_term, limit=3))
                if self.index.db_path.exists():
                    nas_results.extend(self.index.search(query_term, limit=3))
            base_dir = user_inputs[0].get("base_dir")
            file_paths = self._extract_file_paths(str(instructions))
            for rel_path in file_paths[:8]:
                full_path = Path(rel_path)
                if not full_path.is_absolute() and base_dir:
                    full_path = Path(str(base_dir)) / rel_path
                if not full_path.exists():
                    continue
                snippet = self._read_file_excerpt(full_path)
                if not snippet:
                    continue
                source_items.append({
                    "path": str(full_path),
                    "title": rel_path,
                    "snippet": snippet,
                    "collection": "bounty-target",
                    "source": "bounty-target",
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
            user_items=user_inputs,
            source_items=source_items,
        )
        rag_errors = self.rag.drain_errors()
        if rag_errors:
            stats["rag_errors"] = rag_errors
        if source_errors:
            stats["source_errors"] = source_errors
        if user_inputs:
            stats["input_path"] = user_inputs[0].get("path")
        return {
            "rag": rag_results,
            "nas": combined_files,
            "sources": source_items,
            "evidence": evidence,
            "stats": stats,
            "user_inputs": user_inputs,
        }

    def _extract_focus_queries(self, instructions: str) -> list[str]:
        if not instructions:
            return []
        import re
        from pathlib import Path
        queries: list[str] = []
        paths = re.findall(r"(?:programs|crates)/[A-Za-z0-9_./-]+\\.rs", instructions)
        for path in paths:
            queries.append(path)
            queries.append(Path(path).name)
        questions = re.findall(r"\\*\\*Question:\\*\\*\\s*(.+)", instructions)
        for q in questions:
            queries.append(q.strip())
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
        if meta and meta.get("prompt_id"):
            latest = self.store.latest_for_prompt(str(meta.get("prompt_id")))
            if latest:
                return latest
        return self.store.latest()

    def _extract_file_paths(self, instructions: str) -> list[str]:
        if not instructions:
            return []
        import re
        paths = re.findall(r"(?:programs|crates)/[A-Za-z0-9_./-]+\\.rs", instructions)
        unique = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return unique

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
        reasoner_model = plan.get("reasoner") or next(iter(plan.values()), None)
        critic_model = plan.get("critic") or reasoner_model
        config = self.config.raw.get("deliberation", {})
        max_rounds = int(config.get("max_rounds", 3))
        require_agreement = bool(config.get("require_agreement", True))
        context_blob = self._format_context(context)
        instructions = self._user_instructions(context)
        domain = route.get("domain")
        domain_instructions = ""
        if domain == "bounty":
            domain_instructions = (
                "For bounty analysis, only propose findings that are supported by evidence snippets.\n"
                "You must cite file paths and line numbers. If you cannot, say \"INSUFFICIENT EVIDENCE\".\n"
            )
        rounds = []
        reasoner_out = ""
        critic_out = ""
        agreement = False

        for round_idx in range(1, max_rounds + 1):
            analysis_prompt = (
                "You are the reasoner. Provide a careful analysis and propose a decision.\n\n"
                f"Question: {query}\n\nContext:\n{context_blob}\n"
            )
            if round_idx > 1:
                analysis_prompt = (
                    "You are the reasoner. Revise the decision to address the critic's feedback.\n\n"
                    f"Question: {query}\n\nContext:\n{context_blob}\n\n"
                    f"Previous draft:\n{reasoner_out}\n\n"
                    f"Critic feedback:\n{critic_out}\n"
                )
            if domain_instructions or instructions:
                analysis_prompt += f"\n{domain_instructions}\n"
                if instructions:
                    analysis_prompt += f"Instructions from input:\n{instructions}\n"
            reasoner_out = self._call_model(reasoner_model, analysis_prompt, role="reasoner")

            critic_prompt = (
                "You are the critic. Challenge the reasoning, list disagreements and gaps, and suggest fixes.\n"
                "Return sections:\nDisagreements:\n- ...\nGaps:\n- ...\nVerdict:\nAGREE or DISAGREE\n\n"
                f"Question: {query}\n\nReasoner draft:\n{reasoner_out}\n"
            )
            if domain_instructions or instructions:
                critic_prompt += f"\n{domain_instructions}\n"
                if instructions:
                    critic_prompt += f"Instructions from input:\n{instructions}\n"
            critic_out = self._call_model(critic_model, critic_prompt, role="critic")
            agreement = self._critic_agrees(critic_out)
            round_entry = {
                "round": round_idx,
                "agreement": agreement,
                "disagreements": self._extract_disagreements(critic_out),
            }
            rounds.append(round_entry)
            if self._run_id:
                self.store.append_event(self._run_id, {"phase": "deliberate", **round_entry})
            if agreement or not require_agreement:
                break

        return {
            "reasoner": reasoner_out,
            "critic": critic_out,
            "disagreements": self._extract_disagreements(critic_out),
            "rounds": rounds,
            "agreement": agreement,
        }

    def _critic_agrees(self, critic: str) -> bool:
        lower = critic.lower()
        for line in critic.splitlines():
            if "verdict" in line.lower():
                verdict = line.split(":")[-1].strip().lower()
                if verdict in {"agree", "agreed", "yes", "accept", "approved"}:
                    return True
                if verdict in {"disagree", "no", "reject"}:
                    return False
        return "verdict" in lower and "agree" in lower and "disagree" not in lower

    def _summarize(self, query: str, context: Dict[str, Any], deliberation: Dict[str, Any], route: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        summarizer_model = plan.get("summarizer") or plan.get("reasoner")
        context_blob = self._format_context(context)
        instructions = self._user_instructions(context)
        domain = route.get("domain")
        evidence_hint = (
            f"Evidence count: {quality.get('evidence_count', 0)}, "
            f"pdf_ratio: {quality.get('pdf_ratio', 0):.2f}, "
            f"off_domain_ratio: {quality.get('off_domain_ratio', 0):.2f}, "
            f"signal: {quality.get('max_signal_score', 0):.2f}"
        )
        domain_instructions = ""
        if domain == "bounty":
            domain_instructions = (
                "For bounty output, follow this format for each finding:\n"
                "- Location (file:line)\n"
                "- Root cause\n"
                "- Attack scenario\n"
                "- Severity assessment\n"
                "- Why it's not a false positive\n"
                "Only list findings that are directly supported by the provided evidence and cite file paths.\n"
                "If you cannot cite file paths with line numbers from evidence, say \"INSUFFICIENT EVIDENCE\" and list what code files are needed.\n"
            )
        summary_prompt = (
            "You are the summarizer. Produce a final consensus answer with bullet points."
            " Include an Evidence section listing the top sources (file paths or collection names)."
            " Include Risks/Uncertainties and Follow-ups. Include a confidence level (low/medium/high).\n\n"
            f"Question: {query}\n\nContext:\n{context_blob}\n\n"
            f"Evidence quality: {evidence_hint}\n\n"
            f"{domain_instructions}\n"
            f"{'Instructions from input:\\n' + instructions + '\\n' if instructions else ''}"
            f"Reasoner notes:\n{deliberation.get('reasoner', '')}\n\n"
            f"Critic notes:\n{deliberation.get('critic', '')}\n"
        )
        summary = self._call_model(summarizer_model, summary_prompt, role="summarizer")
        fallback_used = False
        if not summary.strip():
            summary = self._fallback_summary(query, deliberation)
            fallback_used = True
        model_conf = self._extract_confidence(summary)
        auto_conf = self._auto_confidence(quality)
        final_conf = self._merge_confidence(model_conf, auto_conf)
        return {
            "answer": summary.strip(),
            "confidence": final_conf,
            "confidence_model": model_conf,
            "confidence_auto": auto_conf,
            "pope": summary.strip().splitlines()[0] if summary.strip() else "",
            "fallback_used": fallback_used,
            "insufficient_evidence": False,
        }

    def _call_model(self, model_id: Optional[str], prompt: str, role: Optional[str] = None) -> str:
        if not model_id:
            return ""
        if model_id.startswith("ollama:"):
            model = model_id.split(":", 1)[1]
            result = self.ollama.generate(model, prompt, temperature=0.2)
            self._record_model_observation(model_id, result)
            audit = self._audit
            run_id = self._run_id
            payload = {
                "role": role,
                "model_id": model_id,
                "ok": result.ok,
                "duration_ms": round(result.duration_ms, 2),
                "error": result.error,
            }
            if audit:
                audit.log("model.call", payload)
            if run_id:
                self.store.append_event(run_id, {"phase": "model", **payload})
            return result.text
        if model_id.startswith("cli:"):
            result = self._call_cli_model(model_id, prompt, role=role)
            self._record_model_observation(model_id, result)
            audit = self._audit
            run_id = self._run_id
            payload = {
                "role": role,
                "model_id": model_id,
                "ok": result.ok,
                "duration_ms": round(result.duration_ms, 2),
                "error": result.error,
                "stderr": result.stderr,
            }
            if audit:
                audit.log("model.call", payload)
            if run_id:
                self.store.append_event(run_id, {"phase": "model", **payload})
            return result.text
        return ""

    def _call_cli_model(self, model_id: str, prompt: str, role: Optional[str] = None) -> Any:
        card = self.registry.get_model(model_id) or {}
        command = card.get("command") or []
        prompt_mode = card.get("prompt_mode", "arg")
        stdin_flag = card.get("stdin_flag")
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
        lines = [
            "**Fallback Consensus Answer:**",
            "",
            f"- **Query**: {query}",
            "",
            "**Reasoner Draft:**",
            reasoner.strip() or "No reasoner output.",
            "",
            "**Critic Disagreements:**",
        ]
        if disagreements:
            for item in disagreements[:5]:
                lines.append(f"- {item}")
        elif critic.strip():
            lines.append(critic.strip()[:800])
        else:
            lines.append("No critic output.")
        lines.append("")
        lines.append("**Confidence Level**: Low")
        return "\n".join(lines)

    def _insufficient_evidence_answer(self, query: str, quality: Dict[str, Any], note: str | None = None) -> Dict[str, Any]:
        details = []
        details.append(f"- Evidence count: {quality.get('evidence_count', 0)} (min {self.config.quality.get('min_evidence', 2)})")
        details.append(f"- Max signal score: {quality.get('max_signal_score', 0):.2f}")
        details.append(f"- PDF ratio: {quality.get('pdf_ratio', 0):.2f}")
        if quality.get("rag_errors"):
            details.append(f"- Retrieval errors: {len(quality.get('rag_errors', []))} (rag.tannner.com)")
        if quality.get("source_errors"):
            details.append(f"- Source fetch errors: {len(quality.get('source_errors', []))} (health/money)")
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
            "- Specify the target collection explicitly (e.g., tax-rag, health-rag).",
            "- Run `conclave index` if NAS context is missing.",
            "- Add/clean high-signal sources (markdown or structured notes).",
            "- Check `rag.tannner.com` availability if retrieval failed.",
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
                    lines.append(stripped.lstrip("- ").strip())
                elif numbered.match(stripped):
                    lines.append(numbered.sub("", stripped).strip())
        return lines

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

    def _validate_bounty_output(self, summary: str) -> tuple[bool, str]:
        if not summary.strip():
            return False, "empty output"
        lower = summary.lower()
        if "insufficient evidence" in lower:
            return False, "model reported insufficient evidence"
        import re
        if "location" not in lower:
            return False, "missing Location field"
        path_line = re.search(r"(programs/|crates/|tests?/).+?:\\d+", summary)
        if not path_line:
            return False, "missing file:line references"
        return True, ""

    def _score_item(
        self,
        item: Dict[str, Any],
        source: str,
        preferred_collections: List[str] | None = None,
        domain: str | None = None,
        domain_paths: Dict[str, List[str]] | None = None,
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
        user_items: List[Dict[str, Any]] | None = None,
        source_items: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        items = []
        for item in rag:
            items.append(self._score_item(item, "rag", preferred_collections, domain, domain_paths))
        for item in nas:
            enriched = self._maybe_attach_line(item)
            items.append(self._score_item(enriched, "nas", preferred_collections, domain, domain_paths))
        if user_items:
            items.extend([
                self._score_item(item, "user", preferred_collections, domain, domain_paths) for item in user_items
            ])
        if source_items:
            items.extend([
                self._score_item(item, "source", preferred_collections, domain, domain_paths) for item in source_items
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
        min_strong = int(self.config.quality.get("min_strong_evidence", 1))
        min_content = int(self.config.quality.get("min_content_evidence", 1))
        min_non_user = int(self.config.quality.get("min_non_user_evidence", 1))
        min_required = int(self.config.quality.get("min_required_collection_hits", 1))
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
        issues = []
        if evidence_count < min_evidence:
            issues.append("insufficient_evidence")
        if max_signal < low_signal_threshold:
            issues.append("low_signal")
        if strong_count < min_strong or content_count < min_content or non_user_count < min_non_user:
            issues.append("low_relevance")
        if required_hits < min_required:
            issues.append("missing_required_evidence")
        if pdf_ratio > pdf_ratio_limit:
            issues.append("pdf_heavy")
        if domain_known and off_domain_ratio > off_domain_limit:
            issues.append("off_domain")
        if stats.get("rag_errors"):
            issues.append("rag_errors")
        if stats.get("source_errors"):
            issues.append("source_errors")
        return {
            "evidence_count": evidence_count,
            "max_signal_score": max_signal,
            "avg_signal_score": avg_signal,
            "pdf_ratio": pdf_ratio,
            "off_domain_ratio": off_domain_ratio,
            "rag_errors": stats.get("rag_errors", []),
            "source_errors": stats.get("source_errors", []),
            "issues": issues,
            "insufficient": bool(issues and ("insufficient_evidence" in issues or "low_signal" in issues or "low_relevance" in issues)),
        }

    def _expand_collections(self, domain: str, base: List[str], explicit: bool = False) -> tuple[List[str], Dict[str, Any]]:
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
