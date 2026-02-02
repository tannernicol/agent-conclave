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
from conclave.rag import RagClient, NasIndex
from conclave.store import DecisionStore
from conclave.audit import AuditLog
from conclave.mcp import load_mcp_servers


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
        self.rag = RagClient(config.rag.get("base_url", "http://localhost:8091"))
        self.index = NasIndex(
            data_dir=config.data_dir,
            allowlist=config.index.get("allowlist", []),
            exclude_patterns=config.index.get("exclude_patterns", []),
            max_file_mb=int(config.index.get("max_file_mb", 2)),
        )
        self.store = DecisionStore(config.data_dir)
        self._audit: AuditLog | None = None
        self._run_id: str | None = None
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
            self.store.append_event(run_id, {"phase": "route", "status": "done", "route": route})
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
                "rag_samples": context["rag"][:3],
                "nas_samples": context["nas"][:3],
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
                        "previous_run_id": self.store.latest().get("id") if self.store.latest() else None,
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
            self.store.append_event(run_id, {"phase": "deliberate", "status": "done"})
            audit.log("deliberate.complete", {
                "reasoner": deliberation.get("reasoner", ""),
                "critic": deliberation.get("critic", ""),
                "disagreements": deliberation.get("disagreements", []),
            })

            consensus = self._summarize(query, context, deliberation, route, quality)
            previous = self.store.latest()
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

    def _calibrate_models(self, run_id: str) -> None:
        calibration = self.config.calibration
        if not calibration.get("enabled", True):
            return
        max_seconds = float(calibration.get("max_seconds", 20))
        prompt = calibration.get("ping_prompt", "Return only: OK")
        start = time.perf_counter()
        for card in self.registry.list_models():
            if time.perf_counter() - start > max_seconds:
                break
            model_id = card.get("id", "")
            if not model_id.startswith("ollama:"):
                continue
            model_name = model_id.split(":", 1)[1]
            result = self.ollama.generate(model_name, prompt, temperature=0)
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
        if any(word in q for word in ["tax", "irs", "1099", "w-2", "w2", "basis", "deduction"]):
            domain = "tax"
        if any(word in q for word in ["health", "lab", "blood", "apoe", "genetic", "medication"]):
            domain = "health"
        if any(word in q for word in ["bounty", "vuln", "exploit", "smart contract", "immunefi"]):
            domain = "bounty"
        base = collections or self.config.rag.get("domain_collections", {}).get(domain) or self.config.rag.get("default_collections", [])
        selected, catalog = self._expand_collections(domain, base, explicit=bool(collections))
        roles = ["router", "reasoner", "critic", "summarizer"]
        plan_with_rationale = self.planner.plan_with_rationale(roles, self.registry.list_models())
        return {
            "domain": domain,
            "collections": selected,
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
        evidence, stats = self._select_evidence(
            rag_results,
            combined_files,
            preferred_collections=route.get("collections", []),
            domain=route.get("domain"),
            domain_paths=self.config.quality.get("domain_paths", {}),
            user_items=user_inputs,
        )
        rag_errors = self.rag.drain_errors()
        if rag_errors:
            stats["rag_errors"] = rag_errors
        if user_inputs:
            stats["input_path"] = user_inputs[0].get("path")
        return {"rag": rag_results, "nas": combined_files, "evidence": evidence, "stats": stats, "user_inputs": user_inputs}

    def _deliberate(self, query: str, context: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        reasoner_model = plan.get("reasoner") or next(iter(plan.values()), None)
        critic_model = plan.get("critic") or reasoner_model

        context_blob = self._format_context(context)
        analysis_prompt = (
            "You are the reasoner. Provide a careful analysis and propose a decision.\n\n"
            f"Question: {query}\n\nContext:\n{context_blob}\n"
        )
        reasoner_out = self._call_model(reasoner_model, analysis_prompt, role="reasoner")

        critic_prompt = (
            "You are the critic. Challenge the reasoning, list disagreements and gaps, and suggest fixes.\n"
            "Return sections:\nDisagreements:\n- ...\nGaps:\n- ...\nVerdict:\n- ...\n\n"
            f"Question: {query}\n\nReasoner draft:\n{reasoner_out}\n"
        )
        critic_out = self._call_model(critic_model, critic_prompt, role="critic")
        return {
            "reasoner": reasoner_out,
            "critic": critic_out,
            "disagreements": self._extract_disagreements(critic_out),
        }

    def _summarize(self, query: str, context: Dict[str, Any], deliberation: Dict[str, Any], route: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        summarizer_model = plan.get("summarizer") or plan.get("reasoner")
        context_blob = self._format_context(context)
        evidence_hint = (
            f"Evidence count: {quality.get('evidence_count', 0)}, "
            f"pdf_ratio: {quality.get('pdf_ratio', 0):.2f}, "
            f"off_domain_ratio: {quality.get('off_domain_ratio', 0):.2f}, "
            f"signal: {quality.get('max_signal_score', 0):.2f}"
        )
        summary_prompt = (
            "You are the summarizer. Produce a final consensus answer with bullet points."
            " Include an Evidence section listing the top sources (file paths or collection names)."
            " Include Risks/Uncertainties and Follow-ups. Include a confidence level (low/medium/high).\n\n"
            f"Question: {query}\n\nContext:\n{context_blob}\n\n"
            f"Evidence quality: {evidence_hint}\n\n"
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
        return ""

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
                title = item.get("title") or item.get("path")
                source = item.get("source", "context").upper()
                meta = f"{item.get('collection', '')}".strip()
                if meta:
                    meta = f" ({meta})"
                lines.append(
                    f"- [{source}] {title}{meta}: {item.get('snippet') or ''} [signal={item.get('signal_score', 0):.2f}]"
                )
            return "\n".join(lines)
        for item in context.get("rag", [])[:12]:
            title = item.get("title") or item.get("name") or item.get("path")
            lines.append(f"- [RAG] {title}: {item.get('snippet') or item.get('match_line') or ''}")
        for item in context.get("nas", [])[:12]:
            lines.append(f"- [NAS] {item.get('title')}: {item.get('snippet', '')}")
        return "\n".join(lines)

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

    def _insufficient_evidence_answer(self, query: str, quality: Dict[str, Any]) -> Dict[str, Any]:
        details = []
        details.append(f"- Evidence count: {quality.get('evidence_count', 0)} (min {self.config.quality.get('min_evidence', 2)})")
        details.append(f"- Max signal score: {quality.get('max_signal_score', 0):.2f}")
        details.append(f"- PDF ratio: {quality.get('pdf_ratio', 0):.2f}")
        if quality.get("rag_errors"):
            details.append(f"- Retrieval errors: {len(quality.get('rag_errors', []))} (rag.tannner.com)")
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
            "collection": "user-input",
        }]

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
        score_val = 0.0
        try:
            score_val = float(item.get("score", 0.0))
        except Exception:
            score_val = 0.0
        ext = Path(path).suffix.lower() if path else ""
        if source == "user":
            signal = 2.0
        else:
            signal = 1.0 + min(score_val, 1.0) * 0.4
        if ext == ".pdf":
            signal -= 0.3
        elif ext in {".md", ".txt", ".json", ".yaml", ".yml", ".toml"}:
            signal += 0.3
        if match_type == "filename" and source != "user":
            signal -= 0.2
        if len(snippet.strip()) < 40 and source != "user":
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
        if source == "user":
            domain_known = True
            on_domain = True
        if preferred_collections and domain_known:
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
        }

    def _select_evidence(
        self,
        rag: List[Dict[str, Any]],
        nas: List[Dict[str, Any]],
        limit: int = 12,
        preferred_collections: List[str] | None = None,
        domain: str | None = None,
        domain_paths: Dict[str, List[str]] | None = None,
        user_items: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        items = [
            self._score_item(item, "rag", preferred_collections, domain, domain_paths) for item in rag
        ] + [
            self._score_item(item, "nas", preferred_collections, domain, domain_paths) for item in nas
        ]
        if user_items:
            items.extend([
                self._score_item(item, "user", preferred_collections, domain, domain_paths) for item in user_items
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
        }
        return selected, stats

    def _evaluate_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stats = context.get("stats", {})
        min_evidence = int(self.config.quality.get("min_evidence", 2))
        pdf_ratio_limit = float(self.config.quality.get("pdf_ratio_limit", 0.7))
        off_domain_limit = float(self.config.quality.get("off_domain_ratio_limit", 0.5))
        low_signal_threshold = float(self.config.quality.get("low_signal_threshold", 0.5))
        evidence_count = int(stats.get("evidence_count", 0))
        max_signal = float(stats.get("max_signal_score", 0))
        avg_signal = float(stats.get("avg_signal_score", 0))
        pdf_ratio = float(stats.get("pdf_ratio", 0))
        off_domain_ratio = float(stats.get("off_domain_ratio", 0))
        domain_known = int(stats.get("domain_known", 0))
        issues = []
        if evidence_count < min_evidence:
            issues.append("insufficient_evidence")
        if max_signal < low_signal_threshold:
            issues.append("low_signal")
        if pdf_ratio > pdf_ratio_limit:
            issues.append("pdf_heavy")
        if domain_known and off_domain_ratio > off_domain_limit:
            issues.append("off_domain")
        if stats.get("rag_errors"):
            issues.append("rag_errors")
        return {
            "evidence_count": evidence_count,
            "max_signal_score": max_signal,
            "avg_signal_score": avg_signal,
            "pdf_ratio": pdf_ratio,
            "off_domain_ratio": off_domain_ratio,
            "rag_errors": stats.get("rag_errors", []),
            "issues": issues,
            "insufficient": bool(issues and ("insufficient_evidence" in issues or "low_signal" in issues)),
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
