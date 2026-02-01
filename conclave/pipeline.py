"""Core decision pipeline for Conclave."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from conclave.config import Config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner
from conclave.models.ollama import OllamaClient
from conclave.rag import RagClient, NasIndex
from conclave.store import DecisionStore
from conclave.audit import AuditLog


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

    def run(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> PipelineResult:
        run_id = run_id or self.store.create_run(query, meta=meta)
        audit = AuditLog(self.store.run_dir(run_id) / "audit.jsonl")
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
            context = self._retrieve_context(query, route)
            self.store.append_event(run_id, {"phase": "retrieve", "status": "done", "context": {"rag": len(context["rag"]), "nas": len(context["nas"])}})
            audit.log("retrieve.complete", {
                "rag_count": len(context["rag"]),
                "nas_count": len(context["nas"]),
                "rag_samples": context["rag"][:3],
                "nas_samples": context["nas"][:3],
            })

            self.store.append_event(run_id, {"phase": "deliberate", "status": "start"})
            deliberation = self._deliberate(query, context, route)
            self.store.append_event(run_id, {"phase": "deliberate", "status": "done"})
            audit.log("deliberate.complete", {
                "reasoner": deliberation.get("reasoner", ""),
                "critic": deliberation.get("critic", ""),
                "disagreements": deliberation.get("disagreements", []),
            })

            consensus = self._summarize(query, context, deliberation, route)
            previous = self.store.latest()
            reconcile = {
                "previous_run_id": previous.get("id") if previous else None,
                "changed": bool(previous and previous.get("consensus", {}).get("answer") != consensus.get("answer")),
            }
            artifacts = {
                "route": route,
                "context": context,
                "deliberation": deliberation,
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
        selected = collections or self.config.rag.get("domain_collections", {}).get(domain) or self.config.rag.get("default_collections", [])
        roles = ["router", "reasoner", "critic", "summarizer"]
        plan_with_rationale = self.planner.plan_with_rationale(roles, self.registry.list_models())
        return {
            "domain": domain,
            "collections": selected,
            "roles": roles,
            "plan": plan_with_rationale["assignments"],
            "rationale": plan_with_rationale["rationale"],
        }

    def _retrieve_context(self, query: str, route: Dict[str, Any]) -> Dict[str, Any]:
        rag_results: List[Dict[str, Any]] = []
        for coll in route.get("collections", []):
            rag_results.extend(self.rag.search(query, collection=coll, limit=10))
        nas_results = []
        file_results = self.rag.search_files(query, limit=10)
        if self.config.index.get("enabled", True):
            self._maybe_refresh_index()
            nas_results = self.index.search(query, limit=10)
        combined_files = file_results + nas_results
        return {"rag": rag_results, "nas": combined_files}

    def _deliberate(self, query: str, context: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        reasoner_model = plan.get("reasoner") or next(iter(plan.values()), None)
        critic_model = plan.get("critic") or reasoner_model

        context_blob = self._format_context(context)
        analysis_prompt = (
            "You are the reasoner. Provide a careful analysis and propose a decision.\n\n"
            f"Question: {query}\n\nContext:\n{context_blob}\n"
        )
        reasoner_out = self._call_model(reasoner_model, analysis_prompt)

        critic_prompt = (
            "You are the critic. Challenge the reasoning, list disagreements and gaps, and suggest fixes.\n"
            "Return sections:\nDisagreements:\n- ...\nGaps:\n- ...\nVerdict:\n- ...\n\n"
            f"Question: {query}\n\nReasoner draft:\n{reasoner_out}\n"
        )
        critic_out = self._call_model(critic_model, critic_prompt)
        return {
            "reasoner": reasoner_out,
            "critic": critic_out,
            "disagreements": self._extract_disagreements(critic_out),
        }

    def _summarize(self, query: str, context: Dict[str, Any], deliberation: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        plan = route.get("plan", {})
        summarizer_model = plan.get("summarizer") or plan.get("reasoner")
        context_blob = self._format_context(context)
        summary_prompt = (
            "You are the summarizer. Produce a final consensus answer with bullet points and cite key evidence."
            " Include a confidence level (low/medium/high).\n\n"
            f"Question: {query}\n\nContext:\n{context_blob}\n\n"
            f"Reasoner notes:\n{deliberation.get('reasoner', '')}\n\n"
            f"Critic notes:\n{deliberation.get('critic', '')}\n"
        )
        summary = self._call_model(summarizer_model, summary_prompt)
        return {
            "answer": summary.strip(),
            "confidence": self._extract_confidence(summary),
            "pope": summary.strip().splitlines()[0] if summary.strip() else "",
        }

    def _call_model(self, model_id: Optional[str], prompt: str) -> str:
        if not model_id:
            return ""
        if model_id.startswith("ollama:"):
            model = model_id.split(":", 1)[1]
            result = self.ollama.generate(model, prompt, temperature=0.2)
            return result.text
        return ""

    def _format_context(self, context: Dict[str, Any]) -> str:
        lines = []
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

    def _extract_disagreements(self, critic: str) -> list[str]:
        lines = []
        capture = False
        for line in critic.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("disagreements"):
                capture = True
                continue
            if capture:
                if stripped.lower().startswith("gaps"):
                    break
                if stripped.startswith("-"):
                    lines.append(stripped.lstrip("- ").strip())
        return lines

    def _maybe_refresh_index(self) -> None:
        db_path = self.index.db_path
        if not db_path.exists():
            self.index.index()
            return
        age_seconds = time.time() - db_path.stat().st_mtime
        if age_seconds > 7 * 24 * 3600:
            self.index.index()
