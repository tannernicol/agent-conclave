"""Command line interface for Conclave."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from conclave.config import get_config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner
from conclave.pipeline import ConclavePipeline, PipelineResult
from conclave.rag import FileIndex, RagClient
from conclave.scheduler import apply_schedule
from conclave.store import DecisionStore
from conclave.mcp_bridge import MCPBridge


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def _fail_on_insufficient(args: argparse.Namespace, config: Any) -> tuple[bool, int]:
    exit_code = int(config.quality.get("strict_exit_code", 0))
    if args.no_fail_on_insufficient:
        return False, exit_code
    if args.fail_on_insufficient:
        return True, exit_code
    if bool(config.quality.get("strict", False)) and exit_code:
        return True, exit_code
    return False, exit_code


def _build_meta(args: argparse.Namespace, input_path: str | None = None) -> dict | None:
    meta: dict[str, Any] = {}
    path = input_path or getattr(args, "input_file", None)
    if path:
        meta["input_path"] = path
    meta["source"] = "cli"
    output_type = getattr(args, "output_type", None)
    if output_type:
        meta["output_type"] = output_type
    if getattr(args, "max_evidence", None):
        meta["evidence_limit"] = args.max_evidence
    if getattr(args, "max_context_chars", None):
        meta["context_char_limit"] = args.max_context_chars
    if getattr(args, "token_budget_total", None) is not None:
        meta["token_budget_total"] = args.token_budget_total
    if getattr(args, "token_budget_remaining", None) is not None:
        meta["token_budget_remaining"] = args.token_budget_remaining
    if getattr(args, "token_budget_used", None) is not None:
        meta["token_budget_used"] = args.token_budget_used
    return meta or None


def _execute_run(
    pipeline: ConclavePipeline,
    query: str,
    collections: list[str] | None,
    meta: dict | None,
    args: argparse.Namespace,
) -> PipelineResult:
    store = pipeline.store
    run_id = store.create_run(query, meta=meta if meta else None)
    stop_event = threading.Event()
    timeout_thread = None
    progress_thread = None
    if getattr(args, "progress", False):
        progress_thread = threading.Thread(
            target=_progress_printer,
            args=(store, run_id, stop_event),
            daemon=True,
        )
        progress_thread.start()
        print(f"[conclave] run_id={run_id}", file=sys.stderr)

    if getattr(args, "max_seconds", None):
        timeout_seconds = int(args.max_seconds)
        def _timeout_guard() -> None:
            if stop_event.wait(timeout_seconds):
                return
            try:
                store.fail_run(run_id, f"run exceeded max_seconds={timeout_seconds}")
            except Exception:
                pass
            print(f"[conclave] run aborted: exceeded max_seconds={timeout_seconds}", file=sys.stderr)
            os._exit(2)
        timeout_thread = threading.Thread(target=_timeout_guard, daemon=True)
        timeout_thread.start()

    try:
        result = pipeline.run(query, collections=collections, meta=meta, run_id=run_id)
    except TimeoutError as exc:
        store.fail_run(run_id, str(exc))
        consensus = {
            "answer": f"### Conclave timed out\n\n{exc}",
            "confidence": "low",
            "confidence_model": "low",
            "confidence_auto": "low",
            "pope": "### Conclave timed out",
            "fallback_used": True,
            "insufficient_evidence": True,
        }
        result = PipelineResult(run_id=run_id, consensus=consensus, artifacts={})
    finally:
        stop_event.set()
        if progress_thread:
            progress_thread.join(timeout=1.0)
        if timeout_thread:
            timeout_thread.join(timeout=1.0)
    return result


def cmd_models(args: argparse.Namespace) -> None:
    config = get_config()
    registry = ModelRegistry.from_config(config.models)
    if args.models_cmd == "list":
        _print({"models": registry.list_models()})
    elif args.models_cmd == "status":
        _print({"models": registry.list_models()})
    elif args.models_cmd == "benchmark":
        pipeline = ConclavePipeline(config)
        pipeline._calibrate_models("cli-benchmark")
        _print({"ok": True})


def cmd_plan(args: argparse.Namespace) -> None:
    config = get_config()
    registry = ModelRegistry.from_config(config.models)
    planner = Planner.from_config(config.planner)
    roles = ["router", "reasoner", "critic", "summarizer"]
    plan = planner.choose_models_for_roles(roles, registry.list_models())
    _print({"roles": roles, "plan": plan})


def cmd_run(args: argparse.Namespace) -> None:
    config = get_config()
    pipeline = ConclavePipeline(config)
    meta = _build_meta(args)
    result = _execute_run(pipeline, args.query, args.collection, meta, args)
    _print({"run_id": result.run_id, "consensus": result.consensus})
    if args.output_md:
        run = pipeline.store.get_run(result.run_id) or {}
        _write_markdown_report(Path(args.output_md), run, result.consensus, result.artifacts)
    fail_on, exit_code = _fail_on_insufficient(args, config)
    if fail_on and result.consensus.get("insufficient_evidence"):
        raise SystemExit(exit_code or 2)


def cmd_iterate(args: argparse.Namespace) -> None:
    config = get_config()
    pipeline = ConclavePipeline(config)
    query = args.query or "Review current state and propose next steps"
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    max_runs = int(args.max_runs or 0)
    run_count = 0
    append_path = Path(args.append_md) if args.append_md else None
    stop_on_insufficient = bool(args.stop_on_insufficient)
    if not max_runs and args.sleep_seconds <= 0:
        args.sleep_seconds = 600
    while True:
        run_count += 1
        meta = _build_meta(args, input_path=args.input_file)
        result = _execute_run(pipeline, query, args.collection, meta, args)
        run = pipeline.store.get_run(result.run_id) or {}
        snapshot_path = output_dir / f"conclave-iter-{result.run_id}.md"
        latest_path = output_dir / "conclave-iter-latest.md"
        _write_markdown_report(snapshot_path, run, result.consensus, result.artifacts)
        _write_markdown_report(latest_path, run, result.consensus, result.artifacts)
        if append_path:
            _append_markdown_report(append_path, snapshot_path)
        print(json.dumps({
            "run_id": result.run_id,
            "status": run.get("status"),
            "insufficient": bool(result.consensus.get("insufficient_evidence")),
            "report": str(snapshot_path),
        }, indent=2))
        if stop_on_insufficient and result.consensus.get("insufficient_evidence"):
            break
        if max_runs and run_count >= max_runs:
            break
        if args.sleep_seconds:
            time.sleep(float(args.sleep_seconds))


def cmd_runs(args: argparse.Namespace) -> None:
    config = get_config()
    store = DecisionStore(config.data_dir)
    if args.runs_cmd == "latest":
        _print(store.latest() or {})
    elif args.runs_cmd == "list":
        _print({"runs": store.list_runs(limit=args.limit)})


def cmd_reconcile(args: argparse.Namespace) -> None:
    config = get_config()
    pipeline = ConclavePipeline(config)
    fail_on, exit_code = _fail_on_insufficient(args, config)
    def _topic_meta(item: dict) -> dict:
        meta = {"topic": item.get("id")}
        input_path = item.get("input_path")
        if input_path:
            meta["input_path"] = str(input_path)
        input_title = item.get("input_title")
        if input_title:
            meta["input_title"] = str(input_title)
        output_type = item.get("output_type")
        if output_type:
            meta["output_type"] = str(output_type)
        return meta
    def _write_topic_reports(item: dict, result: PipelineResult) -> None:
        output_md = item.get("output_md")
        append_md = item.get("append_md")
        if not output_md and not append_md:
            return
        run = pipeline.store.get_run(result.run_id) or {}
        if output_md:
            output_path = Path(str(output_md))
            _write_markdown_report(output_path, run, result.consensus, result.artifacts)
            if append_md:
                _append_markdown_report(Path(str(append_md)), output_path)
            return
        temp_path = config.data_dir / "reports" / f"conclave-{result.run_id}.md"
        _write_markdown_report(temp_path, run, result.consensus, result.artifacts)
        _append_markdown_report(Path(str(append_md)), temp_path)
    def _should_fail(consensus: dict) -> bool:
        return bool(fail_on and consensus.get("insufficient_evidence"))
    if args.topic == "all":
        results = []
        for item in config.topics:
            result = pipeline.run(
                item.get("query", ""),
                collections=item.get("collections"),
                meta=_topic_meta(item),
            )
            results.append({"topic": item.get("id"), "run_id": result.run_id, "consensus": result.consensus})
            _write_topic_reports(item, result)
            if _should_fail(result.consensus):
                _print({"results": results, "error": "insufficient_evidence"})
                raise SystemExit(exit_code or 2)
        _print({"results": results})
        return
    topic = None
    for item in config.topics:
        if item.get("id") == args.topic:
            topic = item
            break
    if not topic:
        raise SystemExit(f"Unknown topic: {args.topic}")
    result = pipeline.run(
        topic.get("query", ""),
        collections=topic.get("collections"),
        meta=_topic_meta(topic),
    )
    _print({"run_id": result.run_id, "consensus": result.consensus})
    _write_topic_reports(topic, result)
    if _should_fail(result.consensus):
        raise SystemExit(exit_code or 2)


def cmd_index(args: argparse.Namespace) -> None:
    config = get_config()
    index = FileIndex(
        data_dir=config.data_dir,
        allowlist=config.index.get("allowlist", []),
        exclude_patterns=config.index.get("exclude_patterns", []),
        max_file_mb=int(config.index.get("max_file_mb", 2)),
    )
    indexed = index.index()
    _print({"indexed": indexed})


def cmd_audit(args: argparse.Namespace) -> None:
    config = get_config()
    from conclave.quality_audit import run_audit
    output_dir = Path(args.output_dir) if args.output_dir else (config.data_dir / "audits")
    result = run_audit(
        config,
        mode=args.mode,
        output_dir=output_dir,
        fetch_sources=not args.no_fetch,
    )
    _print({"ok": True, "audit": result.payload, "json": str(result.json_path), "md": str(result.md_path)})
    if args.fail_on_issues:
        rag = result.payload.get("rag") or {}
        mcp = result.payload.get("mcp") or {}
        issues = 0
        issues += len(rag.get("collections_empty") or [])
        issues += len(rag.get("missing_reliability") or [])
        issues += len(rag.get("allowlist_missing") or {})
        for check in (mcp.get("checks") or []):
            if check.get("ok") is False:
                issues += 1
        if issues:
            raise SystemExit(2)


def cmd_schedule(args: argparse.Namespace) -> None:
    config = get_config()
    if args.schedule_cmd == "list":
        topics = []
        for topic in config.topics:
            topics.append({
                "id": topic.get("id"),
                "schedule": topic.get("schedule", "weekly"),
                "enabled": topic.get("enabled", True),
                "collections": topic.get("collections", []),
            })
        _print({"topics": topics})
        return
    if args.schedule_cmd == "apply":
        result = apply_schedule(
            config.topics,
            unit_dir=Path(args.unit_dir).expanduser(),
            enable=args.enable,
            reload_systemd=not args.no_reload,
            dry_run=args.dry_run,
            validate=not args.no_validate,
            disable_legacy=args.disable_legacy,
        )
        _print({
            "created": [
                {
                    "topic": unit.topic_id,
                    "service": str(unit.service_path),
                    "timer": str(unit.timer_path),
                    "schedule": unit.schedule,
                }
                for unit in result["created"]
            ],
            "enabled": bool(args.enable),
            "dry_run": bool(args.dry_run),
            "warnings": result["warnings"],
            "errors": result["errors"],
        })
        if result["errors"]:
            raise SystemExit(1)
    return


def _check_search_health(config: Any) -> dict:
    verification = config.raw.get("verification", {}) or {}
    search_cfg = verification.get("search", {}) or {}
    base_url = str(search_cfg.get("base_url", "")).rstrip("/")
    endpoint = str(search_cfg.get("endpoint", "/search"))
    if not base_url:
        return {"ok": True, "skipped": True, "reason": "search not configured"}
    try:
        import httpx
        with httpx.Client(timeout=float(search_cfg.get("timeout", 10))) as client:
            resp = client.post(f"{base_url}{endpoint}", data={"q": "health check", "format": "json"})
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])
        return {"ok": True, "results": len(results)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "base_url": base_url}


def _health_payload(config: Any) -> dict:
    pipeline = ConclavePipeline(config)
    required = pipeline._check_required_models()
    rag = RagClient(config.rag.get("base_url", "http://localhost:8091"))
    rag_ok = rag.health_check()
    search = _check_search_health(config)
    ok = bool(required.get("ok", True) and rag_ok and search.get("ok", True))
    return {
        "ok": ok,
        "required_models": required,
        "rag": {"ok": rag_ok, "base_url": config.rag.get("base_url", "http://localhost:8091")},
        "search": search,
    }


def cmd_health(args: argparse.Namespace) -> None:
    config = get_config()
    payload = _health_payload(config)
    _print(payload)
    if not payload.get("ok", False):
        raise SystemExit(2)


def _routing_payload(config: Any) -> dict:
    pipeline = ConclavePipeline(config)
    validation = config.raw.get("routing_validation", {}) or {}
    if not validation.get("enabled", True):
        return {"ok": True, "skipped": True, "reason": "routing_validation disabled"}
    cases = validation.get("cases", []) or []
    results = []
    failures = 0
    for case in cases:
        query = str(case.get("query") or "").strip()
        expected_domain = str(case.get("expect_domain") or "").strip()
        expected_collections = list(case.get("expect_collections") or [])
        if not query or not expected_domain:
            continue
        route = pipeline._route_query(query, collections=None)
        actual_domain = route.get("domain")
        selected = route.get("collections", [])
        ok = actual_domain == expected_domain
        if expected_collections:
            ok = ok and all(item in selected for item in expected_collections)
        if not ok:
            failures += 1
        results.append({
            "query": query,
            "expected_domain": expected_domain,
            "actual_domain": actual_domain,
            "expected_collections": expected_collections,
            "selected_collections": selected,
            "ok": ok,
        })
    return {
        "ok": failures == 0,
        "total": len(results),
        "failures": failures,
        "results": results,
    }


def cmd_validate_routing(args: argparse.Namespace) -> None:
    config = get_config()
    payload = _routing_payload(config)
    _print(payload)
    if not payload.get("ok", False):
        raise SystemExit(2)


def _notify_validation_failure(config: Any, payload: dict) -> None:
    notify_cfg = config.raw.get("validation_notifications", {}) or {}
    if not notify_cfg.get("enabled", False):
        return
    channels = notify_cfg.get("channels") or ["system"]
    title = str(notify_cfg.get("title", "Conclave validation failed"))
    priority = str(notify_cfg.get("priority", "high"))
    tags = notify_cfg.get("tags") or ["warning"]
    service = str(notify_cfg.get("service", "conclave-validate"))

    health = payload.get("health") or {}
    routing = payload.get("routing") or {}
    message = (
        f"health_ok={health.get('ok')} routing_ok={routing.get('ok')}. "
        f"health_failures={health.get('required_models', {}).get('missing') or health.get('required_models', {}).get('failed')} "
        f"routing_failures={routing.get('failures', 0)}"
    )
    mcp = MCPBridge(config_path=config.mcp_config_path)
    try:
        if "system" in channels:
            mcp.call("notifications", "notify_system", {
                "service": service,
                "status": "error",
                "message": message,
            })
        if "phone" in channels:
            mcp.call("notifications", "notify_phone", {
                "title": title,
                "message": message,
                "priority": priority,
                "tags": tags,
            })
        if "desktop" in channels:
            mcp.call("notifications", "notify_desktop", {
                "title": title,
                "message": message,
                "urgency": "critical",
            })
    except Exception:
        # Best-effort notifications; don't fail validation itself.
        return


def cmd_validate(args: argparse.Namespace) -> None:
    config = get_config()
    run_routing = bool(args.routing) or not (args.routing or args.health)
    run_health = bool(args.health) or not (args.routing or args.health)
    results = {}
    failed = False
    if run_health:
        try:
            health_payload = _health_payload(config)
            results["health"] = health_payload
            if not health_payload.get("ok", False):
                failed = True
        except Exception:
            failed = True
            results["health"] = {"ok": False}
    if run_routing:
        try:
            routing_payload = _routing_payload(config)
            results["routing"] = routing_payload
            if not routing_payload.get("ok", False):
                failed = True
        except Exception:
            failed = True
            results["routing"] = {"ok": False}
    _print({"ok": not failed, **results})
    if failed:
        _notify_validation_failure(config, results)
        raise SystemExit(2)


def _eval_baseline_prompt(pipeline: ConclavePipeline, query: str, context: dict, output_type: str | None) -> str:
    context_blob = pipeline._format_context(context or {})
    output_instructions = pipeline._output_instructions(output_type)
    prompt = (
        "You are a single-model baseline. Provide a direct, specific, actionable answer.\n"
        "Use the provided context when available; if context is empty, use your own reasoning.\n"
        "State key assumptions explicitly. Do not refuse or mention model limitations.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_blob or '(no context)'}\n"
    )
    if output_instructions:
        prompt += f"\n{output_instructions}\n"
    return prompt


def _eval_judge_prompt(
    pipeline: ConclavePipeline,
    query: str,
    context: dict,
    output_type: str | None,
    conclave_answer: str,
    baseline_answer: str,
) -> str:
    context_blob = pipeline._format_context(context or {})
    output_instructions = pipeline._output_instructions(output_type)
    prompt = (
        "You are a strict evaluator. Compare Answer A (Conclave) vs Answer B (Baseline).\n"
        "Score each on correctness, specificity, actionability, and alignment to the user's request.\n"
        "Use the provided context as ground truth when relevant. Prefer fewer hallucinations.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"winner\": \"conclave|baseline|tie\",\n"
        "  \"scores\": {\n"
        "    \"conclave\": {\"correctness\": 1-5, \"specificity\": 1-5, \"actionability\": 1-5, \"alignment\": 1-5},\n"
        "    \"baseline\": {\"correctness\": 1-5, \"specificity\": 1-5, \"actionability\": 1-5, \"alignment\": 1-5}\n"
        "  },\n"
        "  \"notes\": [\"short bullet\", \"short bullet\"],\n"
        "  \"confidence\": \"low|medium|high\"\n"
        "}\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_blob or '(no context)'}\n\n"
    )
    if output_instructions:
        prompt += f"{output_instructions}\n\n"
    prompt += f"Answer A (Conclave):\n{conclave_answer}\n\nAnswer B (Baseline):\n{baseline_answer}\n"
    return prompt


def _eval_summary(results: list[dict]) -> dict:
    counts = {"conclave": 0, "baseline": 0, "tie": 0, "unknown": 0}
    for item in results:
        winner = str((item.get("judge") or {}).get("winner") or "").strip().lower()
        if winner in counts:
            counts[winner] += 1
        else:
            counts["unknown"] += 1
    return {"total": len(results), "wins": counts}


def cmd_eval(args: argparse.Namespace) -> None:
    config = get_config()
    eval_cfg = config.raw.get("evaluation", {}) or {}
    if not eval_cfg.get("enabled", True):
        _print({"ok": False, "error": "evaluation disabled"})
        raise SystemExit(2)

    cases = list(eval_cfg.get("cases", []) or [])
    if args.case_id:
        wanted = {item.strip() for item in args.case_id if item}
        cases = [case for case in cases if str(case.get("id") or "").strip() in wanted]
    max_cases = args.limit or eval_cfg.get("max_cases")
    if max_cases:
        cases = cases[: int(max_cases)]

    if not cases:
        _print({"ok": False, "error": "no evaluation cases selected"})
        raise SystemExit(2)

    baseline_model = args.baseline_model or eval_cfg.get("baseline_model") or "cli:codex"
    judge_model = None if args.no_judge else (args.judge_model or eval_cfg.get("judge_model") or "cli:claude")

    pipeline = ConclavePipeline(config)
    results: list[dict] = []
    for case in cases:
        query = str(case.get("query") or "").strip()
        if not query:
            continue
        meta = {"input_title": case.get("id") or "eval_case"}
        output_type = case.get("output_type")
        if output_type:
            meta["output_type"] = output_type
        result = pipeline.run(query, collections=case.get("collections"), meta=meta)
        context = result.artifacts.get("context") or {}
        conclave_answer = result.consensus.get("answer") or ""

        baseline_prompt = _eval_baseline_prompt(pipeline, query, context, output_type)
        baseline_answer = pipeline._call_model(baseline_model, baseline_prompt, role="baseline")

        judge_payload = None
        judge_text = None
        if judge_model:
            judge_prompt = _eval_judge_prompt(
                pipeline,
                query,
                context,
                output_type,
                conclave_answer,
                baseline_answer,
            )
            judge_text = pipeline._call_model(judge_model, judge_prompt, role="eval_judge")
            judge_payload = pipeline._parse_json_payload(judge_text) or {}

        results.append({
            "id": case.get("id"),
            "query": query,
            "domain": (result.artifacts.get("route") or {}).get("domain"),
            "conclave_run_id": result.run_id,
            "conclave_answer": conclave_answer,
            "baseline_model": baseline_model,
            "baseline_answer": baseline_answer,
            "judge_model": judge_model,
            "judge": judge_payload,
            "judge_raw": judge_text,
        })

    summary = _eval_summary(results)
    payload = {
        "ok": True,
        "baseline_model": baseline_model,
        "judge_model": judge_model,
        "summary": summary,
        "results": results,
    }

    output_path = args.output_json
    if not output_path:
        save_dir = eval_cfg.get("save_dir")
        if save_dir:
            ts = time.strftime("%Y%m%d-%H%M%S")
            output_path = str(Path(save_dir) / f"eval-{ts}.json")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        payload["output_json"] = str(path)

    _print(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conclave")
    sub = parser.add_subparsers(dest="command")

    models = sub.add_parser("models")
    models_sub = models.add_subparsers(dest="models_cmd")
    models_sub.add_parser("list")
    models_sub.add_parser("status")
    models_sub.add_parser("benchmark")

    plan = sub.add_parser("plan")
    plan.add_argument("--query", required=True)

    run = sub.add_parser("run")
    run.add_argument("--query", required=True)
    run.add_argument("--collection", action="append")
    run.add_argument("--input-file")
    run.add_argument("--output-md")
    run.add_argument("--output-type", help="report|decision|plan|checklist|build_spec|image_brief|model_3d_brief")
    run.add_argument("--progress", action="store_true")
    run.add_argument("--max-seconds", type=int)
    run.add_argument("--max-evidence", type=int)
    run.add_argument("--max-context-chars", type=int)
    run.add_argument("--token-budget-total", type=float)
    run.add_argument("--token-budget-remaining", type=float)
    run.add_argument("--token-budget-used", type=float)
    run.add_argument("--fail-on-insufficient", action="store_true")
    run.add_argument("--no-fail-on-insufficient", action="store_true")

    iterate = sub.add_parser("iterate", help="Run consensus iteratively on a topic")
    iterate.add_argument("--input-file", required=True)
    iterate.add_argument("--query")
    iterate.add_argument("--collection", action="append")
    iterate.add_argument("--output-dir")
    iterate.add_argument("--append-md")
    iterate.add_argument("--max-runs", type=int)
    iterate.add_argument("--sleep-seconds", type=float, default=600)
    iterate.add_argument("--stop-on-insufficient", action="store_true")
    iterate.add_argument("--progress", action="store_true")
    iterate.add_argument("--max-seconds", type=int)
    iterate.add_argument("--max-evidence", type=int)
    iterate.add_argument("--max-context-chars", type=int)
    iterate.add_argument("--token-budget-total", type=float)
    iterate.add_argument("--token-budget-remaining", type=float)
    iterate.add_argument("--token-budget-used", type=float)

    runs = sub.add_parser("runs")
    runs_sub = runs.add_subparsers(dest="runs_cmd")
    runs_sub.add_parser("latest")
    list_cmd = runs_sub.add_parser("list")
    list_cmd.add_argument("--limit", type=int, default=10)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--topic", required=True)
    reconcile.add_argument("--fail-on-insufficient", action="store_true")
    reconcile.add_argument("--no-fail-on-insufficient", action="store_true")

    sub.add_parser("index")

    audit = sub.add_parser("audit")
    audit.add_argument("--mode", choices=["all", "rag", "mcp", "sources"], default="all")
    audit.add_argument("--output-dir")
    audit.add_argument("--no-fetch", action="store_true")
    audit.add_argument("--fail-on-issues", action="store_true")

    health = sub.add_parser("health")

    validate = sub.add_parser("validate")
    validate.add_argument("--routing", action="store_true")
    validate.add_argument("--health", action="store_true")

    validate_routing = sub.add_parser("validate-routing")

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--case-id", action="append")
    eval_cmd.add_argument("--limit", type=int)
    eval_cmd.add_argument("--baseline-model")
    eval_cmd.add_argument("--judge-model")
    eval_cmd.add_argument("--no-judge", action="store_true")
    eval_cmd.add_argument("--output-json")

    schedule = sub.add_parser("schedule")
    schedule_sub = schedule.add_subparsers(dest="schedule_cmd")
    schedule_sub.add_parser("list")
    apply_cmd = schedule_sub.add_parser("apply")
    apply_cmd.add_argument("--unit-dir", default="~/.config/systemd/user")
    apply_cmd.add_argument("--enable", action="store_true")
    apply_cmd.add_argument("--no-reload", action="store_true")
    apply_cmd.add_argument("--dry-run", action="store_true")
    apply_cmd.add_argument("--no-validate", action="store_true")
    apply_cmd.add_argument("--disable-legacy", action="store_true")

    return parser


def _progress_printer(store: DecisionStore, run_id: str, stop_event: threading.Event) -> None:
    last = 0
    while not stop_event.is_set():
        run = store.get_run(run_id) or {}
        events = run.get("events") or []
        for event in events[last:]:
            phase = event.get("phase") or event.get("event") or "event"
            status = event.get("status") or ""
            role = event.get("role")
            model = event.get("model_id") or event.get("model")
            detail = ""
            if phase == "deliberate":
                round_idx = event.get("round")
                max_rounds = event.get("max_rounds")
                round_text = ""
                if round_idx and max_rounds:
                    round_text = f"round {round_idx}/{max_rounds}"
                elif round_idx:
                    round_text = f"round {round_idx}"
                label = event.get("model_label") or model or ""
                verdict = event.get("verdict")
                duration = event.get("duration_s")
                summary = (event.get("summary") or "").strip().replace("\n", " ")
                if status == "round_start":
                    line = " ".join(part for part in [event.get("timestamp", ""), "deliberate", round_text, "start"] if part)
                    print(line.strip(), file=sys.stderr)
                    continue
                if status.endswith("_start") and label:
                    timeout_s = event.get("timeout_s")
                    timeout_text = f"(timeout {int(timeout_s)}s)" if isinstance(timeout_s, (int, float)) else ""
                    line = " ".join(part for part in [event.get("timestamp", ""), "deliberate", round_text, f"{role}->{label}" if role else label, "thinking...", timeout_text] if part)
                    print(line.strip(), file=sys.stderr)
                    continue
                if status.endswith("_done") and label:
                    duration_text = f"({duration}s)" if isinstance(duration, (int, float)) else ""
                    verdict_text = f"{verdict.upper()}: " if verdict else ""
                    summary_text = f"— {verdict_text}{summary}" if (verdict_text or summary) else ""
                    line = " ".join(part for part in [event.get("timestamp", ""), "deliberate", round_text, f"{role}->{label}" if role else label, "done", duration_text, summary_text] if part)
                    print(line.strip(), file=sys.stderr)
                    continue
                if status == "round_result":
                    agreement = event.get("agreement")
                    disagreements = event.get("disagreements") or []
                    verdict_text = "AGREE" if agreement else "DISAGREE"
                    extra = f"({len(disagreements)} issues)" if disagreements else ""
                    line = " ".join(part for part in [event.get("timestamp", ""), "deliberate", round_text, "result", verdict_text, extra] if part)
                    print(line.strip(), file=sys.stderr)
                    continue
                if status == "stable":
                    consecutive = event.get("consecutive")
                    line = " ".join(part for part in [event.get("timestamp", ""), "deliberate", "stable", f"({consecutive}x)" if consecutive else "", "-", "stopping"] if part)
                    print(line.strip(), file=sys.stderr)
                    continue
                if status == "stop":
                    reason = event.get("reason")
                    line = " ".join(part for part in [event.get("timestamp", ""), "deliberate", "stop", f"reason={reason}" if reason else ""] if part)
                    print(line.strip(), file=sys.stderr)
                    continue
            if role and model:
                label = event.get("model_label") or model
                detail = f"{role}->{label}"
            elif model:
                detail = str(model)
            if phase == "route" and status == "done":
                plan = event.get("models") or (event.get("route") or {}).get("plan_details") or (event.get("route") or {}).get("plan") or {}
                parts = []
                def _label(role_key: str):
                    value = plan.get(role_key)
                    if isinstance(value, dict):
                        return value.get("label") or value.get("id")
                    return value
                if _label("reasoner"):
                    parts.append(f"reasoner->{_label('reasoner')}")
                if _label("critic"):
                    parts.append(f"critic->{_label('critic')}")
                if _label("summarizer"):
                    parts.append(f"summarizer->{_label('summarizer')}")
                if parts:
                    detail = " ".join(parts)
            line = " ".join(part for part in [event.get("timestamp", ""), phase, status, detail] if part)
            print(line.strip(), file=sys.stderr)
        last = len(events)
        if run.get("status") in ("complete", "failed"):
            break
        time.sleep(1.0)


def _write_markdown_report(path: Path, run: dict, consensus: dict, artifacts: dict) -> None:
    if path.is_dir():
        filename = f"conclave-{run.get('id','run')}.md"
        path = path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = run.get("meta", {}) if run else {}
    title = meta.get("input_title") or run.get("query") or "Conclave Decision"
    evidence = (artifacts.get("context", {}) or {}).get("evidence", []) if artifacts else []
    quality = artifacts.get("quality", {}) if artifacts else {}
    lines = [
        f"# {title}",
        "",
        f"- **Run ID**: {run.get('id','')}",
        f"- **Status**: {run.get('status','')}",
        f"- **Prompt ID**: {meta.get('prompt_id','')}",
        f"- **Created**: {run.get('created_at','')}",
        f"- **Completed**: {run.get('completed_at','')}",
        "",
        "## Final Consensus",
        "",
        (consensus or {}).get("answer", "").strip() or "No consensus.",
        "",
    ]
    if evidence:
        lines.append("## Evidence (top)")
        lines.append("")
        for item in evidence[:12]:
            label = item.get("title") or item.get("path") or item.get("name") or "Evidence"
            item_path = item.get("path") or item.get("file_path") or ""
            line = item.get("line")
            score = item.get("signal_score")
            loc = f"{item_path}:{line}" if item_path and line else item_path
            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(f"- {label} ({loc}) [{item.get('collection','')}] {score_text}".strip())
        lines.append("")
    if quality:
        lines.extend([
            "## Quality",
            "",
            f"- Evidence count: {quality.get('evidence_count','')}",
            f"- Max signal: {quality.get('max_signal_score','')}",
            f"- PDF ratio: {quality.get('pdf_ratio','')}",
            f"- Off-domain ratio: {quality.get('off_domain_ratio','')}",
            f"- Issues: {', '.join(quality.get('issues', []) or [])}",
            "",
        ])
    run_dir = Path.home() / ".conclave" / "runs" / str(run.get("id", ""))
    lines.extend([
        "## Files",
        "",
        f"- run.json: {run_dir / 'run.json'}",
        f"- audit.jsonl: {run_dir / 'audit.jsonl'}",
        "",
    ])
    path.write_text("\n".join(lines).strip() + "\n")


def _append_markdown_report(dest_path: Path, source_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    content = source_path.read_text()
    with dest_path.open("a", encoding="utf-8") as handle:
        if dest_path.exists() and dest_path.stat().st_size > 0:
            handle.write("\n\n---\n\n")
        handle.write(content.strip() + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "models":
        cmd_models(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "iterate":
        cmd_iterate(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "reconcile":
        cmd_reconcile(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "validate-routing":
        cmd_validate_routing(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "schedule":
        cmd_schedule(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
