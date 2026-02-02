"""Command line interface for Conclave."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from conclave.config import get_config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner
from conclave.pipeline import ConclavePipeline, PipelineResult
from conclave.rag import NasIndex
from conclave.scheduler import apply_schedule
from conclave.store import DecisionStore


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
    if getattr(args, "max_evidence", None):
        meta["evidence_limit"] = args.max_evidence
    if getattr(args, "max_context_chars", None):
        meta["context_char_limit"] = args.max_context_chars
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
    progress_thread = None
    if getattr(args, "progress", False):
        progress_thread = threading.Thread(
            target=_progress_printer,
            args=(store, run_id, stop_event),
            daemon=True,
        )
        progress_thread.start()
        print(f"[conclave] run_id={run_id}", file=sys.stderr)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"run exceeded max_seconds={args.max_seconds}")

    if getattr(args, "max_seconds", None):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(args.max_seconds))

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
        if getattr(args, "max_seconds", None):
            signal.alarm(0)
        stop_event.set()
        if progress_thread:
            progress_thread.join(timeout=1.0)
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


def cmd_hunt(args: argparse.Namespace) -> None:
    config = get_config()
    pipeline = ConclavePipeline(config)
    query = args.query or "bounty: review current findings and propose next hunt steps"
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
        snapshot_path = output_dir / f"conclave-hunt-{result.run_id}.md"
        latest_path = output_dir / "conclave-hunt-latest.md"
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
    def _should_fail(consensus: dict) -> bool:
        return bool(fail_on and consensus.get("insufficient_evidence"))
    if args.topic == "all":
        results = []
        for item in config.topics:
            result = pipeline.run(item.get("query", ""), collections=item.get("collections"), meta={"topic": item.get("id")})
            results.append({"topic": item.get("id"), "run_id": result.run_id, "consensus": result.consensus})
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
    result = pipeline.run(topic.get("query", ""), collections=topic.get("collections"), meta={"topic": args.topic})
    _print({"run_id": result.run_id, "consensus": result.consensus})
    if _should_fail(result.consensus):
        raise SystemExit(exit_code or 2)


def cmd_index(args: argparse.Namespace) -> None:
    config = get_config()
    index = NasIndex(
        data_dir=config.data_dir,
        allowlist=config.index.get("allowlist", []),
        exclude_patterns=config.index.get("exclude_patterns", []),
        max_file_mb=int(config.index.get("max_file_mb", 2)),
    )
    indexed = index.index()
    _print({"indexed": indexed})


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
    run.add_argument("--progress", action="store_true")
    run.add_argument("--max-seconds", type=int)
    run.add_argument("--max-evidence", type=int)
    run.add_argument("--max-context-chars", type=int)
    run.add_argument("--fail-on-insufficient", action="store_true")
    run.add_argument("--no-fail-on-insufficient", action="store_true")

    hunt = sub.add_parser("hunt")
    hunt.add_argument("--input-file", required=True)
    hunt.add_argument("--query")
    hunt.add_argument("--collection", action="append")
    hunt.add_argument("--output-dir")
    hunt.add_argument("--append-md")
    hunt.add_argument("--max-runs", type=int)
    hunt.add_argument("--sleep-seconds", type=float, default=600)
    hunt.add_argument("--stop-on-insufficient", action="store_true")
    hunt.add_argument("--progress", action="store_true")
    hunt.add_argument("--max-seconds", type=int)
    hunt.add_argument("--max-evidence", type=int)
    hunt.add_argument("--max-context-chars", type=int)

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
            if role and model:
                detail = f"{role}->{model}"
            elif model:
                detail = str(model)
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
    route = artifacts.get("route", {}) if artifacts else {}
    plan = route.get("plan", {}) if isinstance(route, dict) else {}
    deliberation = artifacts.get("deliberation", {}) if artifacts else {}
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
        "## Consensus",
        "",
        (consensus or {}).get("answer", "").strip() or "No consensus.",
        "",
    ]
    if plan:
        lines.extend([
            "## Models",
            "",
            f"- Router: {plan.get('router','')}",
            f"- Reasoner: {plan.get('reasoner','')}",
            f"- Critic: {plan.get('critic','')}",
            f"- Summarizer: {plan.get('summarizer','')}",
            "",
        ])
    disagreements = deliberation.get("disagreements") or []
    if disagreements:
        lines.append("## Disagreements")
        lines.append("")
        for item in disagreements[:10]:
            lines.append(f"- {item}")
        lines.append("")
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
    elif args.command == "hunt":
        cmd_hunt(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "reconcile":
        cmd_reconcile(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "schedule":
        cmd_schedule(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
