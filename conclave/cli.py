"""Command line interface for Conclave."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conclave.config import get_config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner
from conclave.pipeline import ConclavePipeline
from conclave.rag import NasIndex
from conclave.scheduler import apply_schedule
from conclave.store import DecisionStore


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


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
    result = pipeline.run(args.query, collections=args.collection)
    _print({"run_id": result.run_id, "consensus": result.consensus})


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
    if args.topic == "all":
        results = []
        for item in config.topics:
            result = pipeline.run(item.get("query", ""), collections=item.get("collections"), meta={"topic": item.get("id")})
            results.append({"topic": item.get("id"), "run_id": result.run_id, "consensus": result.consensus})
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
        units = apply_schedule(
            config.topics,
            unit_dir=Path(args.unit_dir).expanduser(),
            enable=args.enable,
            reload_systemd=not args.no_reload,
            dry_run=args.dry_run,
        )
        _print({
            "created": [
                {
                    "topic": unit.topic_id,
                    "service": str(unit.service_path),
                    "timer": str(unit.timer_path),
                }
                for unit in units
            ],
            "enabled": bool(args.enable),
            "dry_run": bool(args.dry_run),
        })
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

    runs = sub.add_parser("runs")
    runs_sub = runs.add_subparsers(dest="runs_cmd")
    runs_sub.add_parser("latest")
    list_cmd = runs_sub.add_parser("list")
    list_cmd.add_argument("--limit", type=int, default=10)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--topic", required=True)

    sub.add_parser("index")

    schedule = sub.add_parser("schedule")
    schedule_sub = schedule.add_subparsers(dest="schedule_cmd")
    schedule_sub.add_parser("list")
    apply_cmd = schedule_sub.add_parser("apply")
    apply_cmd.add_argument("--unit-dir", default="~/.config/systemd/user")
    apply_cmd.add_argument("--enable", action="store_true")
    apply_cmd.add_argument("--no-reload", action="store_true")
    apply_cmd.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "models":
        cmd_models(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "run":
        cmd_run(args)
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
