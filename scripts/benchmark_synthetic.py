#!/usr/bin/env python3
"""Generate deterministic synthetic benchmark tables for public documentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = "Agent Conclave"
METRICS = [{"scenario":"3-model consensus","p50_ms":420,"p95_ms":790,"cpu_pct":34,"mem_mb":410},{"scenario":"5-model consensus","p50_ms":690,"p95_ms":1210,"cpu_pct":49,"mem_mb":620},{"scenario":"policy-gated review","p50_ms":510,"p95_ms":930,"cpu_pct":38,"mem_mb":470}]


def to_markdown(rows: list[dict]) -> str:
    lines = [
        f"# {PROJECT} Synthetic Benchmarks",
        "",
        "All metrics are synthetic and reproducible for documentation quality checks.",
        "",
        "| Scenario | p50 (ms) | p95 (ms) | CPU (%) | Memory (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['p50_ms']} | {row['p95_ms']} | {row['cpu_pct']} | {row['mem_mb']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.format == "json":
        content = json.dumps({"project": PROJECT, "synthetic": True, "metrics": METRICS}, indent=2) + "\n"
    else:
        content = to_markdown(METRICS)

    if args.output:
        Path(args.output).write_text(content)
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
