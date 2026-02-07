#!/usr/bin/env python3
"""
Conclave Demo -- Multi-agent consensus on fun questions.

Run:
    python examples/demo.py

Requires a running Conclave instance or configured models.
Set CONCLAVE_CONFIG to point to your config file, or use the default.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure conclave is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline


DEMO_QUESTIONS = [
    {
        "query": "What are the top 3 sci-fi films of all time and why?",
        "domain": "creative",
        "description": "Creative debate -- models defend their picks and reach consensus.",
    },
    {
        "query": (
            "If humanity could only bring one invention to Mars, "
            "what should it be and why?"
        ),
        "domain": "research",
        "description": "Open-ended research question with real-world constraints.",
    },
    {
        "query": (
            "Is functional programming or object-oriented programming better "
            "for building large-scale systems? Give concrete trade-offs."
        ),
        "domain": "code_review",
        "description": "Technical debate requiring nuanced reasoning.",
    },
    {
        "query": (
            "What single invention in the last 100 years had the greatest "
            "impact on everyday life? Defend your choice."
        ),
        "domain": "general",
        "description": "General knowledge question with persuasive argumentation.",
    },
]


def run_demo(index: int | None = None) -> None:
    """Run one or all demo questions through Conclave."""
    config = get_config()
    pipeline = ConclavePipeline(config)

    questions = DEMO_QUESTIONS if index is None else [DEMO_QUESTIONS[index]]

    for i, q in enumerate(questions):
        num = index if index is not None else i
        print(f"\n{'=' * 72}")
        print(f"  Demo {num + 1}: {q['description']}")
        print(f"  Domain: {q['domain']}")
        print(f"{'=' * 72}")
        print(f"\n  Q: {q['query']}\n")

        result = pipeline.run(
            query=q["query"],
            meta={"domain": q["domain"]},
        )

        answer = result.consensus.get("answer", "(no answer)")
        agreement = result.consensus.get("agreement_pct", 0)
        models = result.artifacts.get("participants", [])

        print(f"  Agreement: {agreement}%")
        print(f"  Models: {', '.join(models) if models else 'default'}")
        print(f"\n  Answer:\n")
        for line in answer.splitlines():
            print(f"    {line}")
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Conclave demo questions through multi-agent consensus."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=int,
        choices=range(1, len(DEMO_QUESTIONS) + 1),
        help="Run a specific demo question (1-%d)" % len(DEMO_QUESTIONS),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demo questions and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable demo questions:\n")
        for i, q in enumerate(DEMO_QUESTIONS, 1):
            print(f"  {i}. [{q['domain']}] {q['query']}")
            print(f"     {q['description']}\n")
        return

    idx = (args.question - 1) if args.question else None
    run_demo(idx)


if __name__ == "__main__":
    main()
