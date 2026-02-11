#!/usr/bin/env python3
"""
Multi-agent coordination via the JSONL message bus.

Demonstrates how independent agents share context, coordinate work,
and avoid duplication through the shared bus. No LLM required — this
shows the coordination protocol itself.

Run:
    python examples/multi_agent.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conclave.bus import MessageBus


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bus = MessageBus(tmp)

        # --- Agent 1 discovers a problem and broadcasts it ---
        print("=== Agent 1: security-reviewer ===")
        msg = bus.post(
            "security-reviewer",
            subject="SQL injection in /api/users",
            body=(
                "Parameter `sort_by` is interpolated directly into the ORDER BY "
                "clause without sanitization. Exploitable via `sort_by=1;DROP TABLE users`."
            ),
            msg_type="issue",
            priority="high",
            auto_inject=True,
            refs=["src/api/users.py:47"],
            tags=["sql-injection", "critical-path"],
        )
        print(f"  Posted: {msg.subject} (id={msg.id})")

        # --- Agent 2 reads the bus, sees the issue, posts a fix plan ---
        print("\n=== Agent 2: code-fixer ===")
        new_msgs = bus.read("code-fixer")
        print(f"  Read {len(new_msgs)} new message(s)")
        for m in new_msgs:
            print(f"    [{m.priority}] {m.sender}: {m.subject}")

        bus.post(
            "code-fixer",
            subject="Fix plan: parameterized ORDER BY",
            body=(
                "Will replace string interpolation with a whitelist of allowed "
                "column names. Using parameterized query for the direction (ASC/DESC)."
            ),
            msg_type="decision",
            priority="medium",
            auto_inject=True,
            refs=["src/api/users.py:47"],
        )
        print("  Posted fix plan")

        # --- Agent 3 checks what it should include in the LLM context ---
        print("\n=== Agent 3: test-writer ===")
        context = bus.format_context("test-writer")
        print("  Context for LLM prompt:")
        for line in context.splitlines():
            print(f"    {line}")

        bus.post(
            "test-writer",
            subject="Added regression tests for ORDER BY injection",
            body="3 test cases covering: valid column, invalid column, SQL payload.",
            msg_type="summary",
            priority="low",
            refs=["tests/test_users_api.py"],
        )

        # --- Summary ---
        print(f"\n=== Bus state ===")
        total = sum(1 for _ in bus.all_messages())
        print(f"  Total messages: {total}")
        print(f"  Unread for security-reviewer: {bus.pending_count('security-reviewer')}")
        print(f"  Unread for code-fixer: {bus.pending_count('code-fixer')}")
        print(f"  Unread for test-writer: {bus.pending_count('test-writer')}")


if __name__ == "__main__":
    main()
