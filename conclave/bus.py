"""JSONL message bus for inter-agent coordination.

A lightweight, file-based message bus that lets multiple agents
communicate without a server process. Each agent appends messages
to a shared JSONL file and reads from its own cursor position.

Architecture:
    - Append-only JSONL file (no locking needed for single-writer)
    - Per-agent cursor files track read position
    - Messages have TTL, priority, and auto-inject flags
    - Context injection selects active messages for LLM prompts

Example:
    >>> bus = MessageBus("/tmp/conclave-bus")
    >>> bus.post("agent-1", subject="Found regression", body="Auth test failing")
    >>> msgs = bus.read("agent-2")
    >>> bus.context_for("agent-2")  # returns inject-eligible messages
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class Message:
    """A single bus message."""

    id: str
    ts: float
    sender: str
    recipient: str  # agent name or "all"
    msg_type: str  # summary, question, decision, context, issue
    priority: str  # high, medium, low
    subject: str
    body: str
    status: str = "open"  # open, ack, closed
    refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    ttl_hours: float = 24.0
    auto_inject: bool = False

    @property
    def expired(self) -> bool:
        return time.time() > (self.ts + self.ttl_hours * 3600)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Message:
        data = json.loads(line)
        return cls(**data)


class MessageBus:
    """File-backed JSONL message bus with cursor-based reads.

    Args:
        root: Directory for bus.jsonl and cursor files.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.bus_file = self.root / "bus.jsonl"
        self.cursor_dir = self.root / "cursors"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cursor_dir.mkdir(exist_ok=True)
        self.bus_file.touch(exist_ok=True)

    def post(
        self,
        sender: str,
        *,
        subject: str,
        body: str,
        recipient: str = "all",
        msg_type: str = "context",
        priority: str = "medium",
        ttl_hours: float = 24.0,
        auto_inject: bool = False,
        refs: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Message:
        """Append a message to the bus.

        Args:
            sender: Name of the sending agent.
            subject: Brief message title.
            body: Full message content.
            recipient: Target agent or "all".
            msg_type: One of summary, question, decision, context, issue.
            priority: One of high, medium, low.
            ttl_hours: Hours before message expires.
            auto_inject: Whether to auto-inject into LLM context.
            refs: File paths or identifiers this message references.
            tags: Freeform tags for filtering.

        Returns:
            The posted Message.
        """
        msg = Message(
            id=f"{int(time.time())}-{id(subject) % 100000:05d}",
            ts=time.time(),
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            priority=priority,
            subject=subject,
            body=body,
            ttl_hours=ttl_hours,
            auto_inject=auto_inject,
            refs=refs or [],
            tags=tags or [],
        )
        with open(self.bus_file, "a") as f:
            f.write(msg.to_json() + "\n")
        return msg

    def read(
        self,
        agent: str,
        *,
        limit: int = 0,
        peek: bool = False,
    ) -> list[Message]:
        """Read new messages for an agent since its last cursor position.

        Args:
            agent: Agent name (determines cursor file).
            limit: Max messages to return (0 = all new).
            peek: If True, don't advance the cursor.

        Returns:
            List of new messages.
        """
        cursor = self._get_cursor(agent)
        lines = self.bus_file.read_text().splitlines()
        new_lines = lines[cursor:]

        if limit > 0:
            new_lines = new_lines[:limit]

        messages = []
        for line in new_lines:
            line = line.strip()
            if line:
                messages.append(Message.from_json(line))

        if not peek:
            advance = min(len(new_lines), limit) if limit > 0 else len(new_lines)
            self._set_cursor(agent, cursor + advance)

        return messages

    def context_for(self, agent: str) -> list[Message]:
        """Return active, auto-inject messages visible to an agent.

        Filters for messages that:
        - Have auto_inject=True
        - Are not expired (within TTL)
        - Are not closed
        - Are addressed to this agent or "all"
        """
        lines = self.bus_file.read_text().splitlines()
        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            msg = Message.from_json(line)
            if not msg.auto_inject:
                continue
            if msg.expired:
                continue
            if msg.status == "closed":
                continue
            if msg.recipient not in ("all", agent):
                continue
            result.append(msg)

        # Sort by priority then recency
        priority_order = {"high": 0, "medium": 1, "low": 2}
        result.sort(key=lambda m: (priority_order.get(m.priority, 1), -m.ts))
        return result

    def format_context(self, agent: str) -> str:
        """Format injectable context as text for LLM prompts."""
        messages = self.context_for(agent)
        if not messages:
            return ""
        lines = ["## Agent Bus Context", ""]
        for msg in messages:
            lines.append(f"**[{msg.priority}] {msg.sender}** — {msg.subject}")
            lines.append(msg.body)
            lines.append("")
        return "\n".join(lines)

    def pending_count(self, agent: str) -> int:
        """Return count of unread messages for an agent."""
        cursor = self._get_cursor(agent)
        total = sum(1 for _ in open(self.bus_file))
        return max(0, total - cursor)

    def all_messages(self) -> Iterator[Message]:
        """Iterate all messages on the bus (for debugging/audit)."""
        for line in self.bus_file.read_text().splitlines():
            line = line.strip()
            if line:
                yield Message.from_json(line)

    def _get_cursor(self, agent: str) -> int:
        cursor_file = self.cursor_dir / agent
        if cursor_file.exists():
            try:
                return int(cursor_file.read_text().strip())
            except ValueError:
                return 0
        return 0

    def _set_cursor(self, agent: str, position: int) -> None:
        cursor_file = self.cursor_dir / agent
        cursor_file.write_text(str(position) + "\n")
