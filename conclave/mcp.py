"""Lightweight MCP registry reader (no active tool calls)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json


def load_mcp_servers(path: Path | None = None) -> Dict[str, Any]:
    """Return MCP server definitions from ~/.mcp.json."""
    config_path = path or (Path.home() / ".mcp.json")
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except Exception:
        return {}
    return data.get("mcpServers", {})
