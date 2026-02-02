"""Quality audit for RAG and MCP integrations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json

from conclave.config import Config
from conclave.mcp_bridge import MCPBridge
from conclave.rag import RagClient
from conclave.verification import OnDemandFetcher


@dataclass
class AuditResult:
    payload: Dict[str, Any]
    json_path: Path | None = None
    md_path: Path | None = None


def run_audit(
    config: Config,
    mode: str = "all",
    output_dir: Path | None = None,
    fetch_sources: bool = True,
) -> AuditResult:
    output_dir = output_dir or (config.data_dir / "audits")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "mode": mode,
        "rag": None,
        "mcp": None,
        "on_demand": None,
    }
    if mode in ("all", "rag"):
        payload["rag"] = _audit_rag(config)
    if mode in ("all", "mcp"):
        payload["mcp"] = _audit_mcp(config)
    if mode in ("all", "sources") and fetch_sources:
        payload["on_demand"] = _audit_on_demand(config)

    stamp = timestamp.replace(":", "").replace("-", "")
    json_path = output_dir / f"audit-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = output_dir / f"audit-{stamp}.md"
    md_path.write_text(_render_markdown(payload))
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"
    latest_json.write_text(json.dumps(payload, indent=2))
    latest_md.write_text(_render_markdown(payload))
    return AuditResult(payload=payload, json_path=json_path, md_path=md_path)


def _audit_rag(config: Config) -> Dict[str, Any]:
    rag = RagClient(config.rag.get("base_url", "http://localhost:8091"))
    collections = rag.collections()
    available = {c.get("name"): c for c in collections if c.get("name")}
    empty = [name for name, meta in available.items() if meta.get("file_count", 0) == 0]
    reliability = config.rag.get("collection_reliability", {}) or {}
    missing_reliability = [name for name in available if name not in reliability]
    allowlist = config.rag.get("domain_allowlist", {}) or {}
    allowlist_missing: Dict[str, List[str]] = {}
    for domain, items in allowlist.items():
        missing = [item for item in items if item not in available]
        if missing:
            allowlist_missing[domain] = missing
    audit_queries = config.rag.get("audit_queries", {}) or {}
    query_stats: Dict[str, Any] = {}
    for domain, queries in audit_queries.items():
        if not isinstance(queries, list):
            continue
        domain_collections = allowlist.get(domain) or config.rag.get("domain_collections", {}).get(domain, [])
        domain_hits = []
        for query in queries[:5]:
            total_hits = 0
            for coll in domain_collections:
                results = rag.search(str(query), collection=coll, limit=3)
                total_hits += len(results)
            domain_hits.append({"query": query, "hits": total_hits})
        query_stats[domain] = domain_hits
    errors = rag.drain_errors()
    return {
        "base_url": rag.base_url,
        "collections_total": len(available),
        "collections_empty": empty,
        "missing_reliability": missing_reliability,
        "allowlist_missing": allowlist_missing,
        "query_stats": query_stats,
        "errors": errors,
    }


def _audit_mcp(config: Config) -> Dict[str, Any]:
    bridge = MCPBridge(config_path=config.mcp_config_path)
    servers = bridge.available_servers()
    checks = config.raw.get("verification", {}).get("mcp_health_checks", {}) or {}
    results: List[Dict[str, Any]] = []
    for server in servers:
        check = checks.get(server)
        entry: Dict[str, Any] = {"server": server}
        if check:
            tool = check.get("tool")
            args = check.get("arguments", {}) or {}
            timeout = float(check.get("timeout", 15.0))
            resp = bridge.call(server, tool, args, timeout=timeout)
            entry.update({
                "tool": tool,
                "ok": resp.ok,
                "duration_ms": resp.duration_ms,
                "error": resp.error,
            })
        else:
            entry.update({
                "ok": None,
                "error": "no health check configured",
            })

        tool_list = bridge.list_tools(server, timeout=10.0)
        if tool_list.ok and isinstance(tool_list.result, dict):
            tools = tool_list.result.get("tools", []) or []
            entry["tool_count"] = len(tools)
            entry["tools_sample"] = [t.get("name") for t in tools[:5] if isinstance(t, dict)]
        else:
            entry["tool_count"] = 0
            entry["tools_error"] = tool_list.error

        results.append(entry)
    bridge.close_all()
    return {
        "servers": servers,
        "checks": results,
    }


def _audit_on_demand(config: Config) -> Dict[str, Any]:
    bridge = MCPBridge(config_path=config.mcp_config_path)
    rag = RagClient(config.rag.get("base_url", "http://localhost:8091"))
    fetcher = OnDemandFetcher(config, bridge, rag)
    domains = (config.raw.get("verification", {}) or {}).get("on_demand", {}).keys()
    results = []
    for domain in sorted(domains):
        outcome = fetcher.fetch(domain, f"audit:{domain}")
        results.append({
            "domain": domain,
            "items": len(outcome.items),
            "errors": outcome.errors,
        })
    bridge.close_all()
    return {"domains": results}


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Conclave Audit",
        "",
        f"- **Timestamp**: {payload.get('timestamp')}",
        f"- **Mode**: {payload.get('mode')}",
        "",
    ]
    rag = payload.get("rag") or {}
    if rag:
        lines.extend([
            "## RAG",
            "",
            f"- Collections: {rag.get('collections_total', 0)}",
            f"- Empty collections: {len(rag.get('collections_empty') or [])}",
            f"- Missing reliability labels: {len(rag.get('missing_reliability') or [])}",
            "",
        ])
        if rag.get("collections_empty"):
            lines.append("### Empty Collections")
            for name in rag["collections_empty"][:20]:
                lines.append(f"- {name}")
            lines.append("")
        if rag.get("missing_reliability"):
            lines.append("### Collections Missing Reliability")
            for name in rag["missing_reliability"][:20]:
                lines.append(f"- {name}")
            lines.append("")
        if rag.get("allowlist_missing"):
            lines.append("### Allowlist Missing Collections")
            for domain, items in rag["allowlist_missing"].items():
                lines.append(f"- {domain}: {', '.join(items)}")
            lines.append("")
        if rag.get("query_stats"):
            lines.append("### Query Smoke Tests")
            for domain, stats in rag["query_stats"].items():
                entries = ", ".join([f"{s['query']}({s['hits']})" for s in stats])
                lines.append(f"- {domain}: {entries}")
            lines.append("")

    mcp = payload.get("mcp") or {}
    if mcp:
        lines.extend([
            "## MCP",
            "",
            f"- Servers: {len(mcp.get('servers') or [])}",
            "",
        ])
        checks = mcp.get("checks") or []
        if checks:
            lines.append("### Health Checks")
            for entry in checks:
                status = "ok" if entry.get("ok") else "fail" if entry.get("ok") is False else "skip"
                tool = entry.get("tool") or "n/a"
                error = entry.get("error") or ""
                tools = entry.get("tool_count")
                tools_note = f"tools:{tools}" if tools is not None else "tools:? "
                lines.append(f"- {entry.get('server')}: {status} ({tool}) {tools_note} {error}".strip())
            lines.append("")

    on_demand = payload.get("on_demand") or {}
    if on_demand:
        lines.extend([
            "## On-Demand Sources",
            "",
        ])
        for domain in on_demand.get("domains", []):
            err_count = len(domain.get("errors") or [])
            lines.append(f"- {domain.get('domain')}: {domain.get('items')} items, {err_count} errors")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
