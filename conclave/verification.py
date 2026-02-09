"""On-demand source verification and fetch utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
import httpx

from conclave.config import Config
from conclave.mcp_bridge import MCPBridge
from conclave.rag import RagClient
from conclave.sources import extract_text


@dataclass
class VerificationResult:
    items: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


class OnDemandFetcher:
    def __init__(self, config: Config, mcp: MCPBridge, rag: RagClient) -> None:
        self.config = config
        self.mcp = mcp
        self.rag = rag

    def has_tasks(self, domain: str) -> bool:
        cfg = self.config.raw.get("verification", {}) or {}
        if not cfg.get("enabled", True):
            return False
        tasks = cfg.get("on_demand", {}).get(domain, []) or []
        return bool(tasks)

    def fetch(self, domain: str, query: str) -> VerificationResult:
        cfg = self.config.raw.get("verification", {}) or {}
        if not cfg.get("enabled", True):
            return VerificationResult(items=[], errors=[])
        tasks = cfg.get("on_demand", {}).get(domain, []) or []
        items: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for task in tasks:
            result = self._run_task(task, query)
            if result.get("ok"):
                payload = result.get("item")
                if payload:
                    items.append(payload)
            else:
                error = result.get("error") or "unknown error"
                if not task.get("optional", False):
                    errors.append({
                        "task": task.get("title") or task.get("url") or task.get("tool") or task.get("path"),
                        "error": error,
                    })
        return VerificationResult(items=items, errors=errors)

    def _run_task(self, task: Dict[str, Any], query: str) -> Dict[str, Any]:
        task_type = (task.get("type") or "http").lower()
        title = task.get("title") or task.get("name") or task.get("url") or task.get("tool") or "on-demand"
        collection = task.get("collection") or "on-demand"
        try:
            if task_type == "http":
                return self._http_task(task, title, collection)
            if task_type in {"search", "searxng"}:
                return self._search_task(task, query, title, collection)
            if task_type == "mcp":
                return self._mcp_task(task, title, collection)
            if task_type == "file":
                return self._file_task(task, title, collection)
            if task_type == "rag":
                return self._rag_task(task, query, title, collection)
            return {"ok": False, "error": f"unknown task type: {task_type}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _http_task(self, task: Dict[str, Any], title: str, collection: str) -> Dict[str, Any]:
        url = task.get("url")
        if not url:
            return {"ok": False, "error": "missing url"}
        timeout = float(task.get("timeout", 8.0))
        method = str(task.get("method", "GET")).upper()
        fmt = (task.get("format") or "").lower()
        headers = task.get("headers") or {}
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            text = resp.text
            if fmt == "json" or "application/json" in content_type:
                try:
                    data = resp.json()
                    snippet = json.dumps(data, indent=2)[:1200]
                except Exception:
                    snippet = text[:1200]
            elif fmt == "html" or "text/html" in content_type:
                snippet = extract_text(text)[:1200]
            else:
                snippet = text[:1200]
        return {
            "ok": True,
            "item": {
                "path": url,
                "title": title,
                "snippet": snippet,
                "collection": collection,
            },
        }

    def _search_task(self, task: Dict[str, Any], query: str, title: str, collection: str) -> Dict[str, Any]:
        cfg = self.config.raw.get("verification", {}) or {}
        search_cfg = cfg.get("search", {}) or {}
        base_url = str(task.get("base_url") or search_cfg.get("base_url") or "http://127.0.0.1:8095").rstrip("/")
        endpoint = str(task.get("endpoint") or search_cfg.get("endpoint") or "/search")
        timeout = float(task.get("timeout", search_cfg.get("timeout", 10.0)))
        max_results = int(task.get("max_results", search_cfg.get("max_results", 5)))

        task_query = task.get("query")
        template = task.get("query_template")
        if task_query:
            q = str(task_query)
        elif template:
            q = str(template).replace("{query}", query)
        else:
            prefix = str(task.get("query_prefix") or "").strip()
            suffix = str(task.get("query_suffix") or "").strip()
            q = " ".join([part for part in [prefix, query, suffix] if part]).strip()
        if not q:
            return {"ok": False, "error": "missing query"}

        params = {"q": q, "format": "json"}
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}{endpoint}", data=params)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])[:max_results]
        if not results:
            return {"ok": False, "error": "no search results"}
        lines = []
        for item in results:
            title_val = str(item.get("title") or "").strip()
            url_val = str(item.get("url") or "").strip()
            snippet_val = str(item.get("content") or item.get("snippet") or "").strip()
            if title_val and url_val:
                lines.append(f"- {title_val} — {url_val}")
            elif url_val:
                lines.append(f"- {url_val}")
            if snippet_val:
                lines.append(f"  {snippet_val[:240]}")
        snippet = "\n".join(lines)[:1200]
        return {
            "ok": True,
            "item": {
                "path": f"{base_url}{endpoint}",
                "title": title,
                "snippet": snippet,
                "collection": collection,
            },
        }

    def _mcp_task(self, task: Dict[str, Any], title: str, collection: str) -> Dict[str, Any]:
        server = task.get("server")
        tool = task.get("tool")
        if not server or not tool:
            return {"ok": False, "error": "missing server/tool"}
        arguments = task.get("arguments") or {}
        timeout = float(task.get("timeout", 15.0))
        response = self.mcp.call(str(server), str(tool), arguments, timeout=timeout)
        if not response.ok:
            return {"ok": False, "error": response.error or "mcp call failed"}
        snippet = json.dumps(response.result, indent=2)[:1200] if isinstance(response.result, (dict, list)) else str(response.result)[:1200]
        return {
            "ok": True,
            "item": {
                "path": f"mcp://{server}/{tool}",
                "title": title,
                "snippet": snippet,
                "collection": collection,
            },
        }

    def _file_task(self, task: Dict[str, Any], title: str, collection: str) -> Dict[str, Any]:
        path = task.get("path")
        if not path:
            return {"ok": False, "error": "missing path"}
        from pathlib import Path
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return {"ok": False, "error": f"missing file: {file_path}"}
        text = file_path.read_text(errors="ignore")
        snippet = text[:1200]
        return {
            "ok": True,
            "item": {
                "path": str(file_path),
                "title": title,
                "snippet": snippet,
                "collection": collection,
            },
        }

    def _rag_task(self, task: Dict[str, Any], query: str, title: str, collection: str) -> Dict[str, Any]:
        q = task.get("query") or query
        coll = task.get("collection")
        limit = int(task.get("limit", 5))
        results = self.rag.search(str(q), collection=coll, limit=limit)
        if not results:
            return {"ok": False, "error": "no rag results"}
        snippet = results[0].get("snippet") or results[0].get("match_line") or ""
        return {
            "ok": True,
            "item": {
                "path": results[0].get("path") or results[0].get("name") or f"rag://{coll or 'all'}",
                "title": title,
                "snippet": str(snippet)[:1200],
                "collection": collection,
            },
        }
