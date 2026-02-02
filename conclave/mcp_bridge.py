"""MCP Bridge - Call MCP server tools from Conclave pipeline."""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Response from an MCP tool call."""
    ok: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class MCPServer:
    """A running MCP server process."""
    name: str
    process: subprocess.Popen
    request_id: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    response_queue: queue.Queue = field(default_factory=queue.Queue)
    reader_thread: threading.Thread | None = None
    _shutdown: bool = False

    def __post_init__(self):
        """Start the response reader thread."""
        self.reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self.reader_thread.start()

    def _read_responses(self):
        """Background thread to read responses from MCP server."""
        while not self._shutdown:
            try:
                line = self.process.stdout.readline()
                if not line:
                    if self._shutdown:
                        break
                    time.sleep(0.01)
                    continue
                self.response_queue.put(line)
            except Exception as e:
                if not self._shutdown:
                    logger.warning(f"MCP reader error for {self.name}: {e}")
                break

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def call(self, tool: str, arguments: Dict[str, Any], timeout: float = 30.0) -> MCPResponse:
        """Call a tool on this MCP server."""
        start = time.perf_counter()

        with self.lock:
            request_id = self._next_id()
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": arguments,
                },
            }

            try:
                # Clear any stale responses
                while not self.response_queue.empty():
                    try:
                        self.response_queue.get_nowait()
                    except queue.Empty:
                        break

                # Send request
                request_line = json.dumps(request) + "\n"
                self.process.stdin.write(request_line)
                self.process.stdin.flush()

                # Wait for response with timeout
                try:
                    response_line = self.response_queue.get(timeout=timeout)
                except queue.Empty:
                    return MCPResponse(
                        ok=False,
                        error=f"MCP call timed out after {timeout}s",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                if not response_line:
                    return MCPResponse(
                        ok=False,
                        error="No response from MCP server",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                response = json.loads(response_line)

                if "error" in response:
                    return MCPResponse(
                        ok=False,
                        error=response["error"].get("message", str(response["error"])),
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                result = response.get("result", {})
                # Extract text content if present
                content = result.get("content", [])
                if content and isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    if texts:
                        # Try to parse as JSON
                        combined = "\n".join(texts)
                        try:
                            result = json.loads(combined)
                        except json.JSONDecodeError:
                            result = combined

                return MCPResponse(
                    ok=True,
                    result=result,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            except json.JSONDecodeError as e:
                return MCPResponse(
                    ok=False,
                    error=f"Invalid JSON response: {e}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except BrokenPipeError:
                return MCPResponse(
                    ok=False,
                    error="MCP server pipe broken - server may have crashed",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                return MCPResponse(
                    ok=False,
                    error=str(e),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

    def close(self) -> None:
        """Terminate the MCP server process."""
        self._shutdown = True
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


class MCPBridge:
    """Bridge to call MCP server tools."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or (Path.home() / ".mcp.json")
        self.servers: Dict[str, MCPServer] = {}
        self.config: Dict[str, Any] = {}
        self._config_mtime: float | None = None
        self._server_signatures: Dict[str, str] = {}
        self._load_config()

    def _load_config(self, force: bool = False) -> None:
        """Load MCP server configuration if updated."""
        if not self.config_path.exists():
            return
        try:
            mtime = self.config_path.stat().st_mtime
            if not force and self._config_mtime is not None and mtime <= self._config_mtime:
                return
            data = json.loads(self.config_path.read_text())
            self.config = data.get("mcpServers", {})
            self._config_mtime = mtime
            self._reconcile_running_servers()
        except Exception as e:
            logger.warning(f"Failed to load MCP config: {e}")

    def _server_signature(self, server_config: Dict[str, Any]) -> str:
        command = server_config.get("command")
        args = server_config.get("args", []) or []
        env = server_config.get("env", {}) or {}
        return json.dumps({"command": command, "args": args, "env": env}, sort_keys=True)

    def _reconcile_running_servers(self) -> None:
        """Restart running servers if config changed."""
        for name, server in list(self.servers.items()):
            cfg = self.config.get(name)
            if not cfg:
                continue
            signature = self._server_signature(cfg)
            if self._server_signatures.get(name) != signature:
                logger.info(f"MCP server '{name}' config changed, restarting...")
                server.close()
                del self.servers[name]
                self._server_signatures.pop(name, None)

    def _start_server(self, name: str) -> MCPServer | None:
        """Start an MCP server process."""
        self._load_config()
        if name in self.servers:
            # Check if server is still alive
            if self.servers[name].process.poll() is None:
                return self.servers[name]
            else:
                logger.info(f"MCP server '{name}' died, restarting...")
                del self.servers[name]

        server_config = self.config.get(name)
        if not server_config:
            logger.warning(f"MCP server '{name}' not found in config")
            return None

        signature = self._server_signature(server_config)
        if name in self._server_signatures and self._server_signatures[name] != signature:
            self._server_signatures.pop(name, None)

        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        if not command:
            logger.warning(f"MCP server '{name}' has no command")
            return None

        try:
            import os
            run_env = os.environ.copy()
            run_env.update(env)

            process = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=run_env,
            )

            # Give server a moment to start
            time.sleep(0.1)

            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "conclave", "version": "1.0.0"},
                },
            }
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()

            # Wait for initialize response with timeout
            import select
            ready, _, _ = select.select([process.stdout], [], [], 10.0)
            if not ready:
                process.kill()
                logger.warning(f"MCP server '{name}' did not respond to initialize within 10s")
                return None

            response_line = process.stdout.readline()
            if not response_line:
                process.kill()
                logger.warning(f"MCP server '{name}' did not respond to initialize")
                return None

            # Send initialized notification
            initialized = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            process.stdin.write(json.dumps(initialized) + "\n")
            process.stdin.flush()

            server = MCPServer(name=name, process=process)
            self.servers[name] = server
            self._server_signatures[name] = signature
            logger.info(f"Started MCP server: {name}")
            return server

        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}")
            return None

    def _request(self, server_name: str, method: str, params: Dict[str, Any] | None = None, timeout: float = 30.0) -> MCPResponse:
        """Send a raw MCP request to a server."""
        server = self._start_server(server_name)
        if not server:
            return MCPResponse(ok=False, error=f"MCP server '{server_name}' not available")

        start = time.perf_counter()
        with server.lock:
            request_id = server._next_id()
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }

            try:
                while not server.response_queue.empty():
                    try:
                        server.response_queue.get_nowait()
                    except queue.Empty:
                        break

                server.process.stdin.write(json.dumps(request) + "\n")
                server.process.stdin.flush()

                try:
                    response_line = server.response_queue.get(timeout=timeout)
                except queue.Empty:
                    return MCPResponse(
                        ok=False,
                        error=f"MCP request timed out after {timeout}s",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                if not response_line:
                    return MCPResponse(
                        ok=False,
                        error="No response from MCP server",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                response = json.loads(response_line)
                if "error" in response:
                    return MCPResponse(
                        ok=False,
                        error=response["error"].get("message", str(response["error"])),
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

                return MCPResponse(
                    ok=True,
                    result=response.get("result"),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                return MCPResponse(
                    ok=False,
                    error=str(e),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

    def call(self, server_name: str, tool: str, arguments: Dict[str, Any] = None, timeout: float = 30.0) -> MCPResponse:
        """Call a tool on an MCP server."""
        response = self._request(
            server_name,
            "tools/call",
            {"name": tool, "arguments": arguments or {}},
            timeout=timeout,
        )
        response = self._parse_tool_result(response)
        if response.ok:
            return response
        if response.error and "pipe broken" in response.error.lower():
            server = self.servers.get(server_name)
            if server:
                server.close()
                self.servers.pop(server_name, None)
        return response

    def _parse_tool_result(self, response: MCPResponse) -> MCPResponse:
        """Normalize MCP tool responses that return text payloads."""
        if not response.ok or response.result is None:
            return response
        if isinstance(response.result, dict):
            content = response.result.get("content", [])
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                if texts:
                    combined = "\n".join(texts)
                    try:
                        response.result = json.loads(combined)
                    except json.JSONDecodeError:
                        response.result = combined
        return response

    def list_tools(self, server_name: str, timeout: float = 10.0) -> MCPResponse:
        """List tools exposed by an MCP server."""
        return self._request(server_name, "tools/list", {}, timeout=timeout)

    def available_servers(self) -> List[str]:
        """List available MCP servers from config."""
        self._load_config()
        return list(self.config.keys())

    def close_all(self) -> None:
        """Close all running MCP servers."""
        for server in self.servers.values():
            server.close()
        self.servers.clear()

    # Convenience methods for common operations

    def money_summary(self) -> MCPResponse:
        """Get financial summary."""
        return self.call("money", "money_summary", timeout=15.0)

    def money_networth(self, range: str = "1y") -> MCPResponse:
        """Get net worth history."""
        return self.call("money", "money_networth", {"range": range}, timeout=15.0)

    def money_spending(self) -> MCPResponse:
        """Get current month spending breakdown."""
        return self.call("money", "money_spending", timeout=15.0)

    def bounty_rag_query(self, query: str, category: str = "all", limit: int = 10) -> MCPResponse:
        """Query bounty RAG for vulnerability patterns."""
        return self.call("bounty-training", "bounty_rag_query", {
            "query": query,
            "category": category,
            "limit": limit,
        }, timeout=30.0)

    def bounty_semantic_search(self, query: str, top_k: int = 5) -> MCPResponse:
        """Semantic search for similar vulnerabilities."""
        return self.call("bounty-training", "bounty_semantic_search", {
            "query": query,
            "top_k": top_k,
        }, timeout=30.0)

    def bounty_predict_outcome(self, vuln_type: str, description: str, target: str = None, severity: str = None) -> MCPResponse:
        """Predict submission outcome based on similar findings."""
        args = {"vuln_type": vuln_type, "description": description}
        if target:
            args["target"] = target
        if severity:
            args["severity"] = severity
        return self.call("bounty-training", "bounty_predict_outcome", args, timeout=30.0)

    def bounty_false_positive_patterns(self, vuln_type: str) -> MCPResponse:
        """Get known false positive patterns for a vulnerability type."""
        return self.call("bounty-training", "bounty_false_positive_patterns", {"vuln_type": vuln_type}, timeout=15.0)

    def memory_learn(self, category: str, learning: str, context: str = None, importance: str = "medium") -> MCPResponse:
        """Store a learning for future sessions."""
        args = {"category": category, "learning": learning, "importance": importance}
        if context:
            args["context"] = context
        return self.call("memory", "memory_learn", args, timeout=10.0)

    def memory_recall(self, category: str = None, search: str = None, limit: int = 20) -> MCPResponse:
        """Recall learnings from memory."""
        args = {"limit": limit}
        if category:
            args["category"] = category
        if search:
            args["search"] = search
        return self.call("memory", "memory_recall", args, timeout=15.0)

    def memory_log_action(self, action: str, result: str = None) -> MCPResponse:
        """Log an action to today's session thread."""
        args = {"action": action}
        if result:
            args["result"] = result
        return self.call("memory", "memory_log_action", args, timeout=10.0)
