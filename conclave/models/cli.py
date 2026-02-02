"""CLI-based model runner for Conclave."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import time
import os


@dataclass
class CliResult:
    text: str
    duration_ms: float
    ok: bool
    error: Optional[str] = None
    stderr: Optional[str] = None


class CliClient:
    def run(
        self,
        command: List[str],
        prompt: str,
        prompt_mode: str = "arg",
        stdin_flag: Optional[str] = None,
        timeout_seconds: int = 90,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CliResult:
        if not command:
            return CliResult(text="", duration_ms=0.0, ok=False, error="missing command")
        cmd = list(command)
        input_data = None
        if prompt_mode == "stdin":
            input_data = prompt
            if stdin_flag:
                cmd.append(stdin_flag)
        else:
            cmd.append(prompt)
        run_env = os.environ.copy()
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=cwd,
                env=run_env,
            )
            duration = (time.perf_counter() - start) * 1000
            ok = result.returncode == 0
            text = (result.stdout or "").strip()
            return CliResult(
                text=text,
                duration_ms=duration,
                ok=ok,
                error=None if ok else f"exit {result.returncode}",
                stderr=(result.stderr or "").strip() or None,
            )
        except subprocess.TimeoutExpired:
            duration = (time.perf_counter() - start) * 1000
            return CliResult(text="", duration_ms=duration, ok=False, error="timeout")
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            return CliResult(text="", duration_ms=duration, ok=False, error=str(exc))
