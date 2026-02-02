"""CLI-based model runner for Conclave."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import time
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class CliResult:
    text: str
    duration_ms: float
    ok: bool
    error: Optional[str] = None
    stderr: Optional[str] = None
    retries: int = 0


class CliClient:
    def __init__(
        self,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        retry_backoff: float = 2.0,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

    def run(
        self,
        command: List[str],
        prompt: str,
        prompt_mode: str = "arg",
        stdin_flag: Optional[str] = None,
        timeout_seconds: int = 180,  # Increased from 90 to 180
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
    ) -> CliResult:
        if not command:
            return CliResult(text="", duration_ms=0.0, ok=False, error="missing command")

        max_attempts = (retries if retries is not None else self.max_retries) + 1
        last_result: Optional[CliResult] = None
        total_duration = 0.0

        for attempt in range(max_attempts):
            if attempt > 0:
                delay = self.retry_delay * (self.retry_backoff ** (attempt - 1))
                logger.info(f"CLI retry {attempt}/{max_attempts-1} after {delay:.1f}s delay")
                time.sleep(delay)

            result = self._run_once(
                command=command,
                prompt=prompt,
                prompt_mode=prompt_mode,
                stdin_flag=stdin_flag,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
                env=env,
            )
            total_duration += result.duration_ms
            last_result = result

            if result.ok:
                result.retries = attempt
                result.duration_ms = total_duration
                return result

            # Don't retry on timeout - it's likely a resource issue
            if result.error == "timeout":
                logger.warning(f"CLI timeout after {timeout_seconds}s, not retrying")
                break

            # Check if error is retryable
            if not self._is_retryable(result):
                logger.warning(f"CLI error not retryable: {result.error}")
                break

            logger.warning(f"CLI attempt {attempt+1} failed: {result.error}")

        # Return last result with retry count
        if last_result:
            last_result.retries = max_attempts - 1
            last_result.duration_ms = total_duration
            return last_result

        return CliResult(text="", duration_ms=total_duration, ok=False, error="no attempts made")

    def _run_once(
        self,
        command: List[str],
        prompt: str,
        prompt_mode: str,
        stdin_flag: Optional[str],
        timeout_seconds: int,
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
    ) -> CliResult:
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

    def _is_retryable(self, result: CliResult) -> bool:
        """Determine if an error is worth retrying."""
        if result.ok:
            return False

        error = result.error or ""
        stderr = result.stderr or ""

        # Exit codes that are typically transient
        retryable_exits = ["exit 1", "exit 137", "exit 143"]
        if any(code in error for code in retryable_exits):
            # Check stderr for permanent errors
            permanent_errors = [
                "invalid api key",
                "authentication failed",
                "unauthorized",
                "forbidden",
                "not found",
                "invalid model",
            ]
            stderr_lower = stderr.lower()
            if any(pe in stderr_lower for pe in permanent_errors):
                return False
            return True

        # Network/connection errors are retryable
        retryable_patterns = [
            "connection refused",
            "connection reset",
            "network unreachable",
            "temporary failure",
            "service unavailable",
            "rate limit",
            "too many requests",
        ]
        combined = (error + " " + stderr).lower()
        return any(pattern in combined for pattern in retryable_patterns)
