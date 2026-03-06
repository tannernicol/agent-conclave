"""OpenAI-compatible API client for Conclave.

Works with OpenAI, vLLM, LM Studio, Ollama REST, and any provider
that implements the OpenAI chat completions API.

For keyless local servers (vLLM, LM Studio), set api_key_env to "none".

Config example:
    - id: openai:gpt-4o
      base_url: https://api.openai.com/v1        # optional, defaults to OpenAI
      api_key_env: OPENAI_API_KEY                 # optional, defaults to OPENAI_API_KEY
      capabilities:
        text_reasoning: true
        json_reliability: high
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OpenAIResult:
    """Result from an OpenAI-compatible API call."""

    text: str = ""
    ok: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    usage: Dict[str, Any] | None = None
    stderr: str | None = None


class OpenAICompatClient:
    """Client for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        model: str = "gpt-4o",
        system: str | None = None,
        temperature: float = 0.2,
        timeout: int = 120,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> OpenAIResult:
        key = api_key or self.api_key
        url_base = (base_url or self.base_url).rstrip("/")

        # "none" sentinel allows keyless local servers (vLLM, LM Studio)
        keyless = key.lower() == "none" if key else False
        if not key and not keyless:
            return OpenAIResult(ok=False, error="No API key set (OPENAI_API_KEY)")

        url = f"{url_base}/chat/completions"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if key and not keyless:
            headers["Authorization"] = f"Bearer {key}"

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=body, headers=headers)

            duration_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                error_text = response.text[:500]
                return OpenAIResult(
                    ok=False,
                    error=f"HTTP {response.status_code}: {error_text}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return OpenAIResult(
                    ok=False,
                    error="No choices in response",
                    duration_ms=duration_ms,
                )

            text = choices[0].get("message", {}).get("content") or ""

            usage_data = data.get("usage", {})
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

            return OpenAIResult(
                text=text,
                ok=True,
                duration_ms=duration_ms,
                usage=usage,
            )

        except httpx.TimeoutException:
            duration_ms = (time.perf_counter() - start) * 1000
            return OpenAIResult(
                ok=False,
                error=f"API timeout after {timeout}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return OpenAIResult(
                ok=False,
                error=str(e),
                duration_ms=duration_ms,
            )
