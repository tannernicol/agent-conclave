"""Native Gemini API client for Conclave."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GeminiResult:
    """Result from a Gemini API call."""
    text: str = ""
    ok: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    usage: Dict[str, Any] | None = None
    stderr: str | None = None


class GeminiClient:
    """Native Gemini API client using httpx."""

    MODEL_MAP = {
        "2.5-flash": "gemini-2.5-flash-preview-05-20",
        "2.5-pro": "gemini-2.5-pro-preview-05-06",
        "2.0-flash": "gemini-2.0-flash",
        "1.5-flash": "gemini-1.5-flash",
        "1.5-pro": "gemini-1.5-pro",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        model: str = "2.5-flash",
        system: str | None = None,
        temperature: float = 0.2,
        timeout: int = 120,
    ) -> GeminiResult:
        if not self.api_key:
            return GeminiResult(ok=False, error="GEMINI_API_KEY not set")

        model_id = self.MODEL_MAP.get(model, model)
        url = f"{self.base_url}/models/{model_id}:generateContent?key={self.api_key}"

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=body)

            duration_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                error_text = response.text[:500]
                return GeminiResult(
                    ok=False,
                    error=f"HTTP {response.status_code}: {error_text}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return GeminiResult(
                    ok=False,
                    error="No candidates in response",
                    duration_ms=duration_ms,
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

            usage_meta = data.get("usageMetadata", {})
            usage = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            }

            return GeminiResult(
                text=text,
                ok=True,
                duration_ms=duration_ms,
                usage=usage,
            )

        except httpx.TimeoutException:
            duration_ms = (time.perf_counter() - start) * 1000
            return GeminiResult(
                ok=False,
                error=f"Gemini API timeout after {timeout}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return GeminiResult(
                ok=False,
                error=str(e),
                duration_ms=duration_ms,
            )
