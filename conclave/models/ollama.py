"""Minimal Ollama client for local inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import httpx
import time


@dataclass
class OllamaResult:
    text: str
    duration_ms: float
    ok: bool
    error: Optional[str] = None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return data.get("models", [])
        except Exception:
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> OllamaResult:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                duration = (time.perf_counter() - start) * 1000
                return OllamaResult(text=data.get("response", ""), duration_ms=duration, ok=True)
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            return OllamaResult(text="", duration_ms=duration, ok=False, error=str(exc))
