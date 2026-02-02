"""External source fetchers for Conclave context."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional
import json
import re
import httpx


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.skip = False
        self.block_tags = {"p", "li", "h1", "h2", "h3", "h4", "section", "div"}

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self.skip = True
        if tag in self.block_tags and self.parts:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.skip = False

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def extract_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = " ".join(parser.parts)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class HealthDashboardClient:
    base_url: str
    pages: List[str]
    errors: List[Dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def _record_error(self, action: str, exc: Exception) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append({"action": action, "error": str(exc)})

    def drain_errors(self) -> List[Dict[str, str]]:
        if not self.errors:
            return []
        items = list(self.errors)
        self.errors.clear()
        return items

    def fetch(self) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        base = self.base_url.rstrip("/")
        for page in self.pages:
            url = base + page
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    html = resp.text
                title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else page.strip("/")
                snippet = extract_text(html)[:1200]
                results.append({
                    "path": url,
                    "title": title,
                    "snippet": snippet,
                    "collection": "health-dashboard",
                    "source": "health-dashboard",
                })
            except Exception as exc:
                self._record_error(f"health:{page}", exc)
        return results


@dataclass
class MoneyClient:
    base_url: str
    endpoints: List[str]
    errors: List[Dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def _record_error(self, action: str, exc: Exception) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append({"action": action, "error": str(exc)})

    def drain_errors(self) -> List[Dict[str, str]]:
        if not self.errors:
            return []
        items = list(self.errors)
        self.errors.clear()
        return items

    def fetch(self) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        base = self.base_url.rstrip("/")
        for endpoint in self.endpoints:
            url = base + endpoint
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                title = f"money:{endpoint}"
                snippet = json.dumps(data, indent=2)[:1200]
                results.append({
                    "path": url,
                    "title": title,
                    "snippet": snippet,
                    "collection": "money-api",
                    "source": "money-api",
                })
            except Exception as exc:
                self._record_error(f"money:{endpoint}", exc)
        return results
