"""Configuration loader for Conclave."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import os
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
USER_CONFIG_PATH = Path.home() / ".config" / "conclave" / "config.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    if USER_CONFIG_PATH.exists():
        override = yaml.safe_load(USER_CONFIG_PATH.read_text()) or {}
        data = _deep_merge(data, override)
    # Environment overrides
    host = os.getenv("CONCLAVE_HOST")
    port = os.getenv("CONCLAVE_PORT")
    if host:
        data.setdefault("server", {})["host"] = host
    if port:
        try:
            data.setdefault("server", {})["port"] = int(port)
        except ValueError:
            pass
    return data


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def server(self) -> Dict[str, Any]:
        return self.raw.get("server", {})

    @property
    def data_dir(self) -> Path:
        return Path(self.raw.get("data_dir", "/home/tanner/.conclave"))

    @property
    def models(self) -> Dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def planner(self) -> Dict[str, Any]:
        return self.raw.get("planner", {})

    @property
    def rag(self) -> Dict[str, Any]:
        return self.raw.get("rag", {})

    @property
    def index(self) -> Dict[str, Any]:
        return self.raw.get("index", {})

    @property
    def calibration(self) -> Dict[str, Any]:
        return self.raw.get("calibration", {})

    @property
    def quality(self) -> Dict[str, Any]:
        return self.raw.get("quality", {})

    @property
    def sources(self) -> Dict[str, Any]:
        return self.raw.get("sources", {})

    @property
    def topics(self) -> list[dict]:
        return self.raw.get("topics", [])

    @property
    def deliberation(self) -> Dict[str, Any]:
        return self.raw.get("deliberation", {})


def get_config() -> Config:
    return Config(load_config())
