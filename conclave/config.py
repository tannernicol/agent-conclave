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

    # Environment overrides - Server
    host = os.getenv("CONCLAVE_HOST")
    port = os.getenv("CONCLAVE_PORT")
    if host:
        data.setdefault("server", {})["host"] = host
    if port:
        try:
            data.setdefault("server", {})["port"] = int(port)
        except ValueError:
            pass

    # Environment overrides - Data directory
    data_dir = os.getenv("CONCLAVE_DATA_DIR")
    if data_dir:
        data["data_dir"] = data_dir

    # Environment overrides - RAG
    rag_url = os.getenv("CONCLAVE_RAG_URL")
    if rag_url:
        data.setdefault("rag", {})["base_url"] = rag_url

    # Environment overrides - MCP config path
    mcp_config = os.getenv("CONCLAVE_MCP_CONFIG")
    if mcp_config:
        data["mcp_config_path"] = mcp_config

    # Environment overrides - Quality settings
    strict_mode = os.getenv("CONCLAVE_STRICT")
    if strict_mode is not None:
        data.setdefault("quality", {})["strict"] = strict_mode.lower() in ("true", "1", "yes")

    # Environment overrides - Calibration
    calibration_enabled = os.getenv("CONCLAVE_CALIBRATION")
    if calibration_enabled is not None:
        data.setdefault("calibration", {})["enabled"] = calibration_enabled.lower() in ("true", "1", "yes")

    # Environment overrides - Pipeline settings
    run_timeout = os.getenv("CONCLAVE_RUN_TIMEOUT")
    if run_timeout:
        try:
            data.setdefault("pipeline", {})["run_timeout_seconds"] = int(run_timeout)
        except ValueError:
            pass

    cli_timeout = os.getenv("CONCLAVE_CLI_TIMEOUT")
    if cli_timeout:
        try:
            data.setdefault("pipeline", {})["cli_timeout_seconds"] = int(cli_timeout)
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
        default = str(Path.home() / ".conclave")
        return Path(self.raw.get("data_dir", default))

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

    @property
    def mcp_config_path(self) -> Path | None:
        path = self.raw.get("mcp_config_path")
        return Path(path) if path else None

    @property
    def pipeline(self) -> Dict[str, Any]:
        return self.raw.get("pipeline", {})

    @property
    def run_timeout_seconds(self) -> int:
        """Maximum time for a single pipeline run in seconds. Default 10 minutes."""
        return int(self.pipeline.get("run_timeout_seconds", 600))

    @property
    def cli_timeout_seconds(self) -> int:
        """Timeout for CLI model calls in seconds. Default 3 minutes."""
        return int(self.pipeline.get("cli_timeout_seconds", 180))


def get_config() -> Config:
    return Config(load_config())
