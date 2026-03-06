from pathlib import Path

from conclave.config import Config, validate_config


def test_validate_config_accepts_defaults(tmp_path: Path) -> None:
    cfg = Config({
        "server": {"host": "127.0.0.1", "port": 8000},
        "data_dir": str(tmp_path),
        "pipeline": {"run_timeout_seconds": 60, "cli_timeout_seconds": 30},
        "rag": {"base_url": "http://localhost:9999"},
    })
    result = validate_config(cfg)
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_config_detects_issues(tmp_path: Path) -> None:
    bad_dir = tmp_path / "file.txt"
    bad_dir.write_text("x")
    cfg = Config({
        "server": {"port": "abc"},
        "data_dir": str(bad_dir),
        "pipeline": {"run_timeout_seconds": 0, "cli_timeout_seconds": -1},
        "rag": {"base_url": "ftp://bad"},
        "mcp_config_path": str(tmp_path / "missing.json"),
    })
    result = validate_config(cfg)
    assert result["ok"] is False
    fields = {issue["field"] for issue in result["issues"]}
    assert "server.port" in fields
    assert "data_dir" in fields
    assert "pipeline.run_timeout_seconds" in fields
    assert "pipeline.cli_timeout_seconds" in fields
    assert "rag.base_url" in fields
    assert "mcp_config_path" in fields
