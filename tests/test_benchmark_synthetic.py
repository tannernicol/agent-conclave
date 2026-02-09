import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_synthetic.py"


def test_markdown_output_contains_table():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--format", "markdown"], capture_output=True, text=True, check=True)
    out = proc.stdout
    assert "Synthetic Benchmarks" in out
    assert "| Scenario | p50 (ms) | p95 (ms) | CPU (%) | Memory (MB) |" in out


def test_json_output_has_metrics():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--format", "json"], capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert data["synthetic"] is True
    assert isinstance(data["metrics"], list)
    assert len(data["metrics"]) >= 1
    first = data["metrics"][0]
    assert {"scenario", "p50_ms", "p95_ms", "cpu_pct", "mem_mb"}.issubset(first)
