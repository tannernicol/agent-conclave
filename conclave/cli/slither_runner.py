#!/usr/bin/env python3
"""
Slither static analysis runner for smart contracts.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def run_slither(
    target_path: Path,
    detectors: List[str] = None,
    exclude_detectors: List[str] = None,
) -> Dict[str, Any]:
    """
    Run Slither on a target path.

    Args:
        target_path: Path to .sol file, directory, or contract address
        detectors: Specific detectors to run (None = all)
        exclude_detectors: Detectors to exclude

    Returns:
        Dict with 'success', 'findings', 'errors'
    """
    result = {
        "success": False,
        "findings": [],
        "errors": [],
        "detector_count": 0,
    }

    cmd = ["slither", str(target_path), "--json", "-"]

    # Add detector filters
    if detectors:
        cmd.extend(["--detect", ",".join(detectors)])
    if exclude_detectors:
        cmd.extend(["--exclude", ",".join(exclude_detectors)])

    # Exclude common informational detectors by default
    default_exclude = [
        "naming-convention",
        "pragma",
        "solc-version",
        "external-function",
        "constable-states",
        "dead-code",
        "similar-names",
    ]
    if not exclude_detectors:
        cmd.extend(["--exclude", ",".join(default_exclude)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(target_path.parent) if target_path.is_file() else str(target_path),
        )

        if proc.stdout:
            try:
                slither_output = json.loads(proc.stdout)
                result["success"] = slither_output.get("success", False)
                result["findings"] = parse_slither_findings(slither_output)
                result["detector_count"] = len(slither_output.get("results", {}).get("detectors", []))
            except json.JSONDecodeError as e:
                result["errors"].append(f"JSON parse error: {e}")

        if proc.stderr and "error" in proc.stderr.lower():
            result["errors"].append(proc.stderr[:500])

    except subprocess.TimeoutExpired:
        result["errors"].append("Slither timed out after 5 minutes")
    except FileNotFoundError:
        result["errors"].append("Slither not found. Install with: pip install slither-analyzer")
    except Exception as e:
        result["errors"].append(str(e))

    return result


def parse_slither_findings(slither_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse Slither JSON output into standardized findings."""
    findings = []

    detectors = slither_output.get("results", {}).get("detectors", [])

    for detector in detectors:
        # Map Slither impact to our severity
        impact = detector.get("impact", "").lower()
        severity_map = {
            "high": "high",
            "medium": "medium",
            "low": "low",
            "informational": "info",
            "optimization": "info",
        }
        severity = severity_map.get(impact, "medium")

        # Skip informational by default
        if severity == "info":
            continue

        # Extract location info
        elements = detector.get("elements", [])
        file_path = ""
        line_number = None

        if elements:
            first_elem = elements[0]
            source = first_elem.get("source_mapping", {})
            file_path = source.get("filename_relative", "")
            lines = source.get("lines", [])
            if lines:
                line_number = lines[0]

        finding = {
            "title": detector.get("check", "Unknown"),
            "severity": severity,
            "confidence": _map_confidence(detector.get("confidence", "Medium")),
            "description": detector.get("description", ""),
            "file": file_path,
            "line": line_number,
            "impact": detector.get("impact", ""),
            "detector": detector.get("check", ""),
            "found_by": "slither",
        }

        # Add code snippet if available
        if elements:
            snippets = []
            for elem in elements[:3]:
                name = elem.get("name", "")
                type_elem = elem.get("type", "")
                if name:
                    snippets.append(f"{type_elem}: {name}")
            finding["proof"] = "\n".join(snippets)

        findings.append(finding)

    return findings


def _map_confidence(conf: str) -> float:
    """Map Slither confidence to 0-1 scale."""
    conf_map = {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.5,
    }
    return conf_map.get(conf.lower(), 0.7)


# High-value detectors for bounty hunting
HIGH_VALUE_DETECTORS = [
    "reentrancy-eth",
    "reentrancy-no-eth",
    "reentrancy-benign",
    "unchecked-transfer",
    "arbitrary-send-erc20",
    "arbitrary-send-eth",
    "controlled-delegatecall",
    "controlled-array-length",
    "uninitialized-state",
    "uninitialized-local",
    "uninitialized-storage",
    "weak-prng",
    "msg-value-loop",
    "delegatecall-loop",
    "incorrect-equality",
    "tautology",
    "divide-before-multiply",
    "shadowing-state",
    "tx-origin",
    "suicidal",
    "locked-ether",
]

# Detectors that often produce false positives
NOISY_DETECTORS = [
    "reentrancy-unlimited-gas",
    "calls-loop",
    "assembly",
    "low-level-calls",
    "missing-zero-check",
    "boolean-equal",
    "too-many-digits",
]


def run_focused_slither(target_path: Path) -> Dict[str, Any]:
    """Run Slither with bounty-focused detector selection."""
    return run_slither(
        target_path,
        detectors=HIGH_VALUE_DETECTORS,
        exclude_detectors=NOISY_DETECTORS,
    )


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        result = run_focused_slither(target)
        print(f"Success: {result['success']}")
        print(f"Findings: {len(result['findings'])}")
        for f in result['findings']:
            print(f"  [{f['severity']}] {f['title']}: {f['file']}:{f['line']}")
        if result['errors']:
            print(f"Errors: {result['errors']}")
