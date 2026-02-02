#!/usr/bin/env python3
"""
Generate submission reports from findings.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_submission(
    finding: Dict[str, Any],
    platform: str,
    target_name: str,
    output_dir: Path,
) -> Path:
    """Generate a submission report from a finding."""

    template_file = TEMPLATES_DIR / f"submission_{platform}.md"
    if not template_file.exists():
        template_file = TEMPLATES_DIR / "submission_sherlock.md"  # Default

    template = template_file.read_text()

    # Map finding fields to template variables
    replacements = {
        "{{title}}": finding.get("title", "Untitled"),
        "{{description}}": finding.get("description", ""),
        "{{detail}}": finding.get("description", "") + "\n\n" + finding.get("proof", ""),
        "{{impact}}": finding.get("impact", ""),
        "{{code_snippet}}": _format_code_snippet(finding),
        "{{recommendation}}": _generate_recommendation(finding),
        "{{lines_of_code}}": _format_lines_of_code(finding),
        "{{proof_of_concept}}": finding.get("proof", ""),
        "{{mitigation}}": _generate_recommendation(finding),
    }

    content = template
    for key, value in replacements.items():
        content = content.replace(key, str(value))

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    severity = finding.get("severity", "medium").lower()
    slug = _slugify(finding.get("title", "finding")[:40])
    filename = f"{timestamp}-{severity}-{slug}.md"

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(content)

    return output_path


def _format_code_snippet(finding: Dict[str, Any]) -> str:
    """Format code snippet from finding."""
    file_path = finding.get("file", "")
    line = finding.get("line", "")
    proof = finding.get("proof", "")

    if file_path:
        return f"```solidity\n// {file_path}:{line}\n{proof}\n```"
    return f"```\n{proof}\n```"


def _format_lines_of_code(finding: Dict[str, Any]) -> str:
    """Format lines of code reference."""
    file_path = finding.get("file", "")
    line = finding.get("line", "")

    if file_path and line:
        # Assume GitHub URL format
        return f"https://github.com/xxx/{file_path}#L{line}"
    return file_path


def _generate_recommendation(finding: Dict[str, Any]) -> str:
    """Generate recommendation based on vulnerability type."""
    title = finding.get("title", "").lower()
    description = finding.get("description", "").lower()

    # Common recommendations
    if "reentrancy" in title or "reentrancy" in description:
        return """1. Follow the checks-effects-interactions pattern
2. Use OpenZeppelin's ReentrancyGuard
3. Update state before external calls"""

    if "overflow" in title or "overflow" in description:
        return """1. Use Solidity 0.8+ which has built-in overflow checks
2. If using older versions, use SafeMath library
3. Validate inputs before arithmetic operations"""

    if "access control" in title or "unauthorized" in description:
        return """1. Implement proper access control modifiers
2. Use OpenZeppelin's AccessControl or Ownable
3. Validate msg.sender in sensitive functions"""

    if "oracle" in title or "price" in description:
        return """1. Use time-weighted average prices (TWAP)
2. Implement price deviation checks
3. Use multiple oracle sources"""

    if "flash loan" in title or "flashloan" in description:
        return """1. Add flash loan protection checks
2. Verify contract state consistency within transactions
3. Consider using snapshot-based accounting"""

    return "Review the affected code and implement appropriate security controls."


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')


def generate_all_submissions(
    findings: List[Dict[str, Any]],
    platform: str,
    target_name: str,
    output_dir: Path,
) -> List[Path]:
    """Generate submissions for all findings."""
    paths = []

    for finding in findings:
        # Only generate for high-confidence findings
        if finding.get("confidence", 0) < 0.6:
            continue

        path = generate_submission(finding, platform, target_name, output_dir)
        paths.append(path)
        print(f"  📝 Generated: {path.name}")

    return paths


if __name__ == "__main__":
    # Test
    test_finding = {
        "title": "Reentrancy in withdraw function",
        "severity": "high",
        "confidence": 0.85,
        "description": "The withdraw function makes external calls before updating state.",
        "file": "contracts/Vault.sol",
        "line": 45,
        "impact": "Attacker can drain all funds from the vault.",
        "proof": "balance[msg.sender] = 0;\nmsg.sender.call{value: amount}('');",
    }

    output = generate_submission(
        test_finding,
        "sherlock",
        "Test Protocol",
        Path("/tmp/test-submissions"),
    )
    print(f"Generated: {output}")
