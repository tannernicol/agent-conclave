#!/usr/bin/env python3
"""
Knowledge integration for bug bounty hunting.
Incorporates learnings from ~/bug-bounty-recon.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

BOUNTY_RECON_DIR = Path.home() / "bug-bounty-recon"


@dataclass
class HuntKnowledge:
    """Consolidated hunting knowledge."""

    # From LEARNINGS.md
    HIGH_ROI_PATTERNS = [
        {
            "name": "Post-audit code analysis",
            "description": "Code added AFTER main audit gets less scrutiny",
            "check": "git log --oneline --since=AUDIT_DATE -- '*.sol'",
            "hit_rate": "HIGH",
        },
        {
            "name": "Uninitialized storage",
            "description": "Variables default to 0, caps may be bypassed",
            "patterns": ["cap = 0", "uninitialized", "default value"],
            "hit_rate": "HIGH",
        },
        {
            "name": "Disabled validation",
            "description": "Commented out require statements",
            "patterns": ["//require", "/* require", "// REMOVED"],
            "hit_rate": "HIGH",
        },
        {
            "name": "Parser workarounds",
            "description": "Comments like 'malformatted', 'backwards compat'",
            "patterns": ["malformatted", "backwards", "legacy", "workaround"],
            "hit_rate": "HIGH",
        },
        {
            "name": "Oracle/Price wrappers",
            "description": "Often added post-audit, complex math",
            "patterns": ["getPrice", "latestAnswer", "Oracle", "Feed"],
            "hit_rate": "HIGH",
        },
        {
            "name": "Sentinel values",
            "description": "Special values like type(int256).min for 'withdraw all'",
            "patterns": ["type(int256).min", "type(uint256).max", "WITHDRAW_ALL"],
            "hit_rate": "MEDIUM",
        },
    ]

    LOW_ROI_PATTERNS = [
        "Generic 'reentrancy' without specific call sequence",
        "Generic 'access control' without unauthorized action",
        "Generic 'integer overflow' on Solidity 0.8+",
        "CORS findings (almost always out of scope)",
        "API documentation endpoints (/swagger.json)",
        "Subdomain enumeration without vuln evidence",
    ]

    # Protocol-specific learnings
    PROTOCOL_LEARNINGS = {
        "fluid": {
            "high_value": [
                "Trust comments as red flags: 'almost always', 'extremely unlikely'",
                "Follow copy-paste breadcrumbs for systemic bugs",
                "Sentinel vs explicit path validation",
                "Clamp both raw and user-facing amounts",
                "Unbounded loops + external calls = gas DoS",
                "Check for skipped HF checks",
            ],
            "patterns": ["D4 helpers", "_getHfInfo", "_checkHf", "withdraw all"],
        },
        "sky": {
            "high_value": [
                "Focus on lockstake/* (post-audit)",
                "Oracle wrappers are high-value",
                "_getUrn() vs _getAuthedUrn() - check who pays",
            ],
            "skip": ["Core DSS contracts - well audited"],
        },
        "wormhole": {
            "high_value": [
                "Parser functions with 'malformatted' comments",
                "Disabled length validation",
                "NFTBridge pattern → check TokenBridge",
            ],
            "patterns": ["parsePayload", "parseTransfer", "validateMessageLength"],
        },
    }

    # False positive patterns
    FALSE_POSITIVES = {
        "smart_contracts": [
            {"claim": "Reentrancy in stake()", "reality": "CEI pattern followed"},
            {"claim": "Access control on init()", "reality": "One-time setup by design"},
            {"claim": "Integer overflow", "reality": "Solidity 0.8+ has built-in checks"},
            {"claim": "Delegatecall danger", "reality": "To trusted immutable address"},
        ],
        "web": [
            {"claim": "CORS misconfiguration", "reality": "Intentional for public API"},
            {"claim": "Information disclosure", "reality": "Public documentation"},
            {"claim": "Open redirect", "reality": "To same domain"},
        ],
    }

    # Best practices for LLM usage
    LLM_BEST_PRACTICES = {
        "local_good_for": [
            "Quick triage: 'Is this test code?'",
            "Pattern search: 'Find commented-out require statements'",
            "Code summary: 'What does this function do?'",
        ],
        "local_bad_for": [
            "'Find all vulnerabilities' (too broad, FPs)",
            "'Is this exploitable?' (needs deeper analysis)",
            "Final validation (use Claude instead)",
        ],
        "claude_good_for": [
            "Deep analysis of specific functions",
            "Validating findings from local models",
            "Complex multi-step vulnerability analysis",
            "Writing submission reports",
        ],
    }


def load_indexed_techniques(limit: int = 100) -> List[Dict[str, Any]]:
    """Load indexed vulnerability techniques."""
    techniques_file = BOUNTY_RECON_DIR / "indexed-techniques.jsonl"
    techniques = []

    if techniques_file.exists():
        for i, line in enumerate(techniques_file.read_text().strip().split("\n")):
            if i >= limit:
                break
            try:
                techniques.append(json.loads(line))
            except:
                pass

    return techniques


def load_false_positives() -> Dict[str, Any]:
    """Load false positive patterns from config."""
    fp_file = BOUNTY_RECON_DIR / "false_positive_config.json"
    if fp_file.exists():
        return json.loads(fp_file.read_text())
    return {}


def load_known_findings() -> List[Dict[str, Any]]:
    """Load known findings to avoid duplicates."""
    findings_file = BOUNTY_RECON_DIR / "known_findings.json"
    if findings_file.exists():
        try:
            return json.loads(findings_file.read_text())
        except:
            pass
    return []


def get_protocol_hints(protocol_name: str) -> Dict[str, Any]:
    """Get protocol-specific hunting hints."""
    protocol_lower = protocol_name.lower()

    for key, hints in HuntKnowledge.PROTOCOL_LEARNINGS.items():
        if key in protocol_lower:
            return hints

    return {"high_value": [], "patterns": [], "skip": []}


def should_skip_finding(finding: Dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a finding matches known false positive patterns.
    Returns (should_skip, reason).
    """
    title = finding.get("title", "").lower()
    description = finding.get("description", "").lower()

    # Check LOW_ROI patterns
    for pattern in HuntKnowledge.LOW_ROI_PATTERNS:
        pattern_lower = pattern.lower()
        if pattern_lower in title or pattern_lower in description:
            return True, f"Low ROI pattern: {pattern}"

    # Check smart contract FPs
    for fp in HuntKnowledge.FALSE_POSITIVES["smart_contracts"]:
        claim = fp["claim"].lower()
        if claim in title or claim in description:
            return True, f"Known FP: {fp['reality']}"

    # Check for generic claims without specifics
    generic_claims = [
        ("reentrancy", ["call sequence", "specific function", "step 1", "step 2"]),
        ("access control", ["unauthorized", "attacker can", "allows"]),
        ("overflow", ["exploit", "value exceeds", "at tick", "when"]),
    ]

    for claim, required_specifics in generic_claims:
        if claim in title or claim in description:
            has_specifics = any(s in description for s in required_specifics)
            if not has_specifics:
                return True, f"Generic {claim} claim without specifics"

    return False, ""


def enhance_prompt_with_knowledge(
    step: str,
    target_name: str,
    tech_stack: List[str],
) -> str:
    """Enhance a hunting prompt with relevant knowledge."""
    knowledge_context = []

    # Add protocol-specific hints
    hints = get_protocol_hints(target_name)
    if hints.get("high_value"):
        knowledge_context.append("PROTOCOL-SPECIFIC HIGH-VALUE PATTERNS:")
        for h in hints["high_value"][:5]:
            knowledge_context.append(f"  - {h}")

    if hints.get("patterns"):
        knowledge_context.append(f"KEY PATTERNS TO SEARCH: {', '.join(hints['patterns'])}")

    if hints.get("skip"):
        knowledge_context.append(f"SKIP THESE (well-audited): {', '.join(hints['skip'])}")

    # Add relevant high-ROI patterns based on step
    step_lower = step.lower()
    relevant_patterns = []

    if "oracle" in step_lower or "price" in step_lower:
        relevant_patterns.append(HuntKnowledge.HIGH_ROI_PATTERNS[4])  # Oracle wrappers
    if "overflow" in step_lower or "arithmetic" in step_lower:
        relevant_patterns.append(HuntKnowledge.HIGH_ROI_PATTERNS[1])  # Uninitialized
    if "access" in step_lower or "validation" in step_lower:
        relevant_patterns.append(HuntKnowledge.HIGH_ROI_PATTERNS[2])  # Disabled validation

    if relevant_patterns:
        knowledge_context.append("\nRELEVANT PATTERNS FROM PAST SUCCESSES:")
        for p in relevant_patterns:
            knowledge_context.append(f"  - {p['name']}: {p['description']}")
            if p.get("patterns"):
                knowledge_context.append(f"    Search for: {', '.join(p['patterns'][:3])}")

    # Add false positive warnings
    knowledge_context.append("\nAVOID THESE FALSE POSITIVES:")
    knowledge_context.append("  - Generic claims without specific exploitation path")
    knowledge_context.append("  - Reentrancy where CEI pattern is followed")
    knowledge_context.append("  - Integer overflow in Solidity 0.8+")
    knowledge_context.append("  - Access control on initialization functions")

    return "\n".join(knowledge_context)


def get_high_roi_checks(tech_stack: List[str]) -> List[str]:
    """Get high-ROI checks based on tech stack."""
    checks = []

    # Universal high-ROI checks
    checks.extend([
        "Find code added after main audit (git log analysis)",
        "Search for commented-out require/assert statements",
        "Find 'workaround', 'hack', 'TODO' comments",
        "Check for uninitialized storage variables",
    ])

    # Tech-specific
    if any(t in ["evm", "solidity"] for t in tech_stack):
        checks.extend([
            "Analyze oracle/price wrapper contracts",
            "Check for sentinel values (type(int256).min, max)",
            "Review unbounded loops with external calls",
            "Verify CEI pattern in state-changing functions",
        ])

    if any(t in ["solana", "anchor"] for t in tech_stack):
        checks.extend([
            "Check for missing signer validation",
            "Analyze PDA derivation for collisions",
            "Review account ownership validation",
            "Check for unchecked arithmetic in older code",
        ])

    return checks


# Singleton knowledge instance
_knowledge: Optional[HuntKnowledge] = None


def get_knowledge() -> HuntKnowledge:
    """Get or create knowledge singleton."""
    global _knowledge
    if _knowledge is None:
        _knowledge = HuntKnowledge()
    return _knowledge
