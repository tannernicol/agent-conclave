"""Domain routing defaults and calibration mapping."""
from __future__ import annotations

DOMAIN_TO_CALIBRATION = {
    "security": "security",
    "code_review": "code_review",
    "research": "research",
    "creative": "creative",
    "career": "career",
    "general": "general",
}

KNOWN_DOMAINS = set(DOMAIN_TO_CALIBRATION.keys())
