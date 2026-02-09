"""Lightweight domain instruction hints for Conclave deliberation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainInstructions:
    """Text-only hints injected into deliberation and summarizer prompts."""
    deliberation_hint: str = ""
    summarizer_hint: str = ""


DOMAIN_INSTRUCTIONS: dict[str, DomainInstructions] = {
    "security": DomainInstructions(
        deliberation_hint=(
            "For security analysis, produce a structured report with sections:\n"
            "- Summary (<=6 bullets)\n"
            "- Findings (with severity, description, remediation)\n"
            "- Vulnerable Code Locations (file:line when available)\n"
            "- Recommendations\n"
            "Cite file paths and line numbers when available. "
            "If code context is missing, state assumptions and list needed files.\n"
        ),
        summarizer_hint=(
            "For security output, produce a structured report with sections:\n"
            "Summary, Findings, Vulnerable Code Locations, Recommendations.\n"
            "If you list a finding, include: Location (file:line), Root cause, "
            "Impact, Severity, Remediation.\n"
            "Cite file paths and line numbers when available. "
            "If missing, list required files instead of refusing.\n"
        ),
    ),
    "code_review": DomainInstructions(
        deliberation_hint=(
            "For code review, focus on:\n"
            "- Code quality, readability, and maintainability\n"
            "- Potential bugs and edge cases\n"
            "- Architecture and design pattern concerns\n"
            "- Performance implications\n"
            "Be specific with file paths and line numbers.\n"
        ),
        summarizer_hint="",
    ),
    "research": DomainInstructions(
        deliberation_hint=(
            "For research analysis:\n"
            "- Compare options systematically with clear criteria\n"
            "- Cite sources and evidence when available\n"
            "- Present trade-offs explicitly\n"
            "- Provide a clear recommendation with confidence level\n"
        ),
        summarizer_hint="",
    ),
    "creative": DomainInstructions(
        deliberation_hint=(
            "For creative tasks:\n"
            "- Generate diverse options before converging\n"
            "- Consider the target audience\n"
            "- Balance originality with practicality\n"
            "- Provide concrete examples, not just abstract advice\n"
        ),
        summarizer_hint="",
    ),
    "career": DomainInstructions(
        deliberation_hint=(
            "Provide direct, actionable feedback. Be critical and specific.\n"
            "Score each target role numerically with a one-sentence justification.\n"
            "Identify concrete strengths, gaps, and red flags.\n"
            "Recommend specific, actionable changes rather than generic advice.\n"
        ),
        summarizer_hint="",
    ),
}

_DEFAULT = DomainInstructions(
    deliberation_hint=(
        "RAG/MCP context is helpful but not required. Do not refuse due to missing sources.\n"
        "Provide a best-effort, prescriptive recommendation using your own reasoning.\n"
        "State assumptions explicitly and ask follow-up questions, but still give an initial answer.\n"
        "Do not mention model limitations or capabilities.\n"
    ),
    summarizer_hint="",
)


def get_domain_instructions(domain: str | None) -> DomainInstructions:
    """Look up domain-specific prompt hints. Falls back to generic defaults."""
    if not domain:
        return _DEFAULT
    return DOMAIN_INSTRUCTIONS.get(domain, _DEFAULT)
