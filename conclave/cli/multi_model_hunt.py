#!/usr/bin/env python3
"""
Multi-model hunting engine - uses Claude, Codex, and local models optimally.

Model Roles:
- Local (Ollama): Fast initial triage, pattern scanning (free)
- Codex: Code-focused analysis, static patterns, quick iteration
- Claude: Deep reasoning, complex vulnerabilities, final validation

Strategy:
1. Local model does quick scan of all code
2. Codex analyzes specific patterns in parallel
3. Claude validates and deepens promising findings
4. Cross-validate between models to reduce false positives
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """A potential vulnerability finding."""
    title: str
    severity: str
    confidence: float
    description: str
    file: Optional[str] = None
    line: Optional[int] = None
    impact: Optional[str] = None
    proof: Optional[str] = None
    found_by: str = "unknown"
    validated_by: List[str] = field(default_factory=list)
    consensus_score: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    role: str  # triage, analysis, validation
    command: List[str]
    timeout: int = 180
    cost_per_1k_tokens: float = 0.0


class MultiModelHunter:
    """Coordinates multiple models for optimal hunting."""

    MODELS = {
        "local-triage": ModelConfig(
            name="qwen2.5-coder:7b",
            role="triage",
            command=["ollama", "run", "qwen2.5-coder:7b"],
            timeout=120,
            cost_per_1k_tokens=0.0,
        ),
        "local-bounty": ModelConfig(
            name="bounty-learned",
            role="analysis",
            command=["ollama", "run", "bounty-learned"],
            timeout=300,
            cost_per_1k_tokens=0.0,
        ),
        "codex": ModelConfig(
            name="codex",
            role="analysis",
            command=["codex", "--approval-mode", "full-auto", "-q"],
            timeout=180,
            cost_per_1k_tokens=0.01,  # Approximate
        ),
        "claude": ModelConfig(
            name="claude",
            role="validation",
            command=["claude", "--print", "--dangerously-skip-permissions", "-p"],
            timeout=300,
            cost_per_1k_tokens=0.015,  # Sonnet pricing
        ),
    }

    def __init__(self, target_dir: Path, budget_usd: float = 5.0):
        self.target_dir = target_dir
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.findings: List[Finding] = []
        self.executor = ThreadPoolExecutor(max_workers=4)

    def hunt(self, code_files: List[Path], vulnerability_types: List[str]) -> List[Finding]:
        """
        Run multi-model hunt on code files.

        Phase 1: Local triage (free, fast)
        Phase 2: Parallel Codex + Local-Bounty analysis
        Phase 3: Claude validation of high-confidence findings
        """
        all_findings = []

        # Phase 1: Quick local triage
        print("🔍 Phase 1: Local triage scan...")
        triage_findings = self._run_triage(code_files, vulnerability_types)
        print(f"  Found {len(triage_findings)} potential areas of interest")

        # Phase 2: Parallel deep analysis
        print("\n🔬 Phase 2: Parallel deep analysis (Codex + Local)...")
        analysis_findings = self._run_parallel_analysis(code_files, vulnerability_types, triage_findings)
        print(f"  Found {len(analysis_findings)} potential vulnerabilities")

        # Phase 3: Claude validation (only for high-value findings)
        high_conf = [f for f in analysis_findings if f.confidence >= 0.6]
        if high_conf and self.spent_usd < self.budget_usd * 0.5:
            print(f"\n✅ Phase 3: Claude validation of {len(high_conf)} findings...")
            validated = self._validate_findings(high_conf)
            all_findings.extend(validated)
        else:
            all_findings.extend(analysis_findings)

        # Calculate consensus scores
        self._calculate_consensus(all_findings)

        return sorted(all_findings, key=lambda f: f.consensus_score, reverse=True)

    def _run_triage(self, code_files: List[Path], vuln_types: List[str]) -> List[Dict[str, Any]]:
        """Quick local scan to identify areas of interest."""
        model = self.MODELS["local-triage"]
        results = []

        prompt = self._build_triage_prompt(code_files, vuln_types)

        try:
            result = subprocess.run(
                model.command + [prompt],
                capture_output=True,
                text=True,
                timeout=model.timeout,
                cwd=str(self.target_dir),
            )
            if result.stdout:
                # Parse for areas of interest
                results = self._parse_triage_results(result.stdout)
        except Exception as e:
            logger.warning(f"Triage failed: {e}")

        return results

    def _run_parallel_analysis(
        self,
        code_files: List[Path],
        vuln_types: List[str],
        triage_hints: List[Dict],
    ) -> List[Finding]:
        """Run Codex and local-bounty in parallel."""
        findings = []
        futures = []

        # Submit analysis tasks to both models
        for vuln_type in vuln_types:
            prompt = self._build_analysis_prompt(code_files, vuln_type, triage_hints)

            # Local bounty model (free)
            futures.append(
                self.executor.submit(
                    self._run_model,
                    "local-bounty",
                    prompt,
                    vuln_type,
                )
            )

            # Codex (if budget allows)
            if self.spent_usd < self.budget_usd * 0.3:
                futures.append(
                    self.executor.submit(
                        self._run_model,
                        "codex",
                        prompt,
                        vuln_type,
                    )
                )

        # Collect results
        for future in as_completed(futures):
            try:
                model_findings = future.result()
                findings.extend(model_findings)
            except Exception as e:
                logger.warning(f"Analysis task failed: {e}")

        return self._deduplicate_findings(findings)

    def _validate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Use Claude to validate and refine findings."""
        model = self.MODELS["claude"]
        validated = []

        for finding in findings:
            if self.spent_usd >= self.budget_usd:
                finding.validated_by.append("budget_exhausted")
                validated.append(finding)
                continue

            prompt = self._build_validation_prompt(finding)

            try:
                result = subprocess.run(
                    model.command + [prompt],
                    capture_output=True,
                    text=True,
                    timeout=model.timeout,
                    cwd=str(self.target_dir),
                )

                if result.stdout:
                    # Update finding based on Claude's analysis
                    is_valid, refined = self._parse_validation(result.stdout, finding)
                    if is_valid:
                        refined.validated_by.append("claude")
                        validated.append(refined)

                    # Estimate cost
                    tokens = (len(prompt) + len(result.stdout)) / 4
                    self.spent_usd += (tokens / 1000) * model.cost_per_1k_tokens

            except Exception as e:
                logger.warning(f"Validation failed for {finding.title}: {e}")
                validated.append(finding)

        return validated

    def _run_model(self, model_name: str, prompt: str, context: str) -> List[Finding]:
        """Run a single model and parse findings."""
        model = self.MODELS[model_name]
        findings = []

        try:
            if model_name == "codex":
                # Codex uses different invocation
                result = subprocess.run(
                    model.command + [prompt],
                    capture_output=True,
                    text=True,
                    timeout=model.timeout,
                    cwd=str(self.target_dir),
                )
            else:
                result = subprocess.run(
                    model.command + [prompt],
                    capture_output=True,
                    text=True,
                    timeout=model.timeout,
                    cwd=str(self.target_dir),
                )

            if result.stdout:
                findings = self._parse_findings(result.stdout, model_name)

                # Track cost
                if model.cost_per_1k_tokens > 0:
                    tokens = (len(prompt) + len(result.stdout)) / 4
                    self.spent_usd += (tokens / 1000) * model.cost_per_1k_tokens

        except subprocess.TimeoutExpired:
            logger.warning(f"{model_name} timed out on {context}")
        except Exception as e:
            logger.warning(f"{model_name} failed: {e}")

        return findings

    def _build_triage_prompt(self, code_files: List[Path], vuln_types: List[str]) -> str:
        """Build prompt for quick triage scan."""
        files_list = "\n".join(str(f) for f in code_files[:20])
        vuln_list = ", ".join(vuln_types)

        return f"""Quick security scan of smart contract codebase.

FILES TO SCAN:
{files_list}

VULNERABILITY TYPES TO CHECK:
{vuln_list}

OUTPUT FORMAT:
For each potential issue area, output:
AREA: <file>:<function or line range>
CONCERN: <brief description>
PRIORITY: <high/medium/low>

Only output areas that warrant deeper analysis. Be selective."""

    def _build_analysis_prompt(
        self,
        code_files: List[Path],
        vuln_type: str,
        hints: List[Dict],
    ) -> str:
        """Build detailed analysis prompt."""
        hints_text = ""
        if hints:
            hints_text = "AREAS OF INTEREST:\n" + "\n".join(
                f"- {h.get('file', '?')}: {h.get('concern', '?')}"
                for h in hints[:5]
            )

        return f"""Security audit for {vuln_type} vulnerabilities.

{hints_text}

TASK: Analyze the code for {vuln_type} vulnerabilities.

OUTPUT FORMAT (for each finding):
FINDING:
Title: <descriptive title>
Severity: <critical|high|medium|low>
Confidence: <0.0-1.0>
File: <file path>
Line: <line number>
Description: <detailed description>
Impact: <what can attacker do>
Proof: <code snippet or reasoning>
END_FINDING

If no vulnerabilities found: NO_FINDINGS"""

    def _build_validation_prompt(self, finding: Finding) -> str:
        """Build validation prompt for Claude."""
        return f"""Validate this potential vulnerability finding:

TITLE: {finding.title}
SEVERITY: {finding.severity}
CONFIDENCE: {finding.confidence}
FILE: {finding.file}
DESCRIPTION: {finding.description}
IMPACT: {finding.impact}
PROOF: {finding.proof}

FOUND BY: {finding.found_by}

TASKS:
1. Is this a real vulnerability? Consider edge cases and mitigations.
2. Is the severity accurate?
3. Could this be a false positive?
4. Refine the description and impact if needed.

OUTPUT FORMAT:
VALIDATION:
IsValid: <true|false>
Confidence: <0.0-1.0>
RefinedSeverity: <critical|high|medium|low>
Reasoning: <your analysis>
RefinedDescription: <improved description if needed>
END_VALIDATION"""

    def _parse_findings(self, response: str, model_name: str) -> List[Finding]:
        """Parse findings from model response."""
        import re
        findings = []

        pattern = r'FINDING:\s*\n(.*?)END_FINDING'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            finding_dict = {"found_by": model_name}

            for field in ["Title", "Severity", "Confidence", "File", "Line", "Description", "Impact", "Proof"]:
                field_pattern = rf'{field}:\s*(.+?)(?=\n[A-Z][a-z]+:|$)'
                field_match = re.search(field_pattern, match, re.DOTALL)
                if field_match:
                    finding_dict[field.lower()] = field_match.group(1).strip()

            # Convert confidence
            if "confidence" in finding_dict:
                try:
                    finding_dict["confidence"] = float(finding_dict["confidence"])
                except:
                    finding_dict["confidence"] = 0.5

            if finding_dict.get("title") and finding_dict.get("description"):
                findings.append(Finding(
                    title=finding_dict.get("title", ""),
                    severity=finding_dict.get("severity", "medium"),
                    confidence=finding_dict.get("confidence", 0.5),
                    description=finding_dict.get("description", ""),
                    file=finding_dict.get("file"),
                    line=int(finding_dict["line"]) if finding_dict.get("line", "").isdigit() else None,
                    impact=finding_dict.get("impact"),
                    proof=finding_dict.get("proof"),
                    found_by=model_name,
                ))

        return findings

    def _parse_triage_results(self, response: str) -> List[Dict[str, Any]]:
        """Parse triage scan results."""
        import re
        results = []

        pattern = r'AREA:\s*(.+?)\nCONCERN:\s*(.+?)\nPRIORITY:\s*(.+?)(?=\n\n|\nAREA:|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for file, concern, priority in matches:
            results.append({
                "file": file.strip(),
                "concern": concern.strip(),
                "priority": priority.strip().lower(),
            })

        return results

    def _parse_validation(self, response: str, original: Finding) -> Tuple[bool, Finding]:
        """Parse validation response and update finding."""
        import re

        is_valid = True
        refined = Finding(
            title=original.title,
            severity=original.severity,
            confidence=original.confidence,
            description=original.description,
            file=original.file,
            line=original.line,
            impact=original.impact,
            proof=original.proof,
            found_by=original.found_by,
            validated_by=list(original.validated_by),
        )

        # Parse validation block
        valid_match = re.search(r'IsValid:\s*(true|false)', response, re.IGNORECASE)
        if valid_match:
            is_valid = valid_match.group(1).lower() == "true"

        conf_match = re.search(r'Confidence:\s*([\d.]+)', response)
        if conf_match:
            refined.confidence = float(conf_match.group(1))

        sev_match = re.search(r'RefinedSeverity:\s*(\w+)', response)
        if sev_match:
            refined.severity = sev_match.group(1).lower()

        desc_match = re.search(r'RefinedDescription:\s*(.+?)(?=END_VALIDATION|$)', response, re.DOTALL)
        if desc_match and desc_match.group(1).strip():
            refined.description = desc_match.group(1).strip()

        return is_valid, refined

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings, keeping highest confidence."""
        seen = {}
        for f in findings:
            key = (f.title.lower(), f.file)
            if key not in seen or f.confidence > seen[key].confidence:
                # Merge validated_by lists
                if key in seen:
                    f.validated_by = list(set(f.validated_by + seen[key].validated_by + [seen[key].found_by]))
                seen[key] = f
        return list(seen.values())

    def _calculate_consensus(self, findings: List[Finding]):
        """Calculate consensus score based on multi-model agreement."""
        for f in findings:
            # Base score from confidence
            score = f.confidence

            # Bonus for multi-model agreement
            validators = len(set([f.found_by] + f.validated_by))
            if validators >= 3:
                score *= 1.3  # 30% bonus for 3+ models agreeing
            elif validators >= 2:
                score *= 1.15  # 15% bonus for 2 models

            # Bonus for Claude validation
            if "claude" in f.validated_by:
                score *= 1.1

            f.consensus_score = min(score, 1.0)
