#!/usr/bin/env python3
"""
Conclave Hunt CLI - Autonomous bug bounty target discovery and hunting.

Usage:
    conclave-hunt [options]
    conclave-hunt --resume
    conclave-hunt --status

Options:
    --max-tokens N      Maximum tokens to use (default: 100000, ~$5)
    --max-time M        Maximum time in minutes (default: 45)
    --platform P        Filter platforms: immunefi,code4rena,sherlock,hackerone
    --tech T            Filter tech: solana,evm,rust,move (comma-separated)
    --min-bounty N      Minimum bounty in USD (default: 10000)
    --resume            Resume previous hunt session
    --status            Show current hunt status
    --dry-run           Discover targets but don't start hunting
    --notify            Send phone notification on high-confidence findings
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
BOUNTY_RECON_DIR = Path.home() / "bug-bounty-recon"
TARGETS_DIR = BOUNTY_RECON_DIR / "targets"
STATE_FILE = BOUNTY_RECON_DIR / "hunt-state.json"

# Defaults
DEFAULT_MAX_TOKENS = 100_000  # ~$5 at current rates
DEFAULT_MAX_TIME_MINUTES = 45
DEFAULT_MIN_BOUNTY = 10_000
DEFAULT_CHECKPOINT_INTERVAL = 600  # 10 minutes

# Token costs (approximate, for budgeting)
TOKEN_COSTS = {
    "claude-opus": {"input": 15.0, "output": 75.0},  # per 1M tokens
    "claude-sonnet": {"input": 3.0, "output": 15.0},
    "codex": {"input": 2.0, "output": 8.0},
    "local": {"input": 0.0, "output": 0.0},
}


@dataclass
class Target:
    """A bug bounty target."""
    name: str
    platform: str
    bounty_usd: int
    tech_stack: List[str]
    launch_date: str  # ISO format
    end_date: Optional[str]
    url: str
    repo_url: Optional[str]
    scope: List[str]
    tvl_usd: Optional[int] = None
    competition_score: float = 0.5  # 0=low competition, 1=high
    opportunity_score: float = 0.0  # Calculated: bounty / competition
    days_old: int = 0

    def __post_init__(self):
        # Calculate days old
        try:
            launch = datetime.fromisoformat(self.launch_date.replace('Z', '+00:00'))
            self.days_old = (datetime.now().astimezone() - launch).days
        except:
            self.days_old = 0

        # Calculate opportunity score
        if self.competition_score > 0:
            freshness_bonus = max(0, 1 - (self.days_old / 14))  # Bonus for < 2 weeks old
            self.opportunity_score = (self.bounty_usd / 10000) * (1 - self.competition_score) * (1 + freshness_bonus)


@dataclass
class HuntState:
    """Persistent hunt session state."""
    target: Optional[Dict[str, Any]] = None
    phase: str = "idle"  # idle, discover, setup, hunt, complete
    started_at: Optional[str] = None
    tokens_used: int = 0
    time_elapsed_seconds: int = 0
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    hunt_plan: List[str] = field(default_factory=list)
    current_hunt_index: int = 0

    @classmethod
    def load(cls) -> "HuntState":
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(**data)
            except:
                pass
        return cls()

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    def checkpoint(self, note: str = ""):
        self.checkpoints.append({
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase,
            "tokens_used": self.tokens_used,
            "note": note,
        })
        self.save()


class TokenBudget:
    """Track and enforce token budget."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0
        self.by_model: Dict[str, int] = {}

    def use(self, tokens: int, model: str = "unknown"):
        self.used += tokens
        self.by_model[model] = self.by_model.get(model, 0) + tokens

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    def exhausted(self) -> bool:
        return self.used >= self.max_tokens

    def estimate_cost(self) -> float:
        """Estimate cost in USD."""
        total = 0.0
        for model, tokens in self.by_model.items():
            rates = TOKEN_COSTS.get(model, TOKEN_COSTS["claude-sonnet"])
            # Assume 70% input, 30% output
            total += (tokens * 0.7 * rates["input"] + tokens * 0.3 * rates["output"]) / 1_000_000
        return total

    def status(self) -> str:
        pct = (self.used / self.max_tokens * 100) if self.max_tokens > 0 else 0
        cost = self.estimate_cost()
        return f"{self.used:,}/{self.max_tokens:,} tokens ({pct:.1f}%) ~${cost:.2f}"


class TimeBudget:
    """Track and enforce time budget."""

    def __init__(self, max_minutes: int):
        self.max_seconds = max_minutes * 60
        self.start_time = time.time()

    def elapsed(self) -> int:
        return int(time.time() - self.start_time)

    def remaining(self) -> int:
        return max(0, self.max_seconds - self.elapsed())

    def exhausted(self) -> bool:
        return self.elapsed() >= self.max_seconds

    def status(self) -> str:
        elapsed = self.elapsed()
        remaining = self.remaining()
        return f"{elapsed // 60}m {elapsed % 60}s elapsed, {remaining // 60}m {remaining % 60}s remaining"


class HuntCLI:
    """Main hunt CLI implementation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = HuntState.load()
        self.token_budget = TokenBudget(args.max_tokens)
        self.time_budget = TimeBudget(args.max_time)
        self.mcp_available = self._check_mcp()

    def _check_mcp(self) -> Dict[str, bool]:
        """Check which MCP servers are available."""
        available = {}
        mcp_config = Path.home() / ".mcp.json"
        if mcp_config.exists():
            try:
                data = json.loads(mcp_config.read_text())
                for name in data.get("mcpServers", {}):
                    available[name] = True
            except:
                pass
        return available

    def run(self):
        """Main entry point."""
        if self.args.findings or self.args.findings_target:
            self._list_findings(self.args.findings_target)
            return

        if self.args.status:
            self._show_status()
            return

        if self.args.resume:
            if self.state.phase == "idle":
                logger.error("No hunt session to resume. Start a new one with: conclave-hunt")
                return
            self._resume_hunt()
            return

        # Fresh hunt
        self._run_fresh_hunt()

    def _list_findings(self, target_filter: Optional[str] = None):
        """List all findings, optionally filtered by target."""
        print("\n" + "=" * 70)
        print("ALL FINDINGS" + (f" - {target_filter}" if target_filter else ""))
        print("=" * 70)

        # Read from master index
        index_file = BOUNTY_RECON_DIR / "all-findings.jsonl"
        if not index_file.exists():
            print("\nNo findings yet. Start a hunt with: conclave-hunt")
            return

        findings = []
        for line in index_file.read_text().strip().split("\n"):
            if line:
                try:
                    entry = json.loads(line)
                    if target_filter and target_filter.lower() not in entry.get("target", "").lower():
                        continue
                    findings.append(entry)
                except:
                    pass

        if not findings:
            print("\nNo findings match filter.")
            return

        # Group by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": [], "unknown": []}
        for f in findings:
            sev = f.get("finding", {}).get("severity", "unknown").lower()
            if sev not in by_severity:
                sev = "unknown"
            by_severity[sev].append(f)

        # Display
        for severity in ["critical", "high", "medium", "low", "unknown"]:
            items = by_severity[severity]
            if not items:
                continue

            print(f"\n{'🔴' if severity == 'critical' else '🟠' if severity == 'high' else '🟡' if severity == 'medium' else '🟢'} {severity.upper()} ({len(items)})")
            print("-" * 50)

            for entry in items:
                f = entry.get("finding", {})
                target = entry.get("target", "?")
                conf = f.get("confidence", 0)
                title = f.get("title", "Untitled")[:50]
                ts = entry.get("timestamp", "")[:10]
                print(f"  [{target[:15]:<15}] {title:<50} ({conf:.0%}) {ts}")

        # Summary
        total = len(findings)
        print(f"\n" + "=" * 70)
        print(f"Total: {total} findings across all hunts")
        print(f"\n📁 Finding locations:")
        print(f"   Master index: {index_file}")
        print(f"   Per-target:   ~/bug-bounty-recon/targets/*/findings/")
        print(f"   Submissions:  ~/bug-bounty-recon/submissions/")
        print("=" * 70 + "\n")

    def _show_status(self):
        """Show current hunt status."""
        print("\n" + "=" * 60)
        print("CONCLAVE HUNT STATUS")
        print("=" * 60)

        if self.state.phase == "idle":
            print("\nNo active hunt session.")
            print("Start one with: conclave-hunt")
        else:
            print(f"\nPhase: {self.state.phase}")
            if self.state.target:
                t = self.state.target
                print(f"Target: {t.get('name')} ({t.get('platform')})")
                print(f"Bounty: ${t.get('bounty_usd', 0):,}")
            print(f"Tokens: {self.state.tokens_used:,}")
            print(f"Findings: {len(self.state.findings)}")
            if self.state.hunt_plan:
                print(f"Hunt progress: {self.state.current_hunt_index}/{len(self.state.hunt_plan)}")
            print(f"\nResume with: conclave-hunt --resume")

        print("=" * 60 + "\n")

    def _run_fresh_hunt(self):
        """Run a fresh hunt from discovery."""
        print("\n" + "=" * 60)
        print("🎯 CONCLAVE HUNT - Target Discovery")
        print("=" * 60)
        print(f"Budget: {self.args.max_tokens:,} tokens, {self.args.max_time} minutes")
        print(f"Min bounty: ${self.args.min_bounty:,}")
        if self.args.platform:
            print(f"Platforms: {self.args.platform}")
        if self.args.tech:
            print(f"Tech filter: {self.args.tech}")
        print("=" * 60 + "\n")

        # Auto-refresh platform targets if stale (> 6 hours old)
        self._maybe_refresh_targets()

        # Phase 1: Discover targets
        self.state.phase = "discover"
        self.state.started_at = datetime.now().isoformat()
        self.state.save()

        targets = self._discover_targets()

        if not targets:
            logger.error("No targets found matching criteria.")
            self.state.phase = "idle"
            self.state.save()
            return

        # Phase 2: Interactive selection
        target = self._select_target(targets)

        if not target:
            logger.info("No target selected. Exiting.")
            self.state.phase = "idle"
            self.state.save()
            return

        if self.args.dry_run:
            logger.info("Dry run - stopping before setup.")
            self.state.phase = "idle"
            self.state.save()
            return

        # Reset state for new target
        self.state.target = asdict(target)
        self.state.hunt_plan = []  # Will be regenerated
        self.state.current_hunt_index = 0
        self.state.findings = []
        self.state.tokens_used = 0
        self.state.time_elapsed_seconds = 0
        self.state.save()

        # Phase 3: Setup
        self._setup_target(target)

        # Phase 4: Hunt
        self._hunt_target(target)

    def _maybe_refresh_targets(self):
        """Refresh platform targets if they're stale."""
        platforms_dir = BOUNTY_RECON_DIR / "platform-targets"
        if not platforms_dir.exists():
            self._refresh_targets()
            return

        # Check if any file is older than 6 hours
        stale = False
        for f in platforms_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                updated = data.get("updated", "")
                if updated:
                    updated_dt = datetime.fromisoformat(updated)
                    age_hours = (datetime.now() - updated_dt.replace(tzinfo=None)).total_seconds() / 3600
                    if age_hours > 6:
                        stale = True
                        break
            except:
                stale = True
                break

        if stale:
            print("📡 Refreshing platform targets (data > 6h old)...")
            self._refresh_targets()

    def _refresh_targets(self):
        """Run the platform scraper."""
        try:
            import subprocess
            scraper = Path(__file__).parent / "scrape_platforms.py"
            if scraper.exists():
                subprocess.run(["python3", str(scraper)], capture_output=True, timeout=120)
                print("  ✓ Platform targets refreshed")
        except Exception as e:
            logger.warning(f"Failed to refresh targets: {e}")

    def _discover_targets(self) -> List[Target]:
        """Discover fresh targets from bounty platforms."""
        print("🔍 Discovering fresh targets...\n")

        targets = []
        platforms = self.args.platform.split(",") if self.args.platform else ["immunefi", "code4rena", "sherlock"]

        for platform in platforms:
            print(f"  Checking {platform}...")
            platform_targets = self._fetch_platform_targets(platform)
            targets.extend(platform_targets)

        # Filter by criteria
        filtered = []
        for t in targets:
            if t.bounty_usd < self.args.min_bounty:
                continue
            if self.args.tech:
                tech_filter = set(self.args.tech.lower().split(","))
                if not any(tech.lower() in tech_filter for tech in t.tech_stack):
                    continue
            if t.days_old > 14:  # Skip targets older than 2 weeks
                continue
            filtered.append(t)

        # Sort by opportunity score
        filtered.sort(key=lambda x: x.opportunity_score, reverse=True)

        print(f"\n✓ Found {len(filtered)} matching targets\n")
        return filtered[:10]  # Top 10

    def _fetch_platform_targets(self, platform: str) -> List[Target]:
        """Fetch targets from a specific platform."""
        targets = []

        if platform == "immunefi":
            targets.extend(self._fetch_immunefi())
        elif platform == "code4rena":
            targets.extend(self._fetch_code4rena())
        elif platform == "sherlock":
            targets.extend(self._fetch_sherlock())

        return targets

    def _fetch_immunefi(self) -> List[Target]:
        """Fetch from Immunefi."""
        # Use recon MCP if available, otherwise scrape
        if self.mcp_available.get("recon"):
            # TODO: Add immunefi listing to recon MCP
            pass

        # For now, return manually tracked targets
        # In production, this would scrape/API call Immunefi
        return self._get_cached_targets("immunefi")

    def _fetch_code4rena(self) -> List[Target]:
        """Fetch from Code4rena."""
        return self._get_cached_targets("code4rena")

    def _fetch_sherlock(self) -> List[Target]:
        """Fetch from Sherlock."""
        return self._get_cached_targets("sherlock")

    def _get_cached_targets(self, platform: str) -> List[Target]:
        """Get targets from local cache/tracking file."""
        cache_file = BOUNTY_RECON_DIR / "platform-targets" / f"{platform}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return [Target(**t) for t in data.get("targets", [])]
            except Exception as e:
                logger.warning(f"Failed to load {platform} cache: {e}")
        return []

    def _fetch_contest_details(self, target: Target) -> Dict[str, Any]:
        """Fetch detailed contest info from the platform page."""
        details = {
            "scope_files": [],
            "rules": [],
            "prize_pool": target.bounty_usd,
            "start_date": target.launch_date,
            "end_date": target.end_date,
            "judging_criteria": [],
            "known_issues": [],
            "repo_url": target.repo_url,
        }

        platform = target.platform.lower()
        url = target.url

        try:
            if platform == "sherlock":
                details.update(self._fetch_sherlock_details(url))
            elif platform == "code4rena":
                details.update(self._fetch_code4rena_details(url))
            elif platform == "immunefi":
                details.update(self._fetch_immunefi_details(url))
        except Exception as e:
            logger.warning(f"Failed to fetch contest details: {e}")

        return details

    def _fetch_sherlock_details(self, url: str) -> Dict[str, Any]:
        """Fetch contest details from Sherlock."""
        details = {}
        try:
            import urllib.request
            import re

            # Sherlock contest pages have structure like /contests/1225
            # API endpoint: https://audits.sherlock.xyz/api/contests/{id}
            match = re.search(r'/contests/(\d+)', url)
            if match:
                contest_id = match.group(1)
                api_url = f"https://audits.sherlock.xyz/api/contests/{contest_id}"

                req = urllib.request.Request(api_url, headers={"User-Agent": "conclave-hunt/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())

                    details["title"] = data.get("title", "")
                    details["prize_pool"] = data.get("prizePool", 0)
                    details["start_date"] = data.get("startDate", "")
                    details["end_date"] = data.get("endDate", "")
                    details["repo_url"] = data.get("repoUrl", "")
                    details["docs_url"] = data.get("docsUrl", "")

                    # Scope
                    scope = data.get("scope", {})
                    details["scope_files"] = scope.get("files", [])
                    details["scope_description"] = scope.get("description", "")

                    # Known issues / out of scope
                    details["known_issues"] = data.get("knownIssues", [])
                    details["out_of_scope"] = data.get("outOfScope", [])

                    # Judging
                    details["judging_criteria"] = data.get("judgingCriteria", [])

                    print(f"  ✓ Fetched Sherlock contest #{contest_id}")
        except Exception as e:
            logger.debug(f"Sherlock fetch failed: {e}")

        return details

    def _fetch_code4rena_details(self, url: str) -> Dict[str, Any]:
        """Fetch contest details from Code4rena."""
        details = {}
        try:
            import urllib.request
            import re

            # Code4rena URLs like: https://code4rena.com/contests/jupiter-lend
            # Try to get from their API or scrape
            match = re.search(r'/contests/([^/]+)', url)
            if match:
                contest_slug = match.group(1)
                api_url = f"https://code4rena.com/api/v1/contests/{contest_slug}"

                req = urllib.request.Request(api_url, headers={"User-Agent": "conclave-hunt/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())

                    details["title"] = data.get("title", "")
                    details["prize_pool"] = data.get("prize", 0)
                    details["start_date"] = data.get("startTime", "")
                    details["end_date"] = data.get("endTime", "")
                    details["repo_url"] = data.get("repo", "")

                    details["scope_files"] = data.get("scope", [])
                    details["known_issues"] = data.get("knownIssues", [])

                    print(f"  ✓ Fetched Code4rena contest: {contest_slug}")
        except Exception as e:
            logger.debug(f"Code4rena fetch failed: {e}")

        return details

    def _fetch_immunefi_details(self, url: str) -> Dict[str, Any]:
        """Fetch bounty details from Immunefi."""
        details = {}
        try:
            import urllib.request
            import re

            # Immunefi URLs like: https://immunefi.com/bounty/protocol-name
            match = re.search(r'/bounty/([^/]+)', url)
            if match:
                bounty_slug = match.group(1)
                api_url = f"https://immunefi.com/api/bounty/{bounty_slug}"

                req = urllib.request.Request(api_url, headers={"User-Agent": "conclave-hunt/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())

                    details["title"] = data.get("name", "")
                    details["max_bounty"] = data.get("maxBounty", 0)

                    # Scope
                    assets = data.get("assets", [])
                    details["scope_files"] = [a.get("target", "") for a in assets]
                    details["asset_types"] = list(set(a.get("type", "") for a in assets))

                    # Rules
                    details["rules"] = data.get("programRules", [])
                    details["out_of_scope"] = data.get("outOfScope", [])

                    print(f"  ✓ Fetched Immunefi bounty: {bounty_slug}")
        except Exception as e:
            logger.debug(f"Immunefi fetch failed: {e}")

        return details

    def _select_target(self, targets: List[Target]) -> Optional[Target]:
        """Interactive target selection (or auto-select if --auto)."""
        print("=" * 70)
        print("SELECT TARGET")
        print("=" * 70)
        print(f"{'#':<3} {'Name':<25} {'Bounty':<10} {'Platform':<12} {'Tech':<10} {'Age':<5} {'Score':<6}")
        print("-" * 70)

        for i, t in enumerate(targets, 1):
            tech = ",".join(t.tech_stack[:2]) if t.tech_stack else "?"
            age = f"{t.days_old}d" if t.days_old >= 0 else "soon"
            score = f"{t.opportunity_score:.1f}"
            print(f"{i:<3} {t.name[:24]:<25} ${t.bounty_usd//1000}k{'':<5} {t.platform:<12} {tech:<10} {age:<5} {score:<6}")

        print("-" * 70)

        # Auto-select mode
        if self.args.auto:
            selected = targets[0]
            print(f"\n✓ Auto-selected: {selected.name} (highest opportunity score)\n")
            return selected

        # Direct target specification
        if self.args.target:
            for t in targets:
                if self.args.target.lower() in t.name.lower():
                    print(f"\n✓ Selected: {t.name}\n")
                    return t
            print(f"\n⚠ Target '{self.args.target}' not found in list")
            return None

        print("[0] Cancel")
        print()

        while True:
            try:
                choice = input("Select target (0-{max}): ".format(max=len(targets)))
                idx = int(choice)
                if idx == 0:
                    return None
                if 1 <= idx <= len(targets):
                    selected = targets[idx - 1]
                    print(f"\n✓ Selected: {selected.name}\n")
                    return selected
            except (ValueError, KeyboardInterrupt):
                return None

    def _setup_target(self, target: Target):
        """Setup target directory and initial recon."""
        print("\n" + "=" * 60)
        print(f"🔧 SETUP: {target.name}")
        print("=" * 60 + "\n")

        self.state.phase = "setup"
        self.state.save()

        # Create target directory
        target_dir = TARGETS_DIR / self._slugify(target.name)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save target info
        (target_dir / "target.json").write_text(json.dumps(asdict(target), indent=2))

        # Clone repo if available
        if target.repo_url:
            print(f"📥 Cloning repository...")
            repo_dir = target_dir / "repo"
            if not repo_dir.exists():
                try:
                    subprocess.run(
                        ["git", "clone", "--depth", "1", target.repo_url, str(repo_dir)],
                        check=True,
                        capture_output=True,
                        timeout=120,
                    )
                    print(f"  ✓ Cloned to {repo_dir}")
                except Exception as e:
                    print(f"  ⚠ Clone failed: {e}")

        # Fetch contest details from platform
        print(f"\n📋 Fetching contest details from {target.platform}...")
        contest_details = self._fetch_contest_details(target)

        # Save contest details
        (target_dir / "contest-details.json").write_text(json.dumps(contest_details, indent=2))

        # Run initial analysis
        print(f"\n📊 Running initial analysis...")

        # Generate hunt handoff document with fetched details
        self._generate_handoff(target, target_dir, contest_details)

        # Query bounty RAG for similar vulnerabilities
        if self.mcp_available.get("bounty-training"):
            print(f"  Querying bounty RAG for patterns...")
            self._query_bounty_rag(target, target_dir)

        print(f"\n✓ Setup complete: {target_dir}")
        self.state.checkpoint("setup_complete")

    def _generate_handoff(self, target: Target, target_dir: Path, contest_details: Dict[str, Any] = None):
        """Generate HUNT-HANDOFF.md for the target."""
        contest_details = contest_details or {}

        tech_list = "\n".join(f"- {t}" for t in target.tech_stack)
        tvl_str = f"${target.tvl_usd:,}" if target.tvl_usd else "Unknown"
        end_date = contest_details.get("end_date") or target.end_date or "Ongoing"
        repo_url = contest_details.get("repo_url") or target.repo_url or "Not provided"

        # Use fetched scope if available, otherwise fall back to cached
        scope_files = contest_details.get("scope_files") or target.scope
        scope_list = "\n".join(f"- {s}" for s in scope_files) if scope_files else "See contest page"

        # Known issues / out of scope
        known_issues = contest_details.get("known_issues", [])
        known_issues_list = "\n".join(f"- {i}" for i in known_issues) if known_issues else "None listed"

        out_of_scope = contest_details.get("out_of_scope", [])
        out_of_scope_list = "\n".join(f"- {i}" for i in out_of_scope) if out_of_scope else "See contest rules"

        # Judging criteria
        judging = contest_details.get("judging_criteria", [])
        judging_list = "\n".join(f"- {j}" for j in judging) if judging else "Standard platform rules"

        # Docs
        docs_url = contest_details.get("docs_url", "")
        docs_section = f"**Documentation:** {docs_url}\n" if docs_url else ""

        handoff = f"""# {target.name} - Hunt Handoff

**Platform:** {target.platform}
**Bounty:** ${target.bounty_usd:,}
**Launch:** {target.launch_date}
**End:** {end_date}
**Contest URL:** {target.url}
**Repository:** {repo_url}
{docs_section}
## Tech Stack
{tech_list}

## TVL
{tvl_str}

## In-Scope Files/Contracts
{scope_list}

## Known Issues (Do Not Report)
{known_issues_list}

## Out of Scope
{out_of_scope_list}

## Judging Criteria
{judging_list}

## Hunt Status
- [ ] Initial recon complete
- [ ] Scope files reviewed
- [ ] High-value targets identified
- [ ] Deep analysis started
- [ ] Findings documented
- [ ] Submissions prepared

## Findings
(none yet)

## Notes
Generated by conclave-hunt at {datetime.now().isoformat()}
Contest details fetched from: {target.url}
"""
        (target_dir / "HUNT-HANDOFF.md").write_text(handoff)
        print(f"  ✓ Generated HUNT-HANDOFF.md")

    def _query_bounty_rag(self, target: Target, target_dir: Path):
        """Query bounty RAG for similar vulnerabilities."""
        # Build query from target info
        query_parts = [target.name] + target.tech_stack + target.scope[:3]
        query = " ".join(query_parts)

        # This would call the bounty-training MCP
        # For now, save the query for manual follow-up
        (target_dir / "rag-query.txt").write_text(query)

    def _hunt_target(self, target: Target):
        """Main hunting loop with token/time boxing."""
        print("\n" + "=" * 60)
        print(f"🎯 HUNTING: {target.name}")
        print("=" * 60)
        print(f"Budget: {self.token_budget.status()}")
        print(f"Time: {self.time_budget.status()}")
        print("=" * 60 + "\n")

        self.state.phase = "hunt"
        self.state.save()

        target_dir = TARGETS_DIR / self._slugify(target.name)

        # Generate hunt plan if not exists
        if not self.state.hunt_plan:
            self.state.hunt_plan = self._generate_hunt_plan(target)
            self.state.save()

        # Execute hunt plan with budget checks
        last_checkpoint = time.time()

        while self.state.current_hunt_index < len(self.state.hunt_plan):
            # Check budgets
            if self.token_budget.exhausted():
                print(f"\n⚠ Token budget exhausted: {self.token_budget.status()}")
                break

            if self.time_budget.exhausted():
                print(f"\n⚠ Time budget exhausted: {self.time_budget.status()}")
                break

            # Checkpoint periodically
            if time.time() - last_checkpoint > DEFAULT_CHECKPOINT_INTERVAL:
                self.state.tokens_used = self.token_budget.used
                self.state.time_elapsed_seconds = self.time_budget.elapsed()
                self.state.checkpoint("periodic")
                last_checkpoint = time.time()
                print(f"  📌 Checkpoint saved")

            # Show real-time status update
            self._print_status_line()

            # Execute current hunt step
            step = self.state.hunt_plan[self.state.current_hunt_index]
            print(f"\n[{self.state.current_hunt_index + 1}/{len(self.state.hunt_plan)}] {step}")

            try:
                findings = self._execute_hunt_step(step, target, target_dir)
                if findings:
                    for f in findings:
                        # Persist immediately so nothing is ever lost
                        filename = self._persist_finding(f, target, target_dir)
                        print(f"  💾 Saved: {filename}")

                    self.state.findings.extend(findings)
                    print(f"  💡 Found {len(findings)} potential issue(s)")

                    # Notify on high-confidence findings
                    if self.args.notify:
                        for f in findings:
                            if f.get("confidence", 0) >= 0.7:
                                self._notify_finding(f, target)
            except KeyboardInterrupt:
                print("\n\n⏸ Hunt paused. Resume with: conclave-hunt --resume")
                self.state.checkpoint("user_paused")
                return
            except Exception as e:
                print(f"  ⚠ Step failed: {e}")

            self.state.current_hunt_index += 1
            self.state.save()

        # Hunt complete
        self._complete_hunt(target, target_dir)

    def _print_status_line(self):
        """Print real-time status update."""
        tokens = self.token_budget.status()
        time_status = self.time_budget.status()
        findings_count = len(self.state.findings)
        step_num = self.state.current_hunt_index + 1
        total_steps = len(self.state.hunt_plan)

        # Use ANSI escape codes for in-place update
        status = f"\r📊 Tokens: {tokens} | Time: {time_status} | Findings: {findings_count} | Step: {step_num}/{total_steps}"
        print(status, end="", flush=True)

    def _generate_hunt_plan(self, target: Target) -> List[str]:
        """Generate a hunt plan based on target characteristics."""
        plan = []

        # Common steps
        plan.append("Analyze scope and identify high-value contracts")
        plan.append("Check for common vulnerability patterns")

        # Tech-specific steps
        if "solana" in [t.lower() for t in target.tech_stack]:
            plan.extend([
                "Check for missing signer validation",
                "Analyze PDA derivation for collisions",
                "Review account validation and ownership checks",
                "Check for arithmetic overflow in token calculations",
            ])
        elif "evm" in [t.lower() for t in target.tech_stack] or "solidity" in [t.lower() for t in target.tech_stack]:
            plan.extend([
                "Run Slither static analysis",
                "Check for reentrancy vulnerabilities",
                "Analyze access control and privilege escalation",
                "Review oracle/price feed manipulation vectors",
                "Check flash loan attack surfaces",
            ])

        # DeFi-specific
        if target.tvl_usd and target.tvl_usd > 0:
            plan.extend([
                "Analyze liquidation logic for edge cases",
                "Check interest rate calculation precision",
                "Review withdrawal/deposit race conditions",
            ])

        plan.append("Document findings and prepare submissions")

        return plan

    def _execute_hunt_step(self, step: str, target: Target, target_dir: Path) -> List[Dict[str, Any]]:
        """Execute a single hunt step and return any findings."""
        findings = []

        # Build the analysis prompt
        prompt = self._build_hunt_prompt(step, target, target_dir)

        if self.args.debug:
            print(f"  [DEBUG] Prompt length: {len(prompt)} chars")
            print(f"  [DEBUG] Target dir: {target_dir}")

        # Try local model first (free, faster for initial triage)
        if self.args.local_first and not getattr(self.args, 'claude_only', False):
            if self.args.debug:
                print(f"  [DEBUG] Trying local model first...")
            findings = self._execute_hunt_step_ollama(step, target, target_dir)
            # Local model handles everything when local_first is set
            return findings

        # Call Claude for analysis
        try:
            if self.args.debug:
                print(f"  [DEBUG] Calling Claude CLI...")

            result = subprocess.run(
                ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=180,  # Increased timeout
            )

            if self.args.debug:
                print(f"  [DEBUG] Claude returned: {result.returncode}, stdout: {len(result.stdout)} chars")

            if result.returncode == 0 and result.stdout:
                # Parse findings from response
                findings = self._parse_findings(result.stdout, step, target)

                # Estimate tokens used (rough: ~4 chars per token)
                input_tokens = len(prompt) // 4
                output_tokens = len(result.stdout) // 4
                self.token_budget.use(input_tokens + output_tokens, "claude-sonnet")

                if self.args.debug:
                    print(f"  [DEBUG] Parsed {len(findings)} findings")
            else:
                # Even on failure, count some token usage
                self.token_budget.use(1000, "claude-sonnet")
                if result.stderr and self.args.debug:
                    print(f"  [DEBUG] Claude error: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            print(f"  ⏱ Step timed out after 180s")
            self.token_budget.use(2000, "claude-sonnet")
        except FileNotFoundError:
            # Claude CLI not found, fall back to ollama
            if self.args.debug:
                print(f"  [DEBUG] Claude not found, using Ollama...")
            findings = self._execute_hunt_step_ollama(step, target, target_dir)
        except Exception as e:
            if self.args.debug:
                print(f"  [DEBUG] Hunt step failed: {e}")
            self.token_budget.use(500, "claude-sonnet")

        return findings

    def _build_hunt_prompt(self, step: str, target: Target, target_dir: Path) -> str:
        """Build a focused prompt for the hunt step."""
        # Find relevant code files (code is in repo/ subdir)
        repo_dir = target_dir / "repo"
        search_dir = repo_dir if repo_dir.exists() else target_dir

        code_files = []
        for ext in ["*.sol", "*.rs", "*.move", "*.vy"]:
            code_files.extend(search_dir.rglob(ext))

        # Limit to first 10 files to avoid huge prompts
        code_files = sorted(code_files)[:10]

        # Read handoff for context
        handoff = ""
        handoff_file = target_dir / "HUNT-HANDOFF.md"
        if handoff_file.exists():
            handoff = handoff_file.read_text()[:2000]

        prompt = f"""You are a security researcher hunting for bugs in {target.name}.

CONTEXT:
{handoff}

CURRENT TASK: {step}

INSTRUCTIONS:
1. Analyze the code for this specific vulnerability type
2. If you find potential issues, output them in this EXACT format:

FINDING:
Title: <short descriptive title>
Severity: <critical|high|medium|low>
Confidence: <0.0-1.0>
File: <path to file>
Line: <line number if known>
Description: <detailed description of the vulnerability>
Impact: <what an attacker could do>
Proof: <code snippet or reasoning showing the issue>
END_FINDING

3. You can output multiple findings
4. If no issues found for this step, output: NO_FINDINGS_THIS_STEP
5. Be thorough but avoid false positives - only report real issues

Analyze the code now."""

        return prompt

    def _parse_findings(self, response: str, step: str, target: Target) -> List[Dict[str, Any]]:
        """Parse findings from Claude's response."""
        findings = []

        if "NO_FINDINGS_THIS_STEP" in response:
            return findings

        # Parse FINDING blocks
        import re
        finding_pattern = r'FINDING:\s*\n(.*?)END_FINDING'
        matches = re.findall(finding_pattern, response, re.DOTALL)

        for match in matches:
            finding = {
                "step": step,
                "target": target.name,
                "raw": match.strip(),
            }

            # Parse structured fields
            for field in ["Title", "Severity", "Confidence", "File", "Line", "Description", "Impact", "Proof"]:
                pattern = rf'{field}:\s*(.+?)(?=\n[A-Z][a-z]+:|$)'
                field_match = re.search(pattern, match, re.DOTALL)
                if field_match:
                    value = field_match.group(1).strip()
                    finding[field.lower()] = value

            # Convert confidence to float
            if "confidence" in finding:
                try:
                    finding["confidence"] = float(finding["confidence"])
                except:
                    finding["confidence"] = 0.5

            # Only add if we got essential fields
            if finding.get("title") and finding.get("description"):
                findings.append(finding)

        return findings

    def _execute_hunt_step_ollama(self, step: str, target: Target, target_dir: Path) -> List[Dict[str, Any]]:
        """Use local Ollama for analysis (free, no API tokens)."""
        findings = []

        prompt = self._build_hunt_prompt(step, target, target_dir)

        # Prefer bounty-specific model, fall back to general coder
        models_to_try = ["bounty-learned", "qwen3-coder:30b", "qwen2.5-coder:7b"]

        for model in models_to_try:
            try:
                if self.args.debug:
                    print(f"  [DEBUG] Trying Ollama model: {model}")

                result = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes for large models
                )

                if result.returncode == 0 and result.stdout:
                    if self.args.debug:
                        print(f"  [DEBUG] {model} response: {len(result.stdout)} chars")

                    findings = self._parse_findings(result.stdout, step, target)
                    # Local models are free
                    self.token_budget.use(0, "local")

                    if self.args.debug:
                        print(f"  [DEBUG] Parsed {len(findings)} findings from {model}")
                    break  # Success, don't try other models

            except subprocess.TimeoutExpired:
                if self.args.debug:
                    print(f"  [DEBUG] {model} timed out after 300s")
                continue
            except FileNotFoundError:
                if self.args.debug:
                    print(f"  [DEBUG] Ollama not found")
                break
            except Exception as e:
                if self.args.debug:
                    print(f"  [DEBUG] {model} failed: {e}")
                continue

        return findings

    def _persist_finding(self, finding: Dict[str, Any], target: Target, target_dir: Path):
        """Immediately persist a finding to disk so it's never lost."""
        findings_dir = target_dir / "findings"
        findings_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        severity = finding.get("severity", "unknown").lower()
        slug = self._slugify(finding.get("title", "finding")[:30])
        filename = f"{timestamp}-{severity}-{slug}.json"

        # Save full finding
        finding["persisted_at"] = datetime.now().isoformat()
        finding["target"] = target.name
        (findings_dir / filename).write_text(json.dumps(finding, indent=2))

        # Also append to findings log (append-only, never lost)
        log_file = target_dir / "findings.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(finding) + "\n")

        # Update master findings index
        self._update_findings_index(finding, target)

        return filename

    def _update_findings_index(self, finding: Dict[str, Any], target: Target):
        """Update the master findings index for quick access."""
        index_file = BOUNTY_RECON_DIR / "all-findings.jsonl"
        index_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "target": target.name,
            "platform": target.platform,
            "bounty_usd": target.bounty_usd,
            "finding": finding,
        }

        with open(index_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_existing_findings(self, target_dir: Path) -> List[Dict[str, Any]]:
        """Load existing findings from disk (in case of resume)."""
        findings = []
        log_file = target_dir / "findings.jsonl"
        if log_file.exists():
            for line in log_file.read_text().strip().split("\n"):
                if line:
                    try:
                        findings.append(json.loads(line))
                    except:
                        pass
        return findings

    def _notify_finding(self, finding: Dict[str, Any], target: Target):
        """Send notification for high-confidence finding."""
        title = finding.get("title", "Potential Finding")
        severity = finding.get("severity", "unknown")
        confidence = finding.get("confidence", 0)

        # Always print to console
        print(f"  🔔 FINDING: [{severity.upper()}] {title} (confidence: {confidence:.0%})")

        # Send phone notification via ntfy if enabled
        if self.args.notify:
            try:
                import urllib.request
                import urllib.parse

                # Use ntfy.sh for push notifications
                topic = "conclave-hunt"  # User should subscribe to this topic
                message = f"[{severity.upper()}] {title}\nTarget: {target.name}\nConfidence: {confidence:.0%}"

                data = message.encode('utf-8')
                req = urllib.request.Request(
                    f"https://ntfy.sh/{topic}",
                    data=data,
                    headers={
                        "Title": f"Bounty Finding: {target.name}",
                        "Priority": "high" if severity in ("critical", "high") else "default",
                        "Tags": f"bug,{severity}",
                    }
                )
                urllib.request.urlopen(req, timeout=5)
                print(f"  📱 Phone notification sent")
            except Exception as e:
                logger.debug(f"Notification failed: {e}")

    def _complete_hunt(self, target: Target, target_dir: Path):
        """Complete the hunt and generate summary."""
        print("\n" + "=" * 60)
        print("✅ HUNT COMPLETE")
        print("=" * 60)
        print(f"Target: {target.name}")
        print(f"Tokens used: {self.token_budget.status()}")
        print(f"Time: {self.time_budget.status()}")
        print(f"Findings: {len(self.state.findings)}")

        if self.state.findings:
            print("\nFindings summary:")
            for i, f in enumerate(self.state.findings, 1):
                conf = f.get("confidence", 0)
                sev = f.get("severity", "unknown")
                print(f"  {i}. [{sev}] {f.get('title', 'Untitled')} (confidence: {conf:.0%})")

        # Save final state
        self.state.phase = "complete"
        self.state.tokens_used = self.token_budget.used
        self.state.time_elapsed_seconds = self.time_budget.elapsed()
        self.state.checkpoint("hunt_complete")

        # Generate summary report
        self._generate_summary(target, target_dir)

        # Clear notification about saved files
        print(f"\n📁 ALL FILES SAVED:")
        print(f"   Handoff:     {target_dir}/HUNT-HANDOFF.md")
        print(f"   Summary:     {target_dir}/HUNT-SUMMARY.md")
        print(f"   Findings:    {target_dir}/findings.jsonl ({len(self.state.findings)} findings)")
        print(f"   Master log:  ~/bug-bounty-recon/all-findings.jsonl")
        if self.state.findings:
            print(f"\n⚠️  REVIEW YOUR {len(self.state.findings)} FINDINGS:")
            for i, f in enumerate(self.state.findings[:5], 1):
                print(f"   {i}. [{f.get('severity', '?')}] {f.get('title', 'Untitled')}")
            if len(self.state.findings) > 5:
                print(f"   ... and {len(self.state.findings) - 5} more")
            print(f"\n   Run: conclave-hunt --findings-target '{target.name}'")
        print("=" * 60 + "\n")

    def _generate_summary(self, target: Target, target_dir: Path):
        """Generate hunt summary report."""
        summary = f"""# Hunt Summary: {target.name}

**Completed:** {datetime.now().isoformat()}
**Tokens Used:** {self.token_budget.used:,} (~${self.token_budget.estimate_cost():.2f})
**Time Elapsed:** {self.time_budget.elapsed() // 60}m {self.time_budget.elapsed() % 60}s

## Findings ({len(self.state.findings)})

"""
        for i, f in enumerate(self.state.findings, 1):
            summary += f"""### {i}. {f.get('title', 'Untitled')}
- **Severity:** {f.get('severity', 'Unknown')}
- **Confidence:** {f.get('confidence', 0):.0%}
- **Location:** {f.get('location', 'Unknown')}

{f.get('description', 'No description')}

"""

        summary += f"""## Hunt Plan Executed

{chr(10).join(f'- [x] {step}' for step in self.state.hunt_plan[:self.state.current_hunt_index])}
{chr(10).join(f'- [ ] {step}' for step in self.state.hunt_plan[self.state.current_hunt_index:])}

## Next Steps

1. Review findings for false positives
2. Develop PoC for high-confidence issues
3. Prepare submission reports
"""

        (target_dir / "HUNT-SUMMARY.md").write_text(summary)

    def _resume_hunt(self):
        """Resume a paused hunt."""
        if not self.state.target:
            logger.error("No target in saved state.")
            return

        target = Target(**self.state.target)

        print("\n" + "=" * 60)
        print(f"▶ RESUMING HUNT: {target.name}")
        print("=" * 60)
        print(f"Phase: {self.state.phase}")
        print(f"Progress: {self.state.current_hunt_index}/{len(self.state.hunt_plan)} steps")
        print(f"Previous tokens: {self.state.tokens_used:,}")
        print("=" * 60 + "\n")

        # Restore token count
        self.token_budget.used = self.state.tokens_used

        # Continue from where we left off
        if self.state.phase == "setup":
            self._setup_target(target)
            self._hunt_target(target)
        elif self.state.phase == "hunt":
            self._hunt_target(target)
        else:
            logger.info(f"Hunt already {self.state.phase}. Start a new one with: conclave-hunt")

    def _slugify(self, text: str) -> str:
        """Convert text to slug for directory names."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9]+', '-', text)
        return text.strip('-')


def main():
    parser = argparse.ArgumentParser(
        description="Conclave Hunt - Autonomous bug bounty target discovery and hunting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conclave-hunt                      # Start fresh hunt with defaults
  conclave-hunt --max-tokens 50000   # Limit to 50k tokens (~$2.50)
  conclave-hunt --tech solana        # Only Solana targets
  conclave-hunt --min-bounty 50000   # Only $50k+ bounties
  conclave-hunt --resume             # Resume previous session
  conclave-hunt --status             # Check current status
        """
    )

    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Maximum tokens (default: {DEFAULT_MAX_TOKENS:,})")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME_MINUTES,
                        help=f"Maximum time in minutes (default: {DEFAULT_MAX_TIME_MINUTES})")
    parser.add_argument("--platform", type=str,
                        help="Filter platforms (comma-separated): immunefi,code4rena,sherlock")
    parser.add_argument("--tech", type=str,
                        help="Filter tech stack (comma-separated): solana,evm,rust,move")
    parser.add_argument("--min-bounty", type=int, default=DEFAULT_MIN_BOUNTY,
                        help=f"Minimum bounty USD (default: {DEFAULT_MIN_BOUNTY:,})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume previous hunt session")
    parser.add_argument("--status", action="store_true",
                        help="Show current hunt status")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover targets but don't start hunting")
    parser.add_argument("--notify", action="store_true",
                        help="Send phone notifications on findings")
    parser.add_argument("--findings", action="store_true",
                        help="List all findings across all hunts")
    parser.add_argument("--findings-target", type=str,
                        help="List findings for specific target")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select top target (no interactive prompt)")
    parser.add_argument("--target", type=str,
                        help="Directly specify target name to hunt")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--local-first", action="store_true", default=True,
                        help="Try local Ollama model first (free, faster) [default: True]")
    parser.add_argument("--claude-only", action="store_true",
                        help="Only use Claude (skip local models)")

    args = parser.parse_args()

    try:
        cli = HuntCLI(args)
        cli.run()
    except KeyboardInterrupt:
        print("\n\nHunt interrupted. Resume with: conclave-hunt --resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
