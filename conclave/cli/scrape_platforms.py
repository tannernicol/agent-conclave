#!/usr/bin/env python3
"""
Scrape bug bounty platforms for fresh targets.
Run this periodically (e.g., cron every 6 hours) to keep targets fresh.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BOUNTY_RECON_DIR = Path.home() / "bug-bounty-recon"
TARGETS_DIR = BOUNTY_RECON_DIR / "platform-targets"


def fetch_json(url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Fetch JSON from URL."""
    headers = headers or {}
    headers.setdefault("User-Agent", "conclave-hunt/1.0")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return {}


def scrape_sherlock() -> List[Dict[str, Any]]:
    """Scrape active contests from Sherlock."""
    targets = []

    try:
        # Sherlock API - correct endpoint
        data = fetch_json("https://audits.sherlock.xyz/api/contests")

        # Response has 'items' array
        contests = data.get("items", []) if isinstance(data, dict) else data

        for contest in contests:
            # Filter for active/upcoming contests (API uses uppercase)
            status = contest.get("status", "").upper()
            if status not in ("RUNNING", "UPCOMING", "SHERLOCK_JUDGING"):
                continue

            # Timestamps are unix epoch
            starts_at = contest.get("starts_at", 0)
            ends_at = contest.get("ends_at", 0)

            # Convert to ISO format
            start_date = datetime.fromtimestamp(starts_at).isoformat() if starts_at else ""
            end_date = datetime.fromtimestamp(ends_at).isoformat() if ends_at else ""

            # Calculate days old
            days_old = 999
            if starts_at:
                days_old = (datetime.now() - datetime.fromtimestamp(starts_at)).days

            # Skip old contests (unless still running)
            if days_old > 30 and status != "RUNNING":
                continue

            prize = contest.get("prize_pool") or contest.get("rewards", 0)
            if isinstance(prize, str):
                prize = int(re.sub(r'[^\d]', '', prize) or 0)

            contest_id = contest.get("id", "")

            target = {
                "name": contest.get("title", "Unknown"),
                "platform": "sherlock",
                "bounty_usd": prize,
                "tech_stack": _detect_tech_stack(contest),
                "launch_date": start_date,
                "end_date": end_date,
                "url": f"https://audits.sherlock.xyz/contests/{contest_id}",
                "repo_url": None,  # Need to fetch from contest page
                "scope": [],  # Need to fetch from contest page
                "tvl_usd": 0,
                "competition_score": _estimate_competition(contest),
            }
            targets.append(target)
            logger.info(f"  Sherlock: {target['name']} (${prize:,})")

    except Exception as e:
        logger.error(f"Sherlock scrape failed: {e}")

    return targets


def scrape_code4rena() -> List[Dict[str, Any]]:
    """Scrape active contests from Code4rena."""
    targets = []

    try:
        # Code4rena API
        data = fetch_json("https://code4rena.com/api/v1/contests?status=active,upcoming")

        if not data:
            # Try GraphQL endpoint
            data = fetch_json("https://code4rena.com/api/contests")

        contests = data if isinstance(data, list) else data.get("contests", data.get("data", []))

        for contest in contests:
            start_date = contest.get("startTime") or contest.get("start_time", "")
            end_date = contest.get("endTime") or contest.get("end_time", "")

            days_old = 999
            if start_date:
                try:
                    start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    days_old = (datetime.now().astimezone() - start).days
                except:
                    pass

            if days_old > 14:
                continue

            prize = contest.get("prize") or contest.get("totalPrize", 0)
            if isinstance(prize, str):
                prize = int(re.sub(r'[^\d]', '', prize) or 0)

            slug = contest.get("slug") or contest.get("id", "")

            target = {
                "name": contest.get("title") or contest.get("name", slug),
                "platform": "code4rena",
                "bounty_usd": prize,
                "tech_stack": _detect_tech_stack(contest),
                "launch_date": start_date,
                "end_date": end_date,
                "url": f"https://code4rena.com/contests/{slug}",
                "repo_url": contest.get("repo") or contest.get("repoUrl"),
                "scope": contest.get("scope", []),
                "tvl_usd": _parse_tvl(contest.get("tvl")),
                "competition_score": _estimate_competition(contest),
            }
            targets.append(target)
            logger.info(f"  Code4rena: {target['name']} (${prize:,})")

    except Exception as e:
        logger.error(f"Code4rena scrape failed: {e}")

    return targets


def scrape_immunefi() -> List[Dict[str, Any]]:
    """Scrape bounties from Immunefi."""
    targets = []

    try:
        # Immunefi bounties API
        data = fetch_json("https://immunefi.com/api/bounties")

        bounties = data if isinstance(data, list) else data.get("bounties", [])

        # Filter for recently updated/new bounties
        cutoff = datetime.now() - timedelta(days=30)

        for bounty in bounties:
            # Check if recently updated
            updated = bounty.get("updatedAt") or bounty.get("launchDate", "")
            if updated:
                try:
                    updated_dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                    if updated_dt.replace(tzinfo=None) < cutoff:
                        continue
                except:
                    pass

            max_bounty = bounty.get("maxBounty") or bounty.get("maxReward", 0)
            if isinstance(max_bounty, str):
                max_bounty = int(re.sub(r'[^\d]', '', max_bounty) or 0)

            # Skip low bounties
            if max_bounty < 10000:
                continue

            slug = bounty.get("slug") or bounty.get("id", "")

            target = {
                "name": bounty.get("name") or bounty.get("project", slug),
                "platform": "immunefi",
                "bounty_usd": max_bounty,
                "tech_stack": _detect_tech_stack(bounty),
                "launch_date": bounty.get("launchDate") or datetime.now().isoformat(),
                "end_date": None,  # Immunefi bounties are ongoing
                "url": f"https://immunefi.com/bounty/{slug}",
                "repo_url": bounty.get("repoUrl"),
                "scope": [a.get("target", "") for a in bounty.get("assets", [])],
                "tvl_usd": _parse_tvl(bounty.get("tvl")),
                "competition_score": 0.6,  # Immunefi tends to be competitive
            }
            targets.append(target)
            logger.info(f"  Immunefi: {target['name']} (${max_bounty:,})")

    except Exception as e:
        logger.error(f"Immunefi scrape failed: {e}")

    return targets


def _detect_tech_stack(contest: Dict[str, Any]) -> List[str]:
    """Detect tech stack from contest data."""
    stack = []

    # Check explicit fields
    if contest.get("language"):
        stack.append(contest["language"].lower())
    if contest.get("chain"):
        stack.append(contest["chain"].lower())
    if contest.get("framework"):
        stack.append(contest["framework"].lower())

    # Detect from description/title
    text = f"{contest.get('title', '')} {contest.get('description', '')} {contest.get('name', '')}".lower()

    if "solana" in text or "anchor" in text:
        if "solana" not in stack:
            stack.append("solana")
        if "anchor" in text and "anchor" not in stack:
            stack.append("anchor")
    if "evm" in text or "solidity" in text or "ethereum" in text:
        if "evm" not in stack:
            stack.append("evm")
        if "solidity" not in stack:
            stack.append("solidity")
    if "rust" in text:
        if "rust" not in stack:
            stack.append("rust")
    if "move" in text or "aptos" in text or "sui" in text:
        if "move" not in stack:
            stack.append("move")
    if "vyper" in text:
        if "vyper" not in stack:
            stack.append("vyper")

    # Default to EVM/Solidity if nothing detected
    if not stack:
        stack = ["evm", "solidity"]

    return stack


def _parse_tvl(tvl: Any) -> int:
    """Parse TVL value."""
    if not tvl:
        return 0
    if isinstance(tvl, int):
        return tvl
    if isinstance(tvl, str):
        # Handle "$1.2B", "500M", etc.
        tvl = tvl.upper().replace("$", "").replace(",", "").strip()
        multiplier = 1
        if tvl.endswith("B"):
            multiplier = 1_000_000_000
            tvl = tvl[:-1]
        elif tvl.endswith("M"):
            multiplier = 1_000_000
            tvl = tvl[:-1]
        elif tvl.endswith("K"):
            multiplier = 1_000
            tvl = tvl[:-1]
        try:
            return int(float(tvl) * multiplier)
        except:
            return 0
    return 0


def _estimate_competition(contest: Dict[str, Any]) -> float:
    """Estimate competition level (0=low, 1=high)."""
    # Factors: prize size (higher = more competition), platform popularity
    prize = contest.get("prize") or contest.get("prizePool") or contest.get("maxBounty", 0)
    if isinstance(prize, str):
        prize = int(re.sub(r'[^\d]', '', prize) or 0)

    # Higher prizes attract more hunters
    if prize > 200000:
        return 0.8
    elif prize > 100000:
        return 0.6
    elif prize > 50000:
        return 0.4
    else:
        return 0.3


def save_targets(platform: str, targets: List[Dict[str, Any]]):
    """Save targets to JSON file."""
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "updated": datetime.now().isoformat(),
        "platform": platform,
        "count": len(targets),
        "targets": targets,
    }

    path = TARGETS_DIR / f"{platform}.json"
    path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(targets)} targets to {path}")


def main():
    """Scrape all platforms and save targets."""
    logger.info("=" * 50)
    logger.info("Scraping bug bounty platforms...")
    logger.info("=" * 50)

    # Scrape each platform
    logger.info("\nSherlock:")
    sherlock = scrape_sherlock()
    save_targets("sherlock", sherlock)

    logger.info("\nCode4rena:")
    code4rena = scrape_code4rena()
    save_targets("code4rena", code4rena)

    logger.info("\nImmunefi:")
    immunefi = scrape_immunefi()
    save_targets("immunefi", immunefi)

    # Summary
    total = len(sherlock) + len(code4rena) + len(immunefi)
    logger.info("\n" + "=" * 50)
    logger.info(f"Total: {total} active targets")
    logger.info(f"  Sherlock: {len(sherlock)}")
    logger.info(f"  Code4rena: {len(code4rena)}")
    logger.info(f"  Immunefi: {len(immunefi)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
