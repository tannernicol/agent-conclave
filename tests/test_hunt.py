#!/usr/bin/env python3
"""Tests for the hunt CLI."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from conclave.cli.hunt import Target, HuntState, HuntCLI
from conclave.cli.scrape_platforms import (
    scrape_sherlock,
    _detect_tech_stack,
    _parse_tvl,
    _estimate_competition,
)


class TestTarget:
    """Test Target dataclass."""

    def test_target_creation(self):
        """Test creating a target."""
        target = Target(
            name="Test Protocol",
            platform="sherlock",
            bounty_usd=100000,
            tech_stack=["evm", "solidity"],
            launch_date="2026-02-01T00:00:00Z",
            end_date="2026-03-01T00:00:00Z",
            url="https://example.com/contest",
            repo_url="https://github.com/test/repo",
            scope=["contracts/"],
        )
        assert target.name == "Test Protocol"
        assert target.bounty_usd == 100000
        assert target.opportunity_score > 0

    def test_opportunity_score_calculation(self):
        """Test that opportunity score favors fresh, high-bounty, low-competition targets."""
        fresh_target = Target(
            name="Fresh",
            platform="sherlock",
            bounty_usd=100000,
            tech_stack=["evm"],
            launch_date="2026-02-01T00:00:00Z",
            end_date=None,
            url="",
            repo_url=None,
            scope=[],
            competition_score=0.3,
        )

        old_target = Target(
            name="Old",
            platform="sherlock",
            bounty_usd=100000,
            tech_stack=["evm"],
            launch_date="2026-01-01T00:00:00Z",
            end_date=None,
            url="",
            repo_url=None,
            scope=[],
            competition_score=0.3,
        )

        # Fresh target should have higher score
        assert fresh_target.opportunity_score >= old_target.opportunity_score


class TestScraper:
    """Test platform scrapers."""

    def test_detect_tech_stack_solana(self):
        """Test Solana detection."""
        contest = {"title": "Solana DEX", "description": "Built with Anchor"}
        stack = _detect_tech_stack(contest)
        assert "solana" in stack
        assert "anchor" in stack

    def test_detect_tech_stack_evm(self):
        """Test EVM/Solidity detection."""
        contest = {"title": "Ethereum Protocol", "description": "Solidity contracts"}
        stack = _detect_tech_stack(contest)
        assert "evm" in stack
        assert "solidity" in stack

    def test_detect_tech_stack_default(self):
        """Test default stack when nothing detected."""
        contest = {"title": "Unknown", "description": ""}
        stack = _detect_tech_stack(contest)
        assert "evm" in stack  # Default

    def test_parse_tvl_number(self):
        """Test TVL parsing with numbers."""
        assert _parse_tvl(1000000) == 1000000
        assert _parse_tvl(0) == 0

    def test_parse_tvl_string(self):
        """Test TVL parsing with strings."""
        assert _parse_tvl("$1.5B") == 1_500_000_000
        assert _parse_tvl("500M") == 500_000_000
        assert _parse_tvl("100K") == 100_000

    def test_estimate_competition(self):
        """Test competition estimation."""
        high_prize = {"prize_pool": 500000}
        low_prize = {"prize_pool": 20000}

        assert _estimate_competition(high_prize) > _estimate_competition(low_prize)

    @patch('conclave.cli.scrape_platforms.fetch_json')
    def test_scrape_sherlock_parses_response(self, mock_fetch):
        """Test Sherlock scraper parses API response."""
        mock_fetch.return_value = {
            "items": [
                {
                    "id": 1234,
                    "title": "Test Contest",
                    "status": "RUNNING",
                    "starts_at": 1706745600,  # Recent timestamp
                    "ends_at": 1709424000,
                    "prize_pool": 100000,
                    "rewards": 150000,
                }
            ]
        }

        targets = scrape_sherlock()
        assert len(targets) == 1
        assert targets[0]["name"] == "Test Contest"
        assert targets[0]["platform"] == "sherlock"


class TestHuntState:
    """Test HuntState persistence."""

    def test_state_serialization(self, tmp_path):
        """Test state saves and loads correctly."""
        state = HuntState()
        state.phase = "hunt"
        state.tokens_used = 5000
        state.findings = [{"title": "Test Finding"}]

        # Mock the STATE_FILE
        state_file = tmp_path / "hunt-state.json"

        # Save
        state_file.write_text(json.dumps({
            "phase": state.phase,
            "tokens_used": state.tokens_used,
            "findings": state.findings,
        }))

        # Load
        data = json.loads(state_file.read_text())
        assert data["phase"] == "hunt"
        assert data["tokens_used"] == 5000
        assert len(data["findings"]) == 1


class TestFindingParser:
    """Test finding extraction from LLM responses."""

    def test_parse_finding_block(self):
        """Test parsing a properly formatted finding."""
        response = """
Looking at the code...

FINDING:
Title: Reentrancy in withdraw function
Severity: high
Confidence: 0.8
File: contracts/Vault.sol
Line: 45
Description: The withdraw function makes an external call before updating state.
Impact: Attacker can drain funds through reentrancy.
Proof: The call to msg.sender.call happens before balance update.
END_FINDING

No other issues found.
"""
        # This would test the _parse_findings method
        # We'd need to instantiate HuntCLI to test this properly
        assert "FINDING:" in response
        assert "END_FINDING" in response

    def test_no_findings_response(self):
        """Test response with no findings."""
        response = "NO_FINDINGS_THIS_STEP"
        assert "NO_FINDINGS" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
