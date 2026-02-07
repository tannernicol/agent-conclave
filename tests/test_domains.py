"""Tests for conclave.domains module."""
import unittest

from conclave.domains import (
    DomainInstructions,
    DOMAIN_INSTRUCTIONS,
    get_domain_instructions,
)


class TestDomainInstructions(unittest.TestCase):
    def test_security_domain_returns_hints(self):
        hints = get_domain_instructions("security")
        self.assertIn("security", hints.deliberation_hint.lower())
        self.assertIn("findings", hints.deliberation_hint.lower())
        self.assertTrue(hints.summarizer_hint)

    def test_code_review_domain_returns_hints(self):
        hints = get_domain_instructions("code_review")
        self.assertIn("code", hints.deliberation_hint.lower())

    def test_unknown_domain_returns_default(self):
        hints = get_domain_instructions("unknown_domain_xyz")
        self.assertTrue(hints.deliberation_hint)
        self.assertIn("best-effort", hints.deliberation_hint.lower())

    def test_none_domain_returns_default(self):
        hints = get_domain_instructions(None)
        self.assertTrue(hints.deliberation_hint)

    def test_default_has_empty_summarizer_hint(self):
        hints = get_domain_instructions("general")
        # Default summarizer hint is empty string
        self.assertEqual(hints.summarizer_hint, "")

    def test_career_domain_returns_hints(self):
        hints = get_domain_instructions("career")
        self.assertIn("actionable", hints.deliberation_hint.lower())

    def test_known_domains_are_in_registry(self):
        self.assertIn("security", DOMAIN_INSTRUCTIONS)
        self.assertIn("code_review", DOMAIN_INSTRUCTIONS)
        self.assertIn("career", DOMAIN_INSTRUCTIONS)
        self.assertIn("research", DOMAIN_INSTRUCTIONS)
        self.assertIn("creative", DOMAIN_INSTRUCTIONS)

    def test_creative_domain_returns_hints(self):
        hints = get_domain_instructions("creative")
        self.assertIn("diverse", hints.deliberation_hint.lower())

    def test_research_domain_returns_hints(self):
        hints = get_domain_instructions("research")
        self.assertIn("compare", hints.deliberation_hint.lower())


if __name__ == "__main__":
    unittest.main()
