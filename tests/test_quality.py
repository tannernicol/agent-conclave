import unittest

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline


class QualityTests(unittest.TestCase):
    def test_domain_paths_mark_on_domain(self):
        config = get_config()
        pipeline = ConclavePipeline(config)
        rag = [
            {
                "path": "/data/security-docs/vuln-report.md",
                "collection": "security-docs",
                "snippet": "x" * 80,
                "score": 0.9,
            }
        ]
        file_index = [
            {
                "path": "/data/security-docs/notes.md",
                "collection": "security-docs",
                "snippet": "y" * 80,
            }
        ]
        evidence, stats = pipeline._select_evidence(
            rag,
            file_index,
            preferred_collections=["security-docs"],
            required_collections=["security-docs"],
            domain="security",
            domain_paths=config.quality.get("domain_paths", {}),
        )
        self.assertGreaterEqual(stats.get("domain_known", 0), 1)
        self.assertEqual(stats.get("off_domain", 0), 0)
        self.assertTrue(any(item.get("on_domain") for item in evidence if item.get("path")))

    def test_required_collection_hits(self):
        config = get_config()
        pipeline = ConclavePipeline(config)
        rag = [
            {
                "path": "/data/security-docs/vuln-report.md",
                "collection": "security-docs",
                "snippet": "x" * 120,
                "score": 0.9,
            }
        ]
        evidence, stats = pipeline._select_evidence(
            rag,
            [],
            required_collections=["security-docs"],
            preferred_collections=["security-docs"],
            domain="security",
            domain_paths=config.quality.get("domain_paths", {}),
        )
        self.assertGreaterEqual(stats.get("required_collection_hits", 0), 1)

    def test_required_collections_only_enforced_when_present(self):
        config = get_config()
        pipeline = ConclavePipeline(config)
        rag = [
            {
                "path": "/data/notes/example.md",
                "collection": "notes",
                "snippet": "x" * 120,
                "score": 0.9,
            }
        ]
        evidence, stats = pipeline._select_evidence(
            rag,
            [],
            required_collections=[],
            preferred_collections=["notes"],
            domain="general",
            domain_paths=config.quality.get("domain_paths", {}),
        )
        quality = pipeline._evaluate_quality({"stats": stats})
        self.assertNotIn("missing_required_evidence", quality.get("issues", []))


if __name__ == "__main__":
    unittest.main()
