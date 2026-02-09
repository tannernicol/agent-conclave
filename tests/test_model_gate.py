"""Tests for the minimum 2 intelligent models gate."""
import unittest

from conclave.pipeline import ConclavePipeline, InsufficientModelsError


class TestModelGate(unittest.TestCase):
    def _make_pipeline_stub(self):
        """Create a minimal pipeline stub for testing _count_intelligent_providers."""
        # We only need _run_models_used and _count_intelligent_providers
        class Stub:
            _run_models_used: set = set()
            _count_intelligent_providers = ConclavePipeline._count_intelligent_providers
        return Stub()

    def test_two_intelligent_providers_pass(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"cli:claude", "cli:codex"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, {"claude", "codex"})
        self.assertGreaterEqual(len(result), 2)

    def test_one_intelligent_provider_fails(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"cli:claude"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, {"claude"})
        self.assertLess(len(result), 2)

    def test_ollama_only_fails(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"ollama:qwen2.5-coder:7b", "ollama:qwen2.5:32b"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, set())
        self.assertLess(len(result), 2)

    def test_gemini_api_counts(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"cli:claude", "gemini-api:2.5-flash"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, {"claude", "gemini"})
        self.assertGreaterEqual(len(result), 2)

    def test_cli_gemini_counts(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"cli:codex", "cli:gemini"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, {"codex", "gemini"})

    def test_mixed_ollama_and_intelligent(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"ollama:qwen2.5-coder:7b", "cli:claude", "cli:codex"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, {"claude", "codex"})

    def test_empty_models_used(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = set()
        result = stub._count_intelligent_providers()
        self.assertEqual(result, set())

    def test_insufficient_models_error_exists(self):
        """Verify the exception class exists and is an Exception."""
        self.assertTrue(issubclass(InsufficientModelsError, Exception))

    def test_three_providers(self):
        stub = self._make_pipeline_stub()
        stub._run_models_used = {"cli:claude", "cli:codex", "gemini-api:2.5-flash"}
        result = stub._count_intelligent_providers()
        self.assertEqual(result, {"claude", "codex", "gemini"})
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
