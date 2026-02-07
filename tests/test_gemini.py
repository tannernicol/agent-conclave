"""Tests for conclave.models.gemini module."""
import json
import unittest
from unittest.mock import patch, MagicMock

from conclave.models.gemini import GeminiClient, GeminiResult


class TestGeminiClient(unittest.TestCase):
    def test_no_api_key_returns_error(self):
        client = GeminiClient(api_key="")
        result = client.generate("Hello")
        self.assertFalse(result.ok)
        self.assertIn("GEMINI_API_KEY", result.error)

    def test_available_property(self):
        client = GeminiClient(api_key="test-key")
        self.assertTrue(client.available)

        client_no_key = GeminiClient(api_key="")
        self.assertFalse(client_no_key.available)

    @patch("conclave.models.gemini.httpx.Client")
    def test_successful_generation(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            }
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = GeminiClient(api_key="test-key")
        result = client.generate("Say hello")

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "Hello there!")
        self.assertEqual(result.usage["prompt_tokens"], 5)
        self.assertEqual(result.usage["completion_tokens"], 3)

    @patch("conclave.models.gemini.httpx.Client")
    def test_http_error(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = GeminiClient(api_key="bad-key")
        result = client.generate("test")

        self.assertFalse(result.ok)
        self.assertIn("401", result.error)

    @patch("conclave.models.gemini.httpx.Client")
    def test_no_candidates(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = GeminiClient(api_key="test-key")
        result = client.generate("test")

        self.assertFalse(result.ok)
        self.assertIn("No candidates", result.error)

    def test_model_map_contains_expected_models(self):
        self.assertIn("2.5-flash", GeminiClient.MODEL_MAP)
        self.assertIn("2.5-pro", GeminiClient.MODEL_MAP)
        self.assertIn("2.0-flash", GeminiClient.MODEL_MAP)

    def test_gemini_result_defaults(self):
        result = GeminiResult()
        self.assertEqual(result.text, "")
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
