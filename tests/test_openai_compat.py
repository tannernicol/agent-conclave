"""Tests for OpenAI-compatible API client."""
from unittest.mock import patch, MagicMock

from conclave.models.openai_compat import OpenAICompatClient


def test_missing_api_key():
    """Returns error when no API key is set."""
    client = OpenAICompatClient(api_key="")
    with patch.dict("os.environ", {}, clear=True):
        result = client.generate("hello", model="gpt-4o")
    assert not result.ok
    assert "API key" in (result.error or "")


def test_successful_response():
    """Parses a standard chat completions response."""
    client = OpenAICompatClient(api_key="sk-test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello world"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="gpt-4o")

    assert result.ok
    assert result.text == "Hello world"
    assert result.usage["total_tokens"] == 7
    assert result.duration_ms >= 0


def test_http_error():
    """Handles non-200 responses gracefully."""
    client = OpenAICompatClient(api_key="sk-test")

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="gpt-4o")

    assert not result.ok
    assert "429" in (result.error or "")


def test_timeout():
    """Handles timeout gracefully."""
    import httpx

    client = OpenAICompatClient(api_key="sk-test")

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = httpx.TimeoutException("timed out")

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="gpt-4o", timeout=5)

    assert not result.ok
    assert "timeout" in (result.error or "").lower()


def test_custom_base_url():
    """Supports custom base URLs for vLLM/LM Studio."""
    client = OpenAICompatClient(api_key="sk-test", base_url="http://localhost:8000/v1")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "local model response"}}],
        "usage": {},
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="llama-3.1")

    assert result.ok
    # Verify the URL used localhost
    call_args = mock_client.post.call_args
    assert "localhost:8000" in call_args[0][0]


def test_per_call_override():
    """Supports per-call API key and base URL overrides."""
    client = OpenAICompatClient(api_key="sk-default")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {},
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate(
            "hello", model="gpt-4o",
            api_key="sk-override",
            base_url="https://custom.api.com/v1",
        )

    assert result.ok
    call_args = mock_client.post.call_args
    assert "custom.api.com" in call_args[0][0]
    assert "sk-override" in call_args[1]["headers"]["Authorization"]


def test_empty_choices():
    """Handles response with no choices."""
    client = OpenAICompatClient(api_key="sk-test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [], "usage": {}}

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="gpt-4o")

    assert not result.ok
    assert "choices" in (result.error or "").lower()


def test_keyless_local_server():
    """Supports keyless servers (vLLM, LM Studio) via api_key='none'."""
    client = OpenAICompatClient(api_key="none", base_url="http://localhost:8000/v1")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "local response"}}],
        "usage": {},
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="local-llama")

    assert result.ok
    assert result.text == "local response"
    # Verify no Authorization header was sent
    call_args = mock_client.post.call_args
    assert "Authorization" not in call_args[1]["headers"]


def test_content_null():
    """Handles content: null from tool-call-only responses."""
    client = OpenAICompatClient(api_key="sk-test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": None}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("conclave.models.openai_compat.httpx.Client", return_value=mock_client):
        result = client.generate("hello", model="gpt-4o")

    assert result.ok
    assert result.text == ""  # Not None, not error
