from fastapi.testclient import TestClient

from conclave.server import app


def test_index_page_renders() -> None:
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.text
    assert 'id="messages"' in text
    assert "Conclave" in text
    assert "/static/chat.js" in text
    assert "/static/chat.css" in text
