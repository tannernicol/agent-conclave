from fastapi.testclient import TestClient

from conclave.server import app


def test_index_page_renders() -> None:
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.text
    assert 'id="query"' in text
    assert "Ask Conclave" in text
    assert "/static/ui_state.js" in text
    assert "/static/ws.js" in text
