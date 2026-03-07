"""Integration tests for the Conclave chat deliberation UI."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from conclave.server import (
    app,
    ChatRoom,
    _critic_agrees,
    _extract_disagreements,
    _emit_chat_msg,
    _emit_system_msg,
    chat_room,
)


# ── Unit tests for helpers ──

class TestCriticAgrees:
    def test_agree_only(self):
        assert _critic_agrees("Overall I AGREE with the analysis.") is True

    def test_disagree(self):
        assert _critic_agrees("I DISAGREE with the reasoning.") is False

    def test_both_agree_and_disagree(self):
        # If both present, DISAGREE wins
        assert _critic_agrees("I AGREE on point 1 but DISAGREE on point 2.") is False

    def test_no_keywords(self):
        assert _critic_agrees("The analysis looks good.") is False

    def test_case_insensitive(self):
        assert _critic_agrees("Verdict: agree") is True

    def test_empty(self):
        assert _critic_agrees("") is False


class TestExtractDisagreements:
    def test_basic(self):
        text = (
            "Disagreements:\n"
            "- Missing error handling\n"
            "- No timeout logic\n"
            "Verdict:\n"
            "DISAGREE"
        )
        items = _extract_disagreements(text)
        assert items == ["Missing error handling", "No timeout logic"]

    def test_gaps_section(self):
        text = (
            "Gaps:\n"
            "- Doesn't cover edge case X\n"
            "Verdict:\n"
            "DISAGREE"
        )
        items = _extract_disagreements(text)
        assert items == ["Doesn't cover edge case X"]

    def test_max_items(self):
        lines = "Disagreements:\n" + "\n".join(f"- Item {i}" for i in range(20)) + "\nVerdict:\nDISAGREE"
        items = _extract_disagreements(lines)
        assert len(items) <= 8

    def test_no_disagreements(self):
        assert _extract_disagreements("Everything looks good. AGREE") == []


# ── ChatRoom tests ──

class TestChatRoom:
    def test_add_message_and_get_context(self):
        room = ChatRoom()
        msg = {"id": "msg-1", "sender": "user", "content": "hello"}
        room.add_message("test", msg)
        ctx = room.get_context("test")
        assert len(ctx) == 1
        assert ctx[0]["content"] == "hello"

    def test_history_cap(self):
        room = ChatRoom()
        room.max_history = 5
        for i in range(10):
            room.add_message("test", {"id": f"msg-{i}", "content": str(i)})
        ctx = room.get_context("test", limit=100)
        assert len(ctx) == 5
        assert ctx[0]["content"] == "5"

    def test_rate_limiting(self):
        room = ChatRoom()
        ws = MagicMock()
        assert room.check_rate(ws) is True
        assert room.check_rate(ws) is False  # Too fast

    def test_empty_room_context(self):
        room = ChatRoom()
        assert room.get_context("nonexistent") == []


# ── HTTP endpoint tests ──

class TestHTTPEndpoints:
    def test_root_serves_chat(self):
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert 'id="messages"' in resp.text
        assert 'id="message-input"' in resp.text
        assert "chat.js" in resp.text
        assert "chat.css" in resp.text

    def test_chat_route(self):
        client = TestClient(app)
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert 'id="messages"' in resp.text

    def test_static_css(self):
        client = TestClient(app)
        resp = client.get("/static/chat.css")
        assert resp.status_code == 200
        assert "role-badge" in resp.text
        assert "msg-system" in resp.text

    def test_static_js(self):
        client = TestClient(app)
        resp = client.get("/static/chat.js")
        assert resp.status_code == 200
        assert "VALID_SENDERS" in resp.text
        assert "renderContent" in resp.text
        assert "role-badge" in resp.text


# ── WebSocket tests ──

class TestWebSocket:
    def test_connect_receives_status_and_accepts(self):
        client = TestClient(app)
        with client.websocket_connect("/ws/chat/test-room") as ws:
            data = ws.receive_json()
            assert data["type"] == "status"
            assert "claude" in data["agents"]
            assert "codex" in data["agents"]
            assert "gemini" in data["agents"]
            for agent, status in data["agents"].items():
                assert status == "online"

    def test_invalid_room_id_rejected(self):
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat/invalid room!@#") as ws:
                pass

    def test_send_message_echoed_to_others(self):
        """When user sends a message, it should be stored in history."""
        client = TestClient(app)
        room_id = f"test-echo-{int(time.time())}"

        with client.websocket_connect(f"/ws/chat/{room_id}") as ws1:
            # Drain status message
            ws1.receive_json()

            # Send a message
            ws1.send_json({
                "type": "message",
                "content": "Hello world",
                "mentions": [],
            })

            # Give server time to process
            time.sleep(0.5)

            # Message should be in room history
            ctx = chat_room.get_context(room_id)
            assert len(ctx) >= 1
            user_msgs = [m for m in ctx if m.get("sender") == "user"]
            assert any(m["content"] == "Hello world" for m in user_msgs)

    def test_rate_limiting(self):
        client = TestClient(app)
        room_id = f"test-rate-{int(time.time())}"

        with client.websocket_connect(f"/ws/chat/{room_id}") as ws:
            ws.receive_json()  # status

            # Send messages rapidly
            ws.send_json({"type": "message", "content": "msg1", "mentions": []})
            ws.send_json({"type": "message", "content": "msg2", "mentions": []})
            time.sleep(0.5)

            # Only first should be stored (rate limited)
            ctx = chat_room.get_context(room_id)
            user_msgs = [m for m in ctx if m.get("sender") == "user"]
            assert len(user_msgs) == 1
            assert user_msgs[0]["content"] == "msg1"

    def test_message_size_limit(self):
        client = TestClient(app)
        room_id = f"test-size-{int(time.time())}"

        with client.websocket_connect(f"/ws/chat/{room_id}") as ws:
            ws.receive_json()  # status

            # Send oversized message
            ws.send_json({"type": "message", "content": "x" * 20000, "mentions": []})
            time.sleep(0.3)

            # Should not be stored
            ctx = chat_room.get_context(room_id)
            assert len(ctx) == 0

    def test_history_replay_on_connect(self):
        """Second client connecting to a room with messages gets history."""
        client = TestClient(app)
        room_id = f"test-history-{int(time.time())}"

        # Pre-populate room history
        chat_room.add_message(room_id, {
            "id": "msg-test-1",
            "sender": "claude",
            "content": "Test response",
            "timestamp": "2025-01-01T00:00:00Z",
            "role": "reasoner",
        })

        with client.websocket_connect(f"/ws/chat/{room_id}") as ws:
            status = ws.receive_json()
            assert status["type"] == "status"

            history = ws.receive_json()
            assert history["type"] == "history"
            assert len(history["messages"]) >= 1
            assert history["messages"][0]["content"] == "Test response"
            assert history["messages"][0]["role"] == "reasoner"

    def test_non_message_type_ignored(self):
        client = TestClient(app)
        room_id = f"test-ignore-{int(time.time())}"

        with client.websocket_connect(f"/ws/chat/{room_id}") as ws:
            ws.receive_json()  # status

            # Send non-message type
            ws.send_json({"type": "ping"})
            time.sleep(0.3)

            ctx = chat_room.get_context(room_id)
            assert len(ctx) == 0


# ── CSS/JS content validation ──

class TestFrontendAssets:
    def test_css_has_role_badges(self):
        client = TestClient(app)
        css = client.get("/static/chat.css").text
        assert ".role-badge.role-reasoner" in css
        assert ".role-badge.role-critic" in css
        assert ".role-badge.role-panel" in css

    def test_css_has_agent_colors(self):
        client = TestClient(app)
        css = client.get("/static/chat.css").text
        assert "--agent-claude" in css
        assert "--agent-codex" in css
        assert "--agent-gemini" in css

    def test_js_system_messages_use_render_content(self):
        client = TestClient(app)
        js = client.get("/static/chat.js").text
        assert "renderContent(text)" in js

    def test_js_has_dedup_logic(self):
        client = TestClient(app)
        js = client.get("/static/chat.js").text
        assert "messageIds" in js

    def test_js_history_routes_system_messages(self):
        client = TestClient(app)
        js = client.get("/static/chat.js").text
        assert "sender === 'system'" in js or "sender === \\'system\\'" in js

    def test_html_has_all_elements(self):
        client = TestClient(app)
        html = client.get("/").text
        assert 'id="messages"' in html
        assert 'id="message-input"' in html
        assert 'id="btn-send"' in html
        assert 'id="mention-bar"' in html
        assert 'id="agent-status"' in html
        assert 'id="empty-state"' in html
