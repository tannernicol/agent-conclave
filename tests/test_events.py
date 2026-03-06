from conclave.events import RunEvent, normalize_event


def test_run_event_to_dict() -> None:
    event = RunEvent(phase="route", status="done", data={"models": {"reasoner": "gpt"}})
    payload = event.to_dict()
    assert payload["phase"] == "route"
    assert payload["status"] == "done"
    assert payload["models"]["reasoner"] == "gpt"


def test_normalize_event_accepts_dict() -> None:
    raw = {"phase": "retrieve", "status": "start"}
    assert normalize_event(raw) == raw
