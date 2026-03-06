"""Run event schema helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class RunEvent:
    phase: str
    status: str | None = None
    role: str | None = None
    model_id: str | None = None
    model_label: str | None = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"phase": self.phase}
        if self.status is not None:
            payload["status"] = self.status
        if self.role is not None:
            payload["role"] = self.role
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        if self.model_label is not None:
            payload["model_label"] = self.model_label
        if self.data:
            payload.update(self.data)
        return payload


def normalize_event(event: Dict[str, Any] | RunEvent) -> Dict[str, Any]:
    if isinstance(event, RunEvent):
        return event.to_dict()
    return dict(event)
