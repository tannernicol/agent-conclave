"""Systemd scheduler helpers for Conclave topics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import re
import sys
import textwrap
import subprocess


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "topic"


@dataclass
class ScheduleUnit:
    topic_id: str
    service_path: Path
    timer_path: Path
    schedule: str


def render_service(topic_id: str, python_path: str, workdir: Path) -> str:
    return textwrap.dedent(
        f"""\
        [Unit]
        Description=Conclave topic {topic_id}
        After=network.target

        [Service]
        Type=oneshot
        WorkingDirectory={workdir}
        ExecStart={python_path} -m conclave.cli reconcile --topic {topic_id}
        """
    )


def render_timer(topic_id: str, schedule: str) -> str:
    return textwrap.dedent(
        f"""\
        [Unit]
        Description=Conclave topic {topic_id} timer

        [Timer]
        OnCalendar={schedule}
        Persistent=true

        [Install]
        WantedBy=timers.target
        """
    )


def validate_schedule(schedule: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["systemd-analyze", "calendar", schedule],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return True, "systemd-analyze not available"
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        return False, message or "invalid OnCalendar"
    return True, ""


def apply_schedule(
    topics: Iterable[dict],
    unit_dir: Path,
    python_path: str | None = None,
    workdir: Path | None = None,
    enable: bool = False,
    reload_systemd: bool = True,
    dry_run: bool = False,
    validate: bool = True,
    disable_legacy: bool = False,
) -> dict[str, Any]:
    unit_dir = unit_dir.expanduser()
    unit_dir.mkdir(parents=True, exist_ok=True)
    python_path = python_path or sys.executable
    workdir = workdir or Path.cwd()
    created: list[ScheduleUnit] = []
    errors: list[str] = []
    warnings: list[str] = []
    for topic in topics:
        topic_id = str(topic.get("id", "")).strip()
        if not topic_id:
            continue
        if topic.get("enabled", True) is False:
            continue
        schedule = str(topic.get("schedule", "weekly")).strip()
        if validate:
            ok, message = validate_schedule(schedule)
            if not ok:
                errors.append(f"{topic_id}: invalid schedule '{schedule}' ({message})")
                continue
            if message:
                warnings.append(f"{topic_id}: {message}")
        unit_base = f"conclave-topic-{slugify(topic_id)}"
        service_path = unit_dir / f"{unit_base}.service"
        timer_path = unit_dir / f"{unit_base}.timer"
        service_text = render_service(topic_id, python_path, workdir)
        timer_text = render_timer(topic_id, schedule)
        if not dry_run:
            service_path.write_text(service_text)
            timer_path.write_text(timer_text)
        created.append(ScheduleUnit(topic_id, service_path, timer_path, schedule))

    legacy_user = unit_dir / "conclave-reconcile.timer"
    legacy_system = Path("/etc/systemd/system/conclave-reconcile.timer")
    if legacy_user.exists() or legacy_system.exists():
        warnings.append("legacy conclave-reconcile.timer detected; per-topic timers may duplicate runs")
    if not dry_run and reload_systemd:
        result = subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        if result.returncode != 0:
            errors.append("systemctl --user daemon-reload failed")
        if disable_legacy:
            legacy = subprocess.run(
                ["systemctl", "--user", "disable", "--now", "conclave-reconcile.timer"],
                check=False,
            )
            if legacy.returncode != 0:
                errors.append("failed to disable conclave-reconcile.timer")
        if enable:
            for unit in created:
                result = subprocess.run(
                    ["systemctl", "--user", "enable", "--now", unit.timer_path.name],
                    check=False,
                )
                if result.returncode != 0:
                    errors.append(f"failed to enable {unit.timer_path.name}")
    return {"created": created, "errors": errors, "warnings": warnings}
