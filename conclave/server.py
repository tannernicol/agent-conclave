"""FastAPI server for Conclave."""
from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
from typing import Dict, Set
import asyncio
import re
import json
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline
from conclave.models.registry import ModelRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.config = config
    app.state.pipeline = ConclavePipeline(config)
    app.state.store = app.state.pipeline.store
    app.state.registry = ModelRegistry.from_config(config.models)
    try:
        yield
    finally:
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline:
            try:
                pipeline.mcp.close_all()
            except Exception:
                pass

app = FastAPI(title="Conclave", lifespan=lifespan)

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # run_id -> set of websockets

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()
        self.active_connections[run_id].add(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            self.active_connections[run_id].discard(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.add(connection)
            for conn in dead_connections:
                self.active_connections[run_id].discard(conn)

ws_manager = ConnectionManager()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "input"


def _inputs_dir(config) -> Path:
    path = config.data_dir / "inputs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _artifacts_dir(config) -> Path:
    path = config.data_dir / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _prompts_dir(config) -> Path:
    path = config.data_dir / "prompts"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _uploads_dir(config) -> Path:
    path = config.data_dir / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _prompt_path(config, prompt_id: str) -> Path:
    return _prompts_dir(config) / f"{prompt_id}.json"

def _prompt_input_path(config, prompt_id: str) -> Path:
    return _inputs_dir(config) / f"prompt-{prompt_id}.md"

def _normalize_prompt_field(value: str) -> str:
    return " ".join((value or "").strip().split())

def _normalize_role_overrides(value: dict | None) -> dict:
    if not isinstance(value, dict):
        return {}
    result: dict = {}
    for key, raw in value.items():
        role = _normalize_prompt_field(str(key))
        model_id = _normalize_prompt_field(str(raw))
        if role and model_id:
            result[role] = model_id
    return result


def _merge_capabilities(card: dict) -> dict:
    caps = dict(card.get("capabilities", {}) or {})
    override = card.get("capabilities_override") or {}
    if isinstance(override, dict):
        caps.update(override)
    return caps


def _supports_capability(caps: dict, cap: str) -> bool:
    if cap == "image_understanding":
        return str(caps.get("image_understanding", "")).lower() in {"limited", "full"}
    if cap == "image_generation":
        return bool(caps.get("image_generation"))
    if cap == "tool_use":
        return bool(caps.get("tool_use"))
    if cap == "code_generation":
        return str(caps.get("code_generation", "")).lower() in {"medium", "high"}
    if cap == "json_high":
        return str(caps.get("json_reliability", "")).lower() == "high"
    return False


def _capability_score(caps: dict, prefer: list[str]) -> float:
    score = 0.0
    for cap in prefer:
        if cap == "image_understanding":
            value = str(caps.get("image_understanding", "")).lower()
            if value == "full":
                score += 2.0
            elif value == "limited":
                score += 1.2
        elif cap == "image_generation":
            if caps.get("image_generation"):
                score += 1.4
        elif cap == "tool_use":
            if caps.get("tool_use"):
                score += 1.1
        elif cap == "code_generation":
            value = str(caps.get("code_generation", "")).lower()
            if value == "high":
                score += 1.8
            elif value == "medium":
                score += 1.0
            elif value == "low":
                score += 0.3
        elif cap == "json_high":
            if str(caps.get("json_reliability", "")).lower() == "high":
                score += 0.8
    return score


def _recommend_role_overrides(
    output_type: str,
    route: dict,
    models: list[dict],
    planner,
) -> dict:
    key = (output_type or "").strip().lower()
    if not key:
        return {}
    rules = {
        "image_palette": {"creator": {"require": ["image_understanding"], "prefer": ["image_understanding"]}},
        "image_brief": {"creator": {"prefer": ["image_understanding"]}},
        "webpage_redesign": {"creator": {"prefer": ["code_generation", "tool_use"]}},
        "build_spec": {"creator": {"prefer": ["code_generation", "tool_use"]}},
    }
    spec = rules.get(key)
    if not spec:
        return {}
    overrides: dict = {}
    plan = route.get("plan") or {}
    roles = set(route.get("roles") or [])
    for role, rule in spec.items():
        if roles and role not in roles:
            continue
        require = rule.get("require") or []
        prefer = rule.get("prefer") or []
        current_id = plan.get(role)
        current_card = next((m for m in models if m.get("id") == current_id), None)
        current_caps = _merge_capabilities(current_card or {})
        has_required = all(_supports_capability(current_caps, cap) for cap in require)
        has_prefer = all(_supports_capability(current_caps, cap) for cap in prefer) if prefer else True
        if has_required and has_prefer:
            continue
        best_id = None
        best_score = -1.0
        for card in models:
            caps = _merge_capabilities(card)
            if any(not _supports_capability(caps, cap) for cap in require):
                continue
            base_score = 0.0
            try:
                base_score = float(planner._score(role, card))  # type: ignore[attr-defined]
            except Exception:
                base_score = 0.0
            cap_score = _capability_score(caps, prefer)
            score = base_score + (cap_score * 1.4)
            if score > best_score:
                best_score = score
                best_id = card.get("id")
        if best_id and best_id != current_id:
            overrides[role] = best_id
    return overrides

def _safe_upload_name(filename: str) -> str:
    base = Path(filename).name
    stem = _slugify(Path(base).stem)
    suffix = Path(base).suffix.lower()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    token = hashlib.sha1(f"{base}-{time.time()}".encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{stamp}-{token}{suffix}"

def _agent_sets(config) -> list[dict]:
    cfg = config.raw.get("agent_sets", {}) or {}
    sets = []
    if isinstance(cfg, dict):
        for key, entry in cfg.items():
            if not isinstance(entry, dict):
                continue
            sets.append({
                "id": key,
                "label": entry.get("label") or key,
                "roles": list(entry.get("roles") or []),
                "panel_models": list(entry.get("panel_models") or []),
                "eligible_models": list(entry.get("eligible_models") or []),
                "panel_require_all": entry.get("panel_require_all"),
                "panel_min_ratio": entry.get("panel_min_ratio"),
            })
    return sorted(sets, key=lambda item: item.get("label", item.get("id", "")))

def _use_cases(config) -> list[dict]:
    items = []
    cfg = config.raw.get("use_cases", []) or []
    if isinstance(cfg, list):
        for entry in cfg:
            if not isinstance(entry, dict):
                continue
            case_id = entry.get("id")
            if not case_id:
                continue
            items.append({
                "id": case_id,
                "label": entry.get("label") or case_id,
                "output_type": entry.get("output_type"),
                "agent_set": entry.get("agent_set"),
                "collections": entry.get("collections") or [],
            })
    return items

def _prompt_fingerprint(
    query: str,
    notes: str,
    output_type: str | None,
    artifacts: list[str] | None,
    use_case: str | None = None,
    agent_set: str | None = None,
    role_overrides: dict | None = None,
) -> str:
    payload = {
        "query": _normalize_prompt_field(query),
        "notes": _normalize_prompt_field(notes),
        "output_type": _normalize_prompt_field(output_type or ""),
        "use_case": _normalize_prompt_field(use_case or ""),
        "agent_set": _normalize_prompt_field(agent_set or ""),
        "artifacts": sorted(_normalize_prompt_field(item) for item in (artifacts or []) if _normalize_prompt_field(item)),
        "role_overrides": _normalize_role_overrides(role_overrides),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _load_prompt_fingerprint(prompt: dict) -> str:
    existing = prompt.get("fingerprint")
    if isinstance(existing, str) and existing:
        return existing
    return _prompt_fingerprint(
        prompt.get("query") or "",
        prompt.get("notes") or "",
        prompt.get("output_type"),
        prompt.get("artifacts") or [],
        prompt.get("use_case"),
        prompt.get("agent_set"),
        prompt.get("role_overrides"),
    )

def _find_prompt_by_fingerprint(config, fingerprint: str) -> dict | None:
    for path in sorted(_prompts_dir(config).glob("*.json"), reverse=True):
        try:
            prompt = json.loads(path.read_text())
        except Exception:
            continue
        if _load_prompt_fingerprint(prompt) == fingerprint:
            prompt.setdefault("id", path.stem)
            return prompt
    return None


def _compact_run(run: dict | None) -> dict | None:
    if not run:
        return None
    consensus = run.get("consensus") or {}
    artifacts = run.get("artifacts") or {}
    deliberation = artifacts.get("deliberation") or {}
    return {
        "id": run.get("id"),
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "completed_at": run.get("completed_at"),
        "error": run.get("error"),
        "query": run.get("query"),
        "meta": run.get("meta") or {},
        "consensus": {
            "answer": consensus.get("answer"),
            "confidence": consensus.get("confidence"),
            "insufficient_evidence": consensus.get("insufficient_evidence"),
            "fallback_used": consensus.get("fallback_used"),
        },
        "artifacts": {
            "deliberation": {
                "agreement": deliberation.get("agreement"),
            }
        },
    }

def _write_prompt_input(
    path: Path,
    title: str,
    question: str,
    notes: str,
    artifacts: list[str] | None = None,
    output_type: str | None = None,
    use_case: str | None = None,
    agent_set: str | None = None,
    role_overrides: dict | None = None,
) -> None:
    body = []
    if title:
        body.append(f"# {title}")
        body.append("")
    if question:
        body.append("## Question")
        body.append(question)
        body.append("")
    if output_type:
        body.append("## Output Type")
        body.append(output_type)
        body.append("")
    if use_case:
        body.append("## Use Case")
        body.append(use_case)
        body.append("")
    if agent_set:
        body.append("## Agent Set")
        body.append(agent_set)
        body.append("")
    if role_overrides:
        body.append("## Role Overrides")
        for role, model_id in sorted(role_overrides.items()):
            body.append(f"- {role}: {model_id}")
        body.append("")
    body.append("## Notes")
    body.append(notes or "")
    if artifacts:
        body.append("")
        body.append("## Artifacts")
        for item in artifacts:
            body.append(f"- {item}")
    path.write_text("\n".join(body).strip() + "\n")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("chat.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "conclave"}


@app.websocket("/ws/run/{run_id}")
async def websocket_run(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time run progress updates."""
    await ws_manager.connect(websocket, run_id)
    try:
        # Send current state immediately
        store = websocket.app.state.store
        run = store.get_run(run_id)
        if run:
            await websocket.send_json({"type": "state", "run": run})

        # Keep connection alive and watch for updates
        last_event_count = len(run.get("events", [])) if run else 0
        while True:
            await asyncio.sleep(0.5)  # Poll every 500ms
            run = store.get_run(run_id)
            if not run:
                break

            current_event_count = len(run.get("events", []))
            if current_event_count > last_event_count:
                # Send new events
                new_events = run["events"][last_event_count:]
                for event in new_events:
                    await websocket.send_json({"type": "event", "event": event})
                last_event_count = current_event_count

            # Check if run is complete
            if run.get("status") in ("complete", "failed"):
                await websocket.send_json({"type": "complete", "run": run})
                break

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, run_id)


@app.get("/api/status")
async def status_api(request: Request):
    latest = request.app.state.store.latest()
    return {
        "ok": True,
        "latest": latest,
    }


@app.get("/api/models")
async def models_api(request: Request):
    return {"models": request.app.state.registry.list_models()}


@app.get("/api/collections")
async def collections_api(request: Request):
    """Get available RAG collections for the picker."""
    config = request.app.state.config
    pipeline = request.app.state.pipeline

    # Get collections from RAG
    collections = pipeline.rag.collections()

    # Add reliability info from config
    reliability_map = config.rag.get("collection_reliability", {})
    result = []
    for coll in collections:
        name = coll.get("name", "")
        result.append({
            "name": name,
            "file_count": coll.get("file_count", 0),
            "exists": coll.get("exists", True),
            "reliability": reliability_map.get(name, "other"),
            "description": coll.get("description", ""),
        })

    return {"collections": result}


@app.get("/api/outputs")
async def outputs_api(request: Request):
    pipeline = request.app.state.pipeline
    return {"outputs": pipeline.output_types()}


@app.get("/api/agent-sets")
async def agent_sets_api(request: Request):
    config = request.app.state.config
    return {"agent_sets": _agent_sets(config)}


@app.get("/api/use-cases")
async def use_cases_api(request: Request):
    config = request.app.state.config
    return {"use_cases": _use_cases(config)}


@app.post("/api/plan")
async def plan_api(payload: dict, request: Request):
    query = (payload.get("query") or "").strip() or "general"
    collections = payload.get("collections")
    use_case = (payload.get("use_case") or "").strip()
    agent_set_id = (payload.get("agent_set") or "").strip()
    output_type = (payload.get("output_type") or "").strip()
    role_overrides = _normalize_role_overrides(payload.get("role_overrides"))
    config = request.app.state.config
    pipeline = request.app.state.pipeline
    case_entry = next((item for item in _use_cases(config) if item.get("id") == use_case), None)
    if case_entry:
        if not output_type and case_entry.get("output_type"):
            output_type = case_entry.get("output_type")
        if not agent_set_id and case_entry.get("agent_set"):
            agent_set_id = case_entry.get("agent_set")
        if not collections and case_entry.get("collections"):
            collections = case_entry.get("collections")
    agent_set = pipeline._resolve_agent_set(agent_set_id) if agent_set_id else None
    route = pipeline._route_query(
        query,
        collections,
        budget_context=None,
        agent_set=agent_set,
        role_overrides=role_overrides,
    )
    plan_details = pipeline._plan_details(route.get("plan", {}))
    route["plan_details"] = plan_details
    route["panel_details"] = pipeline._panel_details(route.get("panel_models") or [])
    output_meta = pipeline._output_meta(output_type, route) if output_type else {}
    return {
        "ok": True,
        "route": route,
        "plan": plan_details,
        "models": request.app.state.registry.list_models(),
        "output": output_meta,
    }


@app.post("/api/run")
async def run_api(payload: dict, background: BackgroundTasks, request: Request):
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    collections = payload.get("collections")
    meta = {"source": "api"}
    use_case = (payload.get("use_case") or "").strip()
    agent_set = (payload.get("agent_set") or "").strip()
    input_title = (payload.get("input_title") or "").strip()
    if input_title:
        meta["input_title"] = input_title
    role_overrides = _normalize_role_overrides(payload.get("role_overrides"))
    if role_overrides:
        meta["role_overrides"] = role_overrides
    output_type = (payload.get("output_type") or "").strip()
    if output_type:
        meta["output_type"] = output_type
    if use_case:
        meta["use_case"] = use_case
    if agent_set:
        meta["agent_set"] = agent_set
    prompt_id = (payload.get("prompt_id") or "").strip()
    if prompt_id:
        meta["prompt_id"] = prompt_id
    input_id = payload.get("input_id")
    input_path = payload.get("input_path")
    config = request.app.state.config
    store = request.app.state.store
    pipeline = request.app.state.pipeline
    case_entry = next((item for item in _use_cases(config) if item.get("id") == use_case), None)
    if case_entry:
        if not output_type and case_entry.get("output_type"):
            meta["output_type"] = case_entry.get("output_type")
        if not agent_set and case_entry.get("agent_set"):
            meta["agent_set"] = case_entry.get("agent_set")
        if not collections and case_entry.get("collections"):
            collections = case_entry.get("collections")
    if input_id:
        meta["input_path"] = str(_inputs_dir(config) / input_id)
    if input_path:
        meta["input_path"] = input_path
    run_id = store.create_run(query, meta=meta)
    def _task():
        pipeline.run(query, collections=collections, run_id=run_id, meta=meta)
    background.add_task(_task)
    return {"ok": True, "run_id": run_id}


@app.get("/api/runs")
async def runs_api(request: Request, limit: int = 20):
    return {"runs": request.app.state.store.list_runs(limit=limit)}


@app.get("/api/runs/latest")
async def runs_latest_api(request: Request):
    return request.app.state.store.latest() or {}


@app.get("/api/runs/{run_id}")
async def run_detail_api(run_id: str, request: Request):
    run = request.app.state.store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    return run


@app.get("/api/runs/{run_id}/artifacts")
async def run_artifacts_api(run_id: str, request: Request):
    run = request.app.state.store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    artifacts = (run.get("artifacts") or {}).get("output_artifacts") or []
    return {"artifacts": artifacts}


@app.get("/api/runs/{run_id}/artifacts/{name}")
async def run_artifact_download_api(run_id: str, name: str, request: Request):
    store = request.app.state.store
    if "/" in name or "\\" in name or ".." in name:
        return JSONResponse({"error": "invalid artifact name"}, status_code=400)
    run_dir = store.run_dir(run_id)
    path = run_dir / name
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.post("/api/runs/{run_id}/rerun")
async def run_rerun_api(run_id: str, background: BackgroundTasks, request: Request):
    store = request.app.state.store
    run = store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    query = run.get("query", "")
    if not query:
        return JSONResponse({"error": "query missing"}, status_code=400)
    meta = dict(run.get("meta") or {})
    meta["source"] = meta.get("source", "api")
    collections = None
    artifacts = run.get("artifacts") or {}
    route = artifacts.get("route") or {}
    if route.get("collections"):
        collections = route.get("collections")
    new_run_id = store.create_run(query, meta=meta)
    def _task():
        request.app.state.pipeline.run(query, collections=collections, run_id=new_run_id, meta=meta)
    background.add_task(_task)
    return {"ok": True, "run_id": new_run_id}


@app.delete("/api/runs/{run_id}")
async def run_delete_api(run_id: str, request: Request):
    store = request.app.state.store
    run = store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    prompt_id = (run.get("meta") or {}).get("prompt_id")
    ok = store.delete_run(run_id)
    store.rebuild_latest()
    if prompt_id:
        store.rebuild_prompt_latest(str(prompt_id))
    return {"ok": ok}


@app.post("/api/runs/{run_id}/fail")
async def run_fail_api(run_id: str, request: Request, payload: dict | None = None):
    store = request.app.state.store
    run = store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    reason = None
    if isinstance(payload, dict):
        reason = payload.get("reason")
    store.fail_run(run_id, reason or "stuck_running")
    store.rebuild_latest()
    prompt_id = (run.get("meta") or {}).get("prompt_id")
    if prompt_id:
        store.rebuild_prompt_latest(str(prompt_id))
    return {"ok": True}


@app.post("/api/reconcile")
async def reconcile_api(payload: dict, background: BackgroundTasks, request: Request):
    topic = payload.get("topic")
    if not topic:
        return JSONResponse({"error": "topic required"}, status_code=400)
    def _task():
        config = request.app.state.config
        pipeline = request.app.state.pipeline
        topics = config.topics
        if topic == "all":
            for item in topics:
                pipeline.run(item.get("query", ""), collections=item.get("collections"), meta={"topic": item.get("id")})
            return
        selected = None
        for item in topics:
            if item.get("id") == topic:
                selected = item
                break
        if selected:
            pipeline.run(selected.get("query", ""), collections=selected.get("collections"), meta={"topic": topic})
    background.add_task(_task)
    return {"ok": True}


@app.post("/api/inputs")
async def inputs_api(payload: dict, request: Request):
    content = (payload.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    title = (payload.get("title") or "").strip()
    question = (payload.get("question") or "").strip()
    output_type = (payload.get("output_type") or "").strip()
    use_case = (payload.get("use_case") or "").strip()
    agent_set = (payload.get("agent_set") or "").strip()
    role_overrides = _normalize_role_overrides(payload.get("role_overrides"))
    artifacts = payload.get("artifacts") or []
    artifacts = [str(item).strip() for item in artifacts if str(item).strip()]
    slug = _slugify(title or question or "input")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}-{slug}.md"
    config = request.app.state.config
    path = _inputs_dir(config) / filename
    body = []
    if title:
        body.append(f"# {title}")
        body.append("")
    if question:
        body.append("## Question")
        body.append(question)
        body.append("")
    if output_type:
        body.append("## Output Type")
        body.append(output_type)
        body.append("")
    if use_case:
        body.append("## Use Case")
        body.append(use_case)
        body.append("")
    if agent_set:
        body.append("## Agent Set")
        body.append(agent_set)
        body.append("")
    if role_overrides:
        body.append("## Role Overrides")
        for role, model_id in sorted(role_overrides.items()):
            body.append(f"- {role}: {model_id}")
        body.append("")
    body.append("## Notes")
    body.append(content)
    if artifacts:
        body.append("")
        body.append("## Artifacts")
        for item in artifacts:
            body.append(f"- {item}")
    path.write_text("\n".join(body).strip() + "\n")
    return {"ok": True, "input_id": filename, "path": str(path), "artifacts": artifacts}


@app.post("/api/uploads")
async def uploads_api(request: Request, files: list[UploadFile] = File(...)):
    config = request.app.state.config
    dest_dir = _uploads_dir(config)
    saved = []
    for file in files or []:
        if not file.filename:
            continue
        safe_name = _safe_upload_name(file.filename)
        target = dest_dir / safe_name
        data = await file.read()
        target.write_bytes(data)
        saved.append({
            "name": file.filename,
            "path": str(target),
            "size": len(data),
        })
    return {"ok": True, "files": saved}


@app.get("/api/inputs")
async def inputs_list_api(request: Request, limit: int = 20):
    items = []
    config = request.app.state.config
    for path in sorted(_inputs_dir(config).glob("*.md"), reverse=True)[:limit]:
        items.append({
            "id": path.name,
            "path": str(path),
            "modified_at": path.stat().st_mtime,
        })
    return {"inputs": items}


@app.get("/api/inputs/{input_id}")
async def inputs_get_api(input_id: str, request: Request):
    config = request.app.state.config
    path = _inputs_dir(config) / input_id
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": input_id, "path": str(path), "content": path.read_text()}


@app.post("/api/prompts")
async def prompts_create_api(payload: dict, request: Request):
    config = request.app.state.config
    title = (payload.get("title") or "").strip()
    query = (payload.get("query") or "").strip()
    notes = (payload.get("notes") or "").strip()
    output_type = (payload.get("output_type") or "").strip()
    use_case = (payload.get("use_case") or "").strip()
    agent_set = (payload.get("agent_set") or "").strip()
    role_overrides = _normalize_role_overrides(payload.get("role_overrides"))
    artifacts = payload.get("artifacts") or []
    artifacts = [str(item).strip() for item in artifacts if str(item).strip()]
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    fingerprint = _prompt_fingerprint(query, notes, output_type, artifacts, use_case, agent_set, role_overrides)
    existing = _find_prompt_by_fingerprint(config, fingerprint)
    if existing:
        return {"ok": True, "prompt": existing, "deduped": True}
    prompt_id = (payload.get("id") or "").strip()
    slug = _slugify(title or query or "prompt")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not prompt_id:
        prompt_id = f"{ts}-{slug}"
    path = _prompt_path(config, prompt_id)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    prompt = {
        "id": prompt_id,
        "title": title,
        "query": query,
        "notes": notes,
        "artifacts": artifacts,
        "created_at": created_at,
        "updated_at": created_at,
        "input_path": str(_prompt_input_path(config, prompt_id)),
        "last_run_id": None,
        "output_type": output_type or None,
        "use_case": use_case or None,
        "agent_set": agent_set or None,
        "role_overrides": role_overrides or None,
        "fingerprint": fingerprint,
    }
    _write_prompt_input(
        Path(prompt["input_path"]),
        title,
        query,
        notes,
        artifacts,
        output_type=output_type,
        use_case=use_case,
        agent_set=agent_set,
        role_overrides=role_overrides,
    )
    path.write_text(json.dumps(prompt, indent=2))
    return {"ok": True, "prompt": prompt}


@app.put("/api/prompts/{prompt_id}")
async def prompts_update_api(prompt_id: str, payload: dict, request: Request):
    config = request.app.state.config
    path = _prompt_path(config, prompt_id)
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    prompt = json.loads(path.read_text())
    title = (payload.get("title") or prompt.get("title") or "").strip()
    query = (payload.get("query") or prompt.get("query") or "").strip()
    notes = (payload.get("notes") or prompt.get("notes") or "").strip()
    output_type = (payload.get("output_type") or prompt.get("output_type") or "").strip()
    use_case = (payload.get("use_case") or prompt.get("use_case") or "").strip()
    agent_set = (payload.get("agent_set") or prompt.get("agent_set") or "").strip()
    role_overrides = payload.get("role_overrides")
    if role_overrides is None:
        role_overrides = prompt.get("role_overrides") or {}
    role_overrides = _normalize_role_overrides(role_overrides)
    artifacts = payload.get("artifacts")
    if artifacts is None:
        artifacts = prompt.get("artifacts") or []
    artifacts = [str(item).strip() for item in artifacts if str(item).strip()]
    prompt["title"] = title
    prompt["query"] = query
    prompt["notes"] = notes
    prompt["artifacts"] = artifacts
    prompt["output_type"] = output_type or None
    prompt["use_case"] = use_case or None
    prompt["agent_set"] = agent_set or None
    prompt["role_overrides"] = role_overrides or None
    prompt["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    prompt["fingerprint"] = _prompt_fingerprint(query, notes, output_type, artifacts, use_case, agent_set, role_overrides)
    input_path = Path(prompt.get("input_path") or _prompt_input_path(config, prompt_id))
    prompt["input_path"] = str(input_path)
    _write_prompt_input(
        input_path,
        title,
        query,
        notes,
        artifacts,
        output_type=output_type,
        use_case=use_case,
        agent_set=agent_set,
        role_overrides=role_overrides,
    )
    path.write_text(json.dumps(prompt, indent=2))
    return {"ok": True, "prompt": prompt}


@app.get("/api/prompts")
async def prompts_list_api(request: Request, limit: int = 50):
    config = request.app.state.config
    store = request.app.state.store
    items = []
    seen = set()
    for path in sorted(_prompts_dir(config).glob("*.json"), reverse=True):
        try:
            prompt = json.loads(path.read_text())
        except Exception:
            continue
        fingerprint = _load_prompt_fingerprint(prompt)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        prompt.setdefault("fingerprint", fingerprint)
        prompt_id = prompt.get("id")
        latest_run = store.latest_for_prompt(str(prompt_id)) if prompt_id else None
        if not latest_run and prompt.get("last_run_id"):
            latest_run = store.get_run(str(prompt.get("last_run_id")))
        prompt["latest_run"] = _compact_run(latest_run)
        items.append(prompt)
        if len(items) >= limit:
            break
    return {"prompts": items}


@app.get("/api/prompts/{prompt_id}")
async def prompts_get_api(prompt_id: str, request: Request):
    config = request.app.state.config
    path = _prompt_path(config, prompt_id)
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return json.loads(path.read_text())


@app.delete("/api/prompts/{prompt_id}")
async def prompts_delete_api(prompt_id: str, request: Request, delete_runs: bool = False):
    config = request.app.state.config
    path = _prompt_path(config, prompt_id)
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    prompt = json.loads(path.read_text())
    input_path = prompt.get("input_path")
    if input_path:
        try:
            Path(input_path).unlink()
        except Exception:
            pass
    path.unlink()
    store = request.app.state.store
    store.rebuild_prompt_latest(prompt_id)
    if delete_runs:
        runs_dir = store._runs_dir()
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                run_path = run_dir / "run.json"
                if not run_path.exists():
                    continue
                try:
                    run = json.loads(run_path.read_text())
                except Exception:
                    continue
                if (run.get("meta") or {}).get("prompt_id") == prompt_id:
                    store.delete_run(run.get("id"))
        store.rebuild_latest()
    return {"ok": True}


@app.post("/api/prompts/{prompt_id}/run")
async def prompts_run_api(prompt_id: str, background: BackgroundTasks, request: Request):
    config = request.app.state.config
    path = _prompt_path(config, prompt_id)
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    prompt = json.loads(path.read_text())
    query = (prompt.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    meta = {
        "source": "api",
        "prompt_id": prompt_id,
        "input_title": prompt.get("title") or prompt_id,
        "input_path": prompt.get("input_path"),
    }
    if prompt.get("output_type"):
        meta["output_type"] = prompt.get("output_type")
    if prompt.get("use_case"):
        meta["use_case"] = prompt.get("use_case")
    if prompt.get("agent_set"):
        meta["agent_set"] = prompt.get("agent_set")
    if prompt.get("role_overrides"):
        meta["role_overrides"] = prompt.get("role_overrides")
    case_entry = next((item for item in _use_cases(config) if item.get("id") == prompt.get("use_case")), None)
    if case_entry and not meta.get("output_type") and case_entry.get("output_type"):
        meta["output_type"] = case_entry.get("output_type")
    if case_entry and not meta.get("agent_set") and case_entry.get("agent_set"):
        meta["agent_set"] = case_entry.get("agent_set")
    run_id = request.app.state.store.create_run(query, meta=meta)
    def _task():
        request.app.state.pipeline.run(query, collections=None, run_id=run_id, meta=meta)
    background.add_task(_task)
    prompt["last_run_id"] = run_id
    prompt["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path.write_text(json.dumps(prompt, indent=2))
    return {"ok": True, "run_id": run_id, "prompt": prompt}


## ── Chat Room ──────────────────────────────────────────────

import uuid as _uuid

_CHAT_MAX_MESSAGE_LEN = 10000
_CHAT_MAX_ROOMS = 50
_CHAT_MIN_INTERVAL_SEC = 1.0
_CHAT_ROOM_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,32}$')


class ChatRoom:
    """Manages a multi-agent chat room with conversation history."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.history: Dict[str, list] = {}
        self.max_history = 200
        self._last_msg_time: Dict[WebSocket, float] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        self.connections.setdefault(room_id, set()).add(websocket)
        try:
            await websocket.send_json({
                "type": "status",
                "agents": {aid: "online" for aid in ("claude", "codex", "gemini")},
            })
            if room_id in self.history and self.history[room_id]:
                await websocket.send_json({
                    "type": "history",
                    "messages": self.history[room_id][-50:],
                })
        except Exception:
            pass

    def disconnect(self, websocket: WebSocket, room_id: str):
        self._last_msg_time.pop(websocket, None)
        if room_id in self.connections:
            self.connections[room_id].discard(websocket)
            if not self.connections[room_id]:
                del self.connections[room_id]

    async def broadcast(self, room_id: str, message: dict, exclude: WebSocket | None = None):
        conns = self.connections.get(room_id)
        if not conns:
            return
        dead = set()
        for conn in list(conns):
            if conn is exclude:
                continue
            try:
                await conn.send_json(message)
            except Exception:
                dead.add(conn)
        conns -= dead

    def check_rate(self, websocket: WebSocket) -> bool:
        now = time.time()
        last = self._last_msg_time.get(websocket, 0)
        if now - last < _CHAT_MIN_INTERVAL_SEC:
            return False
        self._last_msg_time[websocket] = now
        return True

    def add_message(self, room_id: str, msg: dict):
        self.history.setdefault(room_id, []).append(msg)
        if len(self.history[room_id]) > self.max_history:
            self.history[room_id] = self.history[room_id][-self.max_history:]

    def get_context(self, room_id: str, limit: int = 20) -> list:
        return (self.history.get(room_id) or [])[-limit:]


chat_room = ChatRoom()

CHAT_AGENT_MODELS = {
    "claude": "cli:claude",
    "codex": "cli:codex",
    "gemini": "cli:gemini",
}

# Role display names
_ROLE_LABELS = {
    "reasoner": "Reasoner",
    "critic": "Critic",
    "panel": "Panel Reviewer",
}

_AGREE_PATTERNS = re.compile(r'\bAGREE\b', re.IGNORECASE)
_DISAGREE_PATTERNS = re.compile(r'\bDISAGREE\b', re.IGNORECASE)


def _extract_disagreements(text: str) -> list[str]:
    """Extract disagreement bullet points from critic output."""
    items = []
    in_section = False
    for line in text.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("disagreement") or lower.startswith("gap"):
            in_section = True
            continue
        if lower.startswith("verdict"):
            in_section = False
            continue
        if in_section and stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items[:8]


def _critic_agrees(text: str) -> bool:
    """Check if critic text indicates agreement."""
    if _AGREE_PATTERNS.search(text) and not _DISAGREE_PATTERNS.search(text):
        return True
    return False


async def _call_cli_model(card: dict, prompt: str) -> str:
    """Call a CLI model asynchronously and return the response text."""
    from conclave.models.cli import CliClient
    client = CliClient(max_retries=0)
    command = list(card.get("command", []))
    if not command:
        logger.warning("CLI model has no command configured")
        return ""
    logger.info(f"Calling CLI model: {command[0]} (prompt_mode={card.get('prompt_mode', 'arg')})")
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.run(
            command, prompt,
            prompt_mode=card.get("prompt_mode", "arg"),
            stdin_flag=card.get("stdin_flag"),
            timeout_seconds=card.get("timeout_seconds", 120),
            env=card.get("env"),
        ),
    )
    if not result.ok:
        logger.warning(f"CLI model failed: {result.error} stderr={result.stderr}")
        return ""
    logger.info(f"CLI model responded: {len(result.text)} chars in {result.duration_ms:.0f}ms")
    return result.text


async def _emit_chat_msg(room_id: str, sender: str, content: str, role: str | None = None, quoting: dict | None = None):
    """Create and broadcast a chat message from an agent."""
    label = sender
    if role:
        label = f"{sender} ({_ROLE_LABELS.get(role, role)})"
    msg = {
        "id": f"msg-{_uuid.uuid4().hex[:12]}",
        "sender": sender,
        "content": content,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "role": role,
    }
    if quoting:
        msg["quoting"] = quoting
    chat_room.add_message(room_id, msg)
    await chat_room.broadcast(room_id, {"type": "message", **msg})
    return msg


async def _emit_system_msg(room_id: str, content: str):
    """Broadcast a system status message."""
    msg = {
        "id": f"msg-{_uuid.uuid4().hex[:12]}",
        "sender": "system",
        "content": content,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    chat_room.add_message(room_id, msg)
    await chat_room.broadcast(room_id, {"type": "system", "content": content})


async def _run_deliberation(room_id: str, query: str, registry: ModelRegistry, user_input_queue: asyncio.Queue):
    """Run the full deliberation loop, broadcasting each step as chat messages.

    The user can inject messages mid-deliberation via user_input_queue.
    """
    config_raw = app.state.config.raw
    delib_cfg = config_raw.get("deliberation", {}) or {}
    max_rounds = int(delib_cfg.get("max_rounds", 5))

    # Assign roles from config or defaults
    planner_cfg = config_raw.get("planner", {}) or {}
    role_overrides = planner_cfg.get("role_overrides", {}) or {}

    reasoner_id = role_overrides.get("reasoner", "cli:codex")
    critic_id = role_overrides.get("critic", "cli:claude")
    panel_ids = list(delib_cfg.get("panel", {}).get("model_ids", []))

    # Map model IDs back to agent names
    id_to_name = {v: k for k, v in CHAT_AGENT_MODELS.items()}

    reasoner_name = id_to_name.get(reasoner_id, "codex")
    critic_name = id_to_name.get(critic_id, "claude")

    reasoner_card = registry.get_model(reasoner_id)
    critic_card = registry.get_model(critic_id)

    if not reasoner_card or not critic_card:
        await _emit_system_msg(room_id, "Could not find required models. Check config.")
        return

    # Announce roles
    panel_names = [id_to_name.get(pid, pid.split(":")[-1]) for pid in panel_ids if registry.get_model(pid)]
    role_msg = f"**Roles assigned for this question:**\n"
    role_msg += f"- **Reasoner**: @{reasoner_name}\n"
    role_msg += f"- **Critic**: @{critic_name}\n"
    if panel_names:
        role_msg += f"- **Panel**: {', '.join('@' + n for n in panel_names)}\n"
    role_msg += f"- **Max rounds**: {max_rounds}\n"
    role_msg += f"\nStarting deliberation..."
    await _emit_system_msg(room_id, role_msg)

    reasoner_out = ""
    critic_out = ""
    previous_disagreements: list[str] = []
    agreement = False

    for round_idx in range(1, max_rounds + 1):
        await _emit_system_msg(room_id, f"**Round {round_idx}/{max_rounds}**")

        # Check for user input injected mid-deliberation
        user_injection = None
        try:
            user_injection = user_input_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # ── Reasoner phase ──
        await chat_room.broadcast(room_id, {"type": "typing", "agent": reasoner_name})

        if round_idx == 1:
            reasoner_prompt = (
                f"You are the reasoner in a multi-model deliberation. "
                f"Provide a careful analysis and propose a decision.\n"
                f"Be specific and prescriptive. Negotiate toward consensus.\n\n"
                f"Question: {query}\n"
            )
            if user_injection:
                reasoner_prompt += f"\nAdditional context from user: {user_injection}\n"
        else:
            reasoner_prompt = (
                f"You are the reasoner. Revise your draft to address the critic's feedback.\n\n"
                f"Question: {query}\n\n"
                f"Your previous draft:\n{reasoner_out[:3000]}\n\n"
                f"Critic feedback:\n{critic_out[:3000]}\n"
            )
            if previous_disagreements:
                reasoner_prompt += "\nUnresolved disagreements (address each):\n"
                for d in previous_disagreements[:6]:
                    reasoner_prompt += f"- {d}\n"
            if user_injection:
                reasoner_prompt += f"\nUser feedback: {user_injection}\n"
            reasoner_prompt += (
                "\nInclude a 'Resolution log' listing each prior disagreement as RESOLVED or ACCEPTED.\n"
                "If all issues are minor, state that clearly.\n"
            )

        try:
            reasoner_out = await _call_cli_model(reasoner_card, reasoner_prompt)
        except Exception:
            reasoner_out = ""

        await chat_room.broadcast(room_id, {"type": "typing_stop", "agent": reasoner_name})

        if not reasoner_out:
            await _emit_chat_msg(room_id, reasoner_name, f"*{reasoner_name} failed to respond this round.*", role="reasoner")
        else:
            await _emit_chat_msg(
                room_id, reasoner_name, reasoner_out, role="reasoner",
                quoting={"sender": "user", "content": query[:200]},
            )

        # ── Critic phase ──
        await chat_room.broadcast(room_id, {"type": "typing", "agent": critic_name})

        critic_prompt = (
            f"You are the critic. Challenge the reasoning, list disagreements, and suggest fixes.\n"
            f"Negotiate toward consensus. If remaining issues are minor, respond with AGREE.\n"
            f"List at most 3 disagreements; consolidate overlapping items.\n"
            f"Return sections:\nDisagreements:\n- ...\nGaps:\n- ...\nVerdict:\nAGREE or DISAGREE\n\n"
            f"Question: {query}\n\n"
            f"Reasoner draft:\n{reasoner_out[:3000]}\n"
        )
        if previous_disagreements:
            critic_prompt += "\nPrevious disagreements (only list if still unresolved):\n"
            for d in previous_disagreements[:6]:
                critic_prompt += f"- {d}\n"
            critic_prompt += "\nIf all prior disagreements are resolved, respond with AGREE.\n"

        try:
            critic_out = await _call_cli_model(critic_card, critic_prompt)
        except Exception:
            critic_out = ""

        await chat_room.broadcast(room_id, {"type": "typing_stop", "agent": critic_name})

        if not critic_out:
            await _emit_chat_msg(room_id, critic_name, f"*{critic_name} failed to respond this round.*", role="critic")
        else:
            await _emit_chat_msg(
                room_id, critic_name, critic_out, role="critic",
                quoting={"sender": reasoner_name, "content": reasoner_out[:200]},
            )

        # ── Check agreement ──
        agreement = _critic_agrees(critic_out) if critic_out else False
        disagreements = _extract_disagreements(critic_out) if critic_out else []
        previous_disagreements = disagreements

        # ── Panel review (if configured) ──
        panel_votes = []
        for panel_model_id in panel_ids:
            panel_card = registry.get_model(panel_model_id)
            if not panel_card:
                continue
            panel_agent = id_to_name.get(panel_model_id, panel_model_id.split(":")[-1])
            await chat_room.broadcast(room_id, {"type": "typing", "agent": panel_agent})

            panel_prompt = (
                f"You are a panel reviewer. Read the reasoner's draft and critic's feedback.\n"
                f"Do you agree with the reasoner's draft? Respond with AGREE or DISAGREE and brief reasoning.\n\n"
                f"Question: {query}\n\n"
                f"Reasoner draft:\n{reasoner_out[:2000]}\n\n"
                f"Critic feedback:\n{critic_out[:2000]}\n"
            )

            try:
                panel_out = await _call_cli_model(panel_card, panel_prompt)
            except Exception:
                panel_out = ""

            await chat_room.broadcast(room_id, {"type": "typing_stop", "agent": panel_agent})

            if panel_out:
                panel_agrees = _critic_agrees(panel_out)
                panel_votes.append(panel_agrees)
                await _emit_chat_msg(room_id, panel_agent, panel_out, role="panel")

        # ── Round verdict ──
        if panel_votes:
            agree_count = sum(1 for v in panel_votes if v)
            total = len(panel_votes)
            panel_agreement = agree_count / total >= 0.6 if total > 0 else False
            overall = agreement and panel_agreement
        else:
            overall = agreement

        if overall:
            await _emit_system_msg(
                room_id,
                f"**Consensus reached in round {round_idx}.** All parties agree."
            )
            break
        elif disagreements:
            summary = "\n".join(f"- {d}" for d in disagreements[:5])
            await _emit_system_msg(
                room_id,
                f"Round {round_idx} — **no consensus yet**. Disagreements:\n{summary}"
            )
        else:
            await _emit_system_msg(room_id, f"Round {round_idx} — continuing deliberation...")

        # Brief pause to let user inject feedback before next round
        try:
            user_injection = await asyncio.wait_for(user_input_queue.get(), timeout=3.0)
            if user_injection:
                await _emit_system_msg(room_id, f"User feedback received, incorporating into next round.")
        except asyncio.TimeoutError:
            pass

    if not agreement:
        await _emit_system_msg(
            room_id,
            f"**Deliberation ended after {max_rounds} rounds without full consensus.** "
            f"The reasoner's latest draft is the best-effort answer."
        )

    # Final summary
    await _emit_system_msg(room_id, "Deliberation complete.")


@app.get("/chat")
async def chat_page(request: Request):
    return TEMPLATES.TemplateResponse("chat.html", {"request": request})


@app.websocket("/ws/chat/{room_id}")
async def chat_websocket(websocket: WebSocket, room_id: str):
    if not _CHAT_ROOM_ID_RE.match(room_id):
        await websocket.close(code=1008)
        return
    if room_id not in chat_room.connections and len(chat_room.connections) >= _CHAT_MAX_ROOMS:
        await websocket.close(code=1008)
        return

    await chat_room.connect(websocket, room_id)
    user_input_queue: asyncio.Queue = asyncio.Queue()
    deliberation_task: asyncio.Task | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if data.get("type") != "message":
                continue

            content = (data.get("content") or "").strip()
            if not content or len(content) > _CHAT_MAX_MESSAGE_LEN:
                continue

            if not chat_room.check_rate(websocket):
                continue

            user_msg = {
                "id": f"msg-{_uuid.uuid4().hex[:12]}",
                "sender": "user",
                "content": content,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            chat_room.add_message(room_id, user_msg)
            await chat_room.broadcast(room_id, {"type": "message", **user_msg}, exclude=websocket)

            # If deliberation is running, inject user feedback into the current round
            if deliberation_task and not deliberation_task.done():
                user_input_queue.put_nowait(content)
                continue

            # Start a new deliberation
            registry: ModelRegistry = websocket.app.state.registry
            # Drain any stale queue items
            while not user_input_queue.empty():
                try:
                    user_input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            async def _guarded_deliberation(rid, query, reg, q):
                """Wrap deliberation with error handling and overall timeout."""
                try:
                    run_timeout = int(app.state.config.raw.get("pipeline", {}).get("run_timeout_seconds", 900))
                    await asyncio.wait_for(
                        _run_deliberation(rid, query, reg, q),
                        timeout=run_timeout,
                    )
                except asyncio.TimeoutError:
                    await _emit_system_msg(rid, "Deliberation timed out.")
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.exception("Deliberation failed")
                    try:
                        await _emit_system_msg(rid, f"Deliberation error: {type(exc).__name__}")
                    except Exception:
                        pass
                finally:
                    # Clear any stuck typing indicators
                    for agent_name in ("claude", "codex", "gemini"):
                        try:
                            await chat_room.broadcast(rid, {"type": "typing_stop", "agent": agent_name})
                        except Exception:
                            pass

            deliberation_task = asyncio.create_task(
                _guarded_deliberation(room_id, content, registry, user_input_queue)
            )

    except WebSocketDisconnect:
        if deliberation_task and not deliberation_task.done():
            deliberation_task.cancel()
        chat_room.disconnect(websocket, room_id)
    except Exception:
        if deliberation_task and not deliberation_task.done():
            deliberation_task.cancel()
        chat_room.disconnect(websocket, room_id)


def main():
    import uvicorn
    config = get_config()
    host = config.server.get("host", "127.0.0.1")
    port = int(config.server.get("port", 8099))
    uvicorn.run("conclave.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
