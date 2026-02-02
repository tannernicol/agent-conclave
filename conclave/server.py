"""FastAPI server for Conclave."""
from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
from typing import Dict, Set
import asyncio
import re
import json

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline
from conclave.store import DecisionStore
from conclave.models.registry import ModelRegistry

app = FastAPI(title="Conclave")

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

def _prompt_path(config, prompt_id: str) -> Path:
    return _prompts_dir(config) / f"{prompt_id}.json"

def _prompt_input_path(config, prompt_id: str) -> Path:
    return _inputs_dir(config) / f"prompt-{prompt_id}.md"

def _write_prompt_input(
    path: Path,
    title: str,
    question: str,
    notes: str,
    artifacts: list[str] | None = None,
    output_type: str | None = None,
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
    body.append("## Notes")
    body.append(notes or "")
    if artifacts:
        body.append("")
        body.append("## Artifacts")
        for item in artifacts:
            body.append(f"- {item}")
    path.write_text("\n".join(body).strip() + "\n")


@app.on_event("startup")
def _startup() -> None:
    config = get_config()
    app.state.config = config
    app.state.pipeline = ConclavePipeline(config)
    app.state.store = DecisionStore(config.data_dir)
    app.state.registry = ModelRegistry.from_config(config.models)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


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


@app.post("/api/run")
async def run_api(payload: dict, background: BackgroundTasks, request: Request):
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    collections = payload.get("collections")
    meta = {"source": "api"}
    input_title = (payload.get("input_title") or "").strip()
    if input_title:
        meta["input_title"] = input_title
    output_type = (payload.get("output_type") or "").strip()
    if output_type:
        meta["output_type"] = output_type
    prompt_id = (payload.get("prompt_id") or "").strip()
    if prompt_id:
        meta["prompt_id"] = prompt_id
    input_id = payload.get("input_id")
    input_path = payload.get("input_path")
    config = request.app.state.config
    store = request.app.state.store
    pipeline = request.app.state.pipeline
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
    body.append("## Notes")
    body.append(content)
    if artifacts:
        body.append("")
        body.append("## Artifacts")
        for item in artifacts:
            body.append(f"- {item}")
    path.write_text("\n".join(body).strip() + "\n")
    return {"ok": True, "input_id": filename, "path": str(path), "artifacts": artifacts}


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
    artifacts = payload.get("artifacts") or []
    artifacts = [str(item).strip() for item in artifacts if str(item).strip()]
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
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
    }
    _write_prompt_input(Path(prompt["input_path"]), title, query, notes, artifacts, output_type=output_type)
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
    artifacts = payload.get("artifacts")
    if artifacts is None:
        artifacts = prompt.get("artifacts") or []
    artifacts = [str(item).strip() for item in artifacts if str(item).strip()]
    prompt["title"] = title
    prompt["query"] = query
    prompt["notes"] = notes
    prompt["artifacts"] = artifacts
    prompt["output_type"] = output_type or None
    prompt["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    input_path = Path(prompt.get("input_path") or _prompt_input_path(config, prompt_id))
    prompt["input_path"] = str(input_path)
    _write_prompt_input(input_path, title, query, notes, artifacts, output_type=output_type)
    path.write_text(json.dumps(prompt, indent=2))
    return {"ok": True, "prompt": prompt}


@app.get("/api/prompts")
async def prompts_list_api(request: Request, limit: int = 50):
    config = request.app.state.config
    items = []
    for path in sorted(_prompts_dir(config).glob("*.json"), reverse=True):
        try:
            prompt = json.loads(path.read_text())
        except Exception:
            continue
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
    run_id = request.app.state.store.create_run(query, meta=meta)
    def _task():
        request.app.state.pipeline.run(query, collections=None, run_id=run_id, meta=meta)
    background.add_task(_task)
    prompt["last_run_id"] = run_id
    prompt["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path.write_text(json.dumps(prompt, indent=2))
    return {"ok": True, "run_id": run_id, "prompt": prompt}


def main():
    import uvicorn
    config = get_config()
    host = config.server.get("host", "127.0.0.1")
    port = int(config.server.get("port", 8099))
    uvicorn.run("conclave.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
