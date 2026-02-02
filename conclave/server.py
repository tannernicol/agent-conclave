"""FastAPI server for Conclave."""
from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
import re

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline
from conclave.store import DecisionStore
from conclave.models.registry import ModelRegistry

app = FastAPI(title="Conclave")
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
    body.append("## Notes")
    body.append(content)
    path.write_text("\n".join(body).strip() + "\n")
    return {"ok": True, "input_id": filename, "path": str(path)}


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


def main():
    import uvicorn
    config = get_config()
    host = config.server.get("host", "127.0.0.1")
    port = int(config.server.get("port", 8099))
    uvicorn.run("conclave.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
