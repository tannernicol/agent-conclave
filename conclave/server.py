"""FastAPI server for Conclave."""
from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional

from conclave.config import get_config
from conclave.pipeline import ConclavePipeline
from conclave.store import DecisionStore
from conclave.models.registry import ModelRegistry

app = FastAPI(title="Conclave")
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

config = get_config()
pipeline = ConclavePipeline(config)
store = DecisionStore(config.data_dir)
registry = ModelRegistry.from_config(config.models)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def status_api():
    latest = store.latest()
    return {
        "ok": True,
        "latest": latest,
    }


@app.get("/api/models")
async def models_api():
    return {"models": registry.list_models()}


@app.post("/api/run")
async def run_api(payload: dict, background: BackgroundTasks):
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    collections = payload.get("collections")
    run_id = store.create_run(query, meta={"source": "api"})
    def _task():
        pipeline.run(query, collections=collections, run_id=run_id)
    background.add_task(_task)
    return {"ok": True, "run_id": run_id}


@app.get("/api/runs")
async def runs_api(limit: int = 20):
    return {"runs": store.list_runs(limit=limit)}


@app.get("/api/runs/latest")
async def runs_latest_api():
    return store.latest() or {}


@app.get("/api/runs/{run_id}")
async def run_detail_api(run_id: str):
    run = store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not found"}, status_code=404)
    return run


@app.post("/api/reconcile")
async def reconcile_api(payload: dict, background: BackgroundTasks):
    topic = payload.get("topic")
    if not topic:
        return JSONResponse({"error": "topic required"}, status_code=400)
    def _task():
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


def main():
    import uvicorn
    host = config.server.get("host", "127.0.0.1")
    port = int(config.server.get("port", 8099))
    uvicorn.run("conclave.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
