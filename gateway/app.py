"""HTTP API and test page (DESIGN.md 3)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .service import Service

log = logging.getLogger("gateway")
STATIC = Path(__file__).parent / "static"


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _every(seconds: float, tick) -> None:
    while True:
        try:
            await tick()
        except Exception:  # keep the loop alive; the next tick retries
            log.exception("%s failed", tick.__name__)
        await asyncio.sleep(seconds)


def create_app(service: Service, run_loops: bool = True) -> FastAPI:
    c = service.c

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        loops = []
        if run_loops:
            loops = [asyncio.create_task(_every(c.batcher_interval_s, service.batcher_tick)),
                     asyncio.create_task(_every(c.poll_interval_s, service.poller_tick))]
        yield
        for task in loops:
            task.cancel()
        service.db.close()

    app = FastAPI(title="sd_service gateway", lifespan=lifespan)
    c.results_dir.mkdir(parents=True, exist_ok=True)

    @app.post("/api/images/gen", status_code=202)
    async def submit(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error(400, "body must be JSON")
        tasks = body.get("tasks") if isinstance(body, dict) else None
        if not isinstance(tasks, list) or not tasks:
            return error(400, 'body must be {"tasks": [...]} with at least one task')
        if len(tasks) > c.max_tasks_per_request:
            return error(400, f"at most {c.max_tasks_per_request} tasks per request")
        return {"tasks": service.submit(tasks)}

    @app.get("/api/images/tasks/{taskid}")
    async def task_status(taskid: str):
        task = service.db.get_task(taskid)
        if task is None:
            return error(404, "taskid not found")
        return service.task_view(task)

    @app.get("/api/images/tasks")
    async def bulk_status(ids: str = ""):
        wanted = [i for i in ids.split(",") if i]
        if not wanted:
            return error(400, "ids query parameter is required, e.g. ?ids=a,b,c")
        if len(wanted) > c.max_ids_per_status:
            return error(400, f"at most {c.max_ids_per_status} ids per call")
        found = service.db.get_tasks(wanted)
        return {"tasks": [service.task_view(found[i]) if i in found else {"taskid": i, "result": "not_found"}
                          for i in wanted]}

    @app.get("/api/health")
    async def health():
        return service.health()

    @app.get("/", include_in_schema=False)
    async def test_page():
        return FileResponse(STATIC / "index.html")

    app.mount("/results", StaticFiles(directory=c.results_dir), name="results")
    return app
