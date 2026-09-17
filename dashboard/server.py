# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI app: routes, SSE streaming, static frontend.

Binds to 127.0.0.1 only (see scripts/run_dashboard.py) -- this is a
single-user local tool, not a shared service.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard import REPO_ROOT, config_editor, health, reports, storage
from dashboard.injection_runner import InjectionParams
from dashboard.runner import get_run_state, start_run, stop_run, stream_events
from exocomet.core.config import PipelineConfig

_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

storage.init_db()

app = FastAPI(title="Exocomet Hunter Dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Force revalidation on every /static/* request.

    This is a local single-user dev tool that changes constantly; a stale
    cached app.js/style.css after an edit is a real, recurring footgun, not
    just a test artifact. ETag-based revalidation is still cheap locally."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache"
    return response


_ASSET_REF = re.compile(r'(href|src)="(/static/[^"]+)"')


@app.get("/")
def index() -> HTMLResponse:
    """Serve index.html with each local /static/* reference cache-busted by
    that file's own mtime -- a stale cached app.js/style.css after an edit
    is a real, recurring footgun for this constantly-changing local tool."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def bust(match: re.Match[str]) -> str:
        attr, ref = match.group(1), match.group(2)
        asset_path = STATIC_DIR / ref.removeprefix("/static/")
        if not asset_path.is_file():
            return match.group(0)
        return f'{attr}="{ref}?v={int(asset_path.stat().st_mtime)}"'

    return HTMLResponse(_ASSET_REF.sub(bust, html))


# -- Health (boot-sequence status checks) --------------------------------


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    checks = health.run_checks()
    return {
        "all_ok": all(c.ok for c in checks),
        "checks": [c.to_dict() for c in checks],
    }


@app.get("/api/health/repair")
async def repair_health() -> StreamingResponse:
    checks = health.run_checks()

    async def event_stream() -> AsyncIterator[str]:
        for event in health.repair(checks):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# -- Target lists --------------------------------------------------------


@app.get("/api/target-lists")
def target_lists() -> dict[str, list[str]]:
    from ingest import get_target_list  # noqa: PLC0415

    return {
        "Kepler": get_target_list("Kepler"),
        "TESS": get_target_list("TESS"),
    }


# -- Config (Research mode) ----------------------------------------------


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return config_editor.get_form_state()


class ConfigUpdate(BaseModel):
    sections: dict[str, dict[str, Any]]


@app.post("/api/config")
def update_config(body: ConfigUpdate) -> dict[str, Any]:
    try:
        return config_editor.save_overrides(body.sections)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/config/reset")
def reset_config() -> dict[str, Any]:
    return config_editor.reset_to_defaults()


# -- Runs ------------------------------------------------------------------


class StartRunRequest(BaseModel):
    mode: str  # "normal" | "research"
    mission: str  # "Kepler" | "TESS"
    targets: list[str] | None = None  # None => use the full real target list
    max_targets: int | None = None
    injected: bool = False
    injection_depth: float = 0.001
    injection_ingress_days: float = 0.3
    injection_egress_days: float = 1.0
    # Research mode only: apply these edited values to this run alone,
    # bypassing config_editor.load_effective_config() and the persisted
    # override file entirely. Never touches dashboard/var/config_override.yaml.
    inline_config: dict[str, dict[str, Any]] | None = None


@app.post("/api/runs")
async def create_run(body: StartRunRequest) -> dict[str, str]:
    from ingest import get_target_list  # noqa: PLC0415

    if body.mode not in ("normal", "research"):
        raise HTTPException(status_code=400, detail="mode must be 'normal' or 'research'")
    if body.mission not in ("Kepler", "TESS"):
        raise HTTPException(status_code=400, detail="mission must be 'Kepler' or 'TESS'")

    targets = body.targets if body.targets else get_target_list(body.mission)
    if body.max_targets is not None:
        targets = targets[: body.max_targets]
    if not targets:
        raise HTTPException(status_code=400, detail="no targets selected")

    if body.mode == "research" and body.inline_config is not None:
        try:
            config = PipelineConfig.from_dict(body.inline_config)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid inline_config: {exc}") from exc
    else:
        config = config_editor.load_effective_config()

    injection_params = None
    if body.injected:
        injection_params = InjectionParams(
            depth=body.injection_depth,
            ingress_duration_days=body.injection_ingress_days,
            egress_duration_days=body.injection_egress_days,
        )

    run_id = start_run(body.mode, body.mission, targets, config, body.injected, injection_params)
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/stop")
def stop(run_id: str, force: bool = False) -> dict[str, str]:
    try:
        stop_run(run_id, force=force)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown run") from exc
    return {"status": "stopping" if not force else "force_stopping"}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return storage.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")
    run["config"] = json.loads(run["config_json"])
    run["target_list"] = json.loads(run["target_list_json"])
    return {
        "run": run,
        "targets": storage.get_run_targets(run_id),
        "candidates": storage.get_candidates(run_id),
    }


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    if get_run_state(run_id) is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return StreamingResponse(stream_events(run_id), media_type="text/event-stream")


@app.get("/api/images/{target_id}")
def get_image(target_id: str) -> FileResponse:
    from dashboard.images import cached_image_path

    path = cached_image_path(target_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="no cached image")
    return FileResponse(path, media_type="image/png")


# -- Reports & exports -------------------------------------------------------


@app.get("/api/runs/{run_id}/report", response_class=FileResponse)
def run_report(run_id: str) -> FileResponse:
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")
    path = reports.build_run_report(run_id)
    return FileResponse(path, media_type="text/html", filename=path.name)


@app.get("/api/runs/{run_id}/targets/{target_id}/report", response_class=FileResponse)
def target_report(run_id: str, target_id: str) -> FileResponse:
    if storage.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="unknown run")
    try:
        path = reports.build_target_report(run_id, target_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"could not build report: {exc}") from exc
    return FileResponse(path, media_type="text/html", filename=path.name)


@app.get("/api/runs/{run_id}/export.csv", response_class=PlainTextResponse)
def export_csv(run_id: str) -> PlainTextResponse:
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")
    filename = f"INJECTED_{run_id}.csv" if run["injected"] else f"{run_id}.csv"
    return PlainTextResponse(
        reports.export_csv(run_id), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/runs/{run_id}/export.json", response_class=PlainTextResponse)
def export_json(run_id: str) -> PlainTextResponse:
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")
    filename = f"INJECTED_{run_id}.json" if run["injected"] else f"{run_id}.json"
    return PlainTextResponse(
        reports.export_json(run_id), media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
