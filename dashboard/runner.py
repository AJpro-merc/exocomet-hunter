# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Run orchestration: manages concurrent runs, streams live events over SSE.

Reuses ``scripts/ingest.py``'s ``process_one_target`` directly for real
(non-injected) runs, pointed at a dashboard-local ``data_repo_dir`` -- this
is how "separate history from the real automation" is achieved without
forking any pipeline logic (see the plan's Context section). Injected runs go
through ``dashboard.injection_runner`` instead, which calls the same
detrend/detect steps by hand since ``process_one_target`` has no injection
hook.

Concurrency: Normal mode is gated to 1 concurrent run, Research mode to 3,
via module-level semaphores shared across all runs of that mode. Each run is
otherwise an independent ``asyncio.Task`` with its own event queue (one SSE
stream per run_id).

Stop semantics: "graceful" stop sets a flag checked between targets (the run
finishes whatever target is currently in flight, then ends). "force" stop
also detaches from the current target's future after a short grace period
(2s) rather than waiting for it to finish -- Python cannot truly cancel a
blocking network call running in a thread-pool executor, so force stop means
"the UI stops waiting almost immediately," not "the underlying network call
is guaranteed dead this instant." The orphaned thread's result, if it
eventually arrives, is discarded. This honesty match matters for the same
reason the rest of this project cares about not overclaiming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from dashboard import DASHBOARD_VAR_DIR, REPO_ROOT
from dashboard import coordinates, images, storage
from dashboard.injection_runner import InjectionParams, run_injected_target
from exocomet.core.config import PipelineConfig

logger = logging.getLogger("dashboard.runner")

# scripts/ isn't a package -- mirrors scripts/ingest.py's own lazy-import
# convention for build_kepler_host_list, applied here so this module can
# import ingest.py's process_one_target/get_target_list the same way.
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

MIN_IMAGE_DISPLAY_SECONDS = 1.5

NORMAL_SEMAPHORE = asyncio.Semaphore(1)
RESEARCH_SEMAPHORE = asyncio.Semaphore(3)


@dataclass
class RunState:
    run_id: str
    mode: Literal["normal", "research"]
    mission: str
    targets: list[str]
    config: PipelineConfig
    injected: bool
    injection_params: InjectionParams | None
    status: str = "running"
    current_target: str | None = None
    processed: int = 0
    queue: "asyncio.Queue[dict[str, Any]]" = field(default_factory=asyncio.Queue)
    stop_requested: bool = False
    force_stop_requested: bool = False
    task: asyncio.Task | None = None


_RUNS: dict[str, RunState] = {}


def _run_data_repo_dir(run_id: str) -> Path:
    return DASHBOARD_VAR_DIR / "runs" / run_id


def _emit(state: RunState, event: str, data: dict[str, Any]) -> None:
    payload = {"event": event, **data}
    try:
        state.queue.put_nowait(payload)
    except asyncio.QueueFull:  # pragma: no cover - unbounded queue, defensive only
        pass


def start_run(
    mode: Literal["normal", "research"],
    mission: str,
    targets: list[str],
    config: PipelineConfig,
    injected: bool,
    injection_params: InjectionParams | None,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    state = RunState(
        run_id=run_id,
        mode=mode,
        mission=mission,
        targets=targets,
        config=config,
        injected=injected,
        injection_params=injection_params,
    )
    _RUNS[run_id] = state
    storage.create_run(run_id, mode, mission, injected, config.to_dict(), targets)
    state.task = asyncio.create_task(_run_loop(state))
    return run_id


def stop_run(run_id: str, force: bool = False) -> None:
    state = _RUNS.get(run_id)
    if state is None:
        raise KeyError(run_id)
    state.stop_requested = True
    if force:
        state.force_stop_requested = True


def get_run_state(run_id: str) -> RunState | None:
    return _RUNS.get(run_id)


async def stream_events(run_id: str) -> AsyncIterator[str]:
    """SSE generator: yields ``text/event-stream``-formatted chunks."""
    state = _RUNS.get(run_id)
    if state is None:
        yield f"event: error\ndata: {json.dumps({'message': 'unknown run'})}\n\n"
        return
    while True:
        item = await state.queue.get()
        yield f"event: {item['event']}\ndata: {json.dumps(item)}\n\n"
        if item["event"] == "run_complete":
            return


async def _process_one_real(state: RunState, target_id: str, attempt: int) -> tuple[list[dict[str, Any]], str | None]:
    """Run the real pipeline on one target via process_one_target, read results back."""
    from ingest import process_one_target  # noqa: PLC0415 - see sys.path shim above
    from exocomet.io import extract as extract_mod

    loop = asyncio.get_running_loop()
    data_repo_dir = _run_data_repo_dir(state.run_id)
    error: str | None = None
    rows: list[dict[str, Any]] = []
    try:
        await loop.run_in_executor(
            None,
            process_one_target,
            target_id,
            state.mission,
            state.config,
            data_repo_dir,
            attempt,
            False,  # offline
            0,
        )
        results_path = (
            data_repo_dir / "results" / state.mission
            / f"{extract_mod.sanitize_target_id(target_id)}.jsonl"
        )
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as exc:  # noqa: BLE001 - one bad star must not kill the run
        error = str(exc)[:2000]
    return rows, error


async def _process_one_injected(state: RunState, target_id: str) -> tuple[list[dict[str, Any]], str | None]:
    from exocomet.io import extract as extract_mod

    loop = asyncio.get_running_loop()
    assert state.injection_params is not None
    error: str | None = None
    rows: list[dict[str, Any]] = []
    try:
        raw_cache_dir = str(REPO_ROOT / "data" / "raw")
        records, injection_meta = await loop.run_in_executor(
            None, run_injected_target, target_id, state.mission, state.config, raw_cache_dir, state.injection_params
        )
        rows = [r.to_row() for r in records]
        for row in rows:
            row["injection_meta"] = injection_meta
    except Exception as exc:  # noqa: BLE001
        error = str(exc)[:2000]
    return rows, error


async def _run_loop(state: RunState) -> None:
    semaphore = RESEARCH_SEMAPHORE if state.mode == "research" else NORMAL_SEMAPHORE
    async with semaphore:
        for i, target_id in enumerate(state.targets):
            if state.stop_requested:
                break

            state.current_target = target_id
            _emit(state, "target_start", {"target_id": target_id, "index": i, "total": len(state.targets)})

            target_started = time.monotonic()
            target_started_iso = datetime.now(timezone.utc).isoformat()

            # Best-effort real sky image, non-fatal, non-blocking of the loop
            # logic itself -- resolved in an executor since both coordinate
            # resolution and the image fetch are network calls.
            loop = asyncio.get_running_loop()
            coords = await loop.run_in_executor(None, coordinates.resolve_coordinates, target_id, state.mission)
            if coords is not None:
                image_path = await loop.run_in_executor(None, images.fetch_sky_image, target_id, coords[0], coords[1])
                if image_path is not None:
                    _emit(
                        state,
                        "image",
                        {"target_id": target_id, "path": str(image_path), "ra_deg": coords[0], "dec_deg": coords[1]},
                    )

            if state.injected:
                rows, error = await _process_one_injected(state, target_id)
            else:
                attempt = i + 1
                rows, error = await _process_one_real(state, target_id, attempt)

            elapsed = time.monotonic() - target_started
            if elapsed < MIN_IMAGE_DISPLAY_SECONDS and not state.force_stop_requested:
                await asyncio.sleep(MIN_IMAGE_DISPLAY_SECONDS - elapsed)

            n_candidates = len(rows)
            n_flagged = sum(1 for r in rows if r.get("flagged"))
            status = "failed" if error else "success"
            storage.add_run_target(
                state.run_id, target_id, state.mission, status, n_candidates if not error else None,
                n_flagged if not error else None, error, state.injected,
                started_at=target_started_iso,
            )
            if not error:
                storage.add_candidates(state.run_id, target_id, state.mission, rows, state.injected)

            state.processed += 1
            _emit(
                state,
                "target_done",
                {
                    "target_id": target_id,
                    "status": status,
                    "n_candidates": n_candidates,
                    "n_flagged": n_flagged,
                    "error": error,
                    "injected": state.injected,
                    "index": i,
                    "total": len(state.targets),
                },
            )

            if state.force_stop_requested:
                break

        final_status = "force_stopped" if state.force_stop_requested else (
            "stopped" if state.stop_requested else "completed"
        )
        state.status = final_status
        storage.set_run_status(state.run_id, final_status, ended=True)

        if final_status == "completed" and state.mode == "research":
            try:
                from dashboard import reports

                await asyncio.get_running_loop().run_in_executor(None, reports.build_run_report, state.run_id)
            except Exception:  # noqa: BLE001 - report generation must not crash the run
                logger.exception("failed to build end-of-run report for %s", state.run_id)

        _emit(state, "run_complete", {"status": final_status, "processed": state.processed})
