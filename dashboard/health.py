# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Real boot-time health checks for the dashboard's status sequence.

Every check here inspects actual on-disk state or imports actual runtime
modules -- nothing here is decorative, unlike the earlier boot-animation
mock's fake "OK" lines. A failing check can genuinely be repaired: files
tracked in this git repo are restored via `git checkout` (falling back to
`git fetch origin` if the local history doesn't have them), and state this
package generates itself (the run database, the image cache directory, a
corrupt config override) is regenerated from scratch. Nothing here ever
fetches from an untracked external source -- this is a single-user local
tool, and the only real source of "missing" tracked files is this repo.
"""

from __future__ import annotations

import importlib
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from dashboard import REPO_ROOT, config_editor, storage
from dashboard.images import IMAGE_CACHE_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"

_STATIC_REQUIRED = ["emblem.svg", "favicon.svg", "app.js", "style.css", "index.html"]

# (repo-relative path, importable module name) -- these are lazily imported
# at request time elsewhere (see server.py/runner.py), not at server startup,
# so unlike the eagerly-imported dashboard/* package they can genuinely be
# missing or broken without the server itself failing to boot.
_MODULE_REQUIRED = [
    ("scripts/ingest.py", "ingest"),
    ("scripts/build_kepler_host_list.py", "build_kepler_host_list"),
    ("src/exocomet/io/extract.py", "exocomet.io.extract"),
]


@dataclass
class Check:
    id: str
    label: str
    ok: bool
    fix: str | None = None  # "git" | "regenerate" | None (unfixable)
    path: str | None = None  # repo-relative path, used by the "git" fix
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_checks() -> list[Check]:
    checks: list[Check] = [Check(id="api", label="BACKEND API", ok=True)]

    for name in _STATIC_REQUIRED:
        p = STATIC_DIR / name
        checks.append(
            Check(
                id=f"static:{name}",
                label=f"STATIC ASSET — {name}",
                ok=p.exists(),
                fix="git" if not p.exists() else None,
                path=str(p.relative_to(REPO_ROOT)),
            )
        )

    watchlist = REPO_ROOT / "config" / "watchlist.txt"
    checks.append(
        Check(
            id="target_lists:watchlist",
            label="TARGET LIST — TESS WATCHLIST",
            ok=watchlist.exists(),
            fix="git" if not watchlist.exists() else None,
            path=str(watchlist.relative_to(REPO_ROOT)),
        )
    )

    for rel_path, module_name in _MODULE_REQUIRED:
        p = REPO_ROOT / rel_path
        exists = p.exists()
        importable = False
        detail = ""
        if exists:
            try:
                importlib.import_module(module_name)
                importable = True
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                detail = str(exc)
        checks.append(
            Check(
                id=f"module:{module_name}",
                label=f"BACKEND MODULE — {module_name}",
                ok=exists and importable,
                fix="git" if not exists else None,
                path=rel_path,
                detail=detail,
            )
        )

    thresholds = REPO_ROOT / "config" / "thresholds.yaml"
    config_ok, config_detail = True, ""
    try:
        config_editor.load_effective_config()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        config_ok, config_detail = False, str(exc)
    checks.append(
        Check(
            id="config",
            label="PIPELINE CONFIG",
            ok=thresholds.exists() and config_ok,
            fix="git" if not thresholds.exists() else ("regenerate" if not config_ok else None),
            path=str(thresholds.relative_to(REPO_ROOT)),
            detail=config_detail,
        )
    )

    db_ok = storage.DB_PATH.exists()
    if db_ok:
        try:
            conn = sqlite3.connect(storage.DB_PATH)
            conn.execute("SELECT 1 FROM runs LIMIT 1")
            conn.close()
        except Exception:  # noqa: BLE001
            db_ok = False
    checks.append(
        Check(
            id="database",
            label="RUN DATABASE",
            ok=db_ok,
            fix="regenerate" if not db_ok else None,
            path=str(storage.DB_PATH.relative_to(REPO_ROOT)),
        )
    )

    cache_ok = IMAGE_CACHE_DIR.is_dir()
    checks.append(
        Check(
            id="cache_dir",
            label="IMAGE CACHE DIRECTORY",
            ok=cache_ok,
            fix="regenerate" if not cache_ok else None,
            path=str(IMAGE_CACHE_DIR.relative_to(REPO_ROOT)),
        )
    )

    return checks


def _git_restore(rel_path: str) -> tuple[bool, str]:
    """Restore one tracked file via git. Scoped to a single path -- never
    touches anything else in the working tree. Tries local history first;
    only reaches out to `origin` if the file truly isn't there yet."""

    def checkout(rev: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "checkout", rev, "--", rel_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    result = checkout("HEAD")
    if result.returncode == 0:
        return True, "restored from local git history"

    fetch = subprocess.run(
        ["git", "fetch", "origin"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
    )
    if fetch.returncode != 0:
        return False, f"git fetch failed: {fetch.stderr.strip()[:200]}"

    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    branch = branch_result.stdout.strip() or "main"
    result = checkout(f"origin/{branch}")
    if result.returncode == 0:
        return True, f"restored from origin/{branch}"
    return False, f"git checkout failed: {result.stderr.strip()[:200]}"


def _regenerate(check_id: str) -> tuple[bool, str]:
    if check_id == "database":
        storage.init_db()
        return storage.DB_PATH.exists(), "reinitialized database schema"
    if check_id == "cache_dir":
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return IMAGE_CACHE_DIR.is_dir(), "created cache directory"
    if check_id == "config":
        if config_editor.OVERRIDE_PATH.exists():
            config_editor.OVERRIDE_PATH.unlink()
        try:
            config_editor.load_effective_config()
            return True, "cleared invalid override, fell back to defaults"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
    return False, "no regeneration path for this check"


def repair(checks: list[Check]) -> Iterator[dict[str, Any]]:
    """Repair every failing, fixable check in order, yielding one real
    event per file as its fix actually completes -- no artificial pacing,
    the timing reflects the real git/filesystem operation."""
    failing = [c for c in checks if not c.ok and c.fix]
    for check in failing:
        yield {"type": "repair_start", "id": check.id, "label": check.label, "path": check.path}
        if check.fix == "git":
            ok, detail = _git_restore(check.path)
        else:
            ok, detail = _regenerate(check.id)
        yield {"type": "repair_done", "id": check.id, "ok": ok, "detail": detail}
    yield {"type": "repair_complete"}
