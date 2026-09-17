# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Local visual dashboard for exocomet-hunter.

Wraps the existing pipeline (``scripts/ingest.py``, ``src/exocomet/``) with a
FastAPI backend and a small custom frontend. Never a second implementation of
the detection pipeline -- every real fetch/detrend/detect call here goes
through the same functions the real automation uses.

Deliberately separate from production state: everything this package writes
(run history, extracted light curves, results, cached sky images) lives under
``dashboard/var/`` by default, never in the real ``exocomet-hunter-data``
clone or the real ``local.db``. See ``dashboard/storage.py``.
"""

from __future__ import annotations

from pathlib import Path

#: Repo root, resolved from this file's location (not the process CWD) --
#: same reasoning as scripts/ingest.py's own raw_cache_dir computation.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: All dashboard-local state lives under here. Never committed (see
#: .gitignore), never read by the real automation.
DASHBOARD_VAR_DIR = Path(__file__).resolve().parent / "var"
