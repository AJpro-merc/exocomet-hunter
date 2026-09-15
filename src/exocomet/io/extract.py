# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Serialize and deserialize :class:`LightCurveData` to/from a local cache.

A light curve fetched from MAST is expensive (network round trip, minutes of
wall clock, see :mod:`exocomet.io.download`) but cheap to store: a few
megabytes of arrays. Once :func:`~exocomet.io.download.fetch_light_curve` has
produced a :class:`~exocomet.core.types.LightCurveData`, this module lets a
resumable pipeline write it to disk exactly once and never re-download it,
even if the process that downloaded it is killed before it finishes using the
data.

This is a straight serialize/deserialize of the type as it already exists --
no separate "raw" or "pre-filter" representation. ``time``/``flux``/
``flux_err`` become parquet columns; ``target_id``, ``mission`` and ``meta``
ride along as parquet file-level (schema) metadata rather than a sidecar file,
because keeping everything in one file means there is only ever one thing to
write atomically and only one thing that can go missing or drift out of sync.
The trade-off is that ``meta`` must be JSON-serializable (already true of
every producer in this codebase -- see ``_to_light_curve_data`` in
:mod:`exocomet.io.download`), and pyarrow is used directly here (not pandas'
``to_parquet``) because schema metadata does not reliably survive a
``pandas.DataFrame.to_parquet`` round trip.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from exocomet.core.types import LightCurveData

__all__ = [
    "extract_path",
    "load_extract",
    "sanitize_target_id",
    "save_extract",
]

#: Schema-metadata keys. Namespaced so this file's metadata never collides
#: with anything pyarrow/pandas itself might stash on the schema.
_META_TARGET_ID = b"exocomet.target_id"
_META_MISSION = b"exocomet.mission"
_META_META_JSON = b"exocomet.meta_json"


def sanitize_target_id(target_id: str) -> str:
    """Make a catalogue identifier safe to use as a filename.

    Target IDs like ``"KIC 3542116"`` contain a space, which is fine in most
    filesystems but awkward to shell-quote and easy to trip over in tooling.
    Spaces and path separators are replaced with underscores.

    This is *not* guaranteed to be reversible on its own (e.g. a target ID
    that already contains an underscore is indistinguishable from one that
    had a space there) -- and it does not need to be: the original,
    unsanitized ``target_id`` is always written into the file's own metadata
    by :func:`save_extract`, so :func:`load_extract` recovers the exact
    string regardless of how the file was named on disk. Sanitization here is
    purely for choosing a safe path, never the source of truth.
    """
    return target_id.replace(" ", "_").replace("/", "_").replace("\\", "_")


def extract_path(base_dir: Path | str, mission: str, target_id: str) -> Path:
    """Compute the standard cache location for one target's extract.

    Layout: ``<base_dir>/extract/<mission>/<sanitized target_id>.parquet``.
    Callers (e.g. ``scripts/ingest.py``) should use this rather than
    inventing their own path, so every producer and consumer agrees on where
    a given target's extract lives.
    """
    return Path(base_dir) / "extract" / mission / f"{sanitize_target_id(target_id)}.parquet"


def save_extract(data: LightCurveData, path: Path) -> None:
    """Write ``data`` to ``path`` as parquet, atomically.

    Atomicity: the table is written to a temporary file in the *same*
    directory as ``path`` (so the later rename stays on one filesystem, which
    is required for atomicity) and then moved into place with
    :func:`os.replace`. ``os.replace`` is atomic on POSIX -- which covers
    both GitHub Actions runners and macOS, the two environments this pipeline
    runs on -- so a reader can only ever observe ``path`` as either
    completely absent or completely written; there is no window in which a
    half-written file is visible at the final name. If anything raises before
    the replace completes (a full disk, a killed process that still runs its
    ``except`` clause, etc.) the temporary file is removed so it can never be
    mistaken for a real, complete extract.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "time": pa.array(data.time, type=pa.float64()),
            "flux": pa.array(data.flux, type=pa.float64()),
            "flux_err": pa.array(data.flux_err, type=pa.float64()),
        }
    )
    table = table.replace_schema_metadata(
        {
            _META_TARGET_ID: data.target_id.encode("utf-8"),
            _META_MISSION: data.mission.encode("utf-8"),
            _META_META_JSON: json.dumps(data.meta, default=str).encode("utf-8"),
        }
    )

    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp", prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX (GitHub Actions runners, macOS)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_extract(path: Path) -> LightCurveData:
    """Read a :class:`LightCurveData` previously written by :func:`save_extract`.

    Reconstructs ``target_id``, ``mission`` and ``meta`` from the file's
    schema metadata (see the module docstring for why they live there rather
    than in a sidecar) and ``time``/``flux``/``flux_err`` from the parquet
    columns. The result round-trips exactly: identical arrays (``float64``
    in, ``float64`` out, no precision loss from parquet) and an identical
    ``meta`` dict for any JSON-serializable input.
    """
    path = Path(path)
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}

    target_id = metadata.get(_META_TARGET_ID, b"").decode("utf-8")
    mission = metadata.get(_META_MISSION, b"").decode("utf-8")
    meta = json.loads(metadata.get(_META_META_JSON, b"{}").decode("utf-8"))

    return LightCurveData(
        target_id=target_id,
        mission=mission,
        time=table.column("time").to_numpy(zero_copy_only=False),
        flux=table.column("flux").to_numpy(zero_copy_only=False),
        flux_err=table.column("flux_err").to_numpy(zero_copy_only=False),
        meta=meta,
    )
