# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Append-only, crash-safe record of what happened to each target.

Replaces the ``results/processed_ledger.json`` pattern used by
``scripts/monitor_new_data.py`` (one JSON object, rewritten whole at the end
of a run) for the resumable ingest pipeline. That pattern loses every
completed record the instant the process is killed mid-run, because the
*previous* good state only exists in memory until the final rewrite lands.
Under a 30-minute GitHub Actions budget where the runner can be killed at any
moment, that is not acceptable: this module instead opens the ledger file in
``"a"`` mode and appends one JSON line per event, so a completed target's
record is durable the moment its ``append_entry`` call returns -- there is no
"rewrite the whole file" step for a kill to interrupt. The house style this
follows is ``scripts/train_classifier.py``'s ``manifest_<mission>.jsonl``
(open in append mode, write ``json.dumps(entry)`` plus a trailing newline).

Because it is append-only, a target that is retried appears multiple times.
:func:`read_ledger` returns the raw flat list; :func:`latest_entries` reduces
it to one entry per target, which is what most callers actually want.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = [
    "LedgerEntry",
    "append_entry",
    "get_status",
    "latest_entries",
    "next_targets",
    "read_ledger",
]

Status = Literal["success", "failed"]


@dataclass(frozen=True)
class LedgerEntry:
    """One event in the ledger: one attempt at processing one target.

    Parameters
    ----------
    target_id, mission
        Together identify the star, matching :class:`LightCurveData`.
    status
        ``"success"`` or ``"failed"``.
    timestamp_utc
        ISO-8601 UTC timestamp of when this attempt concluded.
    attempt
        1-indexed count of how many times this target has now been attempted,
        including this one. Set by the caller (``scripts/ingest.py``), which
        is expected to look up the previous attempt count (e.g. via
        :func:`get_status` / :func:`latest_entries`) and increment it -- the
        ledger itself does not track attempt numbers independently of what
        callers report, so :func:`next_targets` trusts this field rather than
        recomputing it by counting lines.
    error
        Exception message on failure, ``None`` on success.
    extract_path, results_path
        Where this attempt's outputs live, so a later query can find "what
        output belongs to this star" without guessing a path convention.
        ``extract_path`` is expected to match
        :func:`exocomet.io.extract.extract_path`; ``results_path`` is
        whatever downstream candidate/results file this attempt wrote to.
        Both are ``None`` when not applicable (e.g. a failure before any
        output was produced).
    """

    target_id: str
    mission: str
    status: Status
    timestamp_utc: str
    attempt: int
    error: str | None = None
    extract_path: str | None = None
    results_path: str | None = None


def append_entry(ledger_path: Path, entry: LedgerEntry) -> None:
    r"""Append one entry to the ledger as a single JSON line.

    Durability: opens in ``"a"`` mode (POSIX guarantees appended bytes are
    added at the current end-of-file even with concurrent writers, so lines
    from different processes never interleave mid-line as long as each write
    stays under the OS's atomic-write size) and writes exactly one line, then
    explicitly ``flush()``s the Python-level buffer to the OS and
    ``os.fsync()``s the file descriptor. ``flush()`` alone is not enough:
    it moves bytes out of Python's ``io`` buffer but the OS is still free to
    hold them in its page cache indefinitely. ``fsync`` forces them to the
    physical device, so once ``append_entry`` returns, the record survives
    not just a killed Python process but a killed/rebooted runner. A single
    ledger line for this schema is well under a kilobyte, comfortably inside
    every practical atomic-write size, so no partial line can ever land in
    the file even without an fsync -- the fsync is purely about durability
    (surviving a hard kill), not atomicity (surviving a torn write).

    Guard against a *previous* torn write: if the process that wrote the
    current last line was killed mid-``write()``, that line has no trailing
    newline. Appending straight after it would glue this brand-new, valid
    entry onto the end of that broken line, corrupting a record that would
    otherwise be fine -- turning one bad line into two. So before writing,
    a cheap binary-mode check looks at the file's last byte and, if it is
    not ``\\n``, emits one first. The old broken line still reads as garbage
    (as it must -- its data really is gone) but stays contained to itself,
    and every entry appended from this point on is unaffected.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry)) + "\n"

    needs_leading_newline = False
    if ledger_path.exists() and ledger_path.stat().st_size > 0:
        with ledger_path.open("rb") as check:
            check.seek(-1, os.SEEK_END)
            needs_leading_newline = check.read(1) != b"\n"

    with ledger_path.open("a", encoding="utf-8") as handle:
        if needs_leading_newline:
            handle.write("\n")
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_ledger(ledger_path: Path) -> list[LedgerEntry]:
    """Read every entry in the ledger, skipping a corrupt trailing line.

    A process can be killed mid-``write()``, leaving the last line in the
    file truncated (not valid JSON) even though every earlier line -- each
    already ``fsync``'d by a prior, completed :func:`append_entry` call -- is
    intact. Such a line is logged and skipped rather than raising, so a
    reader never loses N-1 good records because the Nth was interrupted.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []

    entries: list[LedgerEntry] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(LedgerEntry(**data))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "skipping corrupt ledger line %d in %s: %s", lineno, ledger_path, exc
                )
                continue
    return entries


def latest_entries(entries: list[LedgerEntry]) -> dict[tuple[str, str], LedgerEntry]:
    """Reduce a flat ledger to the most recent entry per ``(target_id, mission)``.

    Assumes ``entries`` is in append (i.e. chronological) order, which holds
    for anything produced by :func:`read_ledger` reading a file only ever
    written by :func:`append_entry`. The last entry seen for a key wins. This
    is the "what a caller actually wants 90% of the time" view: current
    status per target, not the full retry history.
    """
    latest: dict[tuple[str, str], LedgerEntry] = {}
    for entry in entries:
        latest[(entry.target_id, entry.mission)] = entry
    return latest


def get_status(ledger_path: Path, target_id: str, mission: str) -> Literal["done", "pending"]:
    """Whether ``target_id``/``mission`` needs (re)processing.

    ``"done"`` iff the most recent entry for this target is
    ``status="success"``. Anything else -- never attempted, or most recent
    attempt failed -- is ``"pending"``: retryable failures and untried
    targets are deliberately conflated here, because from a caller's
    perspective both mean "still needs work"; :func:`next_targets` is where
    the max-attempts distinction (retryable vs. exhausted) actually matters.
    """
    entries = read_ledger(ledger_path)
    entry = latest_entries(entries).get((target_id, mission))
    if entry is not None and entry.status == "success":
        return "done"
    return "pending"


def next_targets(
    ledger_path: Path,
    all_targets: list[str],
    mission: str,
    max_attempts: int = 3,
) -> list[str]:
    """Return the targets in ``all_targets`` that still need processing.

    A target is included if it is either never-attempted, or its most recent
    entry is ``status="failed"`` with ``attempt < max_attempts`` (so it still
    has retries left). A target whose most recent entry is
    ``status="success"``, or that has exhausted ``max_attempts``, is
    excluded.

    ``mission`` is required (unlike the ``next_targets(ledger_path,
    all_targets, max_attempts=3)`` sketch) because the ledger's actual key is
    ``(target_id, mission)`` everywhere else in this module (see
    :func:`get_status`, :func:`latest_entries`) -- a bare ``target_id`` is not
    guaranteed unique across missions, and a script processing one mission's
    watchlist per run always has ``mission`` in hand anyway.

    Ordering: never-attempted targets first, in the order given in
    ``all_targets`` -- a bounded 30-minute run should spend its budget on
    fresh ground before spending any of it on retries. Retriable failures
    follow, oldest-failure-first (by ``timestamp_utc``), so a target that has
    been waiting longest for another attempt is retried before one that only
    just failed -- this keeps a single flaky target from being retried over
    and over while an older failure starves.
    """
    entries = read_ledger(ledger_path)
    latest = latest_entries(entries)

    never_attempted: list[str] = []
    retriable: list[tuple[str, str]] = []  # (timestamp_utc, target_id)

    for target_id in all_targets:
        entry = latest.get((target_id, mission))
        if entry is None:
            never_attempted.append(target_id)
        elif entry.status == "failed" and entry.attempt < max_attempts:
            retriable.append((entry.timestamp_utc, target_id))
        # status == "success", or failed and exhausted: excluded.

    retriable.sort(key=lambda pair: pair[0])
    return never_attempted + [target_id for _, target_id in retriable]
