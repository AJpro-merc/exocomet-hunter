# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Resumable hourly ingestion: download, extract, process, persist, forget the FITS.

Designed to run inside a ~30 minute GitHub Actions budget with the runner able
to be killed at any moment, and to pick up cleanly on the next hourly run.

Per-target sequence, in this exact order (the order is the safety mechanism,
not a transaction):

    1. fetch the light curve from MAST (raw FITS kept on local disk so far)
    2. save the extract to disk, atomically (:func:`exocomet.io.extract.save_extract`)
    3. only now, delete the raw FITS
    4. run the existing detection pipeline (detrend + detector) on the extract
    5. write the results file, atomically
    6. only now, append one ledger line recording success

If the process is killed anywhere before step 6, no ledger entry exists for
this attempt, so the target is simply retried next run -- ``save_extract`` and
the results write are both themselves atomic, so a killed run never leaves a
half-written extract or results file behind, only (at worst) a missing one.
The ledger line is written *last*, deliberately, so it can never claim success
for work that did not fully land.

Budget is checked once per target, at the top of the loop -- gating whether to
*start* another target, never interrupting one already in flight. A star is
the atomic unit of work, matching the existing pattern in
``scripts/generate_training_labels.py``.

Concurrency: two runs processing the same target at once is prevented at the
GitHub Actions level (``concurrency: {group: ingest, cancel-in-progress:
false}`` queues a second trigger rather than running it in parallel), not by
a lockfile here. Simpler, and sufficient given the workflow already owns this
guarantee -- an in-process lock would be redundant infrastructure for a
problem the scheduler already solves.

Layout under ``--data-repo-dir`` (this whole tree is meant to be committed to
the separate ``exocomet-hunter-data`` repository, not this one):

    <data-repo-dir>/
        ledger.jsonl                       # append-only, see exocomet.io.ledger
        extract/<mission>/<target>.parquet # layer A: the permanent, detector-independent extract
        results/<mission>/<target>.jsonl   # layer B: this pipeline version's candidate events

Usage
-----
    python scripts/ingest.py --mission Kepler --time-budget-minutes 25
    python scripts/ingest.py --mission TESS --data-repo-dir data-repo
    python scripts/ingest.py --mission Kepler --offline --max-targets 3   # no network, for testing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from exocomet.core.config import PipelineConfig
    from exocomet.core.types import LightCurveData

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

__all__ = ["build_parser", "get_target_list", "main", "process_one_target", "run_ingest"]

MISSIONS = ("Kepler", "TESS")


def _hold_out_and_bad_hosts() -> tuple[frozenset[str], frozenset[str]]:
    """Import the hold-out/known-bad sets lazily.

    Never process these -- the six-event exam and the "quiet host" pipeline
    proof both depend on these two never appearing in generated data.
    Re-imported (not re-declared) so there is exactly one place this set is
    defined; see scripts/build_kepler_host_list.py's own module docstring for
    why these are excluded and what breaks if they leak in.
    """
    from build_kepler_host_list import EXCLUDED_HOLD_OUT_TARGETS, KNOWN_BAD_HOSTS

    return EXCLUDED_HOLD_OUT_TARGETS, KNOWN_BAD_HOSTS


def _parse_target_file(path: Path) -> list[str]:
    """Parse a plain-text target list: one id per line, ``#``-comments stripped.

    Shared by ``config/watchlist.txt`` and the optional catalogue-generated
    ``config/target_list_<mission>.txt`` files -- both use the same
    convention (``scripts/build_target_list.py``'s output lines look like
    ``TIC 12377940  # role=disc pair=hr-9102 ...``, which this strips down to
    just ``TIC 12377940``).
    """
    if not path.exists():
        return []
    lines = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def get_target_list(mission: str) -> list[str]:
    """The full target list for one mission, with the hold-out guard re-enforced here too.

    Kepler comes from ``build_kepler_host_list.KEPLER_HOST_STARS`` (a tuple
    constant that already excludes the hold-out and known-bad sets via its own
    module-level asserts); TESS comes from ``config/watchlist.txt``. Both are
    the original, hand-picked 33-target seed lists.

    **Additive, optional catalogue expansion**: if
    ``config/target_list_<mission>.txt`` exists (written by
    ``scripts/build_target_list.py``), its targets are appended after the
    seed list, deduplicated. Nothing breaks if the file doesn't exist yet --
    ``get_target_list`` returns exactly the original 33 in that case, same as
    before this was wired in. This is deliberately additive rather than a
    replacement, so the original, already-proven seed lists keep working
    regardless of what the catalogue script produces.

    Both sources are re-checked against the hold-out/known-bad sets here as
    well -- defense in depth, so a future third target source cannot silently
    leak a validation star into generated data without a loud failure.
    """
    hold_out, known_bad = _hold_out_and_bad_hosts()
    repo_root = Path(__file__).resolve().parent.parent

    if mission == "Kepler":
        from build_kepler_host_list import KEPLER_HOST_STARS

        targets = list(KEPLER_HOST_STARS)
    elif mission == "TESS":
        targets = _parse_target_file(repo_root / "config" / "watchlist.txt")
    else:
        raise ValueError(f"unknown mission {mission!r}, expected one of {MISSIONS}")

    catalogue_file = repo_root / "config" / f"target_list_{mission.lower()}.txt"
    seen = set(targets)
    for extra in _parse_target_file(catalogue_file):
        if extra not in seen:
            targets.append(extra)
            seen.add(extra)

    leaked = (set(targets) & hold_out) | (set(targets) & known_bad)
    if leaked:
        raise AssertionError(
            f"target list for {mission} contains hold-out/known-bad target(s): {sorted(leaked)} "
            "-- this must never happen, the six-event exam depends on these never being trained on"
        )
    return targets


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synthetic_light_curve(target_id: str, mission: str, seed: int) -> LightCurveData:
    """A flat, noise-only host for ``--offline`` runs -- no network at all.

    Mirrors ``exocomet.cli._synthetic_light_curve``; kept as a separate,
    smaller copy here rather than importing the CLI's version, since that one
    is deliberately tuned for the ``inject`` command's calibration use case
    (seeded noise level as a CLI argument) and this one only needs to be *a*
    valid, detectable light curve so the extract/ledger/results plumbing can
    be exercised end to end without MAST.
    """
    import numpy as np

    from exocomet.core.types import LightCurveData

    cadence_days = 0.02043
    n_points = 4000
    rng = np.random.default_rng(seed)
    time_arr = np.arange(n_points, dtype=np.float64) * cadence_days
    flux = np.ones(n_points, dtype=np.float64) + rng.normal(0.0, 5.0e-5, size=n_points)
    flux_err = np.full(n_points, 5.0e-5, dtype=np.float64)
    return LightCurveData(
        target_id=target_id,
        mission=mission,
        time=time_arr,
        flux=flux,
        flux_err=flux_err,
        meta={"source": "offline synthetic", "seed": seed},
    )


def _write_results_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    """Write candidate rows as JSONL, atomically.

    Written even when ``rows`` is empty: a star with zero detected candidates
    is retained as valid scientific information (a non-detection), and its
    empty results file existing is what distinguishes "processed, found
    nothing" from "never attempted" -- the same distinction the ledger makes
    for whole-target success/failure, applied at the results-file level.
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp", prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def process_one_target(
    target_id: str,
    mission: str,
    config: PipelineConfig,
    data_repo_dir: Path,
    attempt: int,
    offline: bool,
    offline_seed: int = 0,
) -> None:
    """Run the full per-target sequence described in the module docstring.

    Raises on any failure -- the caller (:func:`run_ingest`) is responsible
    for catching it and writing a ``status="failed"`` ledger entry. This
    function itself only ever writes a ``status="success"`` entry, and only
    as its very last action.
    """
    from exocomet.detect.comet_detector import AsymmetricDipDetector
    from exocomet.detrend.detrend import detrend_savgol
    from exocomet.io import extract as extract_mod
    from exocomet.io import ledger as ledger_mod

    ledger_path = data_repo_dir / "ledger.jsonl"
    ex_path = extract_mod.extract_path(data_repo_dir, mission, target_id)
    results_path = (
        data_repo_dir
        / "results"
        / mission
        / f"{extract_mod.sanitize_target_id(target_id)}.jsonl"
    )

    # Step 1: fetch (raw FITS lands on local disk here, not yet deleted).
    # Cache dir is computed from this file's own location, not the process's
    # CWD -- fetch_light_curve's own default (a bare relative "data/raw") is
    # CWD-dependent, and this script is sometimes launched from scripts/
    # itself (matching the sibling scripts' own convention) and sometimes
    # from the repo root, which would otherwise scatter the raw-FITS cache
    # across two different locations depending on how it was invoked.
    raw_cache_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    if offline:
        lc = _synthetic_light_curve(target_id, mission, offline_seed)
    else:
        from exocomet.io.download import fetch_light_curve

        # discard_after_read=False deliberately: the FITS must survive until
        # step 3, *after* the extract is durably on disk (step 2). Deleting it
        # as a side effect of fetch_light_curve itself would collapse steps 1
        # and 3 into one, removing the ordering guarantee this function exists
        # to provide.
        lc = fetch_light_curve(
            target_id, mission=mission, cache_dir=raw_cache_dir, discard_after_read=False
        )

    # Step 2: persist the extract, atomically.
    extract_mod.save_extract(lc, ex_path)
    if not ex_path.exists():
        # Belt-and-braces: save_extract is atomic and raises on any failure,
        # so this should be unreachable -- but a ledger entry claiming
        # success for a file that isn't actually there is a worse failure
        # mode than a loud, immediate one, so it's checked directly rather
        # than trusted.
        raise RuntimeError(f"save_extract returned without raising but {ex_path} is missing")

    # Step 3: only now, delete the raw FITS -- mirrors the cache-wipe that
    # fetch_light_curve(discard_after_read=True) would have done internally,
    # done explicitly here so it happens strictly after step 2 lands.
    if not offline:
        import shutil

        shutil.rmtree(raw_cache_dir / "mastDownload", ignore_errors=True)

    # Step 4: run the existing, unmodified detection pipeline.
    detrended = detrend_savgol(lc, config.detrend)
    records = AsymmetricDipDetector(config).run(detrended)

    # Step 5: persist results, atomically. Empty list is a valid, meaningful
    # result (a non-detection) and is written just the same as a non-empty one.
    _write_results_jsonl([r.to_row() for r in records], results_path)

    # Step 6: only now, the ledger entry. This line is what makes the whole
    # attempt "count" -- everything above can be safely redone if interrupted.
    ledger_mod.append_entry(
        ledger_path,
        ledger_mod.LedgerEntry(
            target_id=target_id,
            mission=mission,
            status="success",
            timestamp_utc=_now_iso(),
            attempt=attempt,
            error=None,
            extract_path=str(ex_path.relative_to(data_repo_dir)),
            results_path=str(results_path.relative_to(data_repo_dir)),
        ),
    )
    logger.info(
        "%s (%s): %d event(s), %d flagged -- extract %s, results %s",
        target_id,
        mission,
        len(records),
        sum(1 for r in records if r.score.flagged),
        ex_path.name,
        results_path.name,
    )


def run_ingest(
    mission: str,
    data_repo_dir: Path,
    time_budget_minutes: float,
    max_attempts: int,
    offline: bool,
    max_targets: int | None,
    config_path: Path | None,
) -> dict[str, int]:
    """Process targets for one mission until the time budget or target list is exhausted.

    Returns a small summary dict (``succeeded``, ``failed``, ``skipped_budget``)
    for the caller to report and for tests to assert on.
    """
    from exocomet.core.config import PipelineConfig
    from exocomet.io import ledger as ledger_mod

    config = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig()
    ledger_path = data_repo_dir / "ledger.jsonl"

    all_targets = get_target_list(mission)
    pending = ledger_mod.next_targets(ledger_path, all_targets, mission, max_attempts=max_attempts)
    targets = pending[:max_targets] if max_targets is not None else pending

    logger.info(
        "%s: %d target(s) total, %d pending, %d queued this run",
        mission,
        len(all_targets),
        len(pending),
        len(targets),
    )

    started = time.monotonic()
    budget_seconds = time_budget_minutes * 60.0
    summary = {"succeeded": 0, "failed": 0, "skipped_budget": 0}
    entries = ledger_mod.read_ledger(ledger_path)
    prior_attempts = {
        (e.target_id, e.mission): e.attempt
        for e in ledger_mod.latest_entries(entries).values()
    }

    for i, target_id in enumerate(targets):
        elapsed = time.monotonic() - started
        if elapsed >= budget_seconds:
            summary["skipped_budget"] = len(targets) - i
            logger.info(
                "time budget (%.1f min) reached, %d target(s) left for next run",
                time_budget_minutes,
                summary["skipped_budget"],
            )
            break

        attempt = prior_attempts.get((target_id, mission), 0) + 1
        try:
            process_one_target(
                target_id,
                mission,
                config,
                data_repo_dir,
                attempt=attempt,
                offline=offline,
                offline_seed=i,
            )
            summary["succeeded"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad star must not kill the run
            logger.warning("FAILED %s (%s), attempt %d: %s", target_id, mission, attempt, exc)
            ledger_mod.append_entry(
                ledger_path,
                ledger_mod.LedgerEntry(
                    target_id=target_id,
                    mission=mission,
                    status="failed",
                    timestamp_utc=_now_iso(),
                    attempt=attempt,
                    error=str(exc)[:2000],
                    extract_path=None,
                    results_path=None,
                ),
            )
            summary["failed"] += 1

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", choices=MISSIONS, required=True)
    parser.add_argument(
        "--data-repo-dir",
        type=Path,
        default=Path("data-repo"),
        help="directory holding ledger.jsonl, extract/, results/ (matches train.yml's clone dir)",
    )
    parser.add_argument("--time-budget-minutes", type=float, default=25.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=None, help="cap for smoke-testing")
    parser.add_argument("--offline", action="store_true", help="no network; synthetic light curves")
    parser.add_argument("--config", type=Path, default=None, help="optional thresholds YAML")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_ingest(
        mission=args.mission,
        data_repo_dir=args.data_repo_dir,
        time_budget_minutes=args.time_budget_minutes,
        max_attempts=args.max_attempts,
        offline=args.offline,
        max_targets=args.max_targets,
        config_path=args.config,
    )
    print(
        f"{args.mission}: {summary['succeeded']} succeeded, {summary['failed']} failed, "
        f"{summary['skipped_budget']} left for next run"
    )
    return 0 if summary["failed"] == 0 or summary["succeeded"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
