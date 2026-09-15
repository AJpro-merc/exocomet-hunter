"""Check the archive for new observations and search them for exocomet transits.

Designed to run unattended on a schedule. TESS is still observing, so new
sectors keep appearing at MAST; this script asks which observations exist for a
watchlist of stars, skips the ones already processed, runs the pipeline over
whatever is new, and appends any candidates to a results file.

State lives in a small JSON ledger rather than a database, so a scheduled cloud
run can commit it back to the repository and the next run picks up where this
one stopped.

Usage
-----
    python scripts/monitor_new_data.py --watchlist config/watchlist.txt
    python scripts/monitor_new_data.py --watchlist config/watchlist.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exocomet.calibration.features import extract_features
from exocomet.core.config import PipelineConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol
from exocomet.io.download import fetch_light_curve

logger = logging.getLogger("monitor")

LEDGER_PATH = Path("results/processed_ledger.json")
CANDIDATES_PATH = Path("results/monitor_candidates.jsonl")


def load_ledger(path: Path) -> dict[str, dict[str, object]]:
    """Read the record of what has already been processed."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(path: Path, ledger: dict[str, dict[str, object]]) -> None:
    """Persist the processed-observations record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def available_products(target_id: str, mission: str) -> int:
    """Return how many data products the archive currently holds for a target.

    Used as a cheap change-detector: if the count has grown since the last run,
    there is new data worth downloading. This avoids re-downloading gigabytes
    only to discover nothing changed.

    Counts only the author/exptime combination :func:`fetch_light_curve`
    actually pins and downloads (Next Steps A2) -- counting every product
    from every pipeline, as an earlier version did, meant a new QLP or TARS
    file could trigger reprocessing even though the pinned SPOC data never
    changed.
    """
    import lightkurve as lk

    search_kwargs: dict[str, object] = {"mission": mission}
    if mission == "TESS":
        search_kwargs["author"] = "SPOC"
        search_kwargs["exptime"] = 120
    else:
        search_kwargs["author"] = "Kepler"
        search_kwargs["cadence"] = "long"

    try:
        return len(lk.search_lightcurve(target_id, **search_kwargs))
    except Exception as exc:
        logger.warning("archive query failed for %s: %s", target_id, exc)
        return -1


def process_target(
    target_id: str, mission: str, config: PipelineConfig
) -> list[dict[str, object]]:
    """Download, detrend, detect and score one target; return flagged candidates."""
    lc = detrend_savgol(
        fetch_light_curve(target_id, mission=mission, discard_after_read=True), config.detrend
    )
    records = AsymmetricDipDetector(config).run(lc)

    candidates: list[dict[str, object]] = []
    for record in records:
        if not record.score.flagged:
            continue
        row = record.to_row()
        row["features"] = extract_features(record, lc)
        row["found_utc"] = datetime.now(UTC).isoformat()
        row["mission"] = mission
        candidates.append(row)
    return candidates


def main() -> int:
    """Entry point for the scheduled monitor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", type=Path, required=True, help="one target per line")
    parser.add_argument("--mission", default="TESS")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--limit", type=int, default=0, help="max targets to process this run")
    parser.add_argument("--dry-run", action="store_true", help="report new data, process nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    targets = [
        line.strip()
        for line in args.watchlist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info("watchlist holds %d targets (%s)", len(targets), args.mission)

    ledger = load_ledger(args.ledger)
    config = PipelineConfig()

    changed = []
    for target in targets:
        count = available_products(target, args.mission)
        if count < 0:
            continue
        seen = int(ledger.get(target, {}).get("n_products", 0))
        if count > seen:
            changed.append((target, seen, count))

    logger.info("%d target(s) have new data", len(changed))
    for target, seen, count in changed:
        logger.info("  %s: %d -> %d products", target, seen, count)

    if args.dry_run:
        return 0

    if args.limit:
        changed = changed[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for target, _, count in changed:
        try:
            candidates = process_target(target, args.mission, config)
        except Exception as exc:
            logger.warning("skipping %s: %s", target, exc)
            continue

        with args.out.open("a", encoding="utf-8") as handle:
            for row in candidates:
                handle.write(json.dumps(row, default=str) + "\n")

        total += len(candidates)
        ledger[target] = {
            "n_products": count,
            "last_checked_utc": datetime.now(UTC).isoformat(),
            "last_candidates": len(candidates),
        }
        logger.info("%s: %d flagged candidate(s)", target, len(candidates))

    save_ledger(args.ledger, ledger)
    logger.info("run complete: %d new candidate(s) across %d target(s)", total, len(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
