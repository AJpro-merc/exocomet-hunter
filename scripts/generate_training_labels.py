"""C1: build the classifier's labelled training set (see Next Steps Part C1).

For each host star: download, then for each label class inject a synthetic
event (or none, for the real-detection classes), run the unmodified detector,
and if it produced a candidate near the injected epoch, extract its 19
features and record one labelled row. The light curve is discarded
(``discard_after_read=True``) as soon as it has been read, so only the small
feature rows -- never raw photometry -- accumulate on disk. Progress is
written incrementally so a run that is interrupted keeps what it already has.

Classes: ``comet``, ``symmetric``, ``reversed`` (asymmetric-dip shapes),
``flare``, ``starspot`` (false-positive lookalikes, Next Steps D1), ``noise``
(no injection -- real detections on real data), ``time_reversed_real`` (a
real light curve, flipped in time).

Usage
-----
    python scripts/generate_training_labels.py [--n-per-class N] [--out PATH]

Deliberately excludes KIC 3542116 and KIC 11084727 (see
``build_kepler_host_list.EXCLUDED_HOLD_OUT_TARGETS``): those are the final
exam for C2/C3 and must never be trained on.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kepler_host_list import EXCLUDED_HOLD_OUT_TARGETS, KEPLER_HOST_STARS

from exocomet.calibration.features import extract_features
from exocomet.calibration.injection import (
    inject_comet_transit,
    inject_flare,
    inject_starspot_modulation,
)
from exocomet.core.config import PipelineConfig, ScoringConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol
from exocomet.io.download import iter_light_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_training_labels")

DEFAULT_OUT = Path("data/training/labels_v1.parquet")
DEFAULT_HOST_LISTS: dict[str, tuple[str, ...]] = {"Kepler": KEPLER_HOST_STARS}

#: Shape classes localised at one epoch, matched to the nearest detection.
LOCALISED_CLASSES = ("comet", "symmetric", "reversed", "flare")
#: Classes with no injection at all -- real detections on real (or
#: time-reversed) data.
NON_INJECTED_CLASSES = ("noise", "time_reversed_real")
#: Whole-light-curve modulation, matched to the nearest detection to a random
#: anchor time (there is no single injected epoch to match against).
WHOLE_CURVE_CLASSES = ("starspot",)
ALL_CLASSES = LOCALISED_CLASSES + WHOLE_CURVE_CLASSES + NON_INJECTED_CLASSES

#: Depth/amplitude grid (ppm, as fractions) and sampling weights. Weighted
#: toward 500-2000 ppm, the boundary region where detection outcome is
#: actually uncertain (Session Log 2026-09-13's completeness table: ~0% below
#: 500 ppm, ~100% above 1500 ppm) -- uniform sampling wastes most attempts on
#: always-fails or always-trivially-succeeds depths. Widened from an initial
#: narrower 750-1500ppm-only proposal per explicit user feedback ("increase
#: the boundary") this session.
DEPTH_GRID_PPM = (200, 300, 500, 750, 1000, 1500, 2000, 3000)
DEPTH_WEIGHTS = (1, 2, 3, 4, 4, 3, 2, 1)
_DEPTH_GRID = tuple(p * 1e-6 for p in DEPTH_GRID_PPM)
_DEPTH_PROBS = tuple(w / sum(DEPTH_WEIGHTS) for w in DEPTH_WEIGHTS)


def _draw_depth(rng: np.random.Generator) -> float:
    """Sample a fractional depth/amplitude from the weighted grid."""
    return float(rng.choice(_DEPTH_GRID, p=_DEPTH_PROBS))


def _draw_shape_durations(label: str, rng: np.random.Generator) -> tuple[float, float]:
    """Sample (ingress_duration, egress_duration) in days for one dip-shape class."""
    # tau/sigma-style width in hours -> days, per Next Steps C1: sigma 1-10 h.
    width_days = float(rng.uniform(1.0, 10.0)) / 24.0
    if label == "comet":
        ingress = width_days
        egress = width_days * float(rng.uniform(1.2, 5.0))
    elif label == "reversed":
        egress = width_days
        ingress = width_days * float(rng.uniform(1.2, 5.0))
    elif label == "symmetric":
        ingress = egress = width_days
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown shape class: {label}")
    return ingress, egress


def _inject_and_label(
    lc,
    label: str,
    rng: np.random.Generator,
    config: PipelineConfig,
    match_tolerance_days: float = 0.5,
) -> dict[str, float] | None:
    """Inject one event of the given label, run the detector, return a feature row.

    Returns ``None`` when the injection produced no matching candidate (this
    is expected and fine for shallow injections -- see the completeness table
    in Session Log 2026-09-13 -- it just means no row is emitted for that
    attempt).
    """
    cfg = config.candidates
    margin_days = cfg.edge_margin_days + 2.0  # generous pad clear of the edges
    lo, hi = lc.time[0] + margin_days, lc.time[-1] - margin_days
    if hi <= lo:
        return None
    t0 = float(rng.uniform(lo, hi))
    amplitude = float("nan")

    if label == "noise":
        injected = lc
    elif label == "time_reversed_real":
        injected = lc.with_flux(lc.flux[::-1].copy())
    elif label in ("comet", "symmetric", "reversed"):
        amplitude = _draw_depth(rng)
        ingress, egress = _draw_shape_durations(label, rng)
        flux = inject_comet_transit(lc.time, lc.flux, t0, amplitude, ingress, egress)
        injected = lc.with_flux(flux)
    elif label == "flare":
        amplitude = _draw_depth(rng)
        rise = float(rng.uniform(0.01, 0.1))
        decay = float(rng.uniform(0.02, 0.3))
        flux = inject_flare(lc.time, lc.flux, t0, amplitude, rise, decay)
        injected = lc.with_flux(flux)
    elif label == "starspot":
        amplitude = float(rng.uniform(100e-6, 5000e-6))  # Next Steps D1: 100-5000 ppm
        period = float(rng.uniform(0.5, 30.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        flux = inject_starspot_modulation(lc.time, lc.flux, period, amplitude, phase)
        injected = lc.with_flux(flux)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown label class: {label}")

    # Detrend AFTER injecting (not before): the fake event must pass through
    # the same Savitzky-Golay filter a real one would, so detrending's known
    # tendency to distort depth/shape (Session Log 2026-09-13, Bug B1) is
    # part of what the classifier sees, not hidden from it. Also fixes a real
    # bug found 2026-09-14: this function never detrended at all -- every row
    # generated before this fix ran the detector on raw, undetrended flux,
    # where any real instrumental trend or stellar variability (confirmed:
    # KIC 1026032 alone produces 319 spurious raw "candidates" up to ~8%
    # depth) could dominate over the injected signal or contaminate the
    # noise/time_reversed_real classes with real uncontrolled structure.
    detrended = detrend_savgol(injected, config.detrend)

    detector = AsymmetricDipDetector(config)
    records = detector.run(detrended)
    if not records:
        return None

    if label in NON_INJECTED_CLASSES:
        record = records[0]
    else:
        # LOCALISED_CLASSES match strictly (a real injected epoch exists);
        # WHOLE_CURVE_CLASSES (starspot) have no single true epoch, so the
        # nearest detection to the random anchor t0 is accepted without the
        # tolerance check -- any detection that survives a periodic
        # modulation is a legitimate example of what that modulation does to
        # the detector, wherever in the cycle it landed.
        record = min(records, key=lambda r: abs(r.event.window.start_time - t0))
        offset = abs(record.event.window.start_time - t0)
        if label in LOCALISED_CLASSES and offset > match_tolerance_days:
            return None

    row = extract_features(record, detrended)
    row["label"] = label
    row["host_star"] = lc.target_id
    row["injected_depth"] = amplitude
    tau_over_sigma = row.get("tau_over_sigma", float("nan"))
    row["injected_tau_over_sigma"] = tau_over_sigma
    row["author"] = lc.meta.get("author")
    row["exptime"] = lc.meta.get("exptime")
    return row


def load_host_list(path: Path) -> tuple[str, ...]:
    """Read a target list: one target per line, '#' comments and blanks ignored.

    Same convention as ``config/watchlist.txt``.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(stripped for raw in lines if (stripped := raw.split("#", 1)[0].strip()))


def _sanitise_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    """Drop rows that fail basic sanity checks or are exact duplicates.

    Rejects a non-positive/NaN ``injected_depth`` on an injected class, a
    missing ``host_star``, and exact-duplicate ``(host_star, label, seed,
    injected_depth)`` combinations. Returns the cleaned rows and how many
    were dropped, so a caller can log it rather than silently lose data.
    """
    seen: set[tuple[object, ...]] = set()
    cleaned: list[dict[str, object]] = []
    dropped = 0
    for row in rows:
        host = row.get("host_star")
        label = row.get("label")
        if not host or not isinstance(host, str):
            dropped += 1
            continue
        if label in LOCALISED_CLASSES or label in WHOLE_CURVE_CLASSES:
            depth = row.get("injected_depth")
            if depth is None or not np.isfinite(depth) or depth <= 0:
                dropped += 1
                continue
        key = (host, label, row.get("seed"), row.get("injected_depth"), row.get("depth_ppm"))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        cleaned.append(row)
    return cleaned, dropped


def _order_hosts_by_coverage(hosts: list[str], prior_rows: list[dict[str, object]]) -> list[str]:
    """Process least-covered hosts first.

    Not literal duplicate-skipping (a fresh random seed each run makes exact
    repeats unlikely already) but the practical form of "don't just keep
    re-hammering the same few hosts": a fixed host-list order combined with a
    time budget that always runs out partway through means the tail of the
    list rarely gets touched across many recurring runs. Sorting by ascending
    existing-row-count spreads generation across the whole list over time
    instead.
    """
    counts: dict[str, int] = {}
    for row in prior_rows:
        host = row.get("host_star")
        if isinstance(host, str):
            counts[host] = counts.get(host, 0) + 1
    return sorted(hosts, key=lambda h: counts.get(h, 0))


def generate(
    n_per_class: int,
    out_path: Path,
    mission: str = "Kepler",
    host_list: Path | None = None,
    seed: int | None = None,
    bootstrap_draws: int = 300,
    time_budget_minutes: float | None = None,
    max_hosts: int | None = None,
    max_new_rows: int | None = None,
    append: bool = True,
) -> pd.DataFrame:
    """Run C1 end-to-end and return the accumulated label rows.

    Parameters
    ----------
    mission
        ``"Kepler"`` or ``"TESS"``. Both are pinned to a single pipeline and
        cadence by ``io/download.py`` (Next Steps A2: SPOC/120s for TESS,
        Kepler/long for Kepler).
    host_list
        Path to a target-list text file (see :func:`load_host_list`). Falls
        back to :data:`DEFAULT_HOST_LISTS` for the given mission when omitted
        (only defined for Kepler; TESS requires an explicit list).
    seed
        RNG seed. ``None`` (the default) draws a fresh seed from OS entropy
        each call -- the right choice for a recurring job that should keep
        adding *new* random injections each run, not repeat the same ones.
        The seed actually used is recorded in every output row and logged.
    time_budget_minutes
        Stop starting new hosts once this many minutes have elapsed since the
        run began (checkpointing after every host means a run stopped this
        way never loses partial progress). ``None`` means no limit.
    max_hosts
        Stop after this many hosts regardless of time remaining. ``None``
        means no limit. Bounds a single run's MAST calls independent of the
        time budget, so a run can't balloon just because hosts are downloading
        unusually fast.
    max_new_rows
        Stop once this many new rows have been added this run. ``None`` means
        no limit.
    append
        When ``True`` (the default) and ``out_path`` already holds a parquet
        file, new rows are appended to it (after host-coverage ordering and
        sanitisation) rather than overwriting.
    bootstrap_draws
        Overrides ``ScoringConfig.bootstrap_draws`` (default 1000). Bootstrap
        significance dominates per-event runtime; the full default is
        needlessly precise for building a training set (as opposed to a
        published result), the same tradeoff ``tests/unit/test_scoring.py``
        makes with its ``FAST`` config (``bootstrap_draws=60``). 300 balances
        precision against host throughput.
    """
    if host_list is not None:
        candidates = load_host_list(host_list)
    elif mission in DEFAULT_HOST_LISTS:
        candidates = DEFAULT_HOST_LISTS[mission]
    else:
        raise ValueError(f"no default host list for mission {mission!r}; pass --host-list")
    hosts = [h for h in candidates if h not in EXCLUDED_HOLD_OUT_TARGETS]
    assert not (set(hosts) & EXCLUDED_HOLD_OUT_TARGETS)

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    logger.info("using seed=%d", seed)

    config = PipelineConfig(scoring=ScoringConfig(bootstrap_draws=bootstrap_draws))
    rng = np.random.default_rng(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prior_rows: list[dict[str, object]] = []
    if append and out_path.exists():
        prior_rows = pd.read_parquet(out_path).to_dict("records")
        logger.info("appending to %d existing rows in %s", len(prior_rows), out_path)

    hosts = _order_hosts_by_coverage(hosts, prior_rows)
    if max_hosts is not None:
        hosts = hosts[:max_hosts]

    new_rows: list[dict[str, object]] = []
    start = time.time()
    n_hosts_processed = 0

    for lc in iter_light_curves(hosts, mission=mission, discard_after_read=True, skip_errors=True):
        elapsed_min = (time.time() - start) / 60.0
        if time_budget_minutes is not None and elapsed_min >= time_budget_minutes:
            logger.info(
                "time budget of %.1f min reached; stopping before %s",
                time_budget_minutes,
                lc.target_id,
            )
            break
        if max_new_rows is not None and len(new_rows) >= max_new_rows:
            logger.info("max_new_rows=%d reached; stopping before %s", max_new_rows, lc.target_id)
            break

        logger.info("processing %s (%d cadences)", lc.target_id, lc.n_points)
        n_hosts_processed += 1
        for label in ALL_CLASSES:
            is_injected = label not in NON_INJECTED_CLASSES
            n_attempts = n_per_class if is_injected else max(n_per_class // 4, 1)
            for _ in range(n_attempts):
                row = _inject_and_label(lc, label, rng, config)
                if row is not None:
                    row["seed"] = seed
                    new_rows.append(row)

        # Write incrementally so an interrupted run keeps what it has.
        clean_new, dropped = _sanitise_rows(new_rows)
        combined, _ = _sanitise_rows(prior_rows + clean_new)
        if combined:
            pd.DataFrame(combined).to_parquet(out_path, index=False)
            logger.info(
                "checkpoint: %d total rows (%d new, %d dropped this run) -> %s",
                len(combined),
                len(clean_new),
                dropped,
                out_path,
            )

    elapsed = time.time() - start
    clean_new, dropped = _sanitise_rows(new_rows)
    df = pd.DataFrame(_sanitise_rows(prior_rows + clean_new)[0])
    new_df = pd.DataFrame(clean_new)
    logger.info(
        "done: %d/%d hosts processed, %d new rows this run (%d dropped), %d total rows, %.1fs\n"
        "new-row class balance:\n%s\ntotal class balance:\n%s",
        n_hosts_processed,
        len(hosts),
        len(new_df),
        dropped,
        len(df),
        elapsed,
        new_df["label"].value_counts().to_string() if not new_df.empty else "(none)",
        df["label"].value_counts().to_string() if not df.empty else "(empty)",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-per-class", type=int, default=20, help="injections attempted per class per host"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mission", choices=["Kepler", "TESS"], default="Kepler")
    parser.add_argument("--host-list", type=Path, default=None, help="target-list text file")
    parser.add_argument("--seed", type=int, default=None, help="omit for a fresh random seed")
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=None,
        help="stop starting new hosts after this many minutes",
    )
    parser.add_argument("--max-hosts", type=int, default=None, help="stop after this many hosts")
    parser.add_argument(
        "--max-new-rows", type=int, default=None, help="stop once this many new rows are added"
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="overwrite --out instead of appending to it",
    )
    args = parser.parse_args()

    df = generate(
        args.n_per_class,
        args.out,
        mission=args.mission,
        host_list=args.host_list,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
        time_budget_minutes=args.time_budget_minutes,
        max_hosts=args.max_hosts,
        max_new_rows=args.max_new_rows,
        append=not args.no_append,
    )
    if args.out.exists():
        print(f"total {len(df)} rows in {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"total {len(df)} rows; {args.out} was never written (no rows produced this run)")


if __name__ == "__main__":
    main()
