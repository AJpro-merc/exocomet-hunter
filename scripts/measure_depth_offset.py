# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the depth of each validation dip four ways, to locate a 22% bias.

Found 2026-09-15: the pipeline recovers all six published exocomet transits in
KIC 3542116 at the right epochs, but every reported depth comes out at roughly
78% of the value in Rappaport et al. (2018). Six out of six low by a similar
factor is a systematic, not noise.

This script **diagnoses only**. It imports the detector and reads its output;
it never changes how depth is computed. Three hypotheses are on the table (see
``docs/research_log/006-depth-offset.md``):

1. **Baseline contamination.** ``asymmetry.measure_event`` takes the local level
   as ``median(baseline[window])`` where ``baseline`` is a rolling median whose
   window straddles the dip. If a meaningful fraction of that window is
   in-transit, the reference level is dragged down with the dip and the
   measured depth is too shallow.
2. **Observed minimum vs fitted depth.** The published numbers may come from a
   profile fit while ours is the observed flux minimum against a local level.
3. **Detrending absorbs the dip.** The Savitzky-Golay pass divides out a
   quadratic over an 8.19 d window; a ~1 d dip is 12% of that window and a
   least-squares quadratic will follow it partway down.

Hypotheses 1 and 3 are separable, so the measurements are laid out as a 2x2
over {rolling baseline, out-of-event baseline} x {detrended flux, raw flux},
plus the fitted depth:

===========  ==================================================================
``current``  detrended flux, rolling baseline -- exactly what the pipeline
             reports today (cross-checked against ``DipEvent.depth_ppm``)
``h1``       detrended flux, out-of-event baseline (hypothesis 1 alone)
``h2``       ``ModelComparison.comet.params["depth"]`` (hypothesis 2)
``h3``       raw normalised flux, rolling baseline (hypothesis 3 alone)
``h1h3``     raw normalised flux, out-of-event baseline (both corrections)
===========  ==================================================================

Sources
-------
``--source cache`` (default)
    Read the FITS products already sitting in ``data/raw/mastDownload`` and
    stitch them locally. **No network call is made.** This is the mode to use
    for everything short of refreshing the data.
``--source synthetic``
    Inject dips of known depth into a flat, noisy series and run the same five
    measurements. Truth is known, so this is how the harness's own arithmetic
    gets checked without touching real photometry.
``--source mast``
    The only mode that talks to MAST. Opt-in, one target, serialized. Run it
    yourself; do not let an agent run it concurrently.

Usage
-----
    python scripts/measure_depth_offset.py                      # cached FITS
    python scripts/measure_depth_offset.py --source synthetic   # logic check
    python scripts/measure_depth_offset.py --source mast        # live refresh
    python scripts/measure_depth_offset.py --markdown           # log-ready table
"""

from __future__ import annotations

import argparse
import glob
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from exocomet.calibration.injection import inject_comet_transit
from exocomet.core.config import PipelineConfig
from exocomet.core.types import DipEvent, FloatArray, LightCurveData
from exocomet.detect.baseline import days_to_cadences, rolling_baseline
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detect.model_comparison import compare_models
from exocomet.detrend.detrend import detrend_savgol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measure_depth_offset")

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_TARGETS = REPO_ROOT / "config" / "validation_targets.yaml"
CACHE_GLOB = "data/raw/mastDownload/*/*/*_llc.fits"

#: Order of the five measurements in every table this script prints.
METHOD_KEYS = ("current", "h1", "h2", "h3", "h1h3")
METHOD_LABELS = {
    "current": "current (detrended, rolling)",
    "h1": "H1 (detrended, out-of-event)",
    "h2": "H2 (fitted comet depth)",
    "h3": "H3 (raw, rolling)",
    "h1h3": "H1+H3 (raw, out-of-event)",
}


@dataclass(frozen=True)
class PublishedDip:
    """One literature epoch and depth to be reproduced."""

    t_min_bkjd: float
    depth_ppm: float
    note: str


@dataclass(frozen=True)
class DepthRow:
    """All five depth measurements for one matched event."""

    note: str
    published_ppm: float
    t_min: float
    measured_ppm: dict[str, float]

    def ratio(self, key: str) -> float:
        """Measured-over-published ratio for one method."""
        value = self.measured_ppm.get(key, float("nan"))
        if not np.isfinite(value) or self.published_ppm <= 0:
            return float("nan")
        return value / self.published_ppm


def load_published_dips(
    target_id: str, path: Path = VALIDATION_TARGETS
) -> tuple[list[PublishedDip], float]:
    """Read the literature epochs and depths for one target.

    Returns
    -------
    dips, match_tolerance_days
        Epochs sorted in time, and the epoch-matching tolerance declared by the
        same file, so the harness and the validation gate agree on what counts
        as "the same event".
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tolerance = float(raw.get("match_tolerance_days", 0.5))
    for target in raw.get("targets", []):
        if target.get("target_id") != target_id:
            continue
        dips = [
            PublishedDip(
                t_min_bkjd=float(d["t_min_bkjd"]),
                depth_ppm=float(d["depth_ppm"]),
                note=str(d.get("note", "")),
            )
            for d in target.get("dips", []) or []
        ]
        return sorted(dips, key=lambda d: d.t_min_bkjd), tolerance
    raise KeyError(f"{target_id} is not listed in {path}")


# --------------------------------------------------------------------------
# Light-curve sources
# --------------------------------------------------------------------------


def load_from_cache(target_id: str, cache_glob: str = CACHE_GLOB) -> LightCurveData:
    """Stitch the already-downloaded FITS products for a target, offline.

    Reproduces what :func:`exocomet.io.download.fetch_light_curve` does after
    its download step -- per-product normalisation via ``stitch()``, then
    division by the global median -- without contacting MAST.

    Unreadable products are skipped with a warning rather than aborting the
    run: at least one file in the current cache is a truncated remnant of the
    download that had to be killed (see the ``io/download`` module docstring),
    and it covers a quarter later than any validation epoch.
    """
    import lightkurve as lk

    paths = sorted(p for p in glob.glob(str(REPO_ROOT / cache_glob)) if _matches(p, target_id))
    if not paths:
        raise FileNotFoundError(
            f"no cached FITS for {target_id} under {cache_glob}; "
            f"run with --source mast yourself to populate the cache"
        )

    products, skipped = [], []
    for path in paths:
        try:
            products.append(lk.read(path, quality_bitmask="default"))
        except Exception as exc:  # noqa: BLE001 - corrupt file, not a bug here
            skipped.append(Path(path).name)
            logger.warning("skipping unreadable cached product %s (%s)", Path(path).name, exc)
    if not products:
        raise FileNotFoundError(f"every cached product for {target_id} failed to read")

    stitched = lk.LightCurveCollection(products).stitch()
    time = np.asarray(stitched.time.value, dtype=np.float64)
    flux = np.asarray(stitched.flux.value, dtype=np.float64)
    flux_err = np.asarray(stitched.flux_err.value, dtype=np.float64)

    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[good], flux[good], flux_err[good]
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    unique = np.concatenate(([True], np.diff(time) > 0))
    time, flux, flux_err = time[unique], flux[unique], flux_err[unique]

    median = float(np.median(flux))
    logger.info(
        "loaded %d cached product(s) for %s (%d skipped), %d cadences, BKJD %.1f-%.1f",
        len(products),
        target_id,
        len(skipped),
        time.size,
        time[0],
        time[-1],
    )
    return LightCurveData(
        target_id=target_id,
        mission="Kepler",
        time=time,
        flux=flux / median,
        flux_err=flux_err / median,
        meta={
            "source": "local FITS cache (no network)",
            "n_products": len(products),
            "skipped_products": skipped,
        },
    )


def _matches(path: str, target_id: str) -> bool:
    """Whether a cached FITS path belongs to ``target_id`` (e.g. 'KIC 3542116')."""
    digits = "".join(ch for ch in target_id if ch.isdigit())
    return bool(digits) and digits.lstrip("0") in Path(path).name.lstrip("kplrtic0")


def build_synthetic(
    dips: list[PublishedDip], seed: int = 42, noise_ppm: float = 50.0
) -> LightCurveData:
    """Inject dips of the published depths into a flat, noisy series.

    Truth is known here, so any deviation of the ``H1+H3`` column from 1.00 is
    the harness's own error rather than a property of the pipeline. Epochs and
    cadence mirror the Kepler long-cadence case so that the window-to-duration
    ratios driving hypotheses 1 and 3 are realistic.
    """
    rng = np.random.default_rng(seed)
    cadence = 0.020434
    time = np.arange(100.0, 1400.0, cadence)
    flux = np.ones_like(time) + rng.normal(0.0, noise_ppm * 1.0e-6, time.size)

    for dip in dips:
        flux = inject_comet_transit(
            time,
            flux,
            t0=dip.t_min_bkjd,
            depth=dip.depth_ppm * 1.0e-6,
            ingress_duration=0.30,
            egress_duration=0.70,
        )

    return LightCurveData(
        target_id="SYNTHETIC",
        mission="Kepler",
        time=time,
        flux=flux,
        flux_err=np.full_like(time, noise_ppm * 1.0e-6),
        meta={"source": "synthetic injection", "noise_ppm": noise_ppm, "seed": seed},
    )


def load_from_mast(target_id: str) -> LightCurveData:
    """Fetch from MAST. The one code path in this file that uses the network."""
    from exocomet.io.download import fetch_light_curve

    logger.warning("contacting MAST for %s -- run this serially, never in parallel", target_id)
    return fetch_light_curve(target_id, mission="Kepler")


# --------------------------------------------------------------------------
# The five depth measurements
# --------------------------------------------------------------------------


def out_of_event_level(
    time: FloatArray,
    flux: FloatArray,
    event: DipEvent,
    span_days: float,
    exclusion_factor: float,
    min_exclusion_days: float,
) -> float:
    """Median flux in two flanking windows that exclude the event entirely.

    This is the hypothesis-1 estimator. The excluded zone is the detected
    window scaled by ``exclusion_factor`` about the minimum -- the window marks
    where the dip is *significant*, and its wings run well past that, so
    excluding only the window would leave tail cadences in the reference
    sample and reintroduce part of the very bias being tested.

    Returns
    -------
    float
        Median of the flanking samples, or ``nan`` if neither flank holds
        enough cadences to be meaningful.
    """
    half = max(0.5 * event.window.duration_days * exclusion_factor, min_exclusion_days)
    lo_a, lo_b = event.t_min - half - span_days, event.t_min - half
    hi_a, hi_b = event.t_min + half, event.t_min + half + span_days

    mask = ((time >= lo_a) & (time <= lo_b)) | ((time >= hi_a) & (time <= hi_b))
    if int(np.count_nonzero(mask)) < 10:
        return float("nan")
    return float(np.median(flux[mask]))


def fitted_depth_ppm(lc: LightCurveData, event: DipEvent, baseline_level: float) -> float:
    """Depth of the fitted comet profile, in ppm.

    Calls :func:`exocomet.detect.model_comparison.compare_models` exactly as
    :mod:`exocomet.detect.scoring` does, then reads
    ``ModelComparison.comet.params["depth"]``. Note that the fit's own
    ``baseline`` is a free parameter seeded at ``baseline_level``; if it is well
    constrained, this depth is immune to hypothesis 1, and if the fitting region
    holds too little out-of-dip flux, it is not. The fitted baseline is
    reported alongside so the two cases can be told apart.
    """
    model = compare_models(lc.time, lc.flux, lc.flux_err, event, baseline_level)
    if model is None or not model.comet.converged:
        return float("nan")
    return float(model.comet.params.get("depth", float("nan"))) * 1.0e6


def measure_all_ways(
    raw_lc: LightCurveData,
    detrended_lc: LightCurveData,
    event: DipEvent,
    config: PipelineConfig,
    exclusion_factor: float,
    min_exclusion_days: float,
) -> dict[str, float]:
    """Compute all five depths, in ppm, for one detected event.

    ``event`` comes from the unmodified detector running on ``detrended_lc``,
    so ``current`` reproduces the shipped number by construction; it is
    recomputed here from first principles anyway and cross-checked against
    ``event.depth_ppm`` by the caller.
    """
    window = event.window
    span = config.baseline.median_window_days

    detrended_baseline = rolling_baseline(
        detrended_lc.flux, days_to_cadences(span, detrended_lc.cadence_days)
    )
    raw_baseline = rolling_baseline(
        raw_lc.flux, days_to_cadences(span, raw_lc.cadence_days)
    )

    detrended_level = float(
        np.median(detrended_baseline[window.start_index : window.end_index + 1])
    )
    raw_level = float(np.median(raw_baseline[window.start_index : window.end_index + 1]))

    detrended_oo = out_of_event_level(
        detrended_lc.time, detrended_lc.flux, event, span, exclusion_factor, min_exclusion_days
    )
    raw_oo = out_of_event_level(
        raw_lc.time, raw_lc.flux, event, span, exclusion_factor, min_exclusion_days
    )

    f_min_detrended = event.f_min
    f_min_raw = float(raw_lc.flux[window.start_index : window.end_index + 1].min())

    return {
        "current": (detrended_level - f_min_detrended) * 1.0e6,
        "h1": (detrended_oo - f_min_detrended) * 1.0e6,
        "h2": fitted_depth_ppm(detrended_lc, event, detrended_level),
        "h3": (raw_level - f_min_raw) * 1.0e6,
        "h1h3": (raw_oo - f_min_raw) * 1.0e6,
    }


#: Rolling-median widths, in days, swept by :func:`sweep_baseline_window`.
#: Spans the shipped 2.0621 d value up to well past the 8.19 d detrending
#: window, which is where residual stellar variability starts leaking back in.
SWEEP_WINDOW_DAYS = (2.0621, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0)


def sweep_baseline_window(
    detrended_lc: LightCurveData,
    matched: list[tuple[PublishedDip, DipEvent]],
    window_days: tuple[float, ...] = SWEEP_WINDOW_DAYS,
) -> list[tuple[float, list[float], float]]:
    """Recompute ``current``-style depths over a range of rolling-median widths.

    This is the direct test of hypothesis 1, and it is sharper than the
    out-of-event comparison because only one thing changes: the width of the
    window the median is taken over. If the shipped depth is shallow because
    the dip sits inside its own reference window, widening that window must
    raise the depth, and must stop raising it once the dip is a small enough
    minority of the window for the median to ignore.

    Note that ``docs/research_log/003`` ran the same sweep and found no effect
    — but on :math:`A_{dur}`, a *ratio* of two durations, which is insensitive
    to where the reference level sits. That result says nothing about depth,
    which is measured against the level directly.

    Returns
    -------
    list of (window_days, depths_ppm, median_ratio)
        One entry per swept width, depths in published-epoch order.
    """
    rows = []
    for width in window_days:
        baseline = rolling_baseline(
            detrended_lc.flux, days_to_cadences(width, detrended_lc.cadence_days)
        )
        depths, ratios = [], []
        for dip, event in matched:
            level = float(
                np.median(baseline[event.window.start_index : event.window.end_index + 1])
            )
            depth_ppm = (level - event.f_min) * 1.0e6
            depths.append(depth_ppm)
            ratios.append(depth_ppm / dip.depth_ppm)
        rows.append((width, depths, float(np.median(ratios))))
    return rows


def print_sweep(
    matched: list[tuple[PublishedDip, DipEvent]],
    sweep: list[tuple[float, list[float], float]],
    markdown: bool,
) -> None:
    """Print the baseline-window sweep as a table of depths and median ratios."""
    header = ["Baseline window (d)"] + [d.note.split(",")[0] for d, _ in matched] + ["median ratio"]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
    else:
        lines.append("  ".join(f"{h:>20}" for h in header))

    for width, depths, ratio in sweep:
        cells = [f"{width:.2f}"] + [f"{d:.0f}" for d in depths] + [f"{ratio:.3f}x"]
        lines.append(
            "| " + " | ".join(cells) + " |" if markdown else "  ".join(f"{c:>20}" for c in cells)
        )
    print("\n".join(lines))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_harness(
    raw_lc: LightCurveData,
    published: list[PublishedDip],
    tolerance_days: float,
    config: PipelineConfig,
    exclusion_factor: float,
    min_exclusion_days: float,
) -> tuple[list[DepthRow], list[PublishedDip], list[tuple[PublishedDip, DipEvent]], LightCurveData]:
    """Detect events, match them to the literature, and measure each five ways.

    Returns
    -------
    rows, unmatched, matched, detrended_lc
        One :class:`DepthRow` per recovered epoch; the published epochs no
        detected event fell within ``tolerance_days`` of; the matched
        (literature, detected) pairs and the detrended light curve, both of
        which :func:`sweep_baseline_window` needs.
    """
    detrended_lc = detrend_savgol(raw_lc, config.detrend)
    detector = AsymmetricDipDetector(config)
    events = detector.detect(detrended_lc)
    logger.info("detector found %d candidate event(s)", len(events))

    rows: list[DepthRow] = []
    unmatched: list[PublishedDip] = []
    matched: list[tuple[PublishedDip, DipEvent]] = []
    for dip in published:
        nearby = [e for e in events if abs(e.t_min - dip.t_min_bkjd) <= tolerance_days]
        if not nearby:
            unmatched.append(dip)
            continue
        event = max(nearby, key=lambda e: e.depth)
        matched.append((dip, event))
        measured = measure_all_ways(
            raw_lc, detrended_lc, event, config, exclusion_factor, min_exclusion_days
        )

        drift = abs(measured["current"] - event.depth_ppm)
        if drift > 1.0:
            logger.warning(
                "'current' column (%.1f ppm) disagrees with DipEvent.depth_ppm (%.1f ppm) "
                "at t=%.2f -- the harness no longer mirrors the pipeline",
                measured["current"],
                event.depth_ppm,
                event.t_min,
            )

        rows.append(
            DepthRow(
                note=dip.note or f"t={dip.t_min_bkjd:.2f}",
                published_ppm=dip.depth_ppm,
                t_min=event.t_min,
                measured_ppm=measured,
            )
        )
    return rows, unmatched, matched, detrended_lc


def _ratio_summary(rows: list[DepthRow], key: str) -> tuple[float, float]:
    """Median and scatter of the measured/published ratio for one method."""
    values = np.array([r.ratio(key) for r in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.std(values))


def print_table(rows: list[DepthRow], markdown: bool) -> None:
    """Print the depth table, in plain text or as a research-log Markdown table."""
    if not rows:
        logger.error("no events matched -- nothing to tabulate")
        return

    header = ["Dip", "Published"] + [METHOD_LABELS[k] for k in METHOD_KEYS]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
    else:
        lines.append("  ".join(f"{h:>30}" for h in header))

    for row in rows:
        cells = [row.note, f"{row.published_ppm:.0f}"]
        for key in METHOD_KEYS:
            value = row.measured_ppm.get(key, float("nan"))
            cells.append(
                "n/a" if not np.isfinite(value) else f"{value:.0f} ({row.ratio(key):.2f}x)"
            )
        lines.append(
            "| " + " | ".join(cells) + " |" if markdown else "  ".join(f"{c:>30}" for c in cells)
        )

    summary = ["**median ratio**" if markdown else "median ratio", ""]
    for key in METHOD_KEYS:
        median, scatter = _ratio_summary(rows, key)
        summary.append(f"{median:.3f}x (sd {scatter:.3f})")
    lines.append(
        "| " + " | ".join(summary) + " |" if markdown else "  ".join(f"{c:>30}" for c in summary)
    )

    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="KIC 3542116", help="validation target to measure")
    parser.add_argument(
        "--source",
        choices=("cache", "synthetic", "mast"),
        default="cache",
        help="where the photometry comes from; only 'mast' touches the network",
    )
    parser.add_argument(
        "--exclusion-factor",
        type=float,
        default=3.0,
        help="detected-window multiple excluded from the out-of-event baseline",
    )
    parser.add_argument(
        "--min-exclusion-days",
        type=float,
        default=1.0,
        help="floor on the half-width of the excluded zone, in days",
    )
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep the rolling-median width -- the sharpest test of hypothesis 1",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional thresholds YAML; pipeline defaults are used when omitted",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    published, tolerance = load_published_dips(args.target)
    if not published:
        parser.error(f"{args.target} has no published dips recorded in {VALIDATION_TARGETS}")

    if args.source == "cache":
        raw_lc = load_from_cache(args.target)
    elif args.source == "synthetic":
        raw_lc = build_synthetic(published)
    else:
        raw_lc = load_from_mast(args.target)

    rows, unmatched, matched, detrended_lc = run_harness(
        raw_lc, published, tolerance, config, args.exclusion_factor, args.min_exclusion_days
    )
    for dip in unmatched:
        logger.warning(
            "no detected event within %.2f d of published epoch %.2f (%s)",
            tolerance,
            dip.t_min_bkjd,
            dip.note,
        )

    print()
    print(f"source: {raw_lc.meta.get('source')}   target: {raw_lc.target_id}")
    print(f"matched {len(rows)}/{len(published)} published epochs")
    print()
    print_table(rows, args.markdown)

    if args.sweep and matched:
        print()
        print("baseline-window sweep (hypothesis 1, depths in ppm):")
        print()
        print_sweep(matched, sweep_baseline_window(detrended_lc, matched), args.markdown)


if __name__ == "__main__":
    main()
