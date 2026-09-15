# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Regression gate for recovery of the published exocomet dips.

The project's headline claim is that this pipeline independently recovers the
comet-shaped transits already reported in the literature for KIC 3542116. Until
now that claim was checked by hand. This script turns it into an exam with an
exit code, so any change to detrending, candidate finding or scoring that loses
a known event fails loudly instead of quietly.

What is checked
---------------
Only **epoch recovery**: for every dip listed in ``config/validation_targets.yaml``
there must be a detected event whose flux minimum lies within
``match_tolerance_days`` of the published epoch. The ``flagged`` boolean is
reported for information but deliberately does *not* decide pass/fail — the
flagging rule is a tunable policy that is expected to move, while "the detector
finds an event at this epoch" is the stable contract the published result rests
on.

Offline verification
--------------------
The live run needs one networked download per target, and downloads are
deliberately serialised (the download layer has no timeout yet). So the entire
matching and pass/fail path is exercisable without touching the network:
``--offline`` builds synthetic light curves that carry injected dips at exactly
the published epochs and runs the real detrend/detect/score pipeline over them.
``--offline-drop-epoch`` omits one of those injected dips, which must make the
gate fail — that is how the gate is shown to be capable of failing at all.

Usage
-----
    # prove the logic, no network:
    python scripts/run_validation.py --offline
    python scripts/run_validation.py --offline --offline-drop-epoch 991.95  # must exit 1

    # the real gate (one MAST download per target, serialised):
    python scripts/run_validation.py

Exit codes
----------
``0``
    Every known dip of every enabled target was recovered.
``1``
    At least one known dip was missed, or a target failed outright.
``2``
    The run could not be carried out (bad config, unreadable file).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exocomet.core.config import PipelineConfig
from exocomet.core.types import LightCurveData
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("run_validation")

DEFAULT_TARGETS_PATH = REPO_ROOT / "config" / "validation_targets.yaml"

#: Ingress/egress used when synthesising a comet-shaped dip offline. Steep in,
#: slow out, matching the profile the detector is meant to recognise.
OFFLINE_INGRESS_DAYS = 0.12
OFFLINE_EGRESS_DAYS = 0.6
#: Photometric scatter of the synthetic series. Well below the shallowest
#: published dip (491 ppm) so that offline mode tests the matching logic rather
#: than the detector's sensitivity limit.
OFFLINE_NOISE_SIGMA = 5.0e-5
OFFLINE_CADENCE_DAYS = 0.02043
OFFLINE_PAD_DAYS = 6.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KnownDip:
    """One dip transcribed from the literature."""

    t_min_bkjd: float
    depth_ppm: float
    note: str = ""


@dataclass(frozen=True)
class ValidationTarget:
    """A star with published dips the pipeline is required to recover."""

    target_id: str
    mission: str
    dips: tuple[KnownDip, ...]
    expected_dip_count: int | None = None
    reference: str = ""

    @property
    def has_known_dips(self) -> bool:
        """Whether this target carries any epoch to check against."""
        return len(self.dips) > 0


@dataclass(frozen=True)
class ValidationSpec:
    """The full contents of ``config/validation_targets.yaml``."""

    match_tolerance_days: float
    targets: tuple[ValidationTarget, ...]


def load_validation_spec(path: Path | str) -> ValidationSpec:
    """Read and validate the known-dip configuration.

    Raises
    ------
    ValueError
        If the file is missing required keys or is structurally wrong. A
        malformed validation config must never degrade into a vacuous pass.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "match_tolerance_days" not in raw:
        raise ValueError(f"{path}: missing 'match_tolerance_days'")
    tolerance = float(raw["match_tolerance_days"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"{path}: match_tolerance_days must be positive and finite")

    raw_targets = raw.get("targets") or []
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(f"{path}: 'targets' must be a non-empty list")

    targets: list[ValidationTarget] = []
    for entry in raw_targets:
        if not isinstance(entry, dict) or "target_id" not in entry:
            raise ValueError(f"{path}: every target needs a 'target_id'")
        dips = tuple(
            KnownDip(
                t_min_bkjd=float(d["t_min_bkjd"]),
                depth_ppm=float(d.get("depth_ppm", float("nan"))),
                note=str(d.get("note", "")),
            )
            for d in (entry.get("dips") or [])
        )
        expected = entry.get("expected_dip_count")
        targets.append(
            ValidationTarget(
                target_id=str(entry["target_id"]),
                mission=str(entry.get("mission", "Kepler")),
                dips=dips,
                expected_dip_count=None if expected is None else int(expected),
                reference=str(entry.get("reference", "")),
            )
        )
    return ValidationSpec(match_tolerance_days=tolerance, targets=tuple(targets))


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DipMatch:
    """Outcome for a single known dip."""

    known: KnownDip
    matched: bool
    detected_t_min: float | None = None
    offset_days: float | None = None
    detected_depth_ppm: float | None = None
    flagged: bool | None = None

    @property
    def status(self) -> str:
        """``PASS`` when the dip was recovered, otherwise ``FAIL``."""
        return "PASS" if self.matched else "FAIL"


def match_dips(
    known: tuple[KnownDip, ...],
    detections: list[dict[str, Any]],
    tolerance_days: float,
) -> list[DipMatch]:
    """Pair published epochs with detected events, nearest-first.

    Each detected event can satisfy at most one known dip: the pairs are taken
    in order of increasing separation, so two published epochs cannot both be
    "recovered" by the same single detection. Detections outside the tolerance
    of every known epoch are simply unused (spurious extra detections are not a
    failure of the recovery gate — missing a known dip is).

    Parameters
    ----------
    known
        Published dips, in config order.
    detections
        Detected events as dictionaries carrying at least ``t_min``; optional
        ``depth_ppm`` and ``flagged`` are reported but never decide the match.
    tolerance_days
        Maximum allowed ``|detected - published|`` separation.

    Returns
    -------
    list of DipMatch
        One entry per known dip, in config order.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, dip in enumerate(known):
        for j, det in enumerate(detections):
            offset = abs(float(det["t_min"]) - dip.t_min_bkjd)
            if offset <= tolerance_days:
                pairs.append((offset, i, j))
    pairs.sort()

    assigned: dict[int, int] = {}
    used_detections: set[int] = set()
    for _, i, j in pairs:
        if i in assigned or j in used_detections:
            continue
        assigned[i] = j
        used_detections.add(j)

    results: list[DipMatch] = []
    for i, dip in enumerate(known):
        j = assigned.get(i)
        if j is None:
            results.append(DipMatch(known=dip, matched=False))
            continue
        det = detections[j]
        results.append(
            DipMatch(
                known=dip,
                matched=True,
                detected_t_min=float(det["t_min"]),
                offset_days=abs(float(det["t_min"]) - dip.t_min_bkjd),
                detected_depth_ppm=(
                    None if det.get("depth_ppm") is None else float(det["depth_ppm"])
                ),
                flagged=(None if det.get("flagged") is None else bool(det["flagged"])),
            )
        )
    return results


@dataclass
class TargetResult:
    """Everything the report needs to know about one validated target."""

    target: ValidationTarget
    matches: list[DipMatch] = field(default_factory=list)
    n_detections: int = 0
    skipped_reason: str | None = None
    error: str | None = None

    @property
    def skipped(self) -> bool:
        """Whether the target carried no epochs to check (a known gap, not a pass)."""
        return self.skipped_reason is not None

    @property
    def n_recovered(self) -> int:
        """How many published dips were recovered."""
        return sum(1 for m in self.matches if m.matched)

    @property
    def n_missed(self) -> int:
        """How many published dips were missed."""
        return sum(1 for m in self.matches if not m.matched)

    @property
    def passed(self) -> bool:
        """A target passes only when it was actually checked and missed nothing.

        A skipped target is explicitly *not* a pass: counting an untested star
        as green is exactly how a gate stops being a gate.
        """
        if self.error is not None or self.skipped:
            return False
        return bool(self.matches) and self.n_missed == 0


# --------------------------------------------------------------------------- #
# Light-curve providers
# --------------------------------------------------------------------------- #


def fetch_target_light_curve(target: ValidationTarget, cache_dir: Path) -> LightCurveData:
    """Download real photometry for one target (the networked path).

    Imported lazily so that ``--offline`` never touches ``lightkurve`` or the
    archive client at all.
    """
    from exocomet.io.download import fetch_light_curve

    return fetch_light_curve(
        target.target_id,
        mission=target.mission,
        cache_dir=cache_dir,
        discard_after_read=False,
    )


def synthetic_light_curve(
    target: ValidationTarget,
    drop_epochs: tuple[float, ...] = (),
    noise_sigma: float = OFFLINE_NOISE_SIGMA,
    seed: int = 11,
) -> LightCurveData:
    """Build a light curve carrying injected dips at the published epochs.

    Reuses the synthetic-data helpers the unit tests already rely on
    (``tests/conftest.py``), so offline mode and the unit suite share one
    definition of "an injected comet-shaped dip".

    Parameters
    ----------
    target
        Target whose published epochs are injected.
    drop_epochs
        Epochs deliberately *not* injected, used to prove the gate fails when a
        known dip is genuinely absent. Matched against the config epochs with a
        small numerical tolerance.
    noise_sigma
        Photometric scatter of the synthetic series.
    seed
        RNG seed, so an offline run is reproducible.
    """
    if not target.has_known_dips:
        raise ValueError(f"{target.target_id}: no known dips to synthesise")

    sys.path.insert(0, str(REPO_ROOT))
    from tests.conftest import inject_asymmetric_dip, make_flat_light_curve

    epochs = [d.t_min_bkjd for d in target.dips]
    t_start = min(epochs) - OFFLINE_PAD_DAYS
    t_end = max(epochs) + OFFLINE_PAD_DAYS
    n_points = int((t_end - t_start) / OFFLINE_CADENCE_DAYS) + 1

    lc = make_flat_light_curve(
        n_points=n_points,
        cadence_days=OFFLINE_CADENCE_DAYS,
        noise_sigma=noise_sigma,
        seed=seed,
        target_id=target.target_id,
    )
    # make_flat_light_curve starts at t=0; shift onto the BKJD epochs.
    time = lc.time + t_start
    flux = lc.flux
    injected: list[float] = []
    for dip in target.dips:
        if any(abs(dip.t_min_bkjd - dropped) < 1.0e-6 for dropped in drop_epochs):
            logger.warning(
                "offline: deliberately NOT injecting %s at t=%.3f",
                target.target_id,
                dip.t_min_bkjd,
            )
            continue
        depth = dip.depth_ppm * 1.0e-6
        if not np.isfinite(depth) or depth <= 0:
            depth = 1.0e-3
        flux = inject_asymmetric_dip(
            time,
            flux,
            t0=dip.t_min_bkjd,
            depth=float(depth),
            ingress_duration=OFFLINE_INGRESS_DAYS,
            egress_duration=OFFLINE_EGRESS_DAYS,
        )
        injected.append(dip.t_min_bkjd)

    return LightCurveData(
        target_id=target.target_id,
        mission=target.mission,
        time=time,
        flux=flux,
        flux_err=lc.flux_err,
        meta={
            "synthetic": True,
            "source": "offline-fixture",
            "injected_epochs": injected,
            "dropped_epochs": list(drop_epochs),
            "noise_sigma": noise_sigma,
        },
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def detect_events(lc: LightCurveData, config: PipelineConfig) -> list[dict[str, Any]]:
    """Run the real detrend + detect + score pipeline and flatten the events.

    The returned dictionaries carry ``t_min`` (the quantity the gate matches
    on), plus ``depth_ppm`` and ``flagged`` for the report only.
    """
    detrended = detrend_savgol(lc, config.detrend)
    records = AsymmetricDipDetector(config).run(detrended)
    return [
        {
            "t_min": float(r.event.t_min),
            "depth_ppm": float(r.event.depth_ppm),
            "flagged": bool(r.score.flagged),
        }
        for r in records
    ]


def validate_target(
    target: ValidationTarget,
    tolerance_days: float,
    config: PipelineConfig,
    *,
    offline: bool,
    drop_epochs: tuple[float, ...] = (),
    cache_dir: Path = Path("data/raw"),
    noise_sigma: float = OFFLINE_NOISE_SIGMA,
) -> TargetResult:
    """Validate one target end to end, never raising on a per-target failure."""
    if not target.has_known_dips:
        return TargetResult(
            target=target,
            skipped_reason="no published epochs transcribed in config (known gap)",
        )

    try:
        lc = (
            synthetic_light_curve(target, drop_epochs=drop_epochs, noise_sigma=noise_sigma)
            if offline
            else fetch_target_light_curve(target, cache_dir)
        )
        detections = detect_events(lc, config)
    # A target that blows up must fail the gate, not abort the whole report.
    except Exception as exc:
        logger.exception("%s: validation could not be completed", target.target_id)
        return TargetResult(target=target, error=f"{type(exc).__name__}: {exc}")

    return TargetResult(
        target=target,
        matches=match_dips(target.dips, detections, tolerance_days),
        n_detections=len(detections),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional number, rendering ``None`` as a right-aligned dash.

    The dash is padded to the same width as the number would have been, so a
    FAIL row stays aligned with the PASS rows above it.
    """
    width = int(spec.split(".")[0] or 0)
    if value is None:
        return "-".rjust(width)
    return format(value, spec)


def print_report(results: list[TargetResult], tolerance_days: float) -> None:
    """Print the per-dip PASS/FAIL table and the summary."""
    print()
    print("=" * 78)
    print("EXOCOMET VALIDATION GATE -- recovery of published dips")
    print(f"match tolerance: +/-{tolerance_days:g} d (epoch recovery only; "
          f"'flagged' is reported, not required)")
    print("=" * 78)

    for result in results:
        target = result.target
        print()
        print(f"{target.target_id}  ({target.mission})")
        if target.reference:
            print(f"  reference: {target.reference}")

        if result.error is not None:
            print(f"  ERROR: {result.error}")
            print("  -> target FAILED (could not be validated)")
            continue

        if result.skipped:
            print(f"  SKIPPED: {result.skipped_reason}")
            expected = target.expected_dip_count
            if expected:
                print(f"  NOT VERIFIED: {expected} published dip(s) remain unchecked "
                      f"for this target")
            continue

        print(f"  detections: {result.n_detections} event(s) from the pipeline")
        print(
            f"  {'status':6}  {'known t_min':>12}  {'detected':>12}  "
            f"{'offset (d)':>10}  {'known ppm':>9}  {'found ppm':>9}  flagged  note"
        )
        for match in result.matches:
            print(
                f"  {match.status:6}  {match.known.t_min_bkjd:12.3f}  "
                f"{_fmt(match.detected_t_min, '12.3f')}  "
                f"{_fmt(match.offset_days, '10.3f')}  "
                f"{match.known.depth_ppm:9.0f}  "
                f"{_fmt(match.detected_depth_ppm, '9.0f')}  "
                f"{str(match.flagged) if match.flagged is not None else '-':>7}  "
                f"{match.known.note}"
            )
        expected = target.expected_dip_count
        if expected is not None and expected != len(target.dips):
            print(
                f"  NOTE: config lists {len(target.dips)} epoch(s) but expects "
                f"{expected}; {expected - len(target.dips)} remain untranscribed"
            )
        print(f"  -> {result.n_recovered}/{len(result.matches)} recovered "
              f"({'PASS' if result.passed else 'FAIL'})")

    checked = [r for r in results if not r.skipped and r.error is None]
    total_dips = sum(len(r.matches) for r in checked)
    total_recovered = sum(r.n_recovered for r in checked)
    skipped = [r for r in results if r.skipped]
    errored = [r for r in results if r.error is not None]

    print()
    print("-" * 78)
    print(f"SUMMARY: {total_recovered}/{total_dips} published dip(s) recovered "
          f"across {len(checked)} validated target(s)")
    if skipped:
        print(f"         {len(skipped)} target(s) SKIPPED (no epochs in config): "
              + ", ".join(r.target.target_id for r in skipped))
        print("         skipped targets are NOT counted as passes")
    if errored:
        print(f"         {len(errored)} target(s) ERRORED: "
              + ", ".join(r.target.target_id for r in errored))
    verdict = "PASS" if all(r.passed for r in results if not r.skipped) and checked else "FAIL"
    print(f"VERDICT: {verdict}")
    print("-" * 78)


def results_to_json(results: list[TargetResult], tolerance_days: float) -> dict[str, Any]:
    """Serialise the run for archiving alongside a results table."""
    return {
        "match_tolerance_days": tolerance_days,
        "targets": [
            {
                "target_id": r.target.target_id,
                "mission": r.target.mission,
                "skipped_reason": r.skipped_reason,
                "error": r.error,
                "n_detections": r.n_detections,
                "n_known": len(r.target.dips),
                "n_recovered": r.n_recovered,
                "passed": r.passed,
                "dips": [
                    {
                        "known_t_min": m.known.t_min_bkjd,
                        "note": m.known.note,
                        "matched": m.matched,
                        "detected_t_min": m.detected_t_min,
                        "offset_days": m.offset_days,
                        "detected_depth_ppm": m.detected_depth_ppm,
                        "flagged": m.flagged,
                    }
                    for m in r.matches
                ],
            }
            for r in results
        ],
    }


def exit_code(results: list[TargetResult], *, strict_skipped: bool = False) -> int:
    """Map results onto the gate's exit status.

    Non-zero whenever a known dip was missed or a target could not be
    validated. Targets with no transcribed epochs are a documented gap: they do
    not fail the gate by default, but ``--strict`` makes them fail too, for use
    once the missing epoch has been read from the paper.
    """
    checked = [r for r in results if not r.skipped and r.error is None]
    if not checked:
        return 1
    if any(r.error is not None for r in results):
        return 1
    if any(r.n_missed > 0 for r in checked):
        return 1
    if strict_skipped and any(r.skipped for r in results):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the validation gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", type=Path, default=DEFAULT_TARGETS_PATH, help="validation targets YAML"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="pipeline thresholds YAML (defaults if omitted)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run against synthetic light curves with dips injected at the published "
        "epochs; exercises all matching and pass/fail logic with no network access",
    )
    parser.add_argument(
        "--offline-drop-epoch",
        type=float,
        action="append",
        default=[],
        metavar="BKJD",
        help="offline only: do not inject this published epoch, so the gate must FAIL "
        "(use to prove the gate is capable of failing)",
    )
    parser.add_argument(
        "--offline-noise",
        type=float,
        default=OFFLINE_NOISE_SIGMA,
        help="offline only: photometric scatter of the synthetic series",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="restrict to these target IDs (repeatable)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--json-out", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a target has no transcribed epochs to check",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        spec = load_validation_spec(args.targets)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"cannot run validation: {exc}", file=sys.stderr)
        return 2

    try:
        config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"cannot load pipeline config: {exc}", file=sys.stderr)
        return 2

    targets = spec.targets
    if args.target:
        wanted = set(args.target)
        targets = tuple(t for t in targets if t.target_id in wanted)
        if not targets:
            print(f"no validation target matches {sorted(wanted)}", file=sys.stderr)
            return 2

    if args.offline_drop_epoch and not args.offline:
        print("--offline-drop-epoch requires --offline", file=sys.stderr)
        return 2

    if not args.offline:
        print(
            "Running the LIVE gate: one MAST download per target, serialised. "
            "Verify with --offline first.",
            file=sys.stderr,
        )

    results = [
        validate_target(
            target,
            spec.match_tolerance_days,
            config,
            offline=args.offline,
            drop_epochs=tuple(args.offline_drop_epoch),
            cache_dir=args.cache_dir,
            noise_sigma=args.offline_noise,
        )
        for target in targets
    ]

    print_report(results, spec.match_tolerance_days)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results_to_json(results, spec.match_tolerance_days), indent=2),
            encoding="utf-8",
        )

    return exit_code(results, strict_skipped=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
