# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for the exocomet-hunter pipeline.

Declared in ``pyproject.toml`` as ``exocomet = "exocomet.cli:main"``, so this is
the first thing anyone who ``pip install``s the package touches. It is a thin
front end and owns no science: each subcommand wires together functions that
already exist and are unit-tested — :func:`exocomet.io.download.fetch_light_curve`,
:func:`exocomet.detrend.detrend.detrend_savgol`,
:class:`exocomet.detect.comet_detector.AsymmetricDipDetector` and
:func:`exocomet.calibration.injection.run_injection_recovery` — in the same
sequence the ``scripts/`` drivers use.

Every heavyweight or network-touching import lives inside the function that
needs it, mirroring the lazy ``import lightkurve as lk`` in
:mod:`exocomet.io.download`. That is what lets ``exocomet --help`` answer
instantly on a machine with no network and no science stack warmed up.

Usage
-----
    exocomet run "KIC 3542116" --mission Kepler
    exocomet validate
    exocomet inject --n 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from exocomet.core.config import PipelineConfig
    from exocomet.core.types import CandidateRecord, LightCurveData

__all__ = ["build_parser", "main"]

logger = logging.getLogger("exocomet")

DEFAULT_VALIDATION_TARGETS = Path("config/validation_targets.yaml")
MISSIONS = ("Kepler", "TESS")

#: Depth grid swept by ``exocomet inject`` when ``--depths`` is not given.
#: Brackets the published KIC 3542116 dips (491-1900 ppm) on both sides so the
#: completeness curve shows both the floor and the saturated end.
DEFAULT_INJECTION_DEPTHS = (2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3)


def _load_config(path: Path | None) -> PipelineConfig:
    """Load a :class:`PipelineConfig`, falling back to the built-in defaults."""
    from exocomet.core.config import PipelineConfig

    return PipelineConfig.from_yaml(path) if path is not None else PipelineConfig()


def _prepare_light_curve(
    target_id: str, mission: str, config: PipelineConfig, detrend: bool = True
) -> LightCurveData:
    """Fetch one target from MAST and detrend it, ready for detection.

    This is the same download-then-detrend sequence used by
    ``scripts/monitor_new_data.py`` and ``scripts/train_classifier.py``.
    """
    from exocomet.detrend.detrend import detrend_savgol
    from exocomet.io.download import fetch_light_curve

    lc = fetch_light_curve(target_id, mission=mission, discard_after_read=True)
    return detrend_savgol(lc, config.detrend) if detrend else lc


def _format_record(record: CandidateRecord) -> str:
    """Render one candidate as a single aligned line of console output."""
    event = record.event
    score = record.score
    mark = "FLAG" if score.flagged else "    "
    asym = f"{event.asymmetry.a_dur:+.3f}" if event.asymmetry is not None else "  n/a"
    return (
        f"  {mark}  t_min={event.t_min:12.4f}  depth={event.depth_ppm:7.1f} ppm  "
        f"dur={event.window.duration_days:5.2f} d  a_dur={asym}  "
        f"sig={score.significance:6.2f}  dBIC={score.delta_bic:8.2f}  "
        f"tau/sigma={score.tau_over_sigma:6.2f}"
    )


def _synthetic_light_curve(n_points: int, noise_sigma: float, seed: int) -> LightCurveData:
    """Build a flat, noise-only light curve to host injections offline.

    ``exocomet inject`` is a calibration command, and calibrating against real
    photometry (``--target``) is always preferable because real systematics are
    then part of the test. This synthetic host exists so the command is still
    runnable — and reviewable — without an archive round trip.
    """
    import numpy as np

    from exocomet.core.types import LightCurveData

    # Kepler long cadence (~29.4 min), matching the sampling the thresholds
    # in config/thresholds.yaml were tuned on.
    cadence_days = 0.02043
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=np.float64) * cadence_days
    flux = np.ones(n_points, dtype=np.float64) + rng.normal(0.0, noise_sigma, size=n_points)
    flux_err = np.full(n_points, max(noise_sigma, 1.0e-12), dtype=np.float64)
    return LightCurveData(
        target_id="SYNTHETIC",
        mission="SYNTHETIC",
        time=time,
        flux=flux,
        flux_err=flux_err,
        meta={"synthetic": True, "noise_sigma": noise_sigma},
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run the detection pipeline over a single target and print its events."""
    from exocomet.detect.comet_detector import AsymmetricDipDetector

    config = _load_config(args.config)
    lc = _prepare_light_curve(args.target, args.mission, config, detrend=not args.no_detrend)

    print(
        f"{lc.target_id} ({lc.mission}): {lc.n_points} cadences, "
        f"{lc.baseline_days:.1f} d baseline, cadence {lc.cadence_days * 1440:.1f} min"
    )

    records = AsymmetricDipDetector(config).run(lc)
    shown = records if args.all_events else [r for r in records if r.score.flagged]

    n_flagged = sum(1 for r in records if r.score.flagged)
    print(f"{len(records)} candidate event(s) detected, {n_flagged} flagged as comet-like")
    for record in sorted(shown, key=lambda r: r.event.t_min):
        print(_format_record(record))

    if args.out is not None:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_row(), default=str) + "\n")
        print(f"wrote {len(records)} row(s) to {args.out}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Re-detect the published dips in the known-truth stars.

    Exits non-zero if any literature dip is missed, so the command doubles as
    the validation gate described in ``docs/methodology.md``.
    """
    import yaml

    from exocomet.detect.comet_detector import AsymmetricDipDetector

    spec: dict[str, Any] = yaml.safe_load(args.targets.read_text(encoding="utf-8")) or {}
    tolerance = float(spec.get("match_tolerance_days", 0.5))
    targets = spec.get("targets") or []
    if not targets:
        print(f"no targets defined in {args.targets}", file=sys.stderr)
        return 1

    config = _load_config(args.config)
    detector = AsymmetricDipDetector(config)

    n_expected = 0
    n_recovered = 0
    for entry in targets:
        target_id = str(entry["target_id"])
        mission = str(entry.get("mission", "Kepler"))
        dips = entry.get("dips") or []
        print(f"\n{target_id} ({mission})")

        if not dips:
            # KIC 11084727 currently carries no transcribed epoch (Next Steps
            # B5). Skipping loudly is correct; counting it as a pass would make
            # the gate a lie, and failing on it would block on a missing number.
            print("  SKIP  no published epochs transcribed for this target")
            continue

        try:
            lc = _prepare_light_curve(target_id, mission, config)
            records = detector.run(lc)
        except Exception as exc:
            print(f"  ERROR {exc}", file=sys.stderr)
            n_expected += len(dips)
            continue

        for dip in dips:
            n_expected += 1
            t_expected = float(dip["t_min_bkjd"])
            matches = [r for r in records if abs(r.event.t_min - t_expected) <= tolerance]
            best = max(matches, key=lambda r: r.score.significance) if matches else None
            if best is None:
                print(f"  MISS  expected t_min={t_expected:.2f} ({dip.get('note', '')})")
                continue
            n_recovered += 1
            print(
                f"  PASS  expected t_min={t_expected:.2f} -> found {best.event.t_min:.2f}, "
                f"{best.event.depth_ppm:.0f} ppm (published {dip.get('depth_ppm', '?')} ppm), "
                f"flagged={best.score.flagged}"
            )

    print(f"\nvalidation: {n_recovered}/{n_expected} published dip(s) recovered")
    return 0 if n_expected and n_recovered == n_expected else 1


def cmd_inject(args: argparse.Namespace) -> int:
    """Measure detection completeness by injection and recovery."""
    from exocomet.calibration.injection import run_injection_recovery
    from exocomet.detect.comet_detector import AsymmetricDipDetector

    config = _load_config(args.config)

    if args.target is not None:
        lc = _prepare_light_curve(args.target, args.mission, config)
        print(f"injecting into real photometry: {lc.target_id} ({lc.mission})")
    else:
        lc = _synthetic_light_curve(args.n_points, args.noise, config.runtime.rng_seed)
        print(
            f"injecting into a synthetic host ({lc.n_points} cadences); "
            f"pass --target for real photometry"
        )

    depths = args.depths or list(DEFAULT_INJECTION_DEPTHS)
    grid = run_injection_recovery(
        lc,
        AsymmetricDipDetector(config),
        depths=depths,
        ingress_duration=args.ingress,
        egress_duration=args.egress,
        n_per_depth=args.n,
        config=config,
    )

    print(f"\n{grid.n_injected} injection(s), {len(depths)} depth(s), {args.n} per depth")
    print(f"{'depth (ppm)':>12}  {'detected':>9}  {'flagged':>8}")
    detected = grid.completeness_by_depth(flagged_only=False)
    flagged = grid.completeness_by_depth(flagged_only=True)
    for depth in sorted(detected):
        print(f"{depth * 1.0e6:12.0f}  {detected[depth]:9.2f}  {flagged[depth]:8.2f}")

    floor = grid.detection_floor()
    print(f"\noverall completeness (flagged): {grid.completeness():.2f}")
    print(
        f"50% detection floor: {floor * 1.0e6:.0f} ppm"
        if floor is not None
        else "50% detection floor: not reached by any tested depth"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser, including every subcommand.

    Separated from :func:`main` so that tests can exercise parsing without
    executing a pipeline or touching the network.
    """
    parser = argparse.ArgumentParser(
        prog="exocomet",
        description="Detect asymmetric, comet-like transit events in Kepler and TESS photometry.",
    )
    parser.add_argument("--version", action="version", version=_version_string())
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="emit INFO-level progress logging"
    )
    sub = parser.add_subparsers(dest="command", metavar="{run,validate,inject}")

    run = sub.add_parser("run", help="run the detection pipeline on one star")
    run.add_argument("target", help='catalogue identifier, e.g. "KIC 3542116"')
    run.add_argument("--mission", choices=MISSIONS, default="Kepler")
    run.add_argument("--config", type=Path, help="YAML overrides, e.g. config/thresholds.yaml")
    run.add_argument(
        "--all-events", action="store_true", help="print every event, not only flagged ones"
    )
    run.add_argument("--no-detrend", action="store_true", help="skip Savitzky-Golay detrending")
    run.add_argument("--out", type=Path, help="also write every event as JSON Lines to this path")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser(
        "validate", help="re-detect the published dips in the known-truth stars"
    )
    validate.add_argument("--targets", type=Path, default=DEFAULT_VALIDATION_TARGETS)
    validate.add_argument("--config", type=Path, help="YAML overrides")
    validate.set_defaults(func=cmd_validate)

    inject = sub.add_parser("inject", help="measure completeness by injection and recovery")
    inject.add_argument("--n", type=int, default=20, help="injections per depth")
    inject.add_argument("--depths", type=float, nargs="+", help="fractional depths to sweep")
    inject.add_argument("--ingress", type=float, default=0.1, help="ingress duration in days")
    inject.add_argument("--egress", type=float, default=0.5, help="egress duration in days")
    inject.add_argument("--target", help="inject into this star's real photometry")
    inject.add_argument("--mission", choices=MISSIONS, default="Kepler")
    inject.add_argument(
        "--n-points", type=int, default=2000, help="synthetic host length, when no --target"
    )
    inject.add_argument(
        "--noise", type=float, default=1.0e-4, help="synthetic host noise sigma, when no --target"
    )
    inject.add_argument("--config", type=Path, help="YAML overrides")
    inject.set_defaults(func=cmd_inject)

    return parser


def _version_string() -> str:
    """Return the installed package version, without importing the science stack."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"exocomet-hunter {version('exocomet-hunter')}"
    except PackageNotFoundError:  # pragma: no cover - source tree without an install
        return "exocomet-hunter (not installed)"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``exocomet`` command.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a failed run (a missed validation dip, an
        unavailable target, a malformed config), ``2`` for a usage error and
        ``130`` when interrupted.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.debug("command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
