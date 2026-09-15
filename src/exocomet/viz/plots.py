# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic and publication figures.

Plots here are working instruments, not decoration: the per-event figure exists
so that a human can check what the asymmetry measurement actually did — where it
put the minimum, where it found each half-depth crossing, and which cadences
went into each slope fit. A detection that looks wrong on this plot is wrong,
whatever the numbers say.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np

from exocomet.core.types import CandidateRecord, DipEvent, FloatArray, LightCurveData

__all__ = [
    "plot_asymmetry_distribution",
    "plot_detrend_comparison",
    "plot_event",
    "plot_light_curve",
    "plot_tail_direction",
]

PPM = 1.0e6


def _save(fig: plt.Figure, path: Path | str | None) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_light_curve(
    lc: LightCurveData,
    events: list[DipEvent] | None = None,
    known_epochs: dict[str, float] | None = None,
    path: Path | str | None = None,
    title: str | None = None,
) -> Path | None:
    """Plot a full light curve, marking detected events and any known epochs.

    Parameters
    ----------
    known_epochs
        Published epochs to overlay, keyed by label. Drawn in a contrasting
        colour so that recovered-versus-missed is visible at a glance.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(lc.time, (lc.flux - 1.0) * PPM, lw=0.4, color="0.35", rasterized=True)

    for event in events or []:
        ax.axvspan(event.window.start_time, event.window.end_time, color="tab:blue", alpha=0.25)

    for label, epoch in (known_epochs or {}).items():
        ax.axvline(epoch, color="tab:red", ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            label,
            xy=(epoch, ax.get_ylim()[1]),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="tab:red",
        )

    ax.set_xlabel(f"Time ({lc.mission} days)")
    ax.set_ylabel("Relative flux (ppm)")
    ax.set_title(title or f"{lc.target_id} — {lc.n_points:,} cadences, {lc.baseline_days:.0f} d")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def plot_detrend_comparison(
    raw: LightCurveData,
    detrended: LightCurveData,
    path: Path | str | None = None,
    window_days: float | None = None,
    centre: float | None = None,
) -> Path | None:
    """Show flux before and after detrending, to verify signal survived it.

    The question this answers is the one that matters for detrending: did the
    filter remove the trend without eating the dip?
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, lc, label in (
        (axes[0], raw, "Raw (normalised)"),
        (axes[1], detrended, "Detrended"),
    ):
        ax.plot(lc.time, (lc.flux - 1.0) * PPM, lw=0.4, color="0.3", rasterized=True)
        ax.set_ylabel("ppm")
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.2)

    if centre is not None and window_days is not None:
        axes[0].set_xlim(centre - window_days / 2, centre + window_days / 2)

    scatter = 1.4826 * float(np.median(np.abs(detrended.flux - np.median(detrended.flux)))) * PPM
    axes[1].set_xlabel(f"Time (days)   |   post-detrend scatter {scatter:.0f} ppm")
    fig.tight_layout()
    return _save(fig, path)


def plot_event(
    lc: LightCurveData,
    record: CandidateRecord,
    baseline: FloatArray | None = None,
    path: Path | str | None = None,
    pad_factor: float = 3.0,
    known_epoch: float | None = None,
) -> Path | None:
    """Plot one event with the full asymmetry measurement drawn on top.

    Marks the fitted minimum, the half-depth level, both half-depth crossings,
    and the ingress/egress slope fits — everything the score depends on, so that
    a wrong number is visible as a wrong picture.
    """
    event = record.event
    window = event.window
    span = max(window.duration_days, 1e-3) * pad_factor
    mask = (lc.time >= event.t_min - span) & (lc.time <= event.t_min + span)

    t = lc.time[mask]
    f = (lc.flux[mask] - 1.0) * PPM

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        t, f, yerr=lc.flux_err[mask] * PPM, fmt="o", ms=2.5, lw=0.5, color="0.25", alpha=0.8
    )

    if baseline is not None:
        ax.plot(t, (baseline[mask] - 1.0) * PPM, color="tab:green", lw=1.0, label="local baseline")
        level = float(np.median(baseline[window.start_index : window.end_index + 1]))
    else:
        level = 1.0

    ax.axvspan(
        window.start_time, window.end_time, color="tab:blue", alpha=0.12, label="detection window"
    )
    ax.axvline(event.t_min, color="tab:blue", ls="-", lw=1.2, label=f"minimum {event.t_min:.3f}")

    if known_epoch is not None:
        ax.axvline(known_epoch, color="tab:red", ls="--", lw=1.2, label=f"published {known_epoch:.2f}")

    asym = event.asymmetry
    if asym is not None:
        half_ppm = (level - 0.5 * event.depth - 1.0) * PPM
        ax.axhline(half_ppm, color="tab:orange", ls=":", lw=1.0, label="half depth")
        for side, colour in ((asym.ingress, "tab:purple"), (asym.egress, "tab:brown")):
            ax.axvline(side.half_depth_time, color=colour, ls=":", lw=1.2)

        ax.set_title(
            f"{event.target_id} @ {event.t_min:.3f}   "
            f"depth {event.depth_ppm:.0f} ppm   "
            f"ingress {asym.ingress.duration_days * 24:.1f} h / "
            f"egress {asym.egress.duration_days * 24:.1f} h\n"
            f"A_dur = {asym.a_dur:+.3f}   A_slope = {asym.a_slope:+.3f}   "
            f"ΔBIC = {record.score.delta_bic:.1f}   flagged = {record.score.flagged}",
            fontsize=9,
        )
    else:
        ax.set_title(
            f"{event.target_id} @ {event.t_min:.3f} — unmeasurable "
            f"({event.unmeasurable_reason})",
            fontsize=9,
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative flux (ppm)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_asymmetry_distribution(
    records: list[CandidateRecord],
    path: Path | str | None = None,
    threshold: float | None = None,
) -> Path | None:
    """Histogram the asymmetry population, separating flagged from unflagged.

    A search result is only interpretable against the distribution it came from:
    this is the figure that shows whether flagged events are a distinct
    population or the tail of an ordinary one.
    """
    values = [
        r.event.asymmetry.a_dur for r in records if r.event.asymmetry is not None
    ]
    flagged = [
        r.event.asymmetry.a_dur
        for r in records
        if r.event.asymmetry is not None and r.score.flagged
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if values:
        bins = np.linspace(-1, 1, 41)
        ax.hist(values, bins=bins, color="0.6", label=f"all events ({len(values)})")
        if flagged:
            ax.hist(flagged, bins=bins, color="tab:blue", label=f"flagged ({len(flagged)})")

    if threshold is not None:
        ax.axvline(threshold, color="tab:red", ls="--", lw=1.0, label="flagging threshold")

    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Duration asymmetry $A_{dur}$   (positive = comet-like: fast fade, slow recovery)")
    ax.set_ylabel("Number of events")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def _inject_ramped_dip(
    time: FloatArray,
    flux: FloatArray,
    t0: float,
    depth: float,
    ingress_duration: float,
    egress_duration: float,
) -> FloatArray:
    """Add one piecewise-linear dip with independently set ramp durations."""
    out = flux.copy()
    falling = (time >= t0 - ingress_duration) & (time <= t0)
    out[falling] -= depth * (1.0 - (t0 - time[falling]) / ingress_duration)
    rising = (time > t0) & (time <= t0 + egress_duration)
    out[rising] -= depth * (1.0 - (time[rising] - t0) / egress_duration)
    return out


def plot_tail_direction(
    path: Path | str | None = None,
    config: "PipelineConfig | None" = None,
    depth: float = 2.0e-3,
    fast_ramp_days: float = 0.1,
    slow_ramp_days: float = 0.5,
    noise_sigma: float = 1.0e-4,
) -> Path | None:
    """Show that the flagging rule discriminates tail DIRECTION, not asymmetry.

    Two synthetic events are built with identical depth and identical ramp
    durations, differing only in which side is the slow one: a trailing tail
    (fast ingress, slow egress — the physical comet case) and a leading tail
    (the time-reverse, which no dust tail produces). Both are strongly
    asymmetric, so a rule built on ``|A_dur|`` flags both. The rule based on
    ``delta_bic`` and ``tau_over_sigma`` flags only the trailing-tail event,
    which is the whole point of the fix.

    Everything here is generated in-process; nothing is read from disk or the
    network, so the figure is reproducible from the source alone.
    """
    from exocomet.core.config import PipelineConfig
    from exocomet.detect.comet_detector import AsymmetricDipDetector

    cfg = config or PipelineConfig()
    detector = AsymmetricDipDetector(cfg)

    cadence = 0.02043  # Kepler long cadence, days
    n_points = 2000
    time = np.arange(n_points, dtype=np.float64) * cadence
    t0 = float(time[n_points // 2])

    panels = (
        ("Trailing tail (comet-like)", fast_ramp_days, slow_ramp_days, 11),
        ("Leading tail (time-reversed)", slow_ramp_days, fast_ramp_days, 12),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (label, ingress, egress, seed) in zip(axes, panels, strict=True):
        rng = np.random.default_rng(seed)
        flux = 1.0 + rng.normal(0.0, noise_sigma, size=n_points)
        flux = _inject_ramped_dip(time, flux, t0, depth, ingress, egress)
        flux_err = np.full(n_points, noise_sigma, dtype=np.float64)

        lc = LightCurveData(
            target_id=f"SYNTHETIC-{label}",
            mission="SYNTHETIC",
            time=time,
            flux=flux,
            flux_err=flux_err,
            meta={"synthetic": True, "noise_sigma": noise_sigma},
        )
        records = detector.run(lc)

        span = 4.0 * max(ingress, egress)
        mask = (time >= t0 - span) & (time <= t0 + span)
        ax.errorbar(
            (time[mask] - t0) * 24.0,
            (flux[mask] - 1.0) * PPM,
            yerr=flux_err[mask] * PPM,
            fmt="o",
            ms=2.0,
            lw=0.4,
            color="0.3",
            alpha=0.8,
        )
        ax.axvline(0.0, color="tab:blue", lw=0.8, ls="-", alpha=0.6)

        if records:
            score = records[0].score
            verdict = "FLAGGED" if score.flagged else "not flagged"
            colour = "tab:green" if score.flagged else "tab:red"
            a_dur = (
                f"{records[0].event.asymmetry.a_dur:+.2f}"
                if records[0].event.asymmetry is not None
                else "n/a"
            )
            stats = (
                rf"$\Delta$BIC = {score.delta_bic:.0f}   "
                rf"$\tau/\sigma$ = {score.tau_over_sigma:.2f}   "
                rf"$A_{{dur}}$ = {a_dur}"
            )
        else:
            verdict, colour = "no event detected", "tab:red"
            stats = "—"

        ax.set_title(
            f"{label}\ningress {ingress * 24:.1f} h / egress {egress * 24:.1f} h",
            fontsize=10,
        )
        ax.annotate(
            f"{verdict}\n{stats}",
            xy=(0.03, 0.06),
            xycoords="axes fraction",
            fontsize=8.5,
            color=colour,
            va="bottom",
        )
        ax.set_xlabel("Hours from minimum")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Relative flux (ppm)")
    fig.suptitle(
        "Flagging on "
        rf"$\Delta$BIC > {cfg.scoring.delta_bic_threshold:g} and "
        rf"$\tau/\sigma$ > {cfg.scoring.tau_over_sigma_threshold:g}"
        " keeps only the trailing tail;\nthe two events are mirror images, so "
        r"$|A_{dur}|$ alone cannot tell them apart",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, path)
