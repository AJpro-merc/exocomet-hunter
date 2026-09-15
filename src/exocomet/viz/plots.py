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
