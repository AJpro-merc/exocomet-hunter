"""Identification of candidate dimming events.

The first pass is deliberately cheap and permissive: mark every cadence sitting
more than a few robust sigma below the local baseline, keep only runs long
enough to rule out single-sample glitches, and grow each run outwards until the
flux has recovered. Deciding whether a surviving event is *comet-shaped* is the
job of :mod:`exocomet.detect.asymmetry`, not of this module.
"""

from __future__ import annotations

import numpy as np

from exocomet.core.config import CandidateConfig
from exocomet.core.types import EventWindow, FloatArray
from exocomet.detect.baseline import days_to_cadences

__all__ = ["depth_significance", "find_candidate_events"]


def depth_significance(flux: FloatArray, baseline: FloatArray, sigma: FloatArray) -> FloatArray:
    """Return how many robust sigma each cadence sits *below* the baseline.

    Positive values are dimmings; negative values are brightenings.
    """
    return (baseline - flux) / sigma


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` index pairs for each run of ``True``."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by at most ``max_gap`` cadences."""
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _expand_window(
    time: FloatArray,
    significance: FloatArray,
    start: int,
    end: int,
    recovery_sigma: float,
    max_window_days: float,
) -> tuple[int, int]:
    """Grow a run outwards until the flux recovers, subject to a duration cap.

    Expansion stops on each side at the first cadence whose depth significance
    has fallen back to ``recovery_sigma`` or below, or when the window would
    exceed ``max_window_days``.
    """
    n = time.size
    centre_time = time[(start + end) // 2]
    half_span = max_window_days / 2.0

    new_start = start
    while (
        new_start - 1 >= 0
        and significance[new_start - 1] > recovery_sigma
        and centre_time - time[new_start - 1] <= half_span
    ):
        new_start -= 1

    new_end = end
    while (
        new_end + 1 < n
        and significance[new_end + 1] > recovery_sigma
        and time[new_end + 1] - centre_time <= half_span
    ):
        new_end += 1

    return new_start, new_end


def find_candidate_events(
    time: FloatArray,
    flux: FloatArray,
    baseline: FloatArray,
    sigma: FloatArray,
    config: CandidateConfig | None = None,
) -> list[EventWindow]:
    """Find candidate dimming events in a detrended light curve.

    Parameters
    ----------
    time
        Observation times in days, strictly increasing.
    flux
        Detrended, normalised flux.
    baseline
        Local baseline from :func:`exocomet.detect.baseline.rolling_baseline`.
    sigma
        Robust per-cadence noise from
        :func:`exocomet.detect.baseline.local_noise`.
    config
        Threshold settings; defaults are used when omitted.

    Returns
    -------
    list of EventWindow
        Non-overlapping windows, ordered in time.

    Notes
    -----
    The thresholding step is fully vectorised; only the outward expansion of
    each surviving run uses a loop, and that loop is bounded by
    ``max_window_days``, so the cost stays linear in the number of cadences.
    """
    cfg = config or CandidateConfig()

    significance = depth_significance(flux, baseline, sigma)
    mask = significance > cfg.sigma_threshold

    runs = [(s, e) for s, e in _find_runs(mask) if e - s + 1 >= cfg.min_consecutive]
    runs = _merge_runs(runs, cfg.max_gap_cadences)

    # The baseline and noise estimates use truncated windows near the ends of
    # the series, where they are measurably noisier. Events that rely on that
    # region are dropped rather than trusted; the same guard covers the leading
    # and trailing cadences of a sector or quarter, a known artefact source.
    n_cadences = time.size
    cadence_days = float(np.median(np.diff(time))) if n_cadences > 1 else 1.0
    margin = max(days_to_cadences(cfg.edge_margin_days, cadence_days), 0)
    first_valid, last_valid = margin, n_cadences - 1 - margin

    windows: list[EventWindow] = []
    last_end = -1
    for start, end in runs:
        if end < first_valid or start > last_valid:
            continue
        new_start, new_end = _expand_window(
            time, significance, start, end, cfg.recovery_sigma, cfg.max_window_days
        )
        # Expansion can make neighbouring windows touch; keep them disjoint so
        # that a single cadence is never attributed to two events.
        new_start = max(new_start, last_end + 1)
        if new_start > new_end:
            continue
        windows.append(
            EventWindow(
                start_index=new_start,
                end_index=new_end,
                start_time=float(time[new_start]),
                end_time=float(time[new_end]),
            )
        )
        last_end = new_end

    return windows
