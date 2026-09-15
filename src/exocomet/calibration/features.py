# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Feature extraction: turning a detected event into a row of numbers.

The classifier never sees a light curve. It sees only the measurements this
module produces — depth, durations, shape ratios, model-comparison evidence.
That is a deliberate design choice with two consequences worth stating:

1. **Interpretability.** Every feature has a physical meaning, so when the model
   ranks an event highly it is possible to say which measured property drove it.
   A network reading raw flux could not be interrogated that way.
2. **Honesty about provenance.** The measurements come from deterministic,
   testable code. The model only weighs evidence it is handed; it never invents
   a detection.
"""

from __future__ import annotations

import numpy as np

from exocomet.core.types import CandidateRecord, LightCurveData

__all__ = ["FEATURE_NAMES", "extract_features", "features_to_matrix"]

#: Order matters and is fixed: a trained model stores coefficients against these
#: positions, so appending is safe but reordering silently corrupts predictions.
FEATURE_NAMES: tuple[str, ...] = (
    "depth_ppm",
    "log_depth_ppm",
    "duration_days",
    "depth_snr",
    "a_dur",
    "a_slope",
    "abs_a_dur",
    "signs_agree",
    "tau_over_sigma",
    "log_tau_over_sigma",
    "delta_bic",
    "significance",
    "a_dur_sigma",
    "ingress_duration_days",
    "egress_duration_days",
    "ingress_slope_snr",
    "egress_slope_snr",
    "n_points_in_window",
    "periodic",
)


def _safe_log(value: float) -> float:
    """Natural log that degrades to ``nan`` instead of raising or returning -inf."""
    return float(np.log(value)) if np.isfinite(value) and value > 0 else float("nan")


def _slope_snr(slope: float, slope_err: float) -> float:
    """How well determined a side's gradient is, in sigma."""
    if not np.isfinite(slope) or not np.isfinite(slope_err) or slope_err <= 0:
        return float("nan")
    return float(abs(slope) / slope_err)


def extract_features(record: CandidateRecord, lc: LightCurveData) -> dict[str, float]:
    """Reduce one scored event to a fixed dictionary of numeric features.

    Parameters
    ----------
    record
        A detected and scored event.
    lc
        The light curve it came from, used for the local noise scale.

    Returns
    -------
    dict
        Keys exactly :data:`FEATURE_NAMES`. Unmeasurable quantities are ``nan``
        rather than a sentinel number, so that a model cannot mistake a missing
        measurement for a real extreme value.
    """
    event = record.event
    score = record.score
    asym = event.asymmetry

    noise = 1.4826 * float(np.median(np.abs(lc.flux - np.median(lc.flux))))
    depth_snr = float(event.depth / noise) if noise > 0 else float("nan")

    features: dict[str, float] = {
        "depth_ppm": float(event.depth_ppm),
        "log_depth_ppm": _safe_log(event.depth_ppm),
        "duration_days": float(event.window.duration_days),
        "depth_snr": depth_snr,
        "a_dur": asym.a_dur if asym else float("nan"),
        "a_slope": asym.a_slope if asym else float("nan"),
        "abs_a_dur": abs(asym.a_dur) if asym else float("nan"),
        "signs_agree": float(score.signs_agree),
        "tau_over_sigma": float(score.tau_over_sigma),
        "log_tau_over_sigma": _safe_log(score.tau_over_sigma),
        "delta_bic": float(score.delta_bic),
        "significance": float(score.significance) if np.isfinite(score.significance) else float("nan"),
        "a_dur_sigma": float(score.a_dur_sigma),
        "ingress_duration_days": asym.ingress.duration_days if asym else float("nan"),
        "egress_duration_days": asym.egress.duration_days if asym else float("nan"),
        "ingress_slope_snr": _slope_snr(asym.ingress.slope, asym.ingress.slope_err)
        if asym
        else float("nan"),
        "egress_slope_snr": _slope_snr(asym.egress.slope, asym.egress.slope_err)
        if asym
        else float("nan"),
        "n_points_in_window": float(event.window.n_points),
        "periodic": float(score.periodic),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def features_to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    """Stack feature dictionaries into a 2-D array in :data:`FEATURE_NAMES` order."""
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.array(
        [[row.get(name, float("nan")) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
