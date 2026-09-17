# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Visual, non-code config editing for Research mode.

Reads the real ``config/thresholds.yaml`` as the default baseline (never
modifies it). A visual edit is saved to a dashboard-local override file;
"Reset to defaults" just deletes that override and the real thresholds.yaml
is used again, untouched throughout.
"""

from __future__ import annotations

from typing import Any

import yaml

from dashboard import DASHBOARD_VAR_DIR, REPO_ROOT
from exocomet.core.config import PipelineConfig

DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "config" / "thresholds.yaml"
OVERRIDE_PATH = DASHBOARD_VAR_DIR / "config_override.yaml"

#: Labels/descriptions for the "fully non-code" form -- one entry per field
#: actually exposed to Research mode. Deliberately excludes runtime.rng_seed
#: and search.* (batch-search knobs the dashboard doesn't use) so the form
#: only shows knobs that change this run's detection behaviour.
FORM_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "detrend": [
        {"name": "window_days", "label": "Detrend window (days)", "type": "float", "step": 0.1, "min": 0.1},
        {"name": "polyorder", "label": "Savitzky-Golay polynomial order", "type": "int", "min": 1, "max": 5},
        {"name": "upper_sigma_clip", "label": "Upper sigma clip", "type": "float", "step": 0.5, "min": 0},
        {"name": "spline_knot_spacing_days", "label": "Spline knot spacing (days)", "type": "float", "step": 0.1, "min": 0.1},
    ],
    "baseline": [
        {"name": "median_window_days", "label": "Baseline median window (days)", "type": "float", "step": 0.1, "min": 0.1},
        {"name": "noise_window_days", "label": "Noise window (days)", "type": "float", "step": 0.1, "min": 0.1},
    ],
    "candidates": [
        {"name": "sigma_threshold", "label": "Detection significance threshold (sigma)", "type": "float", "step": 0.5, "min": 0},
        {"name": "min_consecutive", "label": "Min consecutive cadences", "type": "int", "min": 1},
        {"name": "max_gap_cadences", "label": "Max gap cadences", "type": "int", "min": 0},
        {"name": "max_window_days", "label": "Max event window (days)", "type": "float", "step": 0.1, "min": 0.1},
        {"name": "edge_margin_days", "label": "Edge margin (days)", "type": "float", "step": 0.1, "min": 0},
    ],
    "asymmetry": [
        {"name": "half_depth_fraction", "label": "Half-depth fraction", "type": "float", "step": 0.05, "min": 0.05, "max": 0.95},
        {"name": "min_points_per_side", "label": "Min points per side", "type": "int", "min": 2},
        {"name": "search_padding_factor", "label": "Search padding factor", "type": "float", "step": 0.1, "min": 1.0},
    ],
    "scoring": [
        {"name": "delta_bic_threshold", "label": "ΔBIC threshold (flag rule)", "type": "float", "step": 1.0, "min": 0},
        {"name": "tau_over_sigma_threshold", "label": "τ/σ threshold (flag rule)", "type": "float", "step": 0.1, "min": 0},
        {"name": "bootstrap_draws", "label": "Bootstrap draws", "type": "int", "min": 100, "max": 10000},
        {"name": "periodicity_min_period_days", "label": "Periodicity min period (days)", "type": "float", "step": 0.1, "min": 0.01},
        {"name": "periodicity_max_period_days", "label": "Periodicity max period (days)", "type": "float", "step": 1.0, "min": 1},
    ],
}

#: The remaining `PipelineConfig` fields, deliberately kept out of the
#: front-and-center form above. Shown in Research mode behind a collapsible
#: "Advanced" section instead of replacing the science-threshold layout.
ADVANCED_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "candidates": [
        {"name": "recovery_sigma", "label": "Recovery sigma (post-flag confirmation)", "type": "float", "step": 0.1, "min": 0},
    ],
    "scoring": [
        {"name": "significance_threshold", "label": "Significance threshold", "type": "float", "step": 0.1, "min": 0},
        {"name": "require_sign_agreement", "label": "Require sign agreement", "type": "bool"},
        {"name": "periodicity_fap_threshold", "label": "Periodicity FAP threshold", "type": "float", "step": 0.001, "min": 0, "max": 1},
    ],
    "search": [
        {"name": "target_batch_size", "label": "Target batch size", "type": "int", "min": 1},
        {"name": "shortlist_size", "label": "Shortlist size", "type": "int", "min": 1},
    ],
    "runtime": [
        {"name": "rng_seed", "label": "RNG seed", "type": "int", "min": 0},
    ],
}


def _default_config_dict() -> dict[str, Any]:
    if DEFAULT_THRESHOLDS_PATH.exists():
        return PipelineConfig.from_yaml(DEFAULT_THRESHOLDS_PATH).to_dict()
    return PipelineConfig().to_dict()


def load_effective_config() -> PipelineConfig:
    """The config a run should actually use: override if present, else real defaults."""
    if OVERRIDE_PATH.exists():
        return PipelineConfig.from_yaml(OVERRIDE_PATH)
    return PipelineConfig.from_yaml(DEFAULT_THRESHOLDS_PATH) if DEFAULT_THRESHOLDS_PATH.exists() else PipelineConfig()


def get_form_state() -> dict[str, Any]:
    """Current effective values plus the schema, shaped for the frontend form."""
    effective = load_effective_config().to_dict()
    defaults = _default_config_dict()
    return {
        "schema": FORM_SCHEMA,
        "advanced_schema": ADVANCED_SCHEMA,
        "values": effective,
        "defaults": defaults,
        "is_overridden": OVERRIDE_PATH.exists(),
    }


def save_overrides(partial: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist a (possibly partial) set of section overrides.

    ``partial`` is shaped like ``{"detrend": {"window_days": 9.0}, ...}``.
    Merged onto the current effective config, validated by constructing a
    real ``PipelineConfig`` (raises on an unknown key or bad type), then the
    *complete* resulting config is written to the override file -- so the
    override is always self-contained and never depends on the real
    thresholds.yaml still existing/being unchanged underneath it.
    """
    current = load_effective_config().to_dict()
    for section, values in partial.items():
        if section not in current:
            raise ValueError(f"unknown config section {section!r}")
        current[section] = {**current[section], **values}

    validated = PipelineConfig.from_dict(current)  # raises on bad input
    DASHBOARD_VAR_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDE_PATH.write_text(yaml.safe_dump(validated.to_dict()), encoding="utf-8")
    return get_form_state()


def reset_to_defaults() -> dict[str, Any]:
    OVERRIDE_PATH.unlink(missing_ok=True)
    return get_form_state()
