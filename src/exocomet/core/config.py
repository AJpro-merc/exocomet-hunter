"""Typed configuration for the pipeline.

Defaults here mirror ``config/thresholds.yaml`` so that an installed copy of the
package works without the repository checked out. A YAML file may override any
subset of the values; anything it omits falls back to the default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AsymmetryConfig",
    "BaselineConfig",
    "CandidateConfig",
    "DetrendConfig",
    "PipelineConfig",
    "RuntimeConfig",
    "ScoringConfig",
    "SearchConfig",
]


@dataclass(frozen=True)
class DetrendConfig:
    """Parameters for removing instrumental trends.

    Windows are specified in days, not cadences: Kepler long cadence is
    29.4 min, but TESS products range from 20 s to 1800 s, so a cadence count
    tuned for one means a wildly different physical duration on the other
    (see ``docs/research_log/004-window-too-tight.md``). Values below are
    exactly equivalent to the previous Kepler-only cadence counts
    (``window_length=401``), preserved so existing validation results do not
    move; they are converted to cadences at runtime via
    :func:`exocomet.detect.baseline.days_to_cadences`.
    """

    window_days: float = 8.1871
    polyorder: int = 2
    upper_sigma_clip: float = 5.0
    spline_knot_spacing_days: float = 1.5


@dataclass(frozen=True)
class BaselineConfig:
    """Parameters for the local baseline and robust noise estimate.

    Windows are in days for the same reason as :class:`DetrendConfig`. Values
    are exactly equivalent to the previous ``median_window=101`` on Kepler
    long cadence.
    """

    median_window_days: float = 2.0621
    noise_window_days: float = 2.0621


@dataclass(frozen=True)
class CandidateConfig:
    """Parameters controlling which excursions become candidate events."""

    sigma_threshold: float = 4.0
    min_consecutive: int = 3
    max_gap_cadences: int = 2
    max_window_days: float = 3.0
    recovery_sigma: float = 1.0
    #: Events within this many days of either end of the series are dropped;
    #: equivalent to the previous ``edge_margin_cadences=50`` on Kepler long
    #: cadence. Days, not cadences, for the same reason as the windows above.
    edge_margin_days: float = 1.0208


@dataclass(frozen=True)
class AsymmetryConfig:
    """Parameters for the ingress/egress asymmetry measurement."""

    half_depth_fraction: float = 0.5
    min_points_per_side: int = 3
    #: How far outside the detected window the ingress/egress search may reach,
    #: as a multiple of the window duration. The window marks where the dip is
    #: statistically significant; its wings extend further, and confining the
    #: measurement to the window leaves real events unmeasurable.
    search_padding_factor: float = 2.0


@dataclass(frozen=True)
class ScoringConfig:
    """Parameters for significance estimation and flagging."""

    bootstrap_draws: int = 1000
    significance_threshold: float = 3.0
    require_sign_agreement: bool = True
    periodicity_min_period_days: float = 0.5
    periodicity_max_period_days: float = 50.0
    periodicity_fap_threshold: float = 0.01


@dataclass(frozen=True)
class SearchConfig:
    """Parameters for the batch search over many targets."""

    target_batch_size: int = 40
    shortlist_size: int = 20


@dataclass(frozen=True)
class RuntimeConfig:
    """Determinism and parallelism settings."""

    rng_seed: int = 42
    max_workers: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Complete configuration for a pipeline run."""

    detrend: DetrendConfig = field(default_factory=DetrendConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    asymmetry: AsymmetryConfig = field(default_factory=AsymmetryConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file, falling back to defaults.

        Parameters
        ----------
        path
            Path to a YAML file shaped like ``config/thresholds.yaml``. Sections
            and individual keys may be omitted.

        Returns
        -------
        PipelineConfig
            Configuration with file values layered over the defaults.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PipelineConfig:
        """Build a configuration from a (possibly partial) nested dictionary."""
        sections: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            values = raw.get(f.name) or {}
            if not isinstance(values, dict):
                raise TypeError(f"config section '{f.name}' must be a mapping, got {type(values)}")
            if f.default_factory is dataclasses.MISSING:  # pragma: no cover - defensive
                raise TypeError(f"config section '{f.name}' has no default factory")
            section_cls = type(f.default_factory())
            known = {sf.name for sf in dataclasses.fields(section_cls)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(
                    f"unknown key(s) in config section '{f.name}': {sorted(unknown)}"
                )
            sections[f.name] = section_cls(**values)
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full configuration, for run manifests and logging."""
        return dataclasses.asdict(self)
