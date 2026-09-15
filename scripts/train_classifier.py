# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""C2: train the comet classifier on accumulated C1 labels, with provenance.

Loads one or more labelled-row parquets (from generate_training_labels.py),
trains a calibrated RandomForest to distinguish ``comet`` from every other
class (symmetric/reversed/flare/starspot/noise/time_reversed_real -- all
negative examples), and writes:

- the model itself (joblib)
- a probability threshold, chosen from out-of-fold predictions at a target
  false-positive rate on the negative classes
- a provenance manifest line (row count, class balance, host stars, date,
  package versions, training-set hash, both backtest results) -- the
  "how much has this been trained on" record
- a before/after comparison against the previous manifest entry, for
  visibility into a regression without gating the commit (this session's
  locked decision: always commit, never silently withhold a worse model)

Two backtests, both real, both recorded:
1. **Real-data check**: fetches the real KIC 3542116 light curve (never
   trained on -- see the hard assertion below), matches its 6 published
   comet dips, and reports what fraction score above threshold. This needs
   real network access to MAST; fine on GitHub Actions, not on a Claude Cloud
   routine (see Next Steps and this session's findings on that).
2. **Fake-injection completeness**: uses the out-of-fold cross-validated
   predictions already produced during training, grouped by
   ``injected_depth``, to report recovery fraction vs. depth for the
   classifier itself (not the hand-set flag rule Session Log 2026-09-13
   originally measured).

Runs automatically (in train.yml, weekly) and by hand (for the real review
conversation) -- same code path either way.

Usage
-----
    python scripts/train_classifier.py --in labels_kepler.parquet \
        --out-model models/kepler.joblib --manifest-out models/manifest_kepler.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kepler_host_list import EXCLUDED_HOLD_OUT_TARGETS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exocomet.calibration.features import FEATURE_NAMES, extract_features
from exocomet.core.config import PipelineConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol
from exocomet.io.download import fetch_light_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_classifier")

MIN_ROWS = 30
TARGET_FALSE_POSITIVE_RATE = 0.01
VALIDATION_TARGETS_PATH = Path("config/validation_targets.yaml")


class HoldOutError(RuntimeError):
    """Raise when a hold-out validation star appears in the training data."""


def load_labels(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more label parquets, enforcing the hold-out."""
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    present_holdouts = set(df["host_star"].unique()) & EXCLUDED_HOLD_OUT_TARGETS
    if present_holdouts:
        raise HoldOutError(
            f"hold-out target(s) {sorted(present_holdouts)} present in "
            "training data -- refusing to train"
        )
    return df


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, is_measurable) from FEATURE_NAMES columns, NaN-safe."""
    x = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    is_measurable = np.isfinite(x).astype(np.float64)
    return x, is_measurable


def _build_pipeline_inputs(df: pd.DataFrame) -> np.ndarray:
    """Impute NaN features and append an is_measurable indicator per feature."""
    x, is_measurable = _feature_matrix(df)
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x) if len(df) else x
    return np.hstack([x_imputed, is_measurable])


def train_and_cross_validate(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> tuple[CalibratedClassifierCV, np.ndarray, float]:
    """Fit the final model on all data; return it, out-of-fold probabilities, and the threshold.

    Cross-validated (out-of-fold) predictions come from a *separate* fitting
    pass using GroupKFold by ``host_star`` -- the same star's noise must never
    appear in both a fold's train and validation split (Next Steps C2). The
    final saved model is refit on the full dataset afterward, as is standard:
    the OOF predictions are only used to choose a threshold and measure
    completeness, never to select what gets saved.
    """
    y = (df["label"] == "comet").astype(int).to_numpy()
    x = _build_pipeline_inputs(df)
    groups = df["host_star"].to_numpy()

    n_groups = len(set(groups))
    splits = min(n_splits, n_groups) if n_groups >= 2 else 2
    if n_groups < 2:
        oof_proba = np.full(len(df), np.nan)
    else:
        gkf = GroupKFold(n_splits=splits)
        base = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        )
        oof_proba = cross_val_predict(
            base, x, y, groups=groups, cv=gkf, method="predict_proba"
        )[:, 1]

    final_base = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=random_state
    )
    final_model = CalibratedClassifierCV(final_base, method="isotonic", cv=3)
    final_model.fit(x, y)

    negative_scores = oof_proba[y == 0]
    negative_scores = negative_scores[np.isfinite(negative_scores)]
    if negative_scores.size:
        threshold = float(np.quantile(negative_scores, 1.0 - TARGET_FALSE_POSITIVE_RATE))
    else:
        threshold = 0.5

    return final_model, oof_proba, threshold


def completeness_by_depth(
    df: pd.DataFrame, oof_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Fraction of comet-labelled rows scoring above threshold, grouped by injected depth."""
    is_comet = df["label"] == "comet"
    depths = df.loc[is_comet, "injected_depth"].to_numpy()
    scores = oof_proba[is_comet.to_numpy()]
    result: dict[str, float] = {}
    for depth in sorted(set(depths[np.isfinite(depths)])):
        mask = depths == depth
        valid = np.isfinite(scores[mask])
        if valid.sum() == 0:
            continue
        recovered = float(np.mean(scores[mask][valid] >= threshold))
        result[f"{depth * 1e6:.0f}ppm"] = round(recovered, 3)
    return result


def real_data_backtest(
    model: CalibratedClassifierCV, threshold: float
) -> dict[str, object]:
    """Run the real final exam: check whether the model flags the 6 published dips.

    Never trained on this star (enforced by :func:`load_labels`'s hard
    assertion) -- this is the one check that uses real astronomical ground
    truth rather than synthetic injections.
    """
    targets = yaml.safe_load(VALIDATION_TARGETS_PATH.read_text(encoding="utf-8"))
    primary = next(t for t in targets["targets"] if t["target_id"] == "KIC 3542116")
    tolerance = float(targets.get("match_tolerance_days", 0.5))
    published_epochs = [float(d["t_min_bkjd"]) for d in primary["dips"]]

    try:
        raw = fetch_light_curve("KIC 3542116", mission="Kepler", discard_after_read=True)
        lc = detrend_savgol(raw)
    except Exception as exc:
        logger.warning("real-data backtest could not run: %s", exc)
        return {"ran": False, "reason": str(exc)}

    config = PipelineConfig()
    records = AsymmetricDipDetector(config).run(lc)

    x_cols = list(FEATURE_NAMES)
    matched = 0
    scored_above_threshold = 0
    per_dip: list[dict[str, object]] = []
    for epoch in published_epochs:
        if not records:
            per_dip.append({"epoch": epoch, "matched": False, "probability": None})
            continue
        nearest = min(records, key=lambda r: abs(r.event.window.start_time - epoch))
        offset = abs(nearest.event.window.start_time - epoch)
        if offset > tolerance:
            per_dip.append({"epoch": epoch, "matched": False, "probability": None})
            continue
        matched += 1
        row = extract_features(nearest, lc)
        x = np.array([[row[c] for c in x_cols]], dtype=np.float64)
        is_measurable = np.isfinite(x).astype(np.float64)
        imputer = SimpleImputer(strategy="median")
        x_imputed = imputer.fit_transform(x) if not np.isnan(x).all() else np.nan_to_num(x)
        proba = float(model.predict_proba(np.hstack([x_imputed, is_measurable]))[0, 1])
        above = proba >= threshold
        scored_above_threshold += int(above)
        per_dip.append(
            {
                "epoch": epoch,
                "matched": True,
                "probability": round(proba, 4),
                "above_threshold": above,
            }
        )

    return {
        "ran": True,
        "n_published_dips": len(published_epochs),
        "n_matched": matched,
        "n_scored_above_threshold": scored_above_threshold,
        "per_dip": per_dip,
    }


def _parquet_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.read_bytes())
    return digest.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--threshold-out", type=Path, default=None)
    parser.add_argument("--last-compare-out", type=Path, default=None)
    parser.add_argument("--mission", default="unknown")
    args = parser.parse_args()

    try:
        df = load_labels(args.in_paths)
    except HoldOutError as exc:
        logger.error("%s", exc)
        return 1

    if len(df) < MIN_ROWS:
        logger.error(
            "only %d rows (< %d minimum) -- refusing to train on too little data", len(df), MIN_ROWS
        )
        return 1

    logger.info("training on %d rows across %d host stars", len(df), df["host_star"].nunique())
    model, oof_proba, threshold = train_and_cross_validate(df)
    completeness = completeness_by_depth(df, oof_proba, threshold)
    real_backtest = real_data_backtest(model, threshold)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_names": list(FEATURE_NAMES), "threshold": threshold}
    joblib.dump(payload, args.out_model)

    manifest_entry = {
        "date_utc": datetime.now(UTC).isoformat(),
        "mission": args.mission,
        "n_rows": len(df),
        "class_counts": df["label"].value_counts().to_dict(),
        "n_host_stars": int(df["host_star"].nunique()),
        "host_stars": sorted(df["host_star"].unique().tolist()),
        "training_set_hash": _parquet_hash(args.in_paths),
        "feature_names": list(FEATURE_NAMES),
        "package_versions": {
            "exocomet": _safe_version("exocomet"),
            "scikit-learn": _safe_version("scikit-learn"),
            "pandas": _safe_version("pandas"),
            "numpy": _safe_version("numpy"),
        },
        "threshold": threshold,
        "real_data_backtest": real_backtest,
        "completeness_by_depth": completeness,
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry) + "\n")
    logger.info("appended manifest entry to %s", args.manifest_out)

    if args.threshold_out:
        args.threshold_out.parent.mkdir(parents=True, exist_ok=True)
        args.threshold_out.write_text(
            json.dumps({"threshold": threshold, "date_utc": manifest_entry["date_utc"]}, indent=2),
            encoding="utf-8",
        )

    if args.last_compare_out:
        previous = _read_last_manifest_entry(args.manifest_out, skip_last=1)
        comparison = {
            "current": {
                "n_rows": manifest_entry["n_rows"],
                "real_data_matched": real_backtest.get("n_matched"),
                "real_data_above_threshold": real_backtest.get("n_scored_above_threshold"),
            },
            "previous": {
                "n_rows": previous.get("n_rows") if previous else None,
                "real_data_matched": (previous or {})
                .get("real_data_backtest", {})
                .get("n_matched"),
                "real_data_above_threshold": (previous or {})
                .get("real_data_backtest", {})
                .get("n_scored_above_threshold"),
            }
            if previous
            else None,
        }
        args.last_compare_out.parent.mkdir(parents=True, exist_ok=True)
        args.last_compare_out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    logger.info(
        "done: threshold=%.4f, real backtest matched %s/%s dips "
        "(%s above threshold), completeness=%s",
        threshold,
        real_backtest.get("n_matched"),
        real_backtest.get("n_published_dips"),
        real_backtest.get("n_scored_above_threshold"),
        completeness,
    )
    return 0


def _safe_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _read_last_manifest_entry(manifest_path: Path, skip_last: int) -> dict[str, object] | None:
    """Read the manifest entry ``skip_last`` lines before the end (0 = most recent)."""
    if not manifest_path.exists():
        return None
    text = manifest_path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    idx = len(lines) - 1 - skip_last
    if idx < 0:
        return None
    return json.loads(lines[idx])


if __name__ == "__main__":
    raise SystemExit(main())
