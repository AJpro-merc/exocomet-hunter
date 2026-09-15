# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the published-dip validation gate.

Two layers here, and the split is deliberate:

* the offline tests run everywhere, including the fast unit sweep's sibling
  integration run, and prove that the matching, reporting and exit-code logic
  are correct — including that the gate actually fails when a known dip is
  absent;
* the ``network``-marked test performs the real thing, one serialised MAST
  download per target, and is excluded from any run that cannot reach the
  archive (``-m "not network"``).

The gate is asserted on epoch recovery only. Nothing here asserts on
``flagged``: that boolean encodes a tunable flagging policy, whereas "an event
was detected at the published epoch" is the contract the project's headline
result actually rests on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_validation_module():
    """Import ``scripts/run_validation.py``, which is not an installed package."""
    path = REPO_ROOT / "scripts" / "run_validation.py"
    spec = importlib.util.spec_from_file_location("run_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_validation"] = module
    spec.loader.exec_module(module)
    return module


rv = _load_validation_module()

TARGETS_PATH = REPO_ROOT / "config" / "validation_targets.yaml"
PRIMARY_TARGET = "KIC 3542116"
EMPTY_DIP_TARGET = "KIC 11084727"


@pytest.fixture(scope="module")
def spec():
    return rv.load_validation_spec(TARGETS_PATH)


@pytest.fixture(scope="module")
def primary(spec):
    return next(t for t in spec.targets if t.target_id == PRIMARY_TARGET)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def test_config_carries_six_known_dips(primary, spec):
    assert spec.match_tolerance_days > 0
    assert len(primary.dips) == 6
    assert primary.expected_dip_count == 6


def test_empty_dip_target_is_loaded_but_carries_no_epochs(spec):
    target = next(t for t in spec.targets if t.target_id == EMPTY_DIP_TARGET)
    assert target.dips == ()
    assert target.has_known_dips is False


def test_malformed_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("targets: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rv.load_validation_spec(bad)

    no_targets = tmp_path / "no_targets.yaml"
    no_targets.write_text("match_tolerance_days: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rv.load_validation_spec(no_targets)


# --------------------------------------------------------------------------- #
# Matching logic
# --------------------------------------------------------------------------- #


def test_match_within_tolerance_passes():
    known = (rv.KnownDip(t_min_bkjd=100.0, depth_ppm=500.0),)
    matches = rv.match_dips(known, [{"t_min": 100.4, "depth_ppm": 480.0}], 0.5)
    assert matches[0].matched
    assert matches[0].offset_days == pytest.approx(0.4)


def test_match_outside_tolerance_fails():
    known = (rv.KnownDip(t_min_bkjd=100.0, depth_ppm=500.0),)
    matches = rv.match_dips(known, [{"t_min": 100.6}], 0.5)
    assert not matches[0].matched
    assert matches[0].detected_t_min is None


def test_no_detections_at_all_fails_every_dip(primary, spec):
    matches = rv.match_dips(primary.dips, [], spec.match_tolerance_days)
    assert len(matches) == 6
    assert all(not m.matched for m in matches)


def test_one_detection_cannot_satisfy_two_known_dips():
    known = (
        rv.KnownDip(t_min_bkjd=100.0, depth_ppm=500.0),
        rv.KnownDip(t_min_bkjd=100.3, depth_ppm=500.0),
    )
    matches = rv.match_dips(known, [{"t_min": 100.1}], 0.5)
    assert sum(m.matched for m in matches) == 1
    # The nearer published epoch claims the detection.
    assert matches[0].matched and not matches[1].matched


def test_spurious_extra_detections_do_not_fail_the_gate():
    known = (rv.KnownDip(t_min_bkjd=100.0, depth_ppm=500.0),)
    detections = [{"t_min": 20.0}, {"t_min": 100.05}, {"t_min": 300.0}]
    matches = rv.match_dips(known, detections, 0.5)
    assert matches[0].matched


def test_matching_ignores_the_flagged_boolean():
    """Epoch recovery is the contract; the flagging rule may change freely."""
    known = (rv.KnownDip(t_min_bkjd=100.0, depth_ppm=500.0),)
    for flagged in (True, False):
        matches = rv.match_dips(known, [{"t_min": 100.1, "flagged": flagged}], 0.5)
        assert matches[0].matched
        assert matches[0].flagged is flagged


# --------------------------------------------------------------------------- #
# Exit-code semantics
# --------------------------------------------------------------------------- #


def _result(target, matches=(), skipped=None, error=None):
    return rv.TargetResult(
        target=target, matches=list(matches), skipped_reason=skipped, error=error
    )


def test_exit_zero_only_when_every_dip_recovered(primary):
    good = [rv.DipMatch(known=d, matched=True, detected_t_min=d.t_min_bkjd) for d in primary.dips]
    assert rv.exit_code([_result(primary, good)]) == 0

    missed = list(good)
    missed[3] = rv.DipMatch(known=primary.dips[3], matched=False)
    assert rv.exit_code([_result(primary, missed)]) == 1


def test_errored_target_exits_non_zero(primary):
    assert rv.exit_code([_result(primary, error="FileNotFoundError: nope")]) == 1


def test_all_targets_skipped_exits_non_zero(spec):
    empty = next(t for t in spec.targets if t.target_id == EMPTY_DIP_TARGET)
    assert rv.exit_code([_result(empty, skipped="no epochs")]) == 1


def test_skipped_target_alongside_a_pass(primary, spec):
    """The empty-dip target must not crash, must not count as a pass."""
    empty = next(t for t in spec.targets if t.target_id == EMPTY_DIP_TARGET)
    good = [rv.DipMatch(known=d, matched=True, detected_t_min=d.t_min_bkjd) for d in primary.dips]
    results = [_result(primary, good), _result(empty, skipped="no epochs")]

    skipped_result = results[1]
    assert skipped_result.skipped
    assert skipped_result.passed is False
    assert rv.exit_code(results) == 0
    assert rv.exit_code(results, strict_skipped=True) == 1


def test_empty_dip_target_is_skipped_not_run(spec):
    """Validating it must not attempt a download or a detection."""
    empty = next(t for t in spec.targets if t.target_id == EMPTY_DIP_TARGET)
    from exocomet.core.config import PipelineConfig

    result = rv.validate_target(
        empty, spec.match_tolerance_days, PipelineConfig(), offline=False
    )
    assert result.skipped
    assert result.error is None
    assert result.n_detections == 0
    assert result.passed is False


# --------------------------------------------------------------------------- #
# Offline end-to-end: the real pipeline against injected dips
# --------------------------------------------------------------------------- #


def test_offline_run_recovers_every_published_epoch(capsys):
    code = rv.main(["--offline", "--targets", str(TARGETS_PATH)])
    out = capsys.readouterr().out
    assert code == 0
    assert "VERDICT: PASS" in out
    assert out.count("PASS  ") >= 6
    # The empty-dip target is reported as an unverified gap, not as a pass.
    assert "SKIPPED" in out
    assert EMPTY_DIP_TARGET in out


def test_offline_run_fails_when_a_known_dip_is_absent(capsys):
    """The gate must be capable of failing, or it proves nothing."""
    code = rv.main(
        ["--offline", "--offline-drop-epoch", "991.95", "--targets", str(TARGETS_PATH)]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "VERDICT: FAIL" in out
    assert "FAIL " in out


def test_offline_json_report(tmp_path, capsys):
    out_path = tmp_path / "validation.json"
    code = rv.main(
        ["--offline", "--targets", str(TARGETS_PATH), "--json-out", str(out_path)]
    )
    capsys.readouterr()
    assert code == 0

    import json

    report = json.loads(out_path.read_text(encoding="utf-8"))
    primary = next(t for t in report["targets"] if t["target_id"] == PRIMARY_TARGET)
    assert primary["n_recovered"] == 6
    assert primary["passed"] is True

    empty = next(t for t in report["targets"] if t["target_id"] == EMPTY_DIP_TARGET)
    assert empty["passed"] is False
    assert empty["skipped_reason"]


def test_drop_epoch_requires_offline(capsys):
    assert rv.main(["--offline-drop-epoch", "991.95"]) == 2


# --------------------------------------------------------------------------- #
# The live gate
# --------------------------------------------------------------------------- #


@pytest.mark.network
@pytest.mark.slow
def test_live_recovery_of_published_dips(capsys):
    """Download real Kepler photometry and recover all six published dips.

    This is the claim the project stands on. It is excluded from the fast unit
    run because it needs MAST; the download layer currently has no timeout, so
    run it deliberately and one target at a time.
    """
    code = rv.main(["--targets", str(TARGETS_PATH), "--target", PRIMARY_TARGET])
    out = capsys.readouterr().out
    assert code == 0, f"published-dip recovery regressed:\n{out}"
    assert "6/6 published dip(s) recovered" in out
