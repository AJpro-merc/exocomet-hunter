# Next Steps

Every known task, in order, with enough detail to start without re-deriving anything.
Identical copy kept in `BRAIN/EXOCOMET/Next Steps.md`. Last updated 2026-09-14 (evening).

Status key: 🔴 blocks trustworthy results · 🟠 important · 🟢 later / polish

---

## Plain-language summary

The detector works on Kepler data. A1 is fixed (window sizes now scale to TESS's faster
cadence) but A2 and A4 are still open, so TESS still can't be trusted. Meanwhile, C1
(building the classifier's training data via fake injected comets on real Kepler stars)
is underway, both locally and via two recurring Claude Cloud routines — see the C1
status note below. The big remaining job is still the **classifier**: the program is
good at *spotting* dips but bad at *judging* whether they are comet-shaped.

---

## Session update — 2026-09-14 (evening)

- **A1 done and verified.** Windows are now in days, converted to cadences at runtime
  from each light curve's own measured spacing. 48/48 unit tests pass; a real re-run on
  KIC 3542116 reproduces the exact same 7 events at the same epochs (confirmed no-op on
  Kepler). Committed: `97e2fc1`.
- **Daily survey workflow disabled** (A3 resolved: chose option (a)). Checked first —
  no bad runs had happened; the only run ever was the manual dry run from 2026-09-13.
  Nothing needed quarantining.
- **C1 (labelled training set) underway**, Kepler-only, deliberately ahead of A2/A4 since
  it doesn't touch TESS: `scripts/build_kepler_host_list.py` (24 quiet Kepler hosts,
  excludes the two validation stars) and `scripts/generate_training_labels.py` (inject
  comet/symmetric/reversed/noise/time_reversed_real, run the detector, save 19 features +
  label to parquet, discard the raw light curve). Committed: `461908a`.
- **Two recurring Claude Cloud routines set up** (see
  `docs/cloud-routine-prompt-training.md` in the repo for the exact paste-in text):
  Kepler track (safe) and a TESS track explicitly marked **provisional** pending A2/A4.
  Both run every 2 hours, capped to a 30-minute budget per run, append (not overwrite) to
  `results/training/labels_kepler.parquet` / `labels_tess_provisional.parquet`.
  **Known open issue on the TESS routine:** its host list currently reuses
  `config/watchlist.txt` — stars with *real, plausible* exocomet activity (beta Pic
  etc.), which is the wrong choice for `noise`/`time_reversed_real` injection hosts (risk
  of mislabelling a genuine event as noise). Needs a genuinely quiet TESS host list before
  that routine's negative-class rows are trusted.
- A2, A4 remain open 🔴 — untouched this session, by explicit choice (training proceeds
  Kepler-only in the meantime).

---

## PART A — Fix before trusting any TESS result

### A1. Window sizes are counted in measurements, not in time ✅ done 2026-09-14
- **Problem:** every window in `config/thresholds.yaml` is a number of cadences
  (measurements). They were chosen for Kepler, which measures every 29.4 min.
  TESS measures every 20 s, 120 s, 200 s, 600 s or 1800 s depending on the product.
- **Verified numbers (beta Pic):** at 120 s, the detrend window of 401 cadences is
  **13.4 hours** and the baseline window of 101 cadences is **3.4 hours**. Comet dips last
  about a day, so the detrending filter would flatten the very dips we are looking for.
  At 20 s cadence it is worse: 401 cadences = 2.2 hours.
- **Fix:**
  1. Change config fields to days: `detrend.window_days` (≈8.2), `baseline.median_window_days`
     (≈2.1), `baseline.noise_window_days`, `candidates.edge_margin_days` (≈1.0).
  2. Convert to cadences at runtime with `round(days / lc.cadence_days)`, forced odd.
  3. Touch: `core/config.py`, `config/thresholds.yaml`, `detrend/detrend.py`,
     `detect/comet_detector.py` (`baseline_and_noise`), `detect/candidates.py`,
     `calibration/injection.py` (uses `edge_margin_cadences`).
- **Test to add:** inject the same comet into a 120 s light curve and a 1800 s light curve
  covering the same time span; detection, epoch, and τ/σ must agree within tolerance.

### A2. TESS products from different pipelines get stitched together 🔴
- **Problem:** `lk.search_lightcurve("beta Pic", mission="TESS")` returned 54 products from
  QLP, SPOC, TARS, TASOC, TESS-SPOC and TGLC, at 20 s to 1800 s. `fetch_light_curve` calls
  `download_all().stitch()` on all of them. Overlapping timestamps are then silently
  dropped by the duplicate filter in `_to_light_curve_data`, which hides the problem.
- **Fix in `io/download.py`:**
  1. Add `author` and `exptime` parameters. Default for TESS: `author="SPOC"`,
     `exptime=120`. Default for Kepler: `author="Kepler"`, long cadence.
  2. Never stitch products of different cadence or author.
  3. Record `author`, `exptime`, sector/quarter list in `lc.meta`.
  4. Log (don't silently drop) how many duplicate timestamps were removed.
- **Also fix `scripts/monitor_new_data.py`:** its change detector counts *all* products,
  so a new QLP file triggers reprocessing. Count only the author/exptime actually used.
- **New evidence, 2026-09-14 (TESS training-routine smoke test):** a real run against 10
  watchlist stars produced **zero** usable rows. `lightkurve` warned of "zero-centered"
  flux (median near 0 ppm, not a large positive count) before `stitch()`/`normalize()` --
  at least one HLSP pipeline (TARS; the same one that also failed to download for 49 Ceti
  with a "file may be corrupt" error) reports flux as an already-normalized residual, not
  raw counts. `_to_light_curve_data`'s unconditional `flux / median(flux)` can corrupt the
  scale outright when such a product gets stitched in, not merely misalign cadences. The
  fix needs to exclude these products entirely (not just separate by cadence/author) —
  worth checking whether pinning `author="SPOC"` alone already avoids TARS, or whether an
  explicit exclusion list is needed.

> **Resolved 2026-09-14.** Checked first, before disabling anything: only one run had
> ever happened (`34776888883`, the manual dry run from 2026-09-13) — the 05:00 UTC
> schedule never fired before it was disabled, and no `results/monitor_candidates.jsonl`
> or `results/processed_ledger.json` existed on `origin/main` to quarantine. The two
> docs (`docs/NEXT_STEPS.md`, `docs/SESSION_LOG_2026-09-13.md`) are committed.

### A3. Decide what the daily robot does until A1 and A2 are fixed ✅ done 2026-09-14
- Disabled: `GH_TOKEN=$(gh auth token -u AJpro-merc) gh workflow disable survey.yml`.
- The scheduled workflow (daily 05:00 UTC) will process up to 5 stars per run and commit
  candidates to the public repo. With A2/A4 unfixed, those results would be unreliable.
- Options: (a) disable the schedule until fixed —
  `GH_TOKEN=$(gh auth token -u AJpro-merc) gh workflow disable survey.yml`;
  (b) change the workflow to `--dry-run` only; (c) accept junk runs and clean up later.
  Recommended: (a).

### A4. The `flagged` decision still uses the statistic we proved unreliable 🔴
- **Problem:** `detect/scoring.py` line ~172 sets
  `flagged = significance >= 3.0 and signs_agree`, where significance is `|a_dur| / σ(a_dur)`.
  Research log 003 showed `a_dur` measured at half depth can have the wrong sign. The README
  says ranking is by ΔBIC, but the code does not do that yet.
- **Also:** the rule flags strong asymmetry in *either* direction. A reversed dip (slow in,
  fast out) is currently flagged, and `tests/unit/test_scoring.py` expects that.
- **Fix (interim, before the classifier exists):**
  `flagged = delta_bic > 10 and tau_over_sigma > 1` (config values, not hardcoded).
  Keep `a_dur` significance as an output column only. Update the reversed-asymmetry test
  to expect *not* flagged, and add a test that the six KIC 3542116 dips are flagged.

---

## PART B — Known bugs and loose ends 🟠

### B1. Depths come out at ~78% of published
- All six: 442/491, 461/524, 412/679, 940/1200, 1094/1500, 1445/1900.
- Hypotheses to test, one at a time:
  1. Baseline for depth is the rolling median over the event window, which is pulled
     down by the dip → measure against out-of-event median instead.
  2. Paper's depth is from a fitted model, ours is the observed minimum → compare fitted
     `ModelComparison.comet.params["depth"]` against published.
  3. Savitzky-Golay detrending absorbs part of the dip → compare depth on raw normalised flux.
- Record outcome in `docs/research_log/`.

### B2. Unexplained event at BKJD 1064.245 (531 ppm) in KIC 3542116
- Check: is it near a quarter boundary or data gap? Does it appear in other stars at the
  same time (spacecraft artefact)? What does its plot look like? τ/σ and ΔBIC?
- Do **not** call it a discovery.

### B3. `exocomet` command is declared but does not exist
- `pyproject.toml` has `exocomet = "exocomet.cli:main"`, but `src/exocomet/cli.py` was never
  written. Installing the package creates a command that crashes. Either write the CLI (C3)
  or remove the entry now.

### B4. Wrong GitHub URLs in `pyproject.toml`
- Lines 63–66 point to `github.com/atharvajoshi/exocomet-hunter`. Should be
  `github.com/AJpro-merc/exocomet-hunter`. Docs URL should be
  `https://ajpro-merc.github.io/exocomet-hunter`.

### B5. Second validation star has no epoch
- `config/validation_targets.yaml` → `KIC 11084727` has `dips: []`. Read the epoch from
  Rappaport et al. 2018 (arXiv:1708.06069) and add it.

### B6. Validation is not automated
- The six epochs are in the config, but nothing reads them. Write
  `scripts/run_validation.py` that loads `validation_targets.yaml`, runs the detector,
  matches within `match_tolerance_days`, prints pass/fail per dip, exits non-zero on a miss.
- Add an integration test `tests/integration/test_validation.py` marked `@pytest.mark.network`.

### B7. Missing tests
- No test file for `detrend/detrend.py`, `calibration/injection.py`,
  `calibration/features.py`, `io/download.py`, `viz/plots.py`.
- No property-based (hypothesis) tests yet, though planned.
- Coverage never measured: `pytest --cov=exocomet tests/unit`.
- `mypy` and `ruff` never run: `.venv/bin/ruff check src tests` and `.venv/bin/mypy`.
  Expect a list of errors; fix them before CI enforces them.

### B8. Detection-efficiency numbers are stale
- The table in the session log was measured before the τ/σ change and before A1/A4.
  Re-run `run_injection_recovery` after those fixes and regenerate `overview.png`.

### B9. Research log numbering gap
- Only `003` exists. Write `001-architecture.md` (why a plugin framework, why no lightkurve
  in `detect/`), `002-edge-artefact.md` (Bug 1), `004-window-too-tight.md` (Bug 2),
  and `005-tess-cadence-and-authors.md` once A1/A2 are done.

---

## PART C — The classifier ("training it") 🟠

The main next build. Goal: replace the hand-set flag rule with a model trained on labelled
examples, and **measure** how good it is.

### C1. Labelled training set — `src/exocomet/calibration/labels.py`
- **Where labels come from:** injection. We fake events of known type, so the label is known.
- **Host light curves** (what fakes get injected into):
  - ~300 real, quiet Kepler long-cadence stars (no known planets, no known variability).
    Build list from NASA Exoplanet Archive exclusions + SIMBAD non-variables.
  - Later: TESS SPOC 120 s hosts (only after A1/A2).
  - Keep the synthetic flat-noise generator for unit tests only.
- **Classes to inject:**

| Label | How to make it | Parameter ranges |
|---|---|---|
| `comet` (positive) | Gaussian ingress + exponential egress, `comet_profile` | depth 200–3000 ppm, σ 1–10 h, τ/σ 1.2–5 |
| `symmetric` | Gaussian or trapezoid dip | depth 200–3000 ppm, width 1–24 h |
| `reversed` | exponential ingress + Gaussian egress | same ranges, mirrored |
| `noise` | no injection; take detections that occur anyway | — |
| `time_reversed_real` | flip a real light curve in time | preserves real systematics, flips comet direction |

- **Output:** one row per detected event: 19 features from `calibration/features.py`,
  plus `label`, `host_star`, `injected_depth`, `injected_tau_over_sigma`, `seed`.
  Save as `data/training/labels_v1.parquet` (gitignored; regenerable from seed).
- **Size:** start with ~10,000 events. Injections take ~87 ms each (measured), so this is
  about 15 minutes on one core.

### C2. Train — `src/exocomet/calibration/classifier.py`
- **Model:** `sklearn.ensemble.RandomForestClassifier` first (sklearn 1.9.1 is installed).
  Compare against `HistGradientBoostingClassifier` and logistic regression.
- **Missing values:** features are `nan` when unmeasurable. HistGradientBoosting handles nan
  natively; for Random Forest add a `SimpleImputer` plus an `is_measurable` indicator.
- **Split by host star, not by row** (`GroupKFold` on `host_star`). Otherwise the same
  star's noise appears in both train and test and the score is inflated.
- **Class imbalance:** `class_weight="balanced"`.
- **Calibrated probabilities:** wrap in `CalibratedClassifierCV` so 0.9 means ~90%.
- **Hold out completely:** KIC 3542116 and KIC 11084727. Never inject into them, never
  train on them. They are the final exam.
- **Save:** `joblib` file containing model, `FEATURE_NAMES`, package version, training-set
  hash, and date. Refuse to load if feature names differ.

### C3. Evaluate — must produce these numbers
- ROC-AUC and precision-recall AUC (PR matters more; comets are rare).
- **Recall at a fixed false-positive rate of 1%.**
- Confusion matrix across the five classes.
- Completeness curve vs injected depth and vs τ/σ (compare against the hand rule).
- Permutation feature importance — which measurements actually drive the decision.
- Final exam: all 6 KIC 3542116 dips must score above the chosen threshold.
- Write results to `docs/research_log/006-classifier-v1.md`, including what failed.

### C4. Wire it in
- Add `score.comet_probability` to `EventScore` and `to_row()`.
- `flagged = comet_probability >= threshold` (threshold chosen from C3 at 1% FPR, stored in config).
- Keep the physics rule from A4 as a reported fallback column so the two can be compared.

---

## PART D — Making results defensible 🟠

### D1. False-positive laboratory
Synthetic lookalikes, each with a documented generator, used as extra negative classes:
- **Stellar flare:** fast rise, exponential decay, *brightening* (positive flux).
- **Eclipsing binary:** periodic, symmetric primary + shallower secondary eclipse.
- **Starspot rotation:** sinusoid with period 0.5–30 d, amplitude 100–5000 ppm.
- **δ Scuti pulsation:** multi-period, 30 min – 6 h, relevant because A-type stars like
  beta Pic pulsate.
- **TESS momentum dump:** short flux step/glitch every ~3 d.
- **Sector-edge ramp:** exponential settling at start of each orbit.
Output: confusion matrix "what fraction of each lookalike gets called a comet".

### D2. Vetting module — `src/exocomet/vetting/` (currently empty)
- `catalogs.py`: SIMBAD object type (`astroquery.simbad`), AAVSO VSX variable flag,
  Kepler/TESS eclipsing-binary catalogue membership, Gaia parameters. Cache responses on disk.
- `checks.py`:
  - flare check (sign of excursion),
  - periodicity (already computed — reuse `periodicity_flag`),
  - **cross-target artefact:** same absolute timestamp flagged in ≥3 unrelated stars → artefact,
  - data-gap / sector-boundary proximity,
  - odd/even depth check for binaries.
- Output: `VettingResult` with status and every reason, including "why NOT a comet".

### D3. Blind-search protocol
1. Tune everything only on the development set (validation stars + injections).
2. Commit and **tag** the frozen configuration, e.g. `v0.2-frozen`.
3. Only then run on the search targets. Record the tag in every result.
4. Any change after looking at search results → new tag, and say so in the write-up.

### D4. Ablation study
Measure recall and false-positive rate with each component removed: detrending, robust
noise (swap for plain std), edge guard, search padding, periodicity penalty, bootstrap,
model comparison, vetting. Table goes in the research log.

### D5. Trials factor
Per-event significance ignores how many places were searched. Estimate the empirical
false-alarm rate from time-reversed and control light curves and report it with results.

---

## PART E — Engineering 🟢

- **E1. `pipeline.py`:** `run_target(target_id, mission, detector, detrender, vetter)` →
  `list[CandidateRecord]`, per-target try/except, returns errors instead of raising.
- **E2. `cli.py`:** `exocomet validate`, `exocomet search --targets FILE`,
  `exocomet inject --depths ...`, `exocomet plot TARGET`. Fixes B3.
- **E3. Provenance manifest:** every run writes `results/runs/<timestamp>.json` with config,
  package version, git commit, Python/OS, seed, target-list hash, lightkurve author/exptime,
  output checksums.
- **E4. Memory-bounded batch:** process one star at a time; parallelise with at most
  2–4 workers (`ProcessPoolExecutor`), not `cpu_count()`.
- **E5. CI:** `.github/workflows/ci.yml` — ruff, mypy, pytest on ubuntu + macOS, Python
  3.11/3.12, coverage. Network tests excluded.
- **E6. Target list from catalogues:** `scripts/build_target_list.py` — debris-disc catalogue
  (VizieR) cross-matched to TIC, filtered to A/F stars with SPOC 120 s data. Replaces the
  hand-written `config/watchlist.txt`. Add a matched **control group** of non-disc stars so
  candidate rates can be compared.
- **E7. Benchmarks:** `tests/benchmarks/` with `pytest-benchmark`: runtime vs light-curve
  length, peak memory per star.

## PART F — Science extensions 🟢

- **F1. Cross-mission check:** same algorithm on Kepler vs TESS — does sensitivity change
  with cadence and noise as expected?
- **F2. Physical comet simulator:** nucleus + dust tail with optical depth, impact parameter,
  transit speed, stellar limb darkening → light curve. Replaces arbitrary injected shapes.
- **F3. Independent second detector:** template matching against a bank of comet profiles;
  report agreement with the asymmetric-dip detector.
- **F4. Debris-disc vs control comparison:** candidate rate per star with confidence
  intervals, corrected for detection efficiency. "No significant difference" is a valid result.
- **F5. Sensitivity map:** 2-D completeness over depth × duration (and τ/σ).
- **F6. Detrending experiment:** Savitzky-Golay vs spline vs robust polynomial vs
  Gaussian process — which best preserves asymmetric shape?

## PART G — Release and publication 🟢

- **G1.** `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`.
- **G2.** Docs site (MkDocs Material) with auto API reference; deploy to GitHub Pages.
- **G3.** Enable Zenodo ↔ GitHub integration, tag `v1.0.0` → DOI.
- **G4.** PyPI release workflow.
- **G5.** `paper/paper.md` in JOSS format — summary, statement of need (no tested
  pip-installable exocomet tool exists), functionality, references.
- **G6.** Get one outside person to run it and open an issue.

## PART H — Later, agreed but parked 🟢

- **H1. AI review layer (Gemini free tier).** Rule: the model **never** decides whether
  something is a comet. It receives only computed features (JSON), returns structured JSON:
  evidence for, evidence against, alternative explanations, recommended next test, and must
  be allowed to answer `insufficient_evidence`. Use it as a skeptic, not a cheerleader.
- **H2. Website / dashboard.** Atharva said later.
- **H3. Claude Cloud routine.** Prompt ready in `docs/cloud-routine-prompt.md`. Create it in
  Claude → Routines → New routine → Cloud → repo `AJpro-merc/exocomet-hunter` → daily.
  Only after A1–A4 are fixed.
- **H4. Project 2 — technosignature / anomalous dimming.** New `Detector` subclass
  (Isolation Forest on light-curve features), validated on Boyajian's Star (KIC 8462852),
  reusing `io/`, `detrend/`, `core/`, `vetting/`.
- **H5. Learning pass.** Atharva chose build-first. Still owed: a plain-language walkthrough
  of every module so he can explain every choice himself.

---

## Reference: commands that are easy to forget

```bash
cd "/Users/atharvajoshi/Projects/Atharva Joshi/exocomet-hunter"
source .venv/bin/activate

pytest tests/unit -q                     # 48 tests, ~2 s
ruff check src tests                     # not yet clean
mypy                                     # not yet clean

# gh commands need the right account (active gh account is elpis-app):
GH_TOKEN=$(gh auth token -u AJpro-merc) gh run list --workflow=survey.yml
GH_TOKEN=$(gh auth token -u AJpro-merc) gh workflow run survey.yml -f dry_run=true
GH_TOKEN=$(gh auth token -u AJpro-merc) gh workflow disable survey.yml

# git push already works inside the repo (local credential config set)
git push
```

## Reference: key numbers

- Kepler long cadence 29.4 min; TESS SPOC 120 s / 20 s; FFI 200 s, 600 s, 1800 s.
- Comet dips in KIC 3542116: 491–1900 ppm, ~1 day (deep) and shorter (shallow).
- Noise in KIC 3542116 after detrending: 83 ppm.
- Pipeline speed: 0.8 s per 4-year Kepler star. Injection: ~87 ms each.
- Contour sign-flip point: τ/σ ≈ 1.70.
- ΔBIC > 10 = strong preference for the comet model.
