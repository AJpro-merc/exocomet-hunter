# Next Steps

Every known task, in order, with enough detail to start without re-deriving anything.
Identical copy kept in `docs/NEXT_STEPS.md`. Last updated 2026-09-15 (early
morning — session ran past midnight).

Status key: 🔴 blocks trustworthy results · 🟠 important · 🟢 later / polish

---

## Plain-language summary

The detector now works correctly on **both** Kepler and TESS data — A1 and A2 are both
fixed and verified against real data. A4 is still open. C1 (the classifier's labelled
training data) and C2 (a real training script) are both built; the recurring generation
and training pipeline runs on GitHub Actions, not Claude Cloud routines (those cannot
reach NASA's archive at all — a platform limit, confirmed and documented). One real step
(the trained model's live backtest against KIC 3542116) is implemented but not yet
verified end-to-end — see "FIRST THING NEXT SESSION" below. Full narrative:
[[Session Log 2026-09-14]].

---

## ⚠️ FIRST THING NEXT SESSION (2026-09-15)

0. ~~Push the squashed `exocomet-hunter` git history~~ — **done**, later the same
   2026-09-14 session, once the user approved the harness's permission prompt. Verify:
   `origin/main` is at `044ad99` (or later). Both repos' history is clean, single-commit
   origin, no Claude attribution anywhere except one README credit line.
1. **Verify `scripts/train_classifier.py`'s real-data backtest actually completes.**
   It hung with zero CPU progress for 9+ minutes fetching KIC 3542116 near the end of
   the 2026-09-14 session and had to be killed; a direct retry of the same
   `fetch_light_curve` call also hung and was interrupted before diagnosis finished.
   Every other MAST fetch that session succeeded, including an earlier one for this
   same star — so this is most likely transient, but **unconfirmed**. Run it for real,
   watched, before trusting this script in production:
   ```bash
   cd "exocomet-hunter" && source .venv/bin/activate
   time python3 -c "
   from exocomet.io.download import fetch_light_curve
   lc = fetch_light_curve('KIC 3542116', mission='Kepler', discard_after_read=True)
   print('ok', lc.n_points)
   "
   ```
   If it hangs again, that's a real bug (network timeout missing somewhere in
   `io/download.py`/`lightkurve`) worth root-causing, not retrying blind.
2. **Set up `DATA_REPO_TOKEN`** (the user's own step, not mine to do): a fine-grained
   GitHub PAT scoped to `AJpro-merc/exocomet-hunter-data` only, `Contents: Read and
   write`, added as a repo secret on `AJpro-merc/exocomet-hunter` (Settings → Secrets
   and variables → Actions). Until this exists, `train.yml`'s push-to-data-repo steps
   fail loudly (expected, not a bug).
3. Once both of those are done, **trigger `train.yml` for real** (`gh workflow run
   train.yml`) and watch a complete run — generation, test, commit to the data repo,
   and (if `run_train` weekly-equivalent input is set) a real training run — end to end
   for the first time.
4. **Create Cloud routine 3** (`Exocomet training summary`, reporting-only — text is
   ready in `docs/cloud-routine-prompt-training.md`). Routines 1/2 are dead; don't
   recreate them.
5. Copyright headers across public-facing files — Atharva asked for this, explicitly
   deferred to a later session (not started).

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

### A2. TESS products from different pipelines get stitched together ✅ done 2026-09-14
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
- **Fixed for real, 2026-09-14 (later the same session).** `fetch_light_curve` /
  `iter_light_curves` / `MastLightCurveSource` gained `author`/`exptime` params,
  defaulting to `author="SPOC"`, `exptime=120` for TESS and `author="Kepler"` for
  Kepler. Pinning to a single pipeline turned out to fix *both* problems at once — the
  originally-named cadence mixing, and a second, worse one found the same session: the
  TARS pipeline (the one that also failed to download for 49 Ceti with a "file may be
  corrupt" error) reports flux as an already zero-centered residual, not raw counts, so
  stitching it in could divide by a near-zero median and corrupt the normalisation
  outright — pinning to SPOC excludes it entirely rather than just separating by
  cadence. Verified against real data: beta Pic now resolves to 10 clean SPOC/120s
  products (down from 54 mixed-pipeline products), flux normalises to a sane 0.99-1.01
  range, author/exptime/sectors are recorded in `lc.meta` and in every generated
  training row. `monitor_new_data.py`'s change detector fixed the same way. New test:
  `tests/unit/test_cross_cadence.py` (synthetic, no network) confirms the same injected
  comet is recovered at matching epoch/τσ at both 120s and 1800s cadence. Committed:
  `52ac2ec`.

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

### C1. Labelled training set ✅ built 2026-09-14 — `scripts/generate_training_labels.py`
- **Where labels come from:** injection. We fake events of known type, so the label is known.
- **Host light curves:** `scripts/build_kepler_host_list.py` (hand-curated ~23 Kepler
  hosts; NASA Exoplanet Archive auto-expansion still **not built** — see risk-mitigation
  note in Session Log 2026-09-14 for why this matters, and `scripts/vet_host_list.py`
  for the empirical alternative built instead: run the real pipeline, no injection, flag
  anything suspiciously deep/frequent). TESS hosts still reuse `config/watchlist.txt`
  (wrong list for `noise`-class purposes — flagged, not fixed).
- **Classes implemented (7, not the original 5):** `comet`, `symmetric`, `reversed`,
  `flare`, `starspot` (the last two added 2026-09-14, Next Steps D1 lookalikes),
  `noise`, `time_reversed_real`. Depth/amplitude sampling is weighted toward 500-2000ppm
  (the genuinely uncertain detection region), not uniform.
- **Output:** one row per detected event: 19 features + `label`, `host_star`,
  `injected_depth`, `injected_tau_over_sigma`, `seed`, `author`, `exptime`. Real output
  now lives in the separate `AJpro-merc/exocomet-hunter-data` repo (not this repo's git
  history, not `data/training/` — that path is still used for local/manual runs only).
- **Real bug found and fixed 2026-09-14:** the generator never detrended before running
  the detector — every row from before that fix (including the first 191-row local run)
  ran on raw, undetrended flux. Nothing had been committed anywhere trustworthy yet, so
  nothing needed cleanup, but see Session Log for the full story.
- **Recurring generation:** `.github/workflows/train.yml`, GitHub Actions, every 2h —
  NOT Claude Cloud routines (confirmed platform-level dead end, see H3 below).

### C2. Train — ✅ built 2026-09-14, ⚠️ not fully verified — `scripts/train_classifier.py`
- **Model:** `RandomForestClassifier` wrapped in `CalibratedClassifierCV` (isotonic).
  `HistGradientBoostingClassifier`/logistic-regression comparison not built — C2 always
  called this a "first" model, still true.
- **Missing values:** `SimpleImputer` (median) + an `is_measurable` indicator per feature,
  as specified.
- **Split by host star:** `GroupKFold` on `host_star`, as specified.
- **Class imbalance:** `class_weight="balanced"`.
- **Calibrated probabilities:** `CalibratedClassifierCV`, as specified. Binary target
  (`comet` vs. everything else), not the original 5-class framing — matches C4's
  eventual `comet_probability` output directly.
- **Hold out completely:** hard `assert`/exception (`HoldOutError`) if KIC 3542116 or KIC
  11084727 ever appear in the training data, not just a warning.
- **Save:** `joblib` file with model + `FEATURE_NAMES` + threshold. Provenance manifest
  (`models/manifest_<mission>.jsonl`, append-only) has row count, class counts, host
  stars, date, package versions, training-set hash — the "how much has it trained on"
  record, plus both backtests (see C3 below).
- **⚠️ Not fully verified:** the real-data-backtest step (fetches KIC 3542116 live) hung
  for 9+ minutes with zero CPU progress in testing and was killed before a full run ever
  completed. Every other piece of this script (loading, hold-out check, training,
  completeness calc, manifest/threshold writing) ran and was inspected; this one network
  step needs a real watched run before trusting it. **First thing next session.**

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
cd "exocomet-hunter"
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
