# Session Log — 2026-09-13

Everything that happened in the first build session, in order, with the numbers.
Identical copy kept in `docs/SESSION_LOG_2026-09-13.md`.

---

## Plain-language summary

We built a program that searches NASA telescope data for comets orbiting other stars.
A planet passing its star makes an even, V-shaped dip in brightness. A comet has a dust
tail, so its dip is lopsided — sharp down, slow back up. The program finds lopsided dips.

We tested it on a star (KIC 3542116) where scientists already found 6 comets in 2018.
Without being told the answers, it found all 6. Along the way we found and fixed three
real bugs, one of which was a genuine maths mistake. The code is public on GitHub and
checks NASA's archive for new data every day by itself.

---

## 1. Choosing the project

- Started from "a NASA star-data project for my profile."
- Options considered and rejected: plain exoplanet transit pipeline (too common — it is
  the default `lightkurve` tutorial), gravitational waves (saturated by the Kaggle G2Net
  competition), solar flare prediction (shallow version is an intro-ML dataset), satellite
  debris tracking (strong but not star data).
- Chosen: **exocomet detection** first, **technosignature / anomalous dimming search**
  second, sharing one codebase.
- Context established: Atharva is in **grade 9**; MIT applications are ~4 years away.
  Machine: MacBook M4, 10 cores (4 performance + 6 efficiency), **24 GB free disk**.

## 2. Repository setup

- Location: `exocomet-hunter`
- Python **3.12.13** venv at `.venv/` (system python is 3.9, too old).
- `pip install -e ".[dev]"` — succeeded; took several minutes. An earlier install attempt
  had been killed because it looked hung; it was just slow. `import lightkurve` alone takes
  ~54 s the first time.
- Packaging: `pyproject.toml`, src-layout, setuptools-scm versioning, ruff/mypy/pytest config.

## 3. What got built

| File | Purpose |
|---|---|
| `core/types.py` | All data structures (`LightCurveData`, `DipEvent`, `EventScore`, `ModelComparison`, `CandidateRecord`, ...). No lightkurve dependency. |
| `core/interfaces.py` | Plugin contracts: `Detector` (ABC), `Detrender`, `LightCurveSource`, `Vetter`. |
| `core/config.py` | Typed configuration mirroring `config/thresholds.yaml`. |
| `io/download.py` | MAST download via lightkurve, normalisation, stream-and-discard. |
| `detrend/detrend.py` | Savitzky-Golay (default) and spline detrending. |
| `detect/baseline.py` | Rolling-median baseline, robust MAD noise. |
| `detect/candidates.py` | >4σ thresholding, run merging, window expansion, edge guard. |
| `detect/asymmetry.py` | Half-depth crossings, per-side slope fits, `a_dur` / `a_slope`. |
| `detect/model_comparison.py` | Symmetric Gaussian vs Gaussian-ingress/exponential-egress fit, ΔBIC. |
| `detect/scoring.py` | Bootstrap uncertainty, Lomb-Scargle periodicity flag. |
| `detect/comet_detector.py` | `AsymmetricDipDetector`, the first `Detector`. |
| `calibration/injection.py` | Inject fake comets, measure recovery fraction. |
| `calibration/features.py` | Turns an event into 19 numeric features for a classifier. |
| `viz/plots.py` | Light curve, detrend comparison, per-event, asymmetry histogram plots. |
| `scripts/monitor_new_data.py` | Checks MAST for new data on a watchlist, processes only what changed. |
| `.github/workflows/survey.yml` | Daily cloud run of the monitor. |
| `config/thresholds.yaml` | Every tunable number, commented with its justification. |
| `config/validation_targets.yaml` | Published dip epochs from Rappaport 2018 Table 3. |
| `config/watchlist.txt` | 10 debris-disc / exocomet stars for the TESS survey. |
| `tests/unit/*` | 48 tests on synthetic light curves with analytic answers. |

## 4. Bugs found and fixed (in the order found)

### Bug 1 — rolling-median edge artefact
- **Symptom:** a fake event at the end of every synthetic light curve.
- **Cause:** `scipy.ndimage.median_filter(mode="nearest")` pads by repeating the last
  sample ~50 times. That collapsed the noise estimate at the edges (4.6e-5 vs true 1.0e-4)
  and inflated edge significance to **32σ and 73σ**, versus **14.5σ** for a real injected dip.
- **Fix:** truncated windows via pandas rolling (`min_periods=1`) — no fake padding.
  Edge significance fell to 1.8σ / 3.5σ. Added `edge_margin_cadences = 50`.

### Bug 2 — real events unmeasurable
- **Symptom:** 2 of 7 detections on real Kepler data had no asymmetry measurement.
- **Cause:** measurement confined to the detection window, which is drawn only where the
  dip exceeds 4σ and is narrower than the real event.
- **Fix:** search up to `search_padding_factor = 2.0` window-widths outside the window;
  extend a steep side's fit region to reach 3 points. Result: 0/7 unmeasurable.

### Bug 3 — the half-depth asymmetry has the wrong sign (a maths finding)
- **Symptom:** all 6 published dips recovered, but measured `a_dur` was
  +0.07, −0.60, −0.58, −0.34, +0.24, −0.43. The paper says every dip has a steeper ingress
  and longer egress, so all should be positive.
- **Hypotheses killed:** baseline window too short (sweeping 101→1001 cadences changed
  a_dur by <0.02); detrending distortion (raw and detrended both −0.33 for D1268);
  sign bug in code (synthetic tests recover correct sign).
- **Cause:** at depth fraction f a Gaussian side reaches the contour at σ√(2 ln 1/f) and an
  exponential side at τ ln(1/f). At half depth that is 1.177σ vs 0.693τ, equal at
  **τ/σ ≈ 1.70**. So a dip with a genuinely longer tail can measure negative. The comet
  signature is in the wings, not the core.
- **Evidence:** measuring at 50% → 25% → 12.5% depth moved every dip toward positive
  (D1268: −0.43 → −0.12 → +0.02). Fitted τ/σ: 1.56, 1.58, 2.17, 0.87, 0.99, 1.28.
- **Change:** added `ModelComparison.tau_over_sigma` as the primary shape statistic.
  Full write-up: `docs/research_log/003-half-depth-contour-sign-error.md`.
- **Lesson:** recovering the right epochs is necessary but not sufficient — position matched
  while the main discriminator had the wrong sign.

### Also corrected
- Config originally cited arXiv **1708.06012** — that is an unrelated coding-theory paper.
  The correct ID is **1708.06069**. Fixed everywhere.

## 5. Validation result

| Published (Rappaport 2018, Table 3) | Detected | Epoch error | Our depth |
|---|---|---|---|
| D140 — 139.98, 491 ppm | 140.026 | 0.046 d | 442 ppm |
| D742 — 742.45, 524 ppm | 742.574 | 0.124 d | 461 ppm |
| D793 — 792.78, 679 ppm | 792.895 | 0.115 d | 412 ppm |
| D992 — 991.95, 1200 ppm | 991.979 | 0.029 d | 940 ppm |
| D1176 — 1175.62, 1500 ppm | 1175.671 | 0.051 d | 1094 ppm |
| D1268 — 1268.10, 1900 ppm | 1268.284 | 0.184 d | 1445 ppm |

- One extra detection at **1064.245 (531 ppm)**, not in the paper. Unexplained.
- Deepest dip: comet model beats symmetric model by **ΔBIC = 998**.

## 6. Measurements

- **Real data:** KIC 3542116 = 18 Kepler quarters, 65,263 usable cadences (241 non-finite
  dropped), 1,470 days, 7.7 MB, 62 s to download. Scatter 121 ppm raw → 83 ppm detrended.
- **Speed:** full 4-year star through the whole pipeline with 1,000 bootstrap draws: **0.8 s**.
- **Detection efficiency** (200 injections, 100 ppm noise, before the τ/σ change):

| Depth | SNR | Detected | Flagged comet-like |
|---|---|---|---|
| 200 ppm | 2 | 0% | 0% |
| 300 ppm | 3 | 0% | 0% |
| 500 ppm | 5 | 4% | 0% |
| 750 ppm | 7.5 | 68% | 0% |
| 1000 ppm | 10 | 100% | 4% |
| 1500 ppm | 15 | 100% | 36% |
| 2000 ppm | 20 | 100% | 64% |
| 3000 ppm | 30 | 100% | 96% |

  **Detection is solved above 1000 ppm; classification is not.**

- **Scaling estimates:** full Kepler archive (200,000 stars) = ~1.5 TB, 1–3 weeks of
  downloading, ~6 h of compute. Download is the bottleneck, not compute.

## 7. Decisions made

- **Stream and discard** light curves instead of storing them. Rejected iCloud as storage:
  it evicts files to the cloud and re-downloads on access, which would stall a pipeline.
- **Training data comes from injection**, not hand-labelling. Training does not need
  "10 years of data" — it needs variety of real noise to inject into.
- **Interpretable ML (Random Forest), not a neural network.** Precedent: arXiv:2402.19075.
- **Only space telescopes.** Comet dips are 500–2000 ppm; ground surveys reach ~1%
  (10,000 ppm). Relevant missions: Kepler (frozen), K2 (frozen), **TESS (live)**, PLATO (future).
- **Kepler is exhausted** — a 2025 A&A paper (arXiv:2510.14687) searched all 201,820 stars
  with a neural net. Frame this project as an independent-method cross-check with measured
  sensitivity, not a discovery hunt.
- **Survey runs daily**, not weekly: checking is cheap, processing only runs on new data.
- **GitHub account:** AJpro-merc. **Public** repo.

## 8. Research gathered

- Rappaport et al. 2018, MNRAS 474, 1453, arXiv:1708.06069 — validation source.
- Kennedy et al. 2019, MNRAS 482, 5587, arXiv:1811.03102 — ΔBIC method.
- Norazman et al. 2025, MNRAS 542, 1486, arXiv:2508.04673 — TESS sectors 1–26 search.
- A&A 2025, arXiv:2510.14687 — NN search of all Kepler, 17 high-confidence events (10 new).
- arXiv:2402.19075 — Random Forest exocomet detection in TESS Sector 1.
- Only existing code: `automated_exocomet_hunt` (clone-and-run scripts, no clear licence).
  No pip-installable, tested exocomet package exists — this is the project's "statement of need".
- NASA tools relevant as precedent: Kepler FLTI and KeplerPORTS (detection efficiency),
  Kepler Robovetter (automated vetting).

## 9. Publishing and automation

- Commits: `cc6cff8` (pipeline), `88b9cda` (daily schedule).
- Repo: **https://github.com/AJpro-merc/exocomet-hunter**
- Workflow dry run on GitHub's servers succeeded (run 34776888883): all 10 watchlist stars
  have TESS data — beta Pic 54 products, RZ Psc 11, HD 172555 24, 49 Ceti 16, HR 4796 16,
  Fomalhaut 10, Vega 20, AU Mic 18, HD 181327 22, eta Crv 21 (212 total).
- A Claude **Cloud routine** needs the repo on GitHub; the paste-in prompt is in
  `docs/cloud-routine-prompt.md`. Not yet created in the Routines UI.

## 10. Account quirk (will bite again)

- Two gh accounts are logged in. The **active** one is `elpis-app`, which has no access to
  this repo.
- Fixed for git inside the repo with local config:
  `credential.https://github.com.username = AJpro-merc` and
  `credential.https://github.com.helper = !gh auth git-credential`. `git push` now works.
- **`gh` API commands** (workflow runs, repo settings) still use the active account. Prefix
  them with `GH_TOKEN=$(gh auth token -u AJpro-merc)`.

## 11. Documentation written

- `README.md`, `docs/research_log/003-half-depth-contour-sign-error.md`,
  `docs/cloud-routine-prompt.md`, this log, `docs/NEXT_STEPS.md`.
- BRAIN: `docs/` — Index, The Science, Pipeline Architecture, Findings Log,
  Data Sources and Literature, Session Log, Next Steps.
- Figures: `results/figures/overview.png`, `results/figures/published_dips.png`.
