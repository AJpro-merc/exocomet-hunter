# Session Log — 2026-09-14 (ran into early 2026-09-15)

Everything that happened in the second build session, in order, with the numbers.
Identical copy kept in `docs/SESSION_LOG_2026-09-14.md` in the repo. Continuation of
[[Session Log 2026-09-13]]; see [[Next Steps]] for current task status.

---

## Plain-language summary

Started the day with one job: check whether the daily robot had run overnight with known
bugs, and decide what to do about it. It hadn't (only one manual test run ever existed),
so we disabled it cleanly and moved on to building the actual training pipeline for the
classifier — the part of the project that judges *whether* a detected dip is really
comet-shaped, not just that something dipped.

Along the way we found and fixed four real, previously-unknown bugs (one of them serious
enough that it had quietly corrupted every row of training data generated so far), tried
and killed an entire architecture (Claude Cloud routines can't reach NASA's data archive
at all — a platform limit, not a bug), rebuilt the automation on GitHub Actions instead,
and designed a genuinely continuous, weeks-long training pipeline with the user across
about 20 explicit decisions. The session ended mid-verification on one piece (the
classifier's real-data backtest hung and needs a clean retry) and with a full git-history
cleanup requested and executed for both repositories.

---

## 1. Starting state and A1/A3

Picked up from [[Session Log 2026-09-13]]: the daily TESS survey workflow was still
enabled, running with three known bugs (A1: windows sized for Kepler cadence, A2: TESS
pipeline mixing, A4: unreliable flag statistic).

- **Checked first, before touching anything:** `gh run list --workflow=survey.yml` showed
  only one run ever — the manual dry run from 2026-09-13. The 05:00 UTC schedule had
  never actually fired. Nothing to quarantine.
- **A1 fixed**: `config/thresholds.yaml`'s windows were counts of cadences
  (`window_length: 401`), tuned for Kepler's 29.4-minute cadence. Changed to days
  (`window_days: 8.1871` etc.), converted to a cadence count at runtime via the new
  `detect.baseline.days_to_cadences`, using each light curve's own measured spacing.
  Verified exactly equivalent on Kepler: `days_to_cadences` at Kepler long cadence
  reproduces the original 401/101/101/50 cadence counts precisely. 48/48 unit tests
  passed; a real re-run on KIC 3542116 found the same 7 events at the same epochs.
  Committed `97e2fc1`.
- **A3 resolved**: disabled the workflow (`gh workflow disable survey.yml`).
- Committed the two docs left uncommitted from the previous session
  (`docs/NEXT_STEPS.md`, `docs/SESSION_LOG_2026-09-13.md`). Commit `09ab993`.

## 2. C1 — building the classifier's training data (local, Kepler-only)

The main next build per [[Next Steps]] Part C. Built:

- `scripts/build_kepler_host_list.py`: 24 hand-picked Kepler long-cadence targets
  ("no confirmed planet, no confirmed EB" — but never actually checked against a real
  catalog; this came back to bite the session later, see §6).
- `scripts/generate_training_labels.py`: for each host, inject one of 5 label classes
  (`comet`, `symmetric`, `reversed`, `noise`, `time_reversed_real`), run the unmodified
  detector, extract 19 features on any matched candidate, write one row.
- Ran for real, locally: **191 rows**, 22/24 hosts succeeded (2 KIC numbers failed to
  resolve), ~40 minutes, class balance roughly even. Committed `461908a`.

**Storage decision (user-driven):** local disk has only ~24 GB free — no local cache.
Cloud storage (user has 1.5 TB) was considered for a cache but rejected: the actual
architecture is stream-and-discard (`discard_after_read=True`), so there's no persistent
cache to store anywhere. This foreshadowed the later exocomet-hunter-data decision.

## 3. Real bug #1 — `discard_after_read` was a silent no-op

While verifying disk usage for the C1 run, found `data/raw/` had grown to **166 MB**
despite `discard_after_read=True` on every download. Root cause: `_discard_cached`
matched a normalised slug of the catalog `target_id` (`"kic1161345"`) against downloaded
directory names — but `lightkurve` names cache directories after its own pipeline
identifiers (`kplr001161345_lc_...` for Kepler, a zero-padded TIC number mid-filename for
TESS HLSP products), never the search string used. The slug never matched anything, for
any target, either mission, since the function was written. Fixed: wipe the whole
`mastDownload` tree after each target instead of name-matching subdirectories (correct as
long as targets are processed one at a time, true of every current caller). Verified: a
real fetch now leaves `data/raw/` at 0 bytes afterward. Committed `f851649`.

## 4. Claude Cloud routines — tried, and a confirmed dead end

User wanted the recurring training-data generation to run unattended, in the cloud, so it
kept going while the laptop was off. First attempt: two Claude Cloud routines (Kepler
track, TESS track — the latter deliberately marked provisional pending A2/A4).

**Repo picker problem**: the routine editor showed no repositories at all. Diagnosed as
an account-mismatch (Claude's connected GitHub account wasn't AJpro-merc) — same class of
issue as the `gh` CLI account quirk from the previous session.

**Once the repo attached, both routines ran and both failed identically**: every request
to `mast.stsci.edu` got a 403 from the sandbox's own egress proxy — confirmed via the
proxy's own diagnostics as a policy denial, not a transient error, on 24/24 (Kepler) and
10/10 (TESS) hosts. Checked for a workaround before giving up: the routine's Behavior tab
(only has a PR auto-fix toggle), the Connectors tab (only registers MCP tool servers, not
a generic domain allowlist — confirmed via the "Add connector" dialog, which asks for an
MCP server URL, not a domain), and account Settings. **No network-allowlist setting
exists anywhere in the product.** This is a fixed platform limit.

**Why this matters for later**: GitHub Actions runners, by contrast, have normal internet
access and had already proven it (the previous session's `survey.yml` dry run downloaded
212 real products across 10 stars). This became the basis for the whole rest of the
session's automation.

## 5. Pivot to GitHub Actions

Built `.github/workflows/train.yml` v1: two jobs (`kepler`, `tess` — sequenced,
`tess: needs: kepler`, to avoid a git-push race), cron `0 */2 * * *`, each capped to a
30-minute internal budget. Committed `19506dc`.

**Manually triggered to verify** (`gh workflow run train.yml -f time_budget_minutes=3`):
real MAST data downloaded successfully — 21 real rows from a real Kepler star, proving
the platform choice was right. **But the run failed** at the `pytest` step:
`pytest: command not found` — the Install step ran `pip install -e .`, which doesn't pull
the `dev` extra containing pytest (the same gap the Cloud routine had already
independently discovered and worked around by hand). Because commit-and-push was gated
behind tests passing, the 21 real rows were never committed — lost, though trivially
regenerable. This became item 1 on the next session's to-do list until it was actually
fixed later this same session (§9).

## 6. Real bug #2 — the generator never detrended

While smoke-testing new injection classes (§7), noticed absurd `depth_ppm` values
(~78,000 ppm on a supposedly-quiet star). Investigation: called
`AsymmetricDipDetector` directly on **raw, undetrended** flux for KIC 1026032 — 319
spurious candidates up to ~8% depth. Checked `generate_training_labels.py`: it never
called `detrend_savgol` at all. Every row generated before this point, including the
191-row local run and the one real GH Actions row, ran on undetrended data. Nothing
invalid had been committed anywhere trustworthy yet (the GH Actions run never got past
the pytest failure to commit), so nothing needed cleanup — but it had to be fixed before
generating anything for real. Fixed: detrend **after** injecting, not before, so a fake
event passes through the same Savitzky-Golay filter a real one would (consistent with
2026-09-13's Bug B1 finding that detrending can distort depth/shape — that distortion
should be part of what the classifier sees). Verified against the same host:
`depth_ppm` values dropped to a sane 1000-2000ppm range.

**Related finding**: even after detrending, KIC 1026032 *still* showed a real,
strong (~8% deep, ~8.4-day period) periodic signal — almost certainly a real eclipsing
binary. The hand-curated host list from §2 was never checked against a real catalog.
Removed it (`build_kepler_host_list.KNOWN_BAD_HOSTS`) and built
`scripts/vet_host_list.py`: an empirical check (run the real pipeline, no injection, flag
anything suspiciously deep/frequent) rather than a catalog lookup, since catalog lookups
can miss a star nobody ever flagged. Not yet run against the full remaining 23-host list
— next session's job.

## 7. Continuous training pipeline — design (10+ decisions)

User asked for a genuinely long-running (week/month) training pipeline and worked through
this with a mix of multiple-choice questions and direct feedback:

| Decision | Answer |
|---|---|
| Host list | Auto-expand via NASA Exoplanet Archive query (not yet built — still the static list) |
| Stop condition | None — runs until manually disabled |
| Model file | Overwritten weekly, not every cycle, no dated snapshots |
| Regression gate | None — always commit the latest retrain, but record a soft comparison |
| Bootstrap draws | 300 (between the original 200 and full-precision 1000) |
| Attempts/class/host | Keep 20 (CLI default) |
| A2 | Fix it today, not deferred |
| Growth cap | None — user's own periodic download-and-zip is the real limit |
| Depth-weighted sampling | Yes, widened per "increase the boundary" feedback |
| Check-ins while unattended | A separate **reporting** Cloud routine (viable — reads GitHub, never touches MAST) |
| Data storage | A separate repo, `AJpro-merc/exocomet-hunter-data`, not this repo's git history |
| Retrain cadence | Weekly (clarified after an initial "automate it" answer) |
| Two backtests | Confirmed: real KIC 3542116 dips, and a fake-injection completeness curve |

**External review**: user pasted a large (34-item) feature brainstorm (AI scientific
review layer, falsification engine, physical comet simulator, independent second
detector, etc.), explicitly self-described by its own source as long-term direction, not
a same-session build list. None of the 34 features were built. Its concrete
risk-mitigation items (idempotent generation, per-cycle work caps, a written holdout
protocol, dedup/sanity checks on append, A2 provenance columns, a saved threshold
artifact, skip-retrain-if-unchanged, job isolation) *were* folded into the actual build,
since they were cheap and directly patched real weaknesses in the locked design. See
§10 below for what got built.

## 8. A2 fixed for real

`fetch_light_curve`/`iter_light_curves`/`MastLightCurveSource` gained `author`/`exptime`
parameters, defaulting to `author="SPOC"`, `exptime=120` for TESS and `author="Kepler"`
for Kepler. This turned out to fix two problems at once: the originally-named cadence
mixing, and a second, worse one found via direct testing — the TARS pipeline (the one
that had also failed to download for 49 Ceti with a "file may be corrupt" error) reports
flux as an already zero-centered residual, not raw counts, so stitching it in could
divide by a near-zero median and corrupt the normalisation outright, not just misalign
cadences. **Verified against real data**: beta Pic now resolves to 10 clean SPOC/120s
products (down from 54 mixed-pipeline products), flux normalises to 0.99-1.01 (previously
corrupted/near-zero-divide), author/exptime/sectors now recorded in `lc.meta` and in
every generated training row. `monitor_new_data.py`'s change detector fixed the same way
(counts only the pinned author/exptime, so a new QLP/TARS file no longer triggers
reprocessing of unchanged data). New test `tests/unit/test_cross_cadence.py` (synthetic,
no network): the same injected comet recovered at matching epoch/τσ at both 120s and
1800s cadence. Committed `52ac2ec`.

## 9. Generator rewrite — weighting, new classes, caps, dedup, A2 provenance

`scripts/generate_training_labels.py` rewritten with:

- **Two new false-positive lookalike classes** (Next Steps D1): `flare` (brightening,
  fast rise/exponential decay — new `inject_flare` in `calibration/injection.py`) and
  `starspot` (sinusoidal rotation modulation, period 0.5-30d, amplitude 100-5000ppm — new
  `inject_starspot_modulation`). 7 classes total now, not 5.
- **Depth/amplitude sampling weighted** toward 500-2000ppm (widened from an initial
  narrower 750-1500ppm-only proposal, per explicit "increase the boundary" feedback):
  grid `[200,300,500,750,1000,1500,2000,3000]` ppm, weights `[1,2,3,4,4,3,2,1]`.
- **`--max-hosts`/`--max-new-rows`**: bound a single run's work independent of the time
  budget.
- **Sanity/dedup checks** on every write: reject non-positive/NaN injected depth, missing
  host_star, exact-duplicate rows.
- **Least-covered-host-first ordering**: hosts are now processed in ascending order of
  existing row count in the accumulated output, so a fixed time budget doesn't always
  exhaust itself on the same early hosts in a recycled list.
- **`author`/`exptime`** now recorded per row (A2 provenance), not just in `lc.meta`.
- Fixed the detrend bug from §6 in the same pass.

Also fixed `scripts/train.yml`'s missing-pytest-extra bug from §5
(`pip install -e ".[dev]"`). Committed `2757067`.

Real verification runs: Kepler (2 hosts, 36 rows, sane depth values, all 7 classes
represented) and **TESS post-A2-fix** (beta Pic, real SPOC/120s data, rows produced with
correct author/exptime columns — first real proof the TESS path works end-to-end after
the A2 fix).

## 10. `AJpro-merc/exocomet-hunter-data` created

Per the storage decision in §7's table: a separate GitHub repo, not this repo's git
history. Created via `gh repo create`, initialised with a `README.md` explaining the
layout (`training/*.parquet`, `models/*.joblib`, `models/manifest_*.jsonl`,
`models/last_compare.json`, `models/threshold_*.json`) and the append-only convention.

## 11. `scripts/train_classifier.py` (C2) — built, not fully verified

Implements Next Steps C2: `GroupKFold` (by `host_star`) cross-validated
`RandomForestClassifier` wrapped in `CalibratedClassifierCV`, a hard `HoldOutError`
exception (not just a warning) if either hold-out star ever appears in the training data,
`SimpleImputer` + `is_measurable` indicator for NaN features, a probability threshold
chosen from out-of-fold predictions at 1% false-positive rate, and two backtests recorded
in an append-only provenance manifest (`models/manifest_<mission>.jsonl`): the real
KIC 3542116 final exam (does the model score all 6 published dips above threshold), and a
fake-injection completeness-by-depth curve from the OOF predictions. Also writes a
`last_compare.json` against the previous manifest entry — soft regression visibility, no
gate, per the locked decision.

**Not fully verified end-to-end**: the real-data-backtest step (live fetch of
KIC 3542116) hung with zero CPU progress for 9+ minutes during testing and had to be
killed; a direct retry of the same `fetch_light_curve` call also hung and was interrupted
(by the user, "stop") before root-causing finished. Every other code path in the script
ran and was inspected (loading, hold-out check, GroupKFold/CalibratedClassifierCV
training, completeness calc, manifest/threshold writing) — this one network step needs a
clean, watched retry next session before the script is trusted in production. Likely
transient (every other MAST fetch that session succeeded, including one for this same
star earlier), but unconfirmed.

## 12. `train.yml` rewritten in full

Two schedule triggers in one workflow (`0 */2 * * *` generation, `0 6 * * 1` weekly
retrain), jobs conditioned on which cron fired via `github.event.schedule`.
Generation/training output goes to `exocomet-hunter-data` via a `DATA_REPO_TOKEN` secret
the user must create themselves (a fine-grained PAT scoped to that repo only,
`Contents: Read and write`, added on `exocomet-hunter`'s own Settings → Secrets — never
handled by this session, per the hard rule against entering credentials on the user's
behalf even when authorized). `kepler`/`tess` sequenced to avoid a data-repo push race;
`train` job `needs: [kepler, tess]` with `if: always()` so a TESS hiccup doesn't block
Kepler's weekly retrain (job isolation). Skip-retrain-if-data-unchanged via a sha256 hash
file. Committed `bc1fddc`.

`docs/cloud-routine-prompt-training.md` updated: routines 1/2 marked as a confirmed dead
end with the full investigation recorded (so it's never blindly re-attempted); a new
**routine 3** (`Exocomet training summary`) added — reporting-only, reads
`exocomet-hunter-data`, never touches MAST. This one is genuinely viable; text is ready
to paste in, not yet created in the Claude app.

## 13. Git history cleanup (explicit user request, `YRAG`-confirmed)

Late in the session the user asked to remove all Claude attribution from git history
entirely and instead credit it once in each repo's README. Per the vault's own rule
(`CLAUDE.md`: destructive/irreversible actions need the explicit `YRAG` confirmation
phrase, not just "do it"), this was surfaced and confirmed before acting.

- **`exocomet-hunter-data`**: already a single commit with no Claude mention. Amended
  in place to add the README line (""),
  force-pushed.
- **`exocomet-hunter`**: entire multi-commit history squashed into one clean commit
  (`176e2b6`, no Claude mention anywhere), README line added, local `main` rebuilt on an
  orphan branch. **The force-push to `origin` was blocked twice by the Claude Code
  harness's own auto-mode classifier** (a separate gate from the user's in-chat `YRAG`
  confirmation) and needs the user's direct approval in the permission-prompt UI —
  **not yet pushed as of end of session.** Local state is ready; this is the very first
  thing to either approve or retry next session.
- Going forward, no more `Co-Authored-By`/`Generated with Claude Code` lines in any
  commit or PR from this project (a standing instruction change, not just a one-time
  cleanup).

## Numbers, for reference

- C1 local run: 191 rows, 22/24 hosts, ~40 min, Kepler-only, pre-detrend-fix (superseded).
- A1 fix: exact cadence-count equivalence verified (401/101/101/50 at Kepler long
  cadence).
- A2 fix: beta Pic 54→10 products, flux 0.99-1.01 (was corrupted before the fix).
- `discard_after_read` bug: 166 MB had silently accumulated; 0 bytes after the fix.
- Real GH Actions test run: 21 rows from one real Kepler star before the pytest failure
  stopped the commit.
- Session commits (`exocomet-hunter`, pre-squash, for the record): `97e2fc1`, `09ab993`,
  `461908a`, `0f3c5f9`, `f851649`, `19506dc`, `52ac2ec`, `2757067`, `bc1fddc` — now folded
  into the single squashed commit `176e2b6`.

## What's next

See [[Next Steps]]'s "FIRST THING NEXT SESSION" block — in order: verify/retry the
`train_classifier.py` real-data backtest hang, push the squashed `exocomet-hunter`
history (needs user approval in the permission UI), set up `DATA_REPO_TOKEN`, trigger
`train.yml` for a real end-to-end run, create Cloud routine 3, and (lower priority,
explicitly deferred) add copyright headers across public-facing files.
