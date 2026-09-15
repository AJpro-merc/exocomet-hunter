# Cloud routines for training-data generation

Two separate recurring Claude Cloud routines that generate classifier training rows
(Next Steps Part C1) while the laptop is off. Neither of these routines trains a model
or runs the production TESS survey -- see `docs/cloud-routine-prompt.md` for that.

Set up in Claude app -> **Routines** -> **New routine** -> **Cloud** -> repository
`AJpro-merc/exocomet-hunter` -> trigger **Schedule**, cron `0 */2 * * *` (every 2 hours).

## Routine 1 -- `Exocomet Kepler training generation`

Safe: no open correctness bugs block the Kepler track.

```
You are generating classifier training data for the exocomet-hunter repository,
Kepler track. This runs unattended on a schedule -- be conservative, never guess.

Steps:

1. Set up the environment:
     python -m venv .venv && .venv/bin/pip install -e .

2. Run the generator, capped to a 30-minute budget so this run ends on its own
   even if the schedule fires again before it would naturally finish:
     .venv/bin/python scripts/generate_training_labels.py \
       --mission Kepler \
       --out results/training/labels_kepler.parquet \
       --time-budget-minutes 30 --bootstrap-draws 200

   (no --host-list needed: Kepler falls back to the 24-star list in
   scripts/build_kepler_host_list.py, which already excludes the two
   validation/hold-out targets)

3. Confirm nothing regressed:
     .venv/bin/pytest tests/unit -q

4. If results/training/labels_kepler.parquet changed, commit and push it with a
   message stating how many rows were added this run and the new total.

5. Report back, in plain language, no jargon:
   - how many hosts were processed and how many new labelled rows were added
   - the running total row count and class balance (comet/symmetric/reversed/
     noise/time_reversed_real)
   - whether unit tests still pass
   - anything that failed or looked wrong; one bad host must not end the run

Important:
- Never train a model in this routine -- only generate and append labelled rows.
  Training/evaluation happens later, in a real conversation, on purpose.
- KIC 3542116 and KIC 11084727 must never appear in the host list or output --
  they are the permanent validation hold-out. If either appears, stop and report
  it as a bug rather than continuing.
- This track is Kepler-only and has no open correctness bugs blocking it.
```

## Routine 2 -- `Exocomet TESS training generation (provisional)`

**Every row this routine produces is provisional.** A2 (mixed TESS pipelines) and A4
(unreliable flag rule) are not fixed yet -- see `docs/NEXT_STEPS.md`. This
routine exists because the user chose to accept that tradeoff for training-data volume
now, on the understanding this data may need regenerating once A2/A4 land.

**Known extra caveat (not yet resolved as of this writing):** `config/tess_training_hosts.txt`
currently reuses `config/watchlist.txt`, which is deliberately a list of stars with
*plausible or established real exocomet/debris-disc activity* (beta Pic, the canonical
exocomet system, among them). That is the right list for the production search, but it
is a **questionable choice as an injection host list**: a "noise"-class attempt on a
star that has real, undocumented dimming events risks mislabelling a genuine event as
noise. Next Steps C1 calls for host stars with *no* known activity for exactly this
reason. Flagged here rather than fixed silently -- swap in a genuinely quiet TESS host
list before trusting the `noise` and `time_reversed_real` classes from this routine.

**Second known caveat, found while smoke-testing this routine:** `--time-budget-minutes`
only checks the clock *between* hosts, not during one host's download. A single heavy
TESS star can overrun the budget by itself -- beta Pic alone (first in the current list)
has ~54 mixed-pipeline products (the exact A2 problem) and took several minutes just to
download, well past a 1.5-minute test budget, before the between-host check ever had a
chance to fire. For a 30-minute production budget this is a real but bounded risk
(one slow host, not unbounded runaway) -- worth knowing about, not yet worth fixing given
this routine's output is already provisional pending A2/A4.

**Third finding, from actually running this smoke test: A2 is worse than "windows
mismatch."** A full run against the current 10-star TESS list produced **zero** rows.
`lightkurve` warned that some products are "zero-centered" (median flux near 0 ppm, not
near a large positive count) before `stitch()`/`normalize()` runs -- meaning some TESS
HLSP pipelines (the same TARS pipeline that also failed to download for 49 Ceti with a
"file may be corrupt" error) report flux as an already-normalized residual, not raw
counts. `io/download.py`'s `_to_light_curve_data` unconditionally divides by
`median(flux)` (see its own guard: `if median <= 0: raise ValueError`) -- stitching a
near-zero-median product in with the rest doesn't just misalign cadences, it can
corrupt the resulting flux scale outright. This is a concrete, reproducible new data
point for A2's fix (which already calls for pinning a single `author`/`exptime`, e.g.
SPOC-only) -- the fix needs to exclude TARS-like already-normalized products entirely,
not merely separate them by cadence.

```
You are generating classifier training data for the exocomet-hunter repository,
TESS track. This runs unattended on a schedule -- be conservative, never guess.

IMPORTANT CONTEXT: two known bugs (tracked as A2 and A4 in
docs/NEXT_STEPS.md) are not yet fixed: TESS data from different
processing pipelines can get mixed together, and the significance rule used to
flag events is known to be unreliable. Every row this routine produces is
PROVISIONAL and may need to be regenerated once those are fixed. Never describe
this data as final or ready for training a released model.

Steps:

1. Set up the environment:
     python -m venv .venv && .venv/bin/pip install -e .

2. Run the generator, capped to a 30-minute budget:
     .venv/bin/python scripts/generate_training_labels.py \
       --mission TESS --host-list config/tess_training_hosts.txt \
       --out results/training/labels_tess_provisional.parquet \
       --time-budget-minutes 30 --bootstrap-draws 200

3. Confirm nothing regressed:
     .venv/bin/pytest tests/unit -q

4. If results/training/labels_tess_provisional.parquet changed, commit and push
   it with a message stating how many rows were added and that it is provisional
   pending A2/A4.

5. Report back, in plain language, no jargon:
   - how many hosts were processed and how many new labelled rows were added
   - the running total row count and class balance
   - a one-line reminder that this data is provisional pending A2/A4
   - whether unit tests still pass
   - anything that failed or looked wrong; one bad host must not end the run

Important:
- Never train a model in this routine -- only generate and append labelled rows.
- KIC 3542116 and KIC 11084727 must never appear in the host list or output.
- Do not touch .github/workflows/survey.yml or re-enable it -- that is the
  separate, still-disabled production search survey.
```

## Why cloud and not local

A local routine only runs while the Claude app is open. A cloud routine runs on
Anthropic's machines on schedule, so generation continues while the laptop is off.

## Review cycle

These routines only generate and commit; they never train or fine-tune anything. The
review/fine-tune step is deliberately conversational, not automated: when you come back,
ask to see the accumulated `results/training/*.parquet` files, get a summary (row
counts, class balance, any anomalies), and decide from there whether to move on to C2
(actually training the classifier) or generate more/different data first.

## Update, 2026-09-14 (later the same session): superseded by GitHub Actions

Both routines above are a **confirmed dead end** -- every real run got a 403 from this
platform's own egress proxy on every request to `mast.stsci.edu`. No domain-allowlist
setting exists anywhere in the product (checked: routine Behavior tab, Connectors tab
which only registers MCP servers, and account Settings). This is a fixed platform
limit, not a configuration gap.

Generation and weekly training now run on `.github/workflows/train.yml` instead --
GitHub Actions runners have real internet access, already proven against real MAST data.
See that workflow and `docs/NEXT_STEPS.md`'s session-update section for the
full story. **Do not recreate routines 1/2 above** -- they cannot work on this platform
regardless of how the instructions are written.

## Routine 3 -- `Exocomet training summary` (reporting only -- this one IS viable)

Unlike routines 1/2, this one never touches MAST. It only reads
`AJpro-merc/exocomet-hunter-data` (a normal GitHub repository, which Cloud routines'
native repo integration already handles) and summarizes in plain language. Trigger:
Schedule, daily.

```
You are reporting on the exocomet-hunter classifier training pipeline. You do NOT
generate data or train anything -- that happens elsewhere, in GitHub Actions
(.github/workflows/train.yml in the AJpro-merc/exocomet-hunter repository). Your job
is only to read AJpro-merc/exocomet-hunter-data and summarize what happened.

Steps:

1. In the AJpro-merc/exocomet-hunter-data repository, read:
   - training/labels_kepler.parquet and training/labels_tess.parquet (or note if
     either doesn't exist yet)
   - models/manifest_kepler.jsonl and models/manifest_tess.jsonl (each line is one
     training run's record -- read the last few lines of each)
   - models/last_compare_kepler.json and models/last_compare_tess.json if present

2. Report back, in plain language, no jargon:
   - current row counts for each of Kepler and TESS, and how much grew since you last
     checked (compare against your own last report if you remember it, otherwise just
     state the current totals)
   - class balance (comet/symmetric/reversed/flare/starspot/noise/time_reversed_real)
     for each
   - when each model was last retrained, how many rows it was trained on, and its two
     backtest results: how many of the 6 real published KIC 3542116 dips it correctly
     scores above threshold, and the fake-injection completeness numbers by depth
   - if last_compare shows the real-dip score dropped versus the previous training run,
     say so explicitly and clearly -- "this looks like it got worse," even though
     nothing blocks the model from being committed either way (that's intentional: no
     automated gate, but you should still flag it)
   - any TESS rows whose author/exptime columns are not SPOC/120s (would mean an A2
     regression -- should not happen post-fix, but check)
   - whether generation or training runs appear to have stopped working (e.g. no new
     commits in the last day when you'd expect them)

Important:
- You cannot fix anything from here -- if something looks broken, report it clearly so
  the user can look into it themselves next session.
- Never claim a "discovery" -- this pipeline finds candidates and trains a classifier
  on synthetic + real labelled data; nothing here is a confirmed comet detection.
- If the data repo is empty or the workflow clearly hasn't run yet, say so plainly
  rather than guessing.
```
