# Cloud routine setup

How to run the survey automatically on Anthropic's machines, so it works whether or
not this laptop is on.

## Setting it up

1. Claude → **Routines** → **New routine** → **Cloud**
2. **Name:** `Exocomet survey`
3. **Repository:** `AJpro-merc/exocomet-hunter`
4. **Trigger:** Schedule → daily, 09:00

   Daily rather than weekly because the *check* is cheap and the *work* is not.
   Querying the archive takes seconds; downloading and processing only happens
   when something new actually exists. TESS sectors land roughly monthly, so most
   runs will correctly report "nothing new" — that is the expected result, not a
   failure.
5. **Instructions:** paste the block below

## The prompt

```
You are running a daily exocomet survey on the exocomet-hunter repository.

TESS is still observing, so new sectors keep appearing in NASA's MAST archive.
Your job is to check whether any watched star has new data, run the detection
pipeline over it, and report what turned up.

Steps:

1. Set up the environment:
     python -m venv .venv && .venv/bin/pip install -e .

2. First check what is new without downloading anything:
     .venv/bin/python scripts/monitor_new_data.py \
       --watchlist config/watchlist.txt --mission TESS --dry-run

3. If any target has new data, process at most 5 of them:
     .venv/bin/python scripts/monitor_new_data.py \
       --watchlist config/watchlist.txt --mission TESS --limit 5

4. Confirm nothing regressed:
     .venv/bin/pytest tests/unit -q

5. If results/monitor_candidates.jsonl gained new rows, commit and push
   results/ back to the repository with a message describing the run.

6. Report back, in plain language, no jargon:
   - which stars had new observations, and how many new data products
   - how many events were flagged, with depth in ppm, tau/sigma, and delta_bic
   - whether the unit tests still pass
   - whether anything failed or looked wrong

Important context for interpreting results:

- A flagged event is a CANDIDATE, never a confirmed exocomet. Do not describe
  anything as a discovery. Real exocomet detections are rare and require
  follow-up this pipeline cannot do.
- Most flagged events will turn out to be eclipsing binaries, stellar flares,
  instrumental artefacts or noise. Say so.
- tau/sigma greater than 1 means the dip has a trailing tail, which is the comet
  signature. delta_bic above 10 means the comet-shaped model fits clearly better
  than a symmetric one.
- If nothing has new data, that is a perfectly normal result. Say "no new
  observations this week" and stop. Do not invent activity.
- If a download fails, note it and move on. One unavailable star must not end the
  run.
```

## Why cloud and not local

A local routine only runs while the Claude app is open. A cloud routine runs on
Anthropic's machines on schedule, which is what "continuously monitor TESS" actually
requires.

## The GitHub Actions alternative

`.github/workflows/survey.yml` does the same job using GitHub Actions, free for public
repositories, and needs no Claude session at all. Either is fine; the Actions version
is more robust, the Claude version gives a written summary each week. Running both is
harmless — they share the same ledger file.
