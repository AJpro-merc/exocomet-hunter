# 006 — Depth offset: partially explained, not resolved

**Date:** 2026-09-15
**Status:** Open — root cause not fully identified. Detector NOT changed.
**Affects:** measurement only; `detect/comet_detector.py`, `detect/baseline.py` are candidates for
a future fix, not touched here
**Tooling:** `scripts/measure_depth_offset.py`

## The problem

All six recovered dips in KIC 3542116 report depths at roughly **77% of the Rappaport et al. (2018)
published values.** Six out of six low, by a similar factor, is systematic.

| Dip | Published (ppm) | Reported (ppm) | Ratio |
|---|---|---|---|
| D140, shallow | 491 | 442 | 0.90x |
| D742, shallow | 524 | 461 | 0.88x |
| D793, shallow | 679 | 412 | 0.61x |
| D992, deep | 1200 | 940 | 0.78x |
| D1176, deep | 1500 | 1094 | 0.73x |
| D1268, deep | 1900 | 1445 | 0.76x |
| **median ratio** | | | **0.772x** |

## Three hypotheses, tested against real data with a known-truth check first

1. **Baseline contamination (H1)** — the rolling-median reference window straddles the dip, so the
   reference level is dragged down toward the dip itself.
2. **Fitted vs observed depth (H2)** — compare `ModelComparison.comet.params["depth"]` against the
   observed minimum.
3. **Detrending absorption (H3)** — Savitzky-Golay division over an 8.19 d window may partially
   follow a ~1 d dip.

`scripts/measure_depth_offset.py` measures all four ways (current + H1 + H2 + H3, plus H1+H3
combined) for every recovered event, against both a synthetic light curve with known injected
depths (a check on the harness's own arithmetic) and the real cached KIC 3542116 photometry
(`data/raw/mastDownload/`, offline, no network — one corrupt cached product from the 2026-09-14 hang
was skipped automatically, `0 skipped` in the run below since it had already been quarantined).

## Step 1 — synthetic sanity check (known truth, so any deviation from 1.00x is the harness's own bias)

| Method | median ratio | scatter |
|---|---|---|
| current (detrended, rolling) | 0.946x | 0.041 |
| H1 (detrended, out-of-event) | 0.947x | 0.067 |
| H2 (fitted comet depth) | 1.145x | 0.062 |
| H3 (raw, rolling) | 0.948x | 0.043 |
| H1+H3 (raw, out-of-event) | 1.016x | 0.064 |

**Already informative before touching real data.** Even with perfectly known injected depths, the
`current` method reads **5.4% low**, and neither H1 nor H3 alone move that — both sit within 0.2
points of `current`. Only the *combination* H1+H3 approaches unbiased (1.016x). H2 overshoots by 14.5%.
So on ideal data, baseline contamination and detrending absorption are each small and largely
independent; neither is individually responsible for a bias of this size, and the ~5% floor on
`current` here is a lower bound on any real-data measurement, not evidence against either hypothesis.

## Step 2 — real data (KIC 3542116, cached, offline)

| Method | median ratio | scatter |
|---|---|---|
| current (detrended, rolling) | **0.772x** | 0.098 |
| H1 (detrended, out-of-event) | 0.852x | 0.103 |
| H2 (fitted comet depth) | 1.316x | **1.009** |
| H3 (raw, rolling) | 0.779x | 0.098 |
| H1+H3 (raw, out-of-event) | 0.879x | 0.122 |

Full per-dip table in the script's `--markdown` output; reproducible with
`python scripts/measure_depth_offset.py --source cache --sweep --markdown`.

## Reading the result honestly

**None of the three hypotheses, alone or combined, closes the gap.**

- **H3 (detrending) is not the cause.** 0.779x vs 0.772x for `current` — statistically indistinguishable.
  This matches the synthetic result and rules out hypothesis 3 with reasonable confidence.
- **H1 (baseline contamination) explains roughly a third of the gap.** 0.772x → 0.852x when isolated.
  Real, but far from sufficient on its own.
- **H1+H3 combined reaches 0.879x** — better, but still 12 points short of unbiased, and worse than
  either hypothesis alone would predict from the synthetic check (where H1+H3 nearly closed the gap).
  **The real-data shortfall is larger than the synthetic-data shortfall by more than the difference
  between the two hypotheses tested.** Something present in real KIC 3542116 photometry and absent
  from the synthetic injection is contributing an additional ~10-15% on top.
- **H2 (fitted depth) is not usable as reported.** Median 1.316x looks closer to unbiased than
  `current`, but the scatter (sd = 1.009, essentially as large as the mean) reveals why that number is
  meaningless: individual ratios range from 0.99x (D1176) to **3.47x** (D140 — 1704 ppm fitted against
  a 491 ppm published depth). The comet-profile fit is diverging badly for at least the shallow events.
  This is very likely a **separate, real bug** in `detect/model_comparison.py`'s fit — worth its own
  investigation — not evidence for or against hypothesis 2 as originally posed.

## The baseline-window sweep (sharper test of H1 alone)

Widening the rolling-median window from the shipped 2.06 d out to 20 d:

| window (d) | median ratio |
|---|---|
| 2.06 (shipped) | 0.772x |
| 3.00 | 0.822x |
| 4.00 | 0.846x |
| 5.00 | 0.849x (peak) |
| 6.00 | 0.843x |
| 8.00 | 0.818x |
| 10.00 | 0.792x |
| 14.00 | 0.784x |
| 20.00 | 0.774x |

Non-monotonic, peaking around 5 days then decaying back toward the shipped value by 20 days — the
window first escapes the dip (ratio rises) then starts reincorporating real stellar variability on
longer timescales (ratio falls again). The peak, 0.849x, is the best H1-alone can do at any window
width, consistent with the out-of-event number above. It confirms H1 is real but bounded, not a free
parameter that can be tuned away.

## What remains unexplained

At minimum ~10-15% of the 22.8% published gap survives every combination tested. Candidates not yet
tested, in rough order of suspicion:

1. **The published depths may themselves be model-fitted**, not raw minima — Rappaport et al. may
   report a depth from a full transit-shape fit with different free parameters than this pipeline's
   comet model, which would not be resolved by anything tested here.
2. **Quarter-to-quarter flux offsets from stitching** — 16 separate Kepler quarters are stitched;
   each has its own PDC normalisation, and a systematic per-quarter offset would bias the reference
   level in a way neither H1 nor H3 isolates.
3. **The comet-model fit itself (H2's instability)** deserves its own investigation — if the fit
   diverges for shallow events, the model comparison and ΔBIC ranking for those events may also be
   less trustworthy than assumed, independent of this depth question.

## What this investigation did NOT do

No change to `detect/comet_detector.py`, `detect/baseline.py`, or how depth is reported. This is
measurement only. The 22% low depth is now decomposed into a ~5% harness/methodology floor, a
~real but partial H1 contribution (~8 points), and a **~10-15% unexplained residual specific to real
data** — narrower than "the depth is wrong," not yet narrow enough to fix with confidence.

## Next steps

- Investigate the H2 fit divergence on shallow events as its own bug, independent of this question.
- Test quarter-offset stitching as a fourth hypothesis.
- Do not change the shipped baseline window based on the sweep above — 0.849x at its best point is
  still an 15-point gap, and the non-monotonic shape means "wider is better" is not a safe rule to
  extrapolate from this data alone.

## References

- Rappaport, S., et al. 2018, MNRAS, 474, 1453, "Likely transiting exocomets detected by Kepler"
  (arXiv:1708.06069)
- [003 — The half-depth asymmetry metric has the wrong sign](003-half-depth-contour-sign-error.md) —
  prior instance of a statistic passing epoch recovery while carrying a real quantitative error
