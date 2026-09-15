# 007 — ΔBIC is direction-blind; the τ/σ condition is load-bearing

**Date:** 2026-09-15
**Status:** Resolved — flagging rule changed (Next Steps A4), finding documented
**Affects:** `detect/scoring.py`, `detect/model_comparison.py`, `core/config.py`, `core/types.py`
**Follows from:** [003 — The half-depth asymmetry metric has the wrong sign](003-half-depth-contour-sign-error.md)

## Context

Log 003 replaced the half-depth contour asymmetry $A_{dur}$ with the fitted ratio
$\tau/\sigma$ as the primary shape statistic, and named $\Delta$BIC the primary
ranking metric. The flagging *decision*, however, was never updated to match. Until
today it read

```python
flagged = abs(a_dur) / sigma >= 3.0
```

Because of the `abs()`, this rule is blind to sign. A **reversed** event — slow
ingress, fast egress, the physically backwards direction for a trailing dust tail —
produces a large $|A_{dur}|$ and was therefore flagged as a candidate. The unit
test `test_reversed_asymmetry_is_flagged_with_the_opposite_sign` asserted this
behaviour as correct.

## Hypothesis under test

The A4 fix replaces the rule with the two statistics log 003 established as
trustworthy:

```python
flagged = delta_bic > 10.0 and tau_over_sigma > 1.0
```

The expectation going in was that $\Delta$BIC would do the discriminating work and
the $\tau/\sigma$ term was a cheap secondary guard.

**That expectation was wrong, and the error is instructive.**

## Observation

Two synthetic events were generated as exact mirror images of each other — same
depth, same noise, same total duration, only the time direction of the profile
flipped (`viz/plots.plot_tail_direction`, figure at
`results/figures/tail_direction.png`):

| | Trailing tail (comet-like) | Leading tail (time-reversed) |
|---|---|---|
| ingress / egress | 2.4 h / 12.0 h | 12.0 h / 2.4 h |
| $A_{dur}$ | **+0.57** | **−0.58** |
| $\tau/\sigma$ | **4.15** | **0.23** |
| $\Delta$BIC | **151** | **201** |
| flagged under new rule | yes | no |

Two things to read off that table.

**First, why the old rule failed.** $|A_{dur}|$ is 0.57 versus 0.58 — indistinguishable.
The sign carries all the information and `abs()` discarded exactly it.

**Second, and unexpected: the reversed event scores a *higher* $\Delta$BIC than the
real comet — 201 against 151.** On $\Delta$BIC alone, the physically impossible event
is the *more* confident detection.

## Cause

$\Delta$BIC compares a comet model against a **symmetric** model. It answers "is this
profile asymmetric?" — not "is this profile asymmetric *in the direction a dust tail
produces*?"

A mirror-image event is exactly as asymmetric as the original, so the comet model beats
the symmetric model just as decisively either way. It scores slightly *higher* here
only because the fitter lands marginally better on this particular realisation of the
noise; the two values are not meaningfully different, which is the whole point. There
is no direction information in $\Delta$BIC to extract.

$\tau/\sigma$ is the only statistic in the set that encodes direction: 4.15 versus
0.23, a factor of eighteen, split cleanly across the threshold of 1.

## Resolution

Both conditions are retained, and the ordering of the conjunction is not arbitrary:

- $\Delta$BIC > 10 establishes **that there is a real asymmetric event** rather than noise.
- $\tau/\sigma$ > 1 establishes **that the asymmetry points the right way**.

Neither is sufficient alone. Thresholds moved to `ScoringConfig.delta_bic_threshold`
and `ScoringConfig.tau_over_sigma_threshold` (mirrored in `config/thresholds.yaml`)
rather than being hardcoded, and `ModelComparison.favors_comet`'s previously hardcoded
`delta_bic > 10.0` now reads the same config value, so the ranking metric and the
flagging rule cannot drift apart.

$A_{dur}$ and its significance are still computed and reported as output columns. They
are no longer part of the decision.

`test_reversed_asymmetry_is_flagged_with_the_opposite_sign` is renamed
`test_reversed_asymmetry_is_not_flagged` and its assertion inverted. A second test
confirms `significance` is still populated for reversed events, so the sign information
remains available to the classifier as a feature even though it no longer gates anything.

## Warning to future work

**Do not "simplify" this rule by dropping the $\tau/\sigma$ condition.** It looks
redundant next to a $\Delta$BIC of 151 and is not. Removing it restores precisely the
false-positive generator that A4 was opened to remove, and the resulting failure would
be silent: the pipeline would keep recovering the six published dips (they are genuine
trailing-tail events and pass either way) while also admitting every backwards event it
encounters. Recovery of known events would not catch it — which is the same trap log 003
documented, arriving from a different direction.

## Wider lesson

Log 003 closed by observing that recovering known epochs is *necessary but not
sufficient*. This is the second instance of the same lesson: a statistic can be
completely correct for ranking and completely useless for deciding. $\Delta$BIC was
never wrong — it was answering a different question than the one the flag was asking.

The general form: **before using a statistic as a decision boundary, construct the
adversarial case it cannot distinguish.** Here that case was a two-line change — mirror
the injected profile in time — and it exposed the gap immediately. That test should exist
for every discriminator the pipeline adds.

## References

- Rappaport, S., et al. 2018, MNRAS, 474, 1453, "Likely transiting exocomets detected
  by Kepler" (arXiv:1708.06069)
- Kennedy, G., et al. 2019, MNRAS, 482, 5587, "An automated search for transiting exocomets"
- Kass, R. E. & Raftery, A. E. 1995, JASA, 90, 773, "Bayes Factors" — the ΔBIC > 10
  "strong evidence" convention used for the threshold default
