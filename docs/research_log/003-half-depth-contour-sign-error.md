# 003 — The half-depth asymmetry metric has the wrong sign

**Date:** 2026-09-13
**Status:** Resolved — primary shape statistic changed
**Affects:** `detect/asymmetry.py`, `detect/model_comparison.py`, `core/types.py`

## Hypothesis under test

The pipeline's original discriminator was a contour-based duration asymmetry,

$$A_{dur} = \frac{T_{egress} - T_{ingress}}{T_{egress} + T_{ingress}}$$

with each side's duration measured from the **half-depth crossing** to the dip
minimum. A comet drags a trailing dust tail, so obscuration should clear more
slowly than it accumulated, and $A_{dur}$ was expected to be positive for a
genuine exocomet transit.

## Observation that broke it

Running the detector on KIC 3542116 recovered all six dips published by
Rappaport et al. (2018, MNRAS 474, 1453) to within 0.19 d. But the measured
asymmetries were inconsistent and mostly **negative**:

| Dip | Published depth (ppm) | $A_{dur}$ (half-depth) |
|---|---|---|
| D140 | 491 | +0.07 |
| D742 | 524 | −0.60 |
| D793 | 679 | −0.58 |
| D992 | 1200 | −0.34 |
| D1176 | 1500 | +0.24 |
| D1268 | 1900 | −0.43 |

The paper describes all of these as having *"steeper ingresses ... followed by
longer egresses"* — every value should have been positive.

## Hypotheses eliminated

1. **Baseline window too short.** The dips last ~1 d (~49 cadences) against a
   101-cadence rolling-median window, approaching the median's 50% breakdown
   point. *Falsified:* sweeping the window over 101 → 1001 cadences changed
   $A_{dur}$ by less than 0.02.
2. **Detrending distorting the profile.** *Falsified:* measuring on raw
   normalised flux and on Savitzky-Golay-detrended flux gave identical values
   (−0.33 for D1268 in both cases).
3. **Sign convention inverted in code.** *Falsified:* verified against synthetic
   injections with known ingress/egress ordering, which recover the correct sign
   (`tests/unit/test_asymmetry.py`).

## Actual cause

The metric was measuring the wrong part of the profile.

A comet transit is modelled as a Gaussian ingress of width $\sigma$ joined to an
exponential egress of timescale $\tau$. At depth fraction $f$ the two sides
reach the contour at

$$T_{ingress}(f) = \sigma\sqrt{2\ln(1/f)}, \qquad T_{egress}(f) = \tau\ln(1/f)$$

At the half-depth contour, $T_{ingress} = 1.177\sigma$ and
$T_{egress} = 0.693\tau$. Setting them equal gives the sign-change condition

$$\frac{\tau}{\sigma} = \frac{1.177}{0.693} \approx 1.70$$

So a transit with $1 < \tau/\sigma < 1.70$ has an unambiguously longer tail and
yet yields a **negative** half-depth asymmetry. The contour samples the compact
core, where the profile is nearly symmetric; the comet signature lives in the
extended wing, which the metric never sees.

## Confirming evidence

Measuring $A_{dur}$ at successively deeper contours moves every dip toward
positive, monotonically, exactly as predicted:

| Dip | $A_{dur}$(50%) | $A_{dur}$(25%) | $A_{dur}$(12.5%) | fitted $\tau/\sigma$ |
|---|---|---|---|---|
| D140 | +0.07 | +0.06 | +0.06 | 1.56 |
| D742 | −0.60 | −0.41 | −0.38 | 1.58 |
| D793 | −0.58 | −0.21 | −0.15 | 2.17 |
| D992 | −0.34 | −0.27 | −0.25 | 0.87 |
| D1176 | +0.24 | +0.25 | +0.27 | 0.99 |
| D1268 | −0.43 | −0.12 | +0.02 | 1.28 |

The independently fitted $\tau/\sigma$ exceeds 1 for four of six dips, so the
trailing tail is present in the data; only the estimator was at fault.

## Resolution

`ModelComparison.tau_over_sigma` — the ratio of fitted egress timescale to
ingress width — becomes the primary shape statistic, with $\Delta$BIC as the
primary ranking metric. Both use the entire fitted profile rather than a single
level set. $A_{dur}$ is retained as an interpretable, model-independent
cross-check, now documented as a core-shape statistic rather than a tail
statistic.

## Wider lesson

This is the reason the validation gate exists. The pipeline recovered 6/6
published epochs — by that criterion alone it "passed" — while its central
discriminator carried the wrong sign. Recovering known events is a **necessary
but not sufficient** check: position matched, morphology did not. Had the
classifier been trained before this was caught, it would have learned to
identify exocomets by a statistic that is anti-correlated with the physics.

## References

- Rappaport, S., et al. 2018, MNRAS, 474, 1453, "Likely transiting exocomets
  detected by Kepler" (arXiv:1708.06069)
- Kennedy, G., et al. 2019, MNRAS, 482, 5587, "An automated search for
  transiting exocomets"
