# Findings

Teaching-grade project with a hypothetical sponsor; not an offering. Every number below is in a
file under `results/` (named after each statement) and in `results/tables.md`. The analysis
choices were fixed in `docs/pricing_protocol.md` before the first fit.

## 1. Spread regression on 56 public tranche prices (real data)

Data: 56 distinct layers (58 rows, two duplicate pairs merged) of cat bonds with US named-storm
exposure issued 2017-2025 (15 layers also cover other perils), terms read from Artemis deal pages; 37 note series, 14 sponsors. Outcome
log(spread), regressor log(expected loss), both in percent.

**Slope b (main sample, 95% intervals; `results/pricing/slope_intervals.csv`).**

| Model | b | wild cluster bootstrap, by series | wild cluster bootstrap, by sponsor | HC3 |
|---|---|---|---|---|
| M1: log EL only | 0.553 | [0.094, 0.966] | [0.167, 1.155] | [0.179, 0.926] |
| M2: + issue-year effects | 0.466 | [0.115, 0.830] | [0.146, 0.871] | [0.128, 0.805] |
| M3: + trigger, log size | 0.492 | not computed (protocol) | not computed | [0.141, 0.844] |

Bootstrap: restricted wild cluster bootstrap-t, Rademacher weights, 9,999 draws, interval by
test inversion. M3 by series (CR1, t with 36 degrees of freedom): [0.174, 0.810].

**Pre-specified rule.** Proportionality (b = 1) is rejected only if the series-clustered wild
bootstrap interval excludes 1 in both M1 and M2. In the main sample it does (upper ends 0.966
and 0.830), so by the rule **proportionality is rejected in the main sample**. The M1 rejection
is marginal and not stable (`results/pricing/proportionality.csv`):

| Sample | M1 interval | M2 interval | Rejected |
|---|---|---|---|
| main (56) | [0.094, 0.966] | [0.115, 0.830] | yes |
| without 3 guidance prices (53) | [0.169, 1.016] | [0.112, 0.896] | no |
| without 4 discount notes (52) | [0.174, 1.033] | [0.163, 0.877] | no |
| without 2 launch-plan EL rows (54) | [0.094, 0.972] | [0.171, 0.881] | yes |
| without all 8 flagged rows (48) | [0.262, 1.084] | [0.160, 0.944] | no |

With the sponsor-clustered bootstrap the M1 interval contains 1 even in the main sample
(p = 0.1573), so the rule would not reject if sponsors were the clusters. The rule was fixed with
series as the clusters before the first fit and is not changed after the fact; the two results
are reported side by side. The M2 interval excludes 1 in every sample. Read together: once issue years are
controlled for, spreads rise less than proportionally with EL in this table (at b = 0.466 a
doubling of EL goes with a spread 1.38 times higher), but the interval is wide (0.115 to 0.830
by series in the main sample) and the M1 result depends on which eight rows are kept.

**Year effects dominate the fit.** R-squared on the log scale is 0.208 for M1 and 0.691 for M2
(`results/pricing/fit_stats.csv`). Post hoc, issue-year effects alone give 0.562 (adjusted 0.487)
and log EL alone 0.208 (adjusted 0.193): the issue year explains more of the spread than EL does
(`results/pricing_posthoc/r2_partition.csv`). The median spread/EL multiple by issue year runs from 2.36
(2018) to 6.67 (2024) and is 4.85 in 2025 (`results/pricing/year_multiple.csv`). The year
coefficients of M2 are relative to 2017 (from -0.101 for 2018 to 0.843 for 2023 on the log
scale; `results/pricing/coefficients.csv`). Trigger type and size add little (residual SD
0.253 for M3 against 0.258 for M2).

**Out-of-sample error** (`results/pricing/cv.csv`, main sample). Holding out one series at a
time, the M2 prediction error has an RMSE of 0.322 on the log scale and 3.02 percentage points
of spread (mean absolute error 2.14 points, median 1.45). Holding out a whole sponsor: 0.367
(log) and 3.20 points. M1: 0.398 and 3.33 points by series. The median spread in the table is
7.6%, so a regression-based spread for a new deal is uncertain by 2 to 3 points either way
before any doubt about its EL.

## 2. What these data cannot support

* **Selection of programs.** The table holds the programs that were searched for by name
  (Florida, Texas, North Carolina and Louisiana residual markets and state pools, several Florida
  carriers, a few national sponsors) and only deals Artemis lists. Two sponsors (Florida
  Citizens and TWIA) give 26 of the 56 layers. The slope describes this table, not the cat bond
  market.
* **Region and peril (post hoc).** The table is called US named-storm cat bonds, but of the 56
  layers 21 list only Florida, 15 name Florida among other states, 19 list no Florida (Texas 9,
  North Carolina 5, Louisiana 1, Hawaii 1, US Northeast 1, one list of seven states without
  Florida and one US-excluding-Florida deal) and 1 covers the whole US without naming states; 15
  layers also cover perils other than named storm, 13 of them among the 19 that list no Florida
  (`results/pricing_posthoc/composition.csv`). The protocol has no region or peril term and the
  spread range of the hypothetical Florida layers applies one regression to all of them. Post hoc
  checks (`results/pricing_posthoc/specifications.csv`; not in the protocol, reported whatever
  they show): adding region-group indicators leaves b at 0.50 (CR1 series interval [0.251,
  0.751]) and puts the layers that list no Florida 0.42 log points (about 35%) below the layers
  that name Florida, at the same EL and issue year (CR1 se 0.065); a multi-peril indicator alone
  gives b = 0.51, and its coefficient (named-storm-only layers 0.36 above) is -0.02 once the
  region is controlled, because 13 of the 15 multi-peril layers list no Florida; on the 21
  Florida-only layers alone M1 gives b = 0.33 [-0.16, 0.83]. With region-group indicators the
  median spread of the hypothetical Florida layers is 13.27, 9.44 and 7.02 against 12.07, 8.79 and
  6.67 for the protocol's M2 (`results/pricing_posthoc/florida_layer_spreads.csv`), that is 5% to
  10% higher. The ranges in the README keep the pre-specified models.
* **Related rows.** "56 distinct layers" means not identical in EL, attachment probability and
  spread (protocol). Gateway Re 2025-1 Classes C1 and C2 have the same attachment, exhaustion,
  deductible, attachment probability (2.43%) and EL (1.31%) and differ in note format: C1 is a
  one-season zero-coupon note (USD 50 million, priced at 91.75, spread equivalent 8.25), C2 a
  two-season coupon note (USD 150 million, 9.50); the Artemis page reads as one layer in two
  formats. Both rows are kept, as the protocol says, and C1 is also in the discount-note sample.
  Everglades Re II 2021-1 and 2021-2, and 2023-1 and 2023-2, were placed together (one Artemis
  page each year) but are four series clusters; sponsor clustering merges them.
* **Year effects.** 23 of 56 layers are from 2025 and only 2 each from 2020 and 2022. Issue-year
  dummies absorb the market cycle (reinsurance pricing, loss experience, capital flows) without
  explaining it, and with 2 to 23 layers per year they are imprecise; M2's EL slope is
  identified from differences within years.
* **Offering-stage EL.** The EL is each deal's initial figure from its own risk modeller, on
  the sponsor's exposure at launch; tranche sizes changed before pricing on 16 of 20 re-checked
  pages, and for Alamo Re 2021-1 and 2023-1 the EL was stated for a launch plan that was then
  restructured. What an error in the stated EL does to b depends on its kind, and none of it was
  measured. If the stated EL is the true EL plus independent noise (classical error), b is biased
  toward zero: with the variance of the stated log EL in this table, an error SD of 0.23 in log EL
  (about 23% of the EL) would turn a true slope of 1 into the estimated 0.55 (M1) and 0.47 (M2)
  (post hoc, `results/pricing_posthoc/implied_error_sd.csv`; a large-sample formula). If instead
  the true EL scatters around the stated one (Berkson error), because investors price the stated
  figure, the slope on the stated EL is not attenuated (simulation in
  `results/pricing_posthoc/attenuation.csv`: mean slope 1.00 at every error SD, against 0.68 for
  classical error with SD 0.2). Differences between vendor models are systematic by sponsor and
  modeller, so the direction of the bias is not known. The data cannot tell "spreads rise less
  than proportionally with EL" from "the stated EL is a noisy measure of the EL that was priced".
* **Models differ by deal.** ELs from different vendors and model versions are not on a common
  scale, so part of the residual variation is model difference, not price.
* **Clustering.** Errors are correlated within series and within sponsor; with 14 sponsors the
  sponsor-clustered inference is coarse, and the two clusterings disagree for M1.
* **Prices.** Three prices are guidance and four are 100 minus a discount-note issue price for
  one season; removing them moves b in M1 from 0.553 to 0.619 and 0.635.

## 3. Hypothetical transaction (hypothetical sponsor and portfolio)

Layer rule (protocol): three adjacent layers of the base-case grid with EL closest to 4%, 2% and
1% of limit. Chosen: USD 500 million each, attaching at 0.50%, 1.00% and 1.50% of the USD 100
billion TIV (`results/transaction/layers.csv`; 100,000 simulated years, seed 1).

| Layer | AP % | EP % | EL % | hazard-parameter band (5-95%) |
|---|---|---|---|---|
| bottom, USD 500-1,000 million | 5.67 | 2.70 | 3.90 | 2.19 - 5.34 |
| middle, USD 1,000-1,500 million | 2.69 | 1.45 | 1.98 | 0.97 - 2.85 |
| top, USD 1,500-2,000 million | 1.44 | 0.82 | 1.09 | 0.44 - 1.59 |

The band comes from 200 refits of the hazard model on bootstrap samples of the 1900-2019 years
(10,000 simulated years each); it covers frequency, intensity and landfall-pool uncertainty and
nothing else. Monte Carlo standard errors of the EL are 0.03 to 0.06 points.

**Regression spread range** (`results/transaction/spreads.csv`; median spread with 90%
prediction interval, issue year 2025 in M2; a regression output, not a quote):

| Layer | M1 at EL point | M2 at EL point | M2 median at band ends |
|---|---|---|---|
| bottom | 11.47 [5.84, 22.54] | 12.07 [7.61, 19.16] | 9.22 - 13.97 |
| middle | 7.88 [4.14, 15.00] | 8.79 [5.65, 13.68] | 6.32 - 10.42 |
| top | 5.68 [2.96, 10.89] | 6.67 [4.25, 10.47] | 4.37 - 7.93 |

The prediction interval (residual scatter of public deals) is wider than the range produced by
the hazard-parameter band, and both are narrower than the model-form sensitivities below. The
five nearest public layers by EL and year (`results/transaction/comparables.csv`) priced at 9.75
to 25.50 (bottom), 6.25 to 9.75 (middle) and 4.25 to 8.25 (top).

**Sensitivities** (`results/transaction/sensitivity.csv`, same seed, 100,000 years each; EL of
bottom / middle / top layer in percent):

| Variant | EL bottom | EL middle | EL top | M2 spread change, middle (points) |
|---|---|---|---|---|
| base | 3.90 | 1.98 | 1.09 | 0.00 |
| frequency x 0.8 | 3.15 | 1.64 | 0.91 | -0.75 |
| frequency x 1.2 | 4.57 | 2.29 | 1.25 | 0.63 |
| calibration window 1950-2019 | 2.33 | 1.04 | 0.43 | -2.28 |
| V_half 145.2 kt | 4.67 | 2.51 | 1.44 | 1.03 |
| V_thresh 40 kt | 6.11 | 3.24 | 1.83 | 2.27 |
| V_half 110 kt | 13.98 | 10.32 | 7.95 | 10.21 |

Frequency moves the EL roughly in proportion (x 0.8 gives 0.81 to 0.83 of base, x 1.2 gives 1.14
to 1.17), the calibration window by a factor 0.39 to 0.60, and the damage-function half-damage
wind by up to 7.3. The frequency and window variants are independent Monte Carlo runs: a
different event count or landfall pool changes every simulated event, so only the variants that
change the loss settings reuse the base case's events (common random numbers; see
`results/eventset/MANIFEST.txt`). The Monte Carlo standard error of the base EL is 1.5% to 2.9% of
the EL, of the same order as the 2.5% to 5% by which the x 1.2 ratios fall short of 1.2 (the
annual cap on principal loss can also make EL rise less than in proportion to frequency; the two
effects are not separated). The V_half 110 kt ELs lie far outside the 0.83% to 4.52% EL range of
the public deals, so the spread mapped from them is an extrapolation of the regression.

**What this does not show.** The model's EL is not comparable to a vendor model's EL (no
calibration to insured losses, centroid exposure, no surge, a hypothetical portfolio), so
mapping it through a regression fitted on vendor ELs mixes two scales. The layer choice
depends on the base-case grid; any variant above would select different attachments for the
same EL targets.

## 4. Questions this analysis does not answer

What an investor or reviewer would still ask, and what is missing (none of it was done):

* **Is the modelled loss plausible?** No benchmark of the modelled AAL (0.101% of TIV) or 1-in-100
  occurrence loss against a published Florida industry figure or a sponsor's disclosed PML, and
  no calibration of the damage function to insured losses (2004-05, 2017-18, 2022). V_half 110 kt
  gives a middle-layer EL of 10.32% against 1.98%; the choice between them is not data-driven.
* **Why is b about 0.5?** Risk pricing, differences between vendor models, sponsor effects and
  selection of programs cannot be told apart here. There are no sponsor fixed effects (14
  sponsors) and no within-sponsor slope.
* **Hurricanes that do not make landfall in Florida, or are not hurricanes there.** Ivan 2004,
  Matthew 2016 and five other hurricanes within 50 km of the coastline are not in the event set,
  and tropical storms are excluded (`results/hazard/near_miss_hurricanes.csv`).
* **Clustered years and the term.** Frequency is Poisson with independent years in the primary
  window (dispersion test p = 0.102) although the 1950-2019 window shows overdispersion
  (p = 0.010) and selects a negative binomial (`results/hazard/frequency.csv`); only P(any loss
  in 3 years) is computed, not the term EL with erosion across years or the term exhaustion
  probability. The 1900-1979 fit predicts 25.0 hurricanes for 1980-2019 against 17 observed
  (p = 0.21), and no view is taken on which window an investor should use.
* **Intensity by location.** Intensity is taken as independent of landfall position; whether it
  differs by coast segment was not tested (only heading and speed are reported by segment,
  `results/hazard/motion_by_coast.csv`).
* **An honest prediction interval.** The interval is the classical one on the log scale; it
  ignores clustering and the uncertainty of the model's own EL. Leave-one-sponsor-out errors are
  reported (3.20 points) but no interval is built from them.
