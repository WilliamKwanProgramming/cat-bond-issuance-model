# Pricing protocol (pre-specified)

This document fixes every analysis choice of the spread regression on the public deal table and
of the hypothetical transaction **before** any model was fitted on the real table. It was
committed before `scripts/run_pricing.py` was first run on `data/deals/cat_bond_deals.csv`;
`git log` shows the order. Every model and sample listed here is reported in `results/pricing/`
whatever it shows. No other specification is fitted.

What had been seen of the real table before this protocol: the table itself (identifiers, flags,
EL and spread values) and the numbers in the re-check of the deal data (by-year medians of the
spread/EL multiple, and an OLS slope of spread on EL in levels, 3.18 on 58 rows and 3.75 without
the guidance and discount rows; see `docs/deal_sources.md`). No log-log fit, no clustered or
bootstrap interval and no cross-validation had been computed.

## Sample

* Input: `data/deals/cat_bond_deals.csv`, 58 tranches (55 final prices, 3 guidance prices).
* Unit of analysis: the 56 distinct layers. Integrity Re III 2025-1 Classes A1/A2 and B1/B2 have
  identical expected loss, attachment probability and spread; **A1 and B1 (the first-listed
  class of each pair) are kept, A2 and B2 dropped**. The kept rows' sizes (USD 50 million each,
  against 100 for A2 and B2) enter `log(size)` in M3. The rule is code
  (`catbond.pricing.sample`) that raises an error if the pair stops being identical.
* Clusters: note series = `deal_name` (37 series in the main sample); sponsor = `sponsor` (14
  sponsors; "Citizens Property Insurance Corporation" and "Louisiana Citizens Property
  Insurance Corporation" are different sponsors).

## Variables

* `y = log(spread_pct)`, `x = log(expected_loss_pct)` (both in percent of principal per year).
* Issue-year fixed effects: one dummy per issue year 2018-2025, reference year 2017.
* Trigger: dummy for `industry loss index` (3 rows), reference `indemnity`.
* `log(size_usd_m)`.

## Models (the only ones fitted)

| Model | Right-hand side |
|---|---|
| M1 | a + b log(EL) |
| M2 | M1 + issue-year fixed effects |
| M3 | M2 + trigger type + log(size) |

Ordinary least squares on the log scale. The point prediction exp(x'b) is the conditional
median spread, not the mean.

## Standard errors and intervals

* Classical, HC3 (MacKinnon and White 1985), and CR1 cluster-robust clustered by series
  (primary) and, shown next to it, by sponsor (Cameron and Miller 2015). CR1 small-sample
  factor G/(G-1) (n-1)/(n-k). Analytic intervals for b use t(n-k) for HC3 and t(G-1) for CR1.
* **Wild cluster bootstrap for b in M1 and M2**: restricted wild cluster bootstrap-t (WCR;
  Cameron, Gelbach and Miller 2008) with Rademacher weights, B = 9999 draws, CR1 standard
  errors in the t statistic, symmetric two-sided p-value; the 95% interval is the set of null
  values not rejected at 5% (test inversion, as in Roodman, MacKinnon, Nielsen and Webb 2019),
  end points located to 1e-4 cluster standard errors by bisection with the same weight draws for
  every null value. Computed by series (primary, 37 clusters) and by sponsor (14 clusters; with
  14 clusters there are 2^14 = 16,384 distinct Rademacher weight vectors, so the p-value is
  coarse). Seeds: `numpy.random.default_rng([6001, i])` with i counting (sample, model,
  cluster) in the order of `scripts/run_pricing.py`. Implementation `catbond.pricing.wild`;
  tests in `tests/test_pricing_wave2.py` on synthetic data: p = 1 at the estimate, exact shift of
  the interval when y is shifted by c x, and coverage over 300 synthetic data sets with 12
  clusters and within-cluster correlated regressor and errors (bootstrap interval covered in
  0.94 of them, the CR1 interval with a normal quantile in 0.80).

## Validation

Leave-one-series-out and leave-one-sponsor-out prediction errors for every model and sample:
fit without the held-out group, predict the median spread of each held-out row. Reported: RMSE
and mean absolute error on the log scale and in percentage points of spread (spread minus
predicted median), median absolute error in points, and the number of rows that cannot be
predicted because their issue year or trigger level does not occur in the training rows (those
rows are counted, not dropped silently).

## Sensitivity samples (all reported)

| Sample | Rows removed from the 56 layers |
|---|---|
| main | none |
| without guidance prices | Citrus Re 2017-1, Alamo Re 2019-1, Alamo Re 2021-1 (3) |
| without discount notes | Everglades Re II 2023-1 A, 2023-2 A; Gateway Re 2025-1 AA, C1 (4) |
| without launch-plan EL | Alamo Re 2021-1, Alamo Re 2023-1 (2) |
| without all flagged rows | the union of the three (8; Alamo Re 2021-1 is in two groups) |

Each sample is fitted with M1, M2 and M3; the wild bootstrap is computed for M1 and M2.

## Descriptive table

Spread/EL multiple by issue year on the 56 layers: count, median EL, median spread, median,
minimum and maximum multiple.

## What counts as a finding

* The slope b is reported with its interval (wild bootstrap by series as primary; HC3, CR1 and
  the sponsor bootstrap next to it).
* "Spread is proportional to EL" (b = 1) is **rejected only if the 95% wild cluster bootstrap
  interval by series excludes 1 in both M1 and M2** (main sample). The same rule is evaluated in
  each sensitivity sample and reported, not used to choose a sample.
* No other hypothesis test is a finding. Year, trigger and size coefficients are reported as
  estimates with standard errors, without a significance claim.

## Hypothetical transaction (fixed before any spread fit)

**Layer choice.** From the base-case layer grid (`results/layers/layer_grid.csv`: attachments
0.25% to 4.00% of TIV in steps of 0.25%, limits 0.5%, 1%, 2% and 3% of TIV), choose a stack of
three adjacent layers, bottom to top with EL targets 4%, 2% and 1% of limit: the bottom layer
[A1, A1 + L1], the middle layer attaching at A1 + L1, the top layer attaching where the middle
layer exhausts, every attachment and limit a grid value. Choose the stack that minimizes
sum over the three layers of (log(EL / target))^2; ties go to the lower attachment. This rule
was written after the layer grid (a wave-1 result that does not involve spreads) had been seen
and before any spread model was fitted.

**Reported per layer:** attachment and exhaustion (% of TIV and USD), attachment probability,
exhaustion probability, EL point estimate (100,000 years, seed 1), its Monte Carlo standard
error, and the hazard-parameter band: 5% and 95% quantiles of the layer EL over the 200
parameter-bootstrap replicates (`results/param_uncertainty/grid_summary.csv`). The band covers
frequency, intensity and landfall-pool uncertainty only, not model form.

**Spread mapping.** M1 and M2 fitted on the main sample (56 layers); M2 evaluated at issue year
2025 (the latest year in the table). For each layer: median spread and classical 90% prediction
interval at the EL point estimate; median spread at the 5% and 95% EL band quantiles. The
spread range is a regression output, not a quote. The model's EL comes from a teaching-grade
model and the deal ELs from vendor models; putting one through a regression fitted on the other
assumes they are on the same scale, which is not shown.

**Comparable deals.** The 5 nearest of the 56 layers to each layer by distance
sqrt(((log EL_i - log EL_0)/0.5)^2 + ((year_i - 2025)/2)^2), with their spread and multiple.

**Sensitivities.** From `results/eventset/sensitivity_layers.csv` (same seed and 100,000 years
per variant): frequency x 0.8 and x 1.2; vulnerability V_half 110 kt, V_half 145.2 kt and
V_thresh 40 kt; calibration window 1950-2019 against 1900-2019. For each: the EL of each layer and
the M1 and M2 median spreads at that EL. Every other variant in the file is reported in the same
table.

## References

* Cameron, A. C., J. B. Gelbach and D. L. Miller, 2008: Bootstrap-based improvements for
  inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414-427.
* Cameron, A. C., and D. L. Miller, 2015: A practitioner's guide to cluster-robust inference.
  *Journal of Human Resources*, 50(2), 317-372.
* MacKinnon, J. G., and H. White, 1985: Some heteroskedasticity-consistent covariance matrix
  estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325.
* Roodman, D., J. G. MacKinnon, M. Ø. Nielsen and M. D. Webb, 2019: Fast and wild: Bootstrap
  inference in Stata using boottest. *Stata Journal*, 19(1), 4-60.
