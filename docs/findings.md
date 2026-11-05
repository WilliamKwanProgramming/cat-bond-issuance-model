# Findings

Teaching-grade project with a hypothetical sponsor; not an offering. Every number below is in a
file under `results/` (named after each statement) and in `results/tables.md`. The analysis
choices were fixed in `docs/pricing_protocol.md` before the first fit.

## 1. Spread regression on 56 public tranche prices (real data)

Data: 56 distinct layers (58 rows, two duplicate pairs merged) of US named-storm cat bonds
issued 2017-2025, terms read from Artemis deal pages; 37 note series, 14 sponsors. Outcome
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
(p = 0.1573). The M2 interval excludes 1 in every sample. Read together: once issue years are
controlled for, spreads rise less than proportionally with EL in this table (at b = 0.466 a
doubling of EL goes with a spread 1.38 times higher), but the interval is wide (0.115 to 0.830
by series in the main sample) and the M1 result depends on which eight rows are kept.

**Year effects dominate the fit.** R-squared on the log scale is 0.208 for M1 and 0.691 for M2
(`results/pricing/fit_stats.csv`). The median spread/EL multiple by issue year runs from 2.36
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
* **Year effects.** 23 of 56 layers are from 2025 and only 2 each from 2020 and 2022. Issue-year
  dummies absorb the market cycle (reinsurance pricing, loss experience, capital flows) without
  explaining it, and with 2 to 23 layers per year they are imprecise; M2's EL slope is
  identified from differences within years.
* **Offering-stage EL.** The EL is each deal's initial figure from its own risk modeller, on
  the sponsor's exposure at launch; tranche sizes changed before pricing on 16 of 20 re-checked
  pages, and for Alamo Re 2021-1 and 2023-1 the EL was stated for a launch plan that was then
  restructured. Error in log EL biases b toward zero (attenuation), in the direction of the
  rejection above; its size is unknown.
* **Models differ by deal.** ELs from different vendors and model versions are not on a common
  scale, so part of the residual variation is model difference, not price.
* **Clustering.** Errors are correlated within series and within sponsor; with 14 sponsors the
  sponsor-clustered inference is coarse, and the two clusterings disagree for M1.
* **Prices.** Three prices are guidance and four are 100 minus a discount-note issue price for
  one season; removing them moves b in M1 from 0.553 to 0.619 and 0.635.
