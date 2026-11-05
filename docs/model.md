# Model description

A teaching-grade Florida hurricane loss model for a hypothetical homeowners portfolio, with
per-occurrence cat bond layer analytics and a spread regression. **It is not a vendor
catastrophe model (RMS, Verisk/AIR, CoreLogic).** No loss, attachment probability or expected
loss from it is comparable to the modelled figures in a cat bond offering circular. Results are
in `results/` and are tabulated in `results/tables.md`; this document states the method and the
choices, not the outcomes.

CLIMADA is not used: its LitPop exposure needs nightlight and population rasters that cannot be
downloaded in the environment where this code was built. The damage function is the one CLIMADA
uses for US hurricanes (Emanuel 2011), implemented here directly.

## Pre-specified choices

Fixed before any frequency test, loss, layer or out-of-sample result was looked at.

| Choice | Value | Reason |
|---|---|---|
| Region | Florida, the 67 counties of a US county boundary file | sponsor is a Florida homeowners insurer |
| Calibration window | 1900-2019 | earlier landfall records are less complete; `AL.rda` ends in 2019 |
| Alternative windows | 1950-2019; 1900-2024 with 2020-2024 from `storms.csv` | sensitivity to the window; most recent data |
| Landfall | hybrid definition (below), maximum wind >= 64 kt at landfall | hurricane strength at landfall |
| Event | one per storm per year: its most intense Florida landfall | the simulator draws one landfall per event |
| Frequency family | negative binomial only if the boundary likelihood-ratio test rejects Poisson at 5% | parsimony |
| Trend / low-frequency signal | reported, not modelled; frequency x 0.8 and x 1.2 runs instead | brief |
| Intensity | GPD on the excess over 64 kt (primary), reflected Gaussian KDE (comparison) | smooth upper tail vs data-driven |
| Out-of-sample | fit 1900-1979, test 1980-2019, Saffir-Simpson categories, 90% predictive intervals | brief |
| Coast segments for motion | South lat < 25.9; Northwest lat >= 29, lon < -82.5; East lon > -81.5; West the rest | describe direction and speed by coast |
| Reference layer | attachment = base-case 1-in-50 OEP loss, exhaustion = 1-in-100 OEP loss, each rounded to 0.05% of TIV, then held fixed in USD | typical first-event layer |
| Seeds | event set 1; out-of-sample 3001; convergence 1001-1020 and 2001-2020; parameter bootstrap 5001; synthetic pricing 4001 | |

The 1966-1984 gap in landfall flags (next section) was found while loading the data, before any
of the analyses above were run; the hybrid definition was fixed at that point.

## Data

* **HURDAT2** (NOAA National Hurricane Center best tracks; format and content: Landsea and
  Franklin 2013): 6-hourly positions, maximum 1-minute sustained 10-m wind (kt), pressure and a
  record identifier ("L" = "center of system crossing a coastline"). Read from the copies in the
  timtrice/HURDAT (`AL.rda`, 1851-2019) and tidyverse/dplyr (`storms.csv`, 1975-2024, no record
  identifier, times truncated to the hour) GitHub repositories, pinned by SHA-256 in
  `scripts/fetch_data.py`. The NHC download route in the same script was not exercised here.
* **County boundaries** (`us_counties.geojson`, a GeoJSON copy of Census county boundaries in the
  plotly/datasets repository): Florida land rings, coastline and county centroids.
* **Households** by county, 2020 Census, from a University of Florida BEBR table (see
  `docs/assumptions.md`).

Data quirks found and handled: the first storm in `AL.rda` (14 rows, 1851) has no key and is
dropped; one longitude of -359 is set to missing; in `storms.csv`, an off-synoptic fix that shares
the (truncated) hour of a synoptic fix is moved 30 minutes later (this affected Milton 2024).

## Hazard

**Landfall definitions.** Record definition: an "L" fix within 25 km of the Florida coastline and
not nearer to another state's coast. Geometric definition: a straight segment between consecutive
fixes crosses the Florida coastline from water to land, with the wind linearly interpolated to
the crossing. In `AL.rda` the "L" identifier is nearly absent for 1966-1984 (six "L" records in
19 years, none in Florida, against 36 to 85 per five-year period in 1945-1964 and 1985-2019;
see `results/hazard/l_flags_by_period.csv`), so the
**hybrid definition** uses the record definition outside 1966-1984 and the geometric definition
inside. Its credibility is measured: in the years with flags, the geometric definition finds the
same storms as the record definition (`results/hazard/definition_agreement.csv`).

The coastline is the set of Florida county edges that no other county in the file shares (edges
shared with Georgia and Alabama counties cancel, so the land border is not coast). Distances use
a local equirectangular projection (error under 1% at a few hundred km; tested).

**Frequency.** Annual counts N_t, t = 1..n, zeros included. Tests: index of dispersion
D = sum (N_t - Nbar)^2 / Nbar against chi-square(n - 1); negative binomial (NB2, variance
mu + mu^2/k) fitted by maximum likelihood and compared by a likelihood-ratio test with the
boundary-corrected p-value 0.5 P(chi2_1 > LR); log-linear Poisson trend
log lambda_t = a + b (t - tbar), LR test of b = 0; for a low-frequency (AMO-like) signal, the
dispersion test applied to non-overlapping decade sums and the Ljung-Box test on lags 1-10.
Exact (Garwood) 95% intervals for the mean count.

**Intensity.** Generalized Pareto distribution for y = V - 64 kt,
F(y) = 1 - (1 + xi y / sigma)^(-1/xi), fitted by maximizing the profile likelihood in
theta = xi / sigma (for fixed theta, xi = mean log(1 + theta y), sigma = xi / theta), restricted
to xi >= -1; the result is checked against `scipy.stats.genpareto.fit` in the tests. The
comparison model is a Gaussian KDE with Silverman's bandwidth, reflected at 64 kt. Both are
truncated at 180 kt. HURDAT2 winds come in 5-kt steps and are treated as continuous.

**Landfall position, heading and translation speed.** A historical landfall is drawn at random,
so position, heading and speed keep their joint, coast-dependent distribution; the position is
moved by a 2-D Gaussian (25 km per axis) and snapped to the nearest coastline point, the heading
by N(0, 10 deg), the speed by a log-normal factor exp(N(0, 0.2^2)). Heading and speed come from
the fixes before and after landfall. Intensity is drawn independently of position.

**Wind field.** Holland (1980) profile in its cyclostrophic, peak-wind form,
V(r) = V_m [ (R/r)^B exp(1 - (R/r)^B) ]^(1/2), used directly as a surface wind profile because
the HURDAT2 peak wind is already a 1-minute 10-m wind. Radius of maximum wind and B from
Willoughby and Rahn (2004), equations 12.1-12.2 (V in m/s, phi latitude in degrees):
R = 46.29 exp(-0.0153 V + 0.0166 phi) km, B = 0.886 + 0.0177 V - 0.0094 phi (B clipped to
[1, 2.5]); those fits are to flight-level winds and are applied here to surface winds (a stated
approximation; RMW x 0.8 and x 1.2 sensitivity runs bracket it). Translation asymmetry (own
assumption): V = h(r) max(V_m - V_t (1 - cos theta), 0), peak V_m on the right of the track.
Inland decay (Kaplan and DeMaria 1995): V_m(t) = V_b + (R_f V_0 - V_b) exp(-alpha t) for t > 0
hours after landfall, with V_b = 26.7 kt, R_f = 0.9, alpha = 0.095 per hour (the US East/Gulf
coast constants of the NCEP/NHC implementation `decayo.f90`). Tracks are straight lines at
constant heading and speed; decay continues if a track re-emerges over water. The peak wind at a
site is the maximum over candidate track positions (landfall point, closest approach +/- 2 R in
steps of R/8, eyewall passages); against a dense search it is within 1% where winds exceed
50 kt (tested).

## Exposure and vulnerability

Hypothetical portfolio: TIV USD 100 billion over the 67 county centroids in proportion to 2020
Census households. Damage: Emanuel (2011), f(V) = v_n^3 / (1 + v_n^3),
v_n = max(V - V_thresh, 0) / (V_half - V_thresh), V_thresh = 50 kt, V_half = 150 kt (the paper's
main values; its sensitivity case is V_half = 110 kt; CLIMADA's default corresponds to
145.2 kt). Applied to the peak 1-minute wind at the county centroid. Policy terms per county:
loss ratio = min(max(f - d, 0), l), d = 2% of TIV, l = 100% (simplification: the deductible is
applied to the county mean damage ratio, not policy by policy).

## Event set and loss statistics

N simulated years (default 100,000); per year a Poisson (or NB) count; per event the attributes
above and a portfolio loss L = sum over counties of loss ratio x county TIV. Two independent
random streams (numpy `SeedSequence.spawn`): counts, and event attributes; results are identical
for a given seed. Per year: occurrence loss M_y = max event loss, aggregate loss S_y = sum.
OEP(x) = P(M > x), AEP(x) = P(S > x); return-period loss for T years = empirical (1 - 1/T)
quantile. Standard errors: bootstrap over simulated years (200 resamples) for quantiles; the SE
of a mean of independent years for AAL and layer EL.

## Layers

Layer [A, E] per occurrence: event layer loss l = min(max(L - A, 0), E - A). **Aggregation
(stated convention):** every event in a year can erode principal, and the annual principal loss
is P_y = min(sum of l over the year's events, E - A). Metrics:

* attachment probability AP = P(at least one event in the year with L >= A);
* exhaustion probability EP = P(P_y >= E - A) (whole principal lost in the year, including by
  several partial events); the occurrence-only P(max L >= E) is reported as `ep_occurrence`;
* expected loss EL = E[P_y] / (E - A), annual, as a fraction of the limit;
* for a term of T = 3 years: P(any principal loss) = fraction of non-overlapping 3-year blocks
  with a loss (years are independent, so this matches 1 - (1 - AP)^3 up to Monte Carlo error).

Stacking: layer losses add up exactly per event; with the annual cap they add up exactly only in
years with one loss event, and otherwise the stacked total is at most the full-layer loss
(tested).

**Market definitions checked.** Artemis glossary: attachment probability is "the probability
that a catastrophe bond ... will attach and experience some losses during a given period,
typically one year"; probability of exhaustion is "the likelihood that a catastrophe bond, or
transaction will suffer a complete loss"; expected loss is "the average loss catastrophe bond
investors can expect to transpire over a certain period" divided by the capital. Verisk
Touchstone Re documentation: probability of attachment is "the probability (%) of a
program/layer experiencing some loss", probability of exhaustion "experiencing a complete loss",
expected loss "the percentage of loss each bond/layer is expected to incur in any given year".
Bodoff and Gan (2013) use expected loss (annual average loss) as a percentage of the bond amount.
The conventions above follow these (annual, fraction of the limit). Where they leave a choice
open: AP uses L >= A (an event exactly at the attachment counts, although its layer loss is
zero; immaterial with continuous losses), and EP counts aggregate exhaustion within the year.

## Parameter uncertainty and convergence

Parameter uncertainty: 200 bootstrap samples of the 120 calibration years (years, not events,
are resampled, so the number of landfalls varies), the hazard model refitted on each (same
frequency family and intensity model), 10,000 simulated years per replicate. The spread of a
replicate statistic combines parameter uncertainty and Monte Carlo noise at 10,000 years; the
latter is measured by the convergence study (20 independent seeds at N = 10,000 and 100,000),
which also compares the across-seed SD with the within-run standard errors.

## Out-of-sample check

Fit lambda and the intensity model on 1900-1979 only; predict counts by category for the 40
years 1980-2019 as Poisson(40 lambda p_k). Reported: plug-in 90% predictive interval, a
parameter-bootstrap 90% predictive interval (2,000 resamples of the fit years), the exact 95%
interval for the observed count's Poisson mean, and a two-sided p-value from the bootstrap
predictive distribution. A test shows that changing or deleting rows after 1979 does not change
the prediction. Fitting on part of the same record and predicting the rest checks stability; it
does not validate losses.

## Pricing

Spread multiple = spread / EL. Regression on a deals table (one row per tranche; columns of the
deals interface: deal_name, tranche, sponsor, issue_year, issue_month, size_usd_m, peril, region,
trigger, expected_loss_pct, attachment_prob_pct, exhaustion_prob_pct, spread_pct, spread_basis,
rating, source_url, access_date, notes):
log(spread) = b0 + b1 log(EL) + optional covariates (issue year, log size, trigger, peril,
region dummies), by OLS with classical, HC3 and sponsor-clustered (CR1) standard errors,
exact leave-one-out errors e_i / (1 - h_ii), and a normal-theory prediction interval on the log
scale. Rows with missing or non-positive EL or spread, a missing covariate, or (optionally) a
spread basis other than the one requested are dropped and counted by reason. Comparable-deal
screen: distance combining log EL and log size differences (scales 0.5 and 1.0) with penalties
for a different peril (2.0) or region (1.0). In wave 1 the regression is run only on synthetic
deal tables with a known generating relation (`results/pricing_synthetic/`).

## References

* Bodoff, N. M., and Y. Gan, 2013: An analysis of the market price of cat bonds. *Variance*,
  6(2), 161-177.
* Emanuel, K., 2011: Global warming effects on U.S. hurricane damage. *Weather, Climate, and
  Society*, 3, 261-268. doi:10.1175/WCAS-D-11-00007.1
* Holland, G. J., 1980: An analytic model of the wind and pressure profiles in hurricanes.
  *Monthly Weather Review*, 108(8), 1212-1218.
* Kaplan, J., and M. DeMaria, 1995: A simple empirical model for predicting the decay of tropical
  cyclone winds after landfall. *Journal of Applied Meteorology*, 34, 2499-2512. Constants as in
  the NCEP/NHC routine `decayo.f90` (nhcg v4.2.1).
* Landsea, C. W., and J. L. Franklin, 2013: Atlantic hurricane database uncertainty and
  presentation of a new database format. *Monthly Weather Review*, 141, 3576-3592.
* Willoughby, H. E., and M. E. Rahn, 2004: Parametric representation of the primary hurricane
  vortex. Part I: Observations and evaluation of the Holland (1980) model. *Monthly Weather
  Review*, 132, 3033-3048.
* Artemis.bm glossary entries "Attachment probability", "Probability of exhaustion" and
  "Expected loss" (accessed 2026-11-05).
* Verisk Touchstone Re 12.0 help, "Expected loss" (accessed 2026-11-05).
* CLIMADA documentation, `climada.entity.impact_funcs.trop_cyclone.ImpfTropCyclone.from_emanuel_usa`
  (accessed 2026-11-05).
* National Hurricane Center, Saffir-Simpson Hurricane Wind Scale (category bounds 64, 83, 96, 113
  and 137 kt), archived copy of nhc.noaa.gov/aboutsshws.php (2014).
