# Assumptions and their sources

Every input that is not estimated from data, with its value, its source and where it is used.
"Own assumption" means a stated choice with no external source. The model is teaching-grade and
the portfolio is hypothetical; see `docs/model.md`.

## Portfolio (hypothetical)

**The portfolio is hypothetical.** No real insurer's exposure is used, and a real sponsor's book
is concentrated differently: by company footprint, distance to the coast, construction, coverage
and value. Here:

| Item | Value | Source / reason |
|---|---|---|
| Total insured value | USD 100 billion | own assumption (a round number; results are also shown as % of TIV) |
| Allocation to counties | proportional to 2020 Census households (occupied housing units) | public count; see below |
| Location within a county | area-weighted centroid of the county polygon | own assumption; county polygons from the boundary file in `scripts/fetch_data.py` |
| Line of business | homeowners (residential buildings) | brief |
| Deductible | 2% of TIV, applied to the county mean damage ratio | own assumption; 0% and 5% sensitivity runs |
| Policy limit | 100% of TIV | own assumption |

**Household counts.** University of Florida, Bureau of Economic and Business Research (BEBR),
Table 1 "Number of Households and Average Household Size in Florida and Its Counties 2025, 2020,
and 2010", column 2020 (Census, April 1, 2020),
<https://bebr.ufl.edu/wp-content/uploads/2025/12/households_2025.pdf>, accessed 2026-11-05. The
table defines households as occupied housing units (seasonal residents and tourists excluded).
The 67 county values were transcribed into `src/catbond/data/fl_county_households_2020.csv`;
they sum exactly to the state total printed in the same table (8,529,067), which the test
`tests/test_exposure.py` checks. Total housing units (which include vacant and seasonal homes)
were the preferred count, but no source for them could be read from the build environment (the
Census Bureau API was not reachable). Households understate the insured stock of counties with
many seasonal homes. The table actually used (counts, shares, TIV and centroids) is generated
into `results/exposure/county_exposure.csv` and reproduced in `results/tables.md`.

## Hazard

| Item | Value | Source |
|---|---|---|
| Track data | HURDAT2 Atlantic best tracks | NOAA NHC; Landsea and Franklin (2013); GitHub copies pinned by SHA-256 |
| Region boundary and coastline | Florida county polygons from a GeoJSON copy of Census county boundaries | plotly/datasets repository file `geojson-counties-fips.json` (SHA-256 pinned) |
| Hurricane threshold | 64 kt (1-min, 10 m) | Saffir-Simpson Hurricane Wind Scale, NHC |
| Category bounds | 64, 83, 96, 113, 137 kt | Saffir-Simpson Hurricane Wind Scale, NHC |
| Landfall distance tolerance | 25 km from the Florida coastline, nearer than any other state's coast | own assumption |
| "L" flag gap | 1966-1984 treated with the geometric definition | data (`results/hazard/l_flags_by_period.csv`) |
| Intensity cap | 180 kt | own assumption |
| Position jitter | 2-D Gaussian, 25 km per axis, then snapped to the coast | own assumption |
| Heading jitter | Gaussian, SD 10 degrees | own assumption |
| Speed jitter | log-normal factor, SD of log 0.2 | own assumption |
| Minimum translation speed | 1 km/h | own assumption (numerical floor) |
| Wind profile | Holland (1980), cyclostrophic peak-wind form, surface wind | Holland (1980) |
| RMW | 46.29 exp(-0.0153 V + 0.0166 phi) km, V in m/s | Willoughby and Rahn (2004) eq. 12.1 |
| Holland B | 0.886 + 0.0177 V - 0.0094 phi, clipped to [1, 2.5] | Willoughby and Rahn (2004) eq. 12.2; clipping own assumption |
| RMW scatter | none (deterministic); x 0.8 and x 1.2 sensitivity runs | own assumption |
| Translation asymmetry | h(r) max(V_m - V_t (1 - cos theta), 0) | own assumption |
| Inland decay | V_b = 26.7 kt, R_f = 0.9, alpha = 0.095 per hour | Kaplan and DeMaria (1995); constants from NCEP/NHC `decayo.f90` |
| Track shape | straight line through the landfall point, constant heading and speed | own assumption (allowed by the brief) |
| Terrain, gusts, surge, rain | not modelled | own assumption |

## Vulnerability

| Item | Value | Source |
|---|---|---|
| Damage function | f = v_n^3 / (1 + v_n^3) | Emanuel (2011) eq. 1 |
| V_thresh | 50 kt | Emanuel (2011); CLIMADA uses 25.7 m/s (50 kt) |
| V_half | 150 kt (base); 110 kt and 145.2 kt sensitivity runs | Emanuel (2011) main and sensitivity values; 74.7 m/s CLIMADA default |
| Wind used | peak 1-minute wind at the county centroid | own assumption |

## Simulation and statistics

| Item | Value | Source |
|---|---|---|
| Simulated years | 100,000 (base); 10,000 per parameter-bootstrap replicate | own assumption |
| Bootstrap for return-period SEs | 200 resamples of simulated years | own assumption |
| Parameter bootstrap | 200 resamples of calibration years | brief (at least 200) |
| Convergence | 20 seeds at N = 10,000 and 100,000 | brief |
| Out-of-sample bootstrap | 2,000 resamples; 90% predictive intervals | own assumption |
| Term for multi-year probability | 3 years | own assumption |

## Pricing

| Item | Value | Source |
|---|---|---|
| Models | M1 log spread on log EL; M2 + issue-year fixed effects; M3 + trigger and log size | brief; fixed in `docs/pricing_protocol.md` |
| Sample | 56 distinct layers (Integrity Re III A2 and B2 dropped as duplicates of A1 and B1) | `docs/pricing_protocol.md` |
| Standard errors | classical, HC3, CR1 clustered by series and by sponsor | MacKinnon and White (1985); Cameron and Miller (2015) |
| Wild cluster bootstrap | restricted, Rademacher weights, 9,999 draws, interval by test inversion, seed 6001 | Cameron, Gelbach and Miller (2008); Roodman et al. (2019) |
| Prediction interval | classical, 90%, log scale; M2 evaluated at issue year 2025 | own assumption, pre-specified |
| Comparable-deal distance | log EL scale 0.5, issue-year scale 2 years, k = 5 | own assumption, pre-specified |
| Transaction layers | three adjacent grid layers with EL closest to 4%, 2%, 1% of limit | `docs/pricing_protocol.md` |
| Synthetic deal generator | log spread = 1.0 + 0.65 log EL - 0.01 (year - 2000) + 0.2 parametric + noise | own assumption, used only to test the code |
