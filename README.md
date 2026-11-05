# Illustrative Cat Bond Issuance: Structuring and Pricing

> **Status: work in progress (wave 1).** The hazard, exposure, loss, layer and pricing code exists and is tested; the hazard is calibrated on real HURDAT2 Florida landfalls and the loss model runs on a hypothetical portfolio. CLIMADA is not used (its LitPop exposure rasters were not available); the Emanuel (2011) damage function that CLIMADA uses is implemented directly. This is a teaching-grade model, not a vendor catastrophe model (RMS, Verisk/AIR, CoreLogic): none of its numbers is comparable to the modelled expected loss of a real cat bond. The spread regression has been run only on synthetic deal tables so far. Method: `docs/model.md`; assumptions and sources: `docs/assumptions.md`; generated result tables: `results/tables.md`.

| Feature | Implemented | Tested | Run on synthetic data | Run on real data |
|---|---|---|---|---|
| HURDAT2 loaders, Florida landfall definitions | yes | yes | yes (toy tracks) | yes (1900-2024) |
| Frequency tests (Poisson, NB, trend, low-frequency) | yes | yes | yes | yes (1900-2019, 1950-2019, 1900-2024) |
| Intensity models (GPD, KDE) | yes | yes | yes | yes |
| Out-of-sample check (fit 1900-1979, test 1980-2019) | yes | yes (incl. look-ahead guard) | yes | yes |
| Wind field (Holland, Willoughby-Rahn, Kaplan-DeMaria) | yes | yes | yes | used in the event set |
| Exposure (hypothetical, 2020 Census households) | yes | yes | n/a | public counts; portfolio hypothetical |
| Event set, OEP/AEP, return-period losses | yes | yes (closed forms) | yes | hazard from real data, exposure hypothetical |
| Layer metrics and candidate layer grid | yes | yes (closed forms, invariants) | yes | hazard from real data, exposure hypothetical |
| Parameter uncertainty (year bootstrap) | yes | yes | yes | yes |
| Spread regression, comparable-deal screen | yes | yes (statsmodels, coverage) | yes | not run (wave 2) |
| Excel workbook, pitchbook PDF | no | no | no | not run (wave 2) |

Tests: `pytest -q`. Reproduce all results: `bash scripts/reproduce.sh` (fetches the public data first). Mutation check of the tests: `python3 tests/mutation_check.py --out results/mutation_check.txt`.

## Goal

Walk through how a catastrophe bond is structured and priced for a hypothetical sponsor, end to end: loss model, layer, expected loss, spread and investor presentation.

## Planned scope

- **Sponsor and layer.** A hypothetical Florida homeowners insurer and a per-occurrence named-storm layer.
- **Loss model.** Simulated hurricane seasons combined with public exposure and vulnerability assumptions, for example the exposure (LitPop) and vulnerability-curve tools in the CLIMADA risk-modelling package from ETH Zurich.
- **Structure.** Attachment and exhaustion points chosen from the aggregate and occurrence exceedance-probability curves (AEP and OEP). Attachment probability, exhaustion probability and expected loss reported for each candidate layer.
- **Pricing.** Spread set as a multiple of expected loss, using a regression on publicly reported deal terms (size, expected loss, spread) and a comparable-deal screen.
- **Deliverable.** A pitchbook-style investor presentation covering the structure, loss curves, spread sensitivities and the comparable-deal table.

## Data

Public data and stated assumptions only. Deal terms come from public reporting. No proprietary model output is used.

## Disclaimer

Illustrative and educational only. This is not an offering of securities, a recommendation or investment advice.
