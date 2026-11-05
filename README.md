# Illustrative Cat Bond Issuance: Structuring and Pricing

> **Status: work in progress.** This repository currently holds the project plan only. No code or results have been published yet.

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
