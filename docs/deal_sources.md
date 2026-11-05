# Deal-terms table: sources, rules and checks

`data/deals/cat_bond_deals.csv` lists publicly reported terms of catastrophe bond tranches with
US named-storm exposure, one row per tranche, for use as the input of a regression of spread on
expected loss. It holds 58 tranches from 37 note series (35 source pages), issued from 2017 to
2025, by 14 sponsors. Every number in it was read on the page named in that row's `source_url`.
`scripts/validate_deals.py` checks it (run by `tests/test_deals_data.py`).

## Terms of use and attribution

The table compiles individual facts (names, dates, sizes, percentages) from public web pages and
gives the page for every row. No text from the pages is reproduced; the `notes` column is short
factual wording written for this table. The source sites' own terms of use apply to anyone who
reuses the table. William should read those terms before relying on the table commercially or
redistributing it beyond this repository.

## Sources

| Source | What it provided | Rows |
|---|---|---|
| Artemis deal directory, `www.artemis.bm/deal-directory/<deal>/` (one page per note series) | sponsor, date of issue, trigger, perils, rating, tranche size at pricing, initial expected loss, initial attachment probability, initial and final price guidance | 58 |

Only Artemis deal pages were used. Sponsor press releases, rating-agency releases and Swiss Re
Capital Markets reports were not used, so no row has a second, independent source and no
aggregate cross-check against a market report was made. The pages were read on 2026-11-05 with
a web-fetch tool that returns a summary of the page; every extraction asked for the exact phrase
containing each number, and the numbers were taken from those phrases.

Pages that could not be read (the fetch returned a client error; the guessed address was
probably wrong, and the series was found under another address where one exists):
`alamo-re-ltd-series-2020-1` (series found as `alamo-re-ii-pte-ltd-series-2020-1`),
`everglades-re-ltd-series-2017-1` (found as `everglades-re-ii-ltd-series-2017-1`),
`cape-lookout-re-ltd-series-2018-1` (not found; no row).

## Rules applied

- Qualifying tranche: the perils listed on the page include US named storm (hurricane / tropical
  cyclone), alone or with other perils, and the page states both an expected loss and a price.
  Fifteen rows are multi-peril (`peril` lists several perils separated by `;`).
- Programs were chosen by name, not by a census of the directory: the Florida, Texas, North
  Carolina and Louisiana residual-market and state-pool sponsors, several Florida private
  carriers, and a few national multi-peril sponsors. The table is therefore not a complete list
  of US named-storm cat bonds in 2017-2025.
- `expected_loss_pct` and `attachment_prob_pct` are the initial (offering) figures that the page
  states, sometimes called "base" expected loss. One series (Purple Re 2025-1) restated its
  layers before pricing; there the restated expected loss is used, the attachment probability is
  left empty because it was not restated, and `notes` gives the earlier pair.
- `spread_pct` is the final price when the page states one (`spread_basis = final`, 54 rows).
  When the page gives only the latest guidance and does not confirm pricing, that guidance is
  used with `spread_basis = guidance` (4 rows). The pages call this number a spread, a coupon or
  simply the price; `notes` records when the page calls it a coupon.
- Discount and zero-coupon notes (Everglades Re II 2023-1 and 2023-2, Gateway Re 2025-1
  Classes AA and C1, 4 rows): `spread_pct` is the coupon or spread equivalent that the page
  itself states next to the issue price. For Everglades Re II 2023 the page calls it approximate.
- `size_usd_m` is the tranche size at pricing. Where the page gives only a deal-level size for a
  single-tranche series, or only the at-a-glance size, `notes` says so.
- `issue_year` and `issue_month` are the page's "date of issue" field.
- `rating` is the page's rating field (`NR` = not rated); empty where the page shows none.
- `exhaustion_prob_pct` is empty in every row: none of the pages states it (they give the
  exhaustion point in dollars instead).
- `tranche` is the class name used on the page; `single tranche` where the page names none.

## Excluded

- Everglades Re II 2020-1 Class A and Everglades Re II 2024-2: withdrawn, not issued.
- Integrity Re III 2025-1 Class E: dropped before pricing (no price).
- Any tranche whose page does not state both expected loss and price.

## Re-verification

After collection, 10 rows were drawn with `random.Random(20261105).sample(range(58), 10)` over the
data rows in file order (file lines 8, 10, 11, 28, 40, 41, 42, 45, 52, 53). Each source page was
fetched again with a differently worded request that did not reveal the recorded values, and seven
numeric fields per row were compared: `issue_year`, `issue_month`, `size_usd_m`,
`expected_loss_pct`, `attachment_prob_pct`, `exhaustion_prob_pct`, `spread_pct` (70 fields).

| Rows checked | Fields checked | Discrepancies | Fixed |
|---|---|---|---|
| 10 | 70 | 1 | 1 |

The discrepancy: Alamo Re 2019-1 `size_usd_m` had been left empty because the article does not
confirm a final size, but the page's at-a-glance field states USD 200 million; the value was
filled with a note. The same rule was then applied to the one other row with an empty size
(Citrus Re 2017-1, at-a-glance USD 125 million, outside the sample). The discrepancy rate (1 of
70 fields, under 10%) did not trigger a full re-check.

## Limitations

- Selection: only deals that Artemis lists, and within those only the programs chosen above.
  Programs with many series (Florida Citizens: 17 rows; TWIA: 9 rows) carry more weight than
  others in any pooled regression.
- Vintage: 25 of 58 rows are from 2025 and only 2 each from 2020 and 2022, so the table does
  not cover each year evenly; spread levels move with market conditions by year.
- Survivorship: withdrawn tranches are excluded, so the table shows only prices at which
  investors bought.
- Expected loss belongs to each deal's risk modeller (different firms and model versions, on the
  sponsor's exposure at the time); it is not an estimate made by this project, and figures from
  different deals are not on a common model basis.
- Price wording differs between pages (spread, coupon, price), and four rows use a stated spread
  equivalent of a discount note. The table does not convert between them.
- Extraction used a summarising fetch tool. The re-verification bounds, but does not remove, the
  risk of a misread number in the 48 rows that were not re-checked.
