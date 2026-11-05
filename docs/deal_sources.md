# Deal-terms table: sources, rules and checks

`data/deals/cat_bond_deals.csv` lists publicly reported terms of catastrophe bond tranches with
US named-storm exposure, one row per tranche, for use as the input of a regression of spread on
expected loss. It holds 58 tranches from 37 note series (35 source pages), issued from 2017 to
2025, by 14 sponsors. Every number in it was read on the page named in that row's `source_url`,
except the final Bowline Re 2018-1 coupon, whose source is named in that row's `notes`.
`scripts/validate_deals.py` checks it (run by `tests/test_deals_data.py`).

## Terms of use and attribution

The table compiles individual facts (names, dates, sizes, percentages) from public web pages and
gives the page for every row. No text from the pages is reproduced; the `notes` column is short
factual wording written for this table. The source sites' own terms of use apply to anyone who
reuses the table. William should read those terms before relying on the table commercially or
redistributing it beyond this repository. A web search during an independent review did not find
the site's terms of use. The table (58 rows taken from one site's deal directory) is the kind of
compilation that a site's terms or database-right rules can address; if they object, it can be
reduced to the source URLs plus the extraction script.

## Sources

| Source | What it provided | Rows |
|---|---|---|
| Artemis deal directory, `www.artemis.bm/deal-directory/<deal>/` (one page per note series) | sponsor, date of issue, trigger, perils, rating, tranche size at pricing, initial expected loss, initial attachment probability, initial and final price guidance | 58 |

Only Artemis pages were used for the values in the table: the deal directory pages and, for one
price (Bowline Re 2018-1), one Artemis news item. Sponsor press releases, rating-agency releases
and Swiss Re Capital Markets reports were not used, so no row's values rest on a second,
independent source and no aggregate cross-check against a market report was made (the
independent re-check below found sponsor documents that agree on size and rate on line for two
rows). The pages were read on 2026-11-05 with
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
- `spread_pct` is the final price when the page states one (`spread_basis = final`, 55 rows).
  When the page gives only the latest guidance and does not confirm pricing, that guidance is
  used with `spread_basis = guidance` (3 rows: Citrus Re 2017-1, Alamo Re 2019-1 and Alamo Re
  2021-1). Bowline Re 2018-1 is `final`: its deal page shows guidance only, but an Artemis news
  item of 2018-05-11 (named in `notes`) reports it as priced at the 4.5 mid-point. The pages call
  this number a spread, a coupon or simply the price; `notes` records when the page calls it a
  coupon.
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

## Independent re-check

A second reader (a separate agent, run after the table was built) re-checked 20 further rows. They were
drawn before any page was fetched with `random.Random(20261106).sample(sorted(range(48)), 20)` from
the 48 data rows that the first check had not covered (file lines 2, 3, 5, 6, 7, 12, 13, 14, 15,
18, 20, 22, 23, 25, 31, 48, 49, 51, 54 and 56 before the corrections below). Ten fields per row
were compared with the page: `issue_year`, `issue_month`, `size_usd_m`, `expected_loss_pct`,
`attachment_prob_pct`, `spread_pct`, `rating`, `trigger`, `sponsor` and `tranche`. The pages were
read with the same kind of summarising fetch tool, each request asked for the exact sentence, and
every doubtful field was fetched again with different wording.

| Rows checked | Fields checked | Discrepancies | Fixed |
|---|---|---|---|
| 20 | 200 | 2 (1.0%; Wilson 95% interval 0.27% to 3.57%) | 1 corrected, 1 left empty on purpose |

None of the discrepancies is in an expected-loss, spread or size cell: 0 of 60 (95% upper bound
6.0%). Counting the first check, 3 of 270 fields were wrong (1.1%), but the first check was made by
the person who built the table, so the two checks are not independent. Both read the same Artemis
pages, so they test transcription, not whether Artemis is right.

- Integrity Re 2019-1: the page says its notes are Series 2019-1 Class A; the table had `single
  tranche`. Corrected to `Class A`.
- Everglades Re II 2023-2 (and 2023-1, which shares the page and was not in the sample): the page
  prints 1.23% (and 1.92%) under the label attachment point and gives the dollar attachment
  point separately. The percentages are 1.2 to 1.3 times the expected losses in the table, so they
  are probably attachment probabilities with a wrong label, but `attachment_prob_pct` stays empty
  rather than guessing.

Two further corrections came out of the same pass (these cells were not among the 200):

- Bowline Re 2018-1: `spread_basis` changed from `guidance` to `final` (value unchanged, 4.5),
  because the Artemis news item cited in `notes` reports the bond as priced at the mid-point of
  4.25 to 4.75.
- Citrus Re 2025-1 `region`: Class A covers named storm in the southeast states (AL, FL, GA, MS,
  NC, SC), and Class B covers named storm in Hawaii only; both rows had listed all seven states.
- Gateway Re 2025-1 Class AAA `region` (found by a later review of 10 further rows): the page
  lists Alabama, Louisiana, Mississippi, New York, North Carolina, South Carolina and Texas for
  Class AAA; Florida is listed only for the other classes. `region` is not used by any model.

**Second sources.** For six sampled rows (Alamo Re 2024-1 Classes A, B and C; Everglades Re II
2018-1 Class A, 2021-1 Class B and 2024-1 Class B) the second reader looked for a source other than
Artemis.

- Everglades Re II 2018-1 Class A: Citizens Property Insurance Corporation's board summary of
  2020-06-24 describes a USD 250 million capital-markets layer of its 2018 coverage at a rate on
  line of 4.73% (the document does not name the series, so the match is by size and year). The table
  has a size of 250 and a spread of 4.75.
- Everglades Re II 2021-1 Class B: Citizens' board summary of 2021-07-13 gives USD 275 million at
  a gross rate on line of 6.75%, the same as the table.
- Everglades Re II 2024-1 and Alamo Re 2024-1: only deal totals were found outside Artemis (Citizens:
  USD 1.1 billion in three tranches; TWIA: a USD 1.4 billion bond), which are consistent with the
  class sizes in the table, but no class-level terms.
- No expected loss was found in any non-Artemis source. A sponsor's rate on line is not the same
  number as an Artemis spread (4.73 against 4.75 here), so such documents are a plausibility
  check, not a substitute.

**Rows to treat separately in any analysis.**

- Three `guidance` prices (above). For Citrus Re 2017-1 the news says the price was fixed at the
  bottom of the lowered guidance range without giving a number; 6 is the bottom of the 6 to 6.5
  range. For Alamo Re 2019-1 the press reports the mid-point (4.5); for Alamo Re 2021-1 the press
  reports the price fixed at the low end of the 4% guidance and no source found says it priced.
- Four discount-note rows (Everglades Re II 2023-1 and 2023-2, Gateway Re 2025-1 Classes AA and
  C1): the value is 100 minus the issue price, a rate-on-line type figure for one wind season, not
  a spread over a money-market rate on a par note, and it ignores collateral yield and the
  sub-year term.
- Expected loss from the launch plan: Alamo Re 2023-1 (the figures were stated for a two-series
  USD 250 million plan; the second series was dropped and the deal doubled, and the page does not
  restate them) and Alamo Re 2021-1 (the table has 1.76; an outside article's wording implies 1.93
  for the doubled deal; unresolved).
- Size changes: on 16 of the 20 re-checked pages the tranche size at pricing differed from the launch
  size (15 up, 1 down), so the expected loss and attachment probability, which are offering-stage
  figures, may not describe the notes as issued.
- Duplicate layers: Integrity Re III 2025-1 A1 and A2, and B1 and B2, have identical expected loss,
  attachment probability and spread, so the 58 rows are 56 distinct layers. A regression should use
  the distinct layers. "Distinct" is this rule and no more: Gateway Re 2025-1 Classes C1 and C2
  have the same attachment and exhaustion points, deductible, attachment probability (2.43%) and
  expected loss (1.31%) but differ in note format and price (the Artemis page: C1 is a one-season
  zero-coupon note of USD 50 million priced at 91.75, C2 a two-season note of USD 150 million at
  a 9.5% coupon), so they are one layer of risk in two formats and both rows are kept. Everglades
  Re II 2021-1 and 2021-2, and 2023-1 and 2023-2, share one Artemis page per year but are
  separate series.
- Sponsor strings: "Citizens Property Insurance Corporation" (Florida) and "Louisiana Citizens
  Property Insurance Corporation" are different entities, so grouping by a substring would merge
  them.
- Spread-to-expected-loss ratios differ by issue year (compute them from the table), so a pooled
  regression needs a control for the year, and its standard errors should allow for correlation
  within series and within sponsor.

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
  risk of a misread number in the 28 rows that were not re-checked in either round.
