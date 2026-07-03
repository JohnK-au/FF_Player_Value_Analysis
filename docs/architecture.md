# Architecture

A layered pipeline; dependencies point **downward** (upper layers import lower,
never the reverse). See [data_sources.md](data_sources.md) for how each source is
scraped, [data_dictionary.md](data_dictionary.md) for what each asset contains,
and [methodology/](methodology/) for the V2 component-framework specs.

```
config  ──►  data (ingest + parse)  ──►  dataset (unified features)  ──►  models/{v1, components/}  ──►  app (Streamlit) + viz
```

## Layers & modules

| Layer | Module | Responsibility |
| --- | --- | --- |
| **Config** | [`src/config.py`](../src/config.py) | Single source of truth for paths + league constants (`TEAMS`, seasons, `CAP_TOTAL`, `DEAD_CAP_RATE`). No internal deps. |
| **Ingest/parse** | `data/sheets.py` | Google Sheet workbook (Option A, by tab name) |
| | `data/espn.py` | Authenticated ESPN league (`get_league`) |
| | `data/nflverse.py` | nflverse player attributes via `nfl_data_py` (age, exp, draft, combine) |
| | `data/contracts.py` | Parse active rosters + extensions; `build_2026_contracts` |
| | `data/cap.py` | Parse cap sections (summary/rookies/tags/IR/cuts); dead-cap model; `reconcile`; `cap_breakdown`; salary aggregations |
| | `data/players.py` | Contract↔ESPN crosswalk (+ nflverse attributes join) |
| | `data/performance.py` | ESPN season + weekly (wk 1–13) fantasy points |
| | `data/scoring.py` | League scoring rules + nflverse-based fantasy-points reconstruction |
| | `data/advanced.py` | pbp/NGS/PFR/snaps aggregations, windowed to wk 1–13 |
| **Dataset** | `data/dataset.py` | Joins salary + position + age + production → model-ready table |
| | `data/population.py` | All-NFL skill-player training frame (2016–2025; 5,598 player-seasons) |
| | `data/context.py` | Per-player baseline/delta/z-scores + `year_type` flag (up/par/down/rookie/partial) |
| **Models — V1 engine** | `models/production.py` | Per-position production model (HistGBR, OOF R² 0.81) + replacement levels / VOR |
| | `models/projection.py` | Next-season PPG projection (OOF R² 0.48) + positional age curves |
| | `models/value.py` | Two-lens fair value (production-anchored VOR + market-fit) → `player_value_2026.csv` |
| **Models — V2 framework** | `models/components/production.py` | Per-position predicted PPG → [0, 100], recency-weighted |
| | `models/components/team.py` | Per-position empirical residual regression → multiplier band |
| | `models/components/age.py` | Logistic decay sigmoid per position |
| | `models/components/injury.py` | Durability score from games played + IR designation |
| | `models/components/position.py` | Cross-position importance constant per position (VORP-Deep Total Impact; see [position.md](methodology/position.md)) |
| | `models/components/intangibles.py` | Stub at neutral 50 |
| | `models/components/combine.py` | On-Field Value (Production × Team multiplier) + Dynasty Value combine method |
| | `models/components/framework.py` | Orchestrator: builds `player_value_v2_2026.csv` (490 players × 6 components) |
| **Models — Cap pricing** | `models/pricing.py` | Cap-unit pricing LAYER on V2 quality scores. 4-stage pipeline: per-position replacement baseline → above-baseline DV → non-linear scarcity (α exponent) → rate × age multiplier × multi-year age decay. Writes `data/processed/player_pricing_2026.csv` joined to V2 master by espn_id. V2 master itself is never modified. See [pricing.md](methodology/pricing.md). |
| **Viz** | `viz/contracts.py`, `viz/cap.py` | Figures (contract timeline, cap distribution/projection, salary-by-position) |
| | `viz/value.py`, `viz/value_interactive.py`, `viz/summary.py` | V1 fair-value scatters + interactive HTMLs (read V1 outputs) |
| | `viz/position_components.py` | V2 per-position component grids (one HTML per position) |
| | `viz/cross_position_variants.py` | V2 weight-tuning explorer — multiple Position-weight variants in one HTML each |
| | `viz/combine_variants.py` | V2 combine-method workshop viz — 5 candidate Dynasty Value combine methods |
| | `viz/pricing_variants.py` | Cap-pricing workshop viz — 4 preset parameter combinations |
| **App** | `app/Home.py` + `app/pages/*.py` | Streamlit app — over/under board, Player Card, Market/Driver Explorer, Roster, Auction, Trade |

## Conventions

- Shared paths/constants live **only** in `src/config.py`.
- Every cross-source join keys on **`espn_id`** (contract sheet ↔ ESPN ↔ nflverse).
- Caches under `data/` and figures under `figures/` are **git-ignored**; secrets
  live only in `.env`.
- Parsers locate sheet columns by **header label**, not fixed offsets.
- PPG basis: 13-week fantasy PPG for value reporting; advanced metrics stay
  full-season; don't mix bases in one calculation. See
  [data_dictionary.md → PPG basis policy](data_dictionary.md#ppg-basis-policy).

## V1 vs V2 — coexistence

The V1 fair-value engine (`models/value.py`) and the V2 component framework
(`models/components/`) currently coexist:

- **V1** is shipped, validated, and powers the Streamlit app today via
  `data/processed/player_value_2026.csv`.
- **V2** is the active development line — interpretable per-component scores
  per position (Production / Team / OFV / Age / Injury / Position /
  Intangibles → Dynasty Value), written to `data/processed/player_value_v2_2026.csv`.

The two share upstream data, models (`models/production.py`,
`models/projection.py`), and `data/context.py`. V1 will be archived once
the Streamlit app migrates to V2 outputs (planned Phase 6/7).

## Known coupling

`cap.py` does three things — sheet-section *parsing*, the *cap model*
(dead cap / reconcile), and cross-domain *aggregation* (`player_salaries_2026`,
`salary_by_position`, `position_salary_tables`, which pull in
`players.attributes_table`). Since `players.current_players` also imports cap's
parsers, there's a `cap ↔ players` cycle resolved with function-level imports.
Cleanup is deferred until after the V1/V2 migration churn settles.
