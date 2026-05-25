# Architecture

A layered pipeline; dependencies point **downward** (upper layers import lower,
never the reverse). See [data_sources.md](data_sources.md) for how each source is
scraped, and [analysis_plan.md](analysis_plan.md) for the roadmap.

```
config  ──►  data (ingest + parse)  ──►  dataset (unified features)  ──►  models (future)
                                    └──►  viz (figures)
```

## Layers & modules

| Layer | Module | Responsibility |
| --- | --- | --- |
| **Config** | [`src/config.py`](../src/config.py) | Single source of truth for paths + league constants (`TEAMS`, seasons, `CAP_TOTAL`, `DEAD_CAP_RATE`). No internal deps. |
| **Ingest/parse** | `data/sheets.py` | Google Sheet workbook (Option A, by tab name) |
| | `data/espn.py` | Authenticated ESPN league (`get_league`) |
| | `data/nflverse.py` | nflverse player attributes via `nfl_data_py` (age, exp) |
| | `data/contracts.py` | Parse active rosters + extensions; `build_2026_contracts` |
| | `data/cap.py` | Parse cap sections (summary/rookies/tags/IR/cuts); dead-cap model; `reconcile`; `cap_breakdown`; salary aggregations |
| | `data/players.py` | Contract↔ESPN crosswalk (+ nflverse attributes join) |
| | `data/performance.py` | ESPN season + weekly (wk 1–13) fantasy points |
| **Dataset** | `data/dataset.py` | Joins salary + position + age + production → model-ready table |
| **Viz** | `viz/contracts.py`, `viz/cap.py` | Figures (contract timeline, cap distribution/projection, salary-by-position) |
| **Models** | *(future)* `src/models/` | Fair-value model, projections, roster optimization |

## Conventions
- Shared paths/constants live **only** in `src/config.py`.
- Every cross-source join keys on **`espn_id`** (contract sheet ↔ ESPN ↔ nflverse).
- Caches under `data/` and figures under `figures/` are **git-ignored**; secrets
  live only in `.env`.
- Parsers locate sheet columns by **header label**, not fixed offsets.

## Known coupling / planned refactor
`cap.py` currently does three things — sheet-section *parsing*, the *cap model*
(dead cap / reconcile), and cross-domain *aggregation* (`player_salaries_2026`,
`salary_by_position`, `position_salary_tables`, which pull in
`players.attributes_table`). Since `players.current_players` also imports cap's
parsers, there's a `cap ↔ players` cycle resolved with function-level imports.

**Planned (deferred to avoid churn before the value model):** move the
cross-domain "combining" functions up into the dataset/analysis layer so `cap.py`
stays pure parsing + cap modelling, breaking the cycle.

## Where upcoming features land
- **B3 — advanced metrics** (target share, aDOT, snaps, RYOE, …): extend
  `data/nflverse.py` or add `data/advanced.py`; join on `espn_id`, aggregate
  weekly over wk 1–13 to match the fantasy season.
- **Phase C — fair-value model**: new `src/models/value.py` consuming
  `dataset.build_player_dataset()`; outputs predicted fair salary + surplus.
- **Phase E — roster optimization**: `src/models/optimize.py` over the cap ledger
  (`cap.py`) + value model.
