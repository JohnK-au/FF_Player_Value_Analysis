# Data Dictionary

Reference for the data this project has access to: what each asset is, its grain and
**scope**, the coverage constraints to keep in mind, and a glossary of the stats
(raw and **derived**) used across the pipeline. See
[data_sources.md](data_sources.md) for how the raw data is scraped/pulled, and
[architecture.md](architecture.md) for where each lives in code.

> **Scope** legend — **NFL-wide** = every rostered/FA NFL player; **rostered** = only
> players on an in-league roster (~the startable universe for our 8 teams);
> **league** = our 8-team dynasty league only.

## Data assets

### Production — ESPN fantasy points, in our custom scoring
| Asset | Grain | Coverage | Scope |
| --- | --- | --- | --- |
| `data/processed/performance.csv` | player × season | 2022–2025 · 6,946 rows · 2,732 players · full-season points/PPG/games | **NFL-wide** |
| `data/processed/performance_weekly.csv` | player × week | 2022–2025 · wk 1–13 · 11,931 rows · 572 players · per-week points + `started` | **rostered** |

Two PPG bases coexist: **13-week fantasy PPG** (weeks 1–13 = the league regular season;
from the weekly box scores) and **full-season PPG** (up to 17 games; from the season pull).
Both are per-game. See the [PPG policy](#ppg-basis-policy).

### Advanced metrics — nflverse (pbp / NGS / PFR / snaps), windowed to wk 1–13
| Asset | Grain | Coverage | Scope |
| --- | --- | --- | --- |
| `data/processed/advanced/advanced_{2022..2025}.parquet` | player × season | ~1,640–2,225 players/season · 19 metrics | **NFL-wide** |

### Player attributes — nflverse (joined on `espn_id`)
| Asset | Content | Scope |
| --- | --- | --- |
| ages (`attributes/ages_{season}.parquet`) | `age` (per season), `years_exp` | NFL-wide |
| draft capital (`context/draft_capital.parquet`) | `draft_round`, `draft_pick`, `draft_value`, `undrafted` | NFL-wide |
| combine (`context/combine.parquet`) | `forty`, `vertical`, `broad_jump` | NFL-wide |

### Contracts / cap — the league's Google Sheet
| Asset | Content | Scope |
| --- | --- | --- |
| `data/processed/contracts_active.csv` | 168 contracts, 8 teams: `salary`, `years_remaining`, `contract_slot` | league |
| `data/processed/contract_extensions.csv` | 15 extensions: `salary`, `years`, `effective_season`, `years_until_end` | league |
| `contracts.build_2026_contracts` | 2026 salary + **`years_2026`** (remaining contract years from 2026) | league |
| cap ledger (`cap.py`) | CAP USED/SPACE, rookies, tags, IR, cuts/dead cap (reconciled to the sheet) | league |
| `data/processed/player_crosswalk.csv` | 242 contract players → `espn_id`, position (100% matched) | bridge |

### Combined / model tables
| Asset | Content |
| --- | --- |
| `data/processed/training_frame.csv` | **2,659 skill player-seasons** (2022–25, 1,051 players) = production + 19 advanced + draft + combine + age — the modeling base |
| `data/processed/player_dataset_2026.csv` | 180 contract players: salary + age + 2024/25 production |
| `data/processed/fair_value_2026.csv` | 155 priced players, both lenses (`prod_fair`/`surplus_prod`, `market_fair`/`surplus_market`, `vor`, `downside`) |

## Coverage constraints (read before modeling)
1. **"NFL-wide" is not uniform.** Full-season PPG and advanced metrics are NFL-wide, but
   **weekly scoring (consistency/variance) exists only for the ~572 rostered players** — no
   weekly basis for the full replacement pool without reconstructing it from nflverse.
2. **Only 4 seasons (2022–2025)** → just **3 year-over-year transitions** for any
   projection / age-curve model. Robustness is a real concern; treat projections as directional.
3. **Scoring is ESPN-custom** (IDP, head-coach, special teams, unusual passing: +0.35/completion,
   −1/attempt, −0.66/incompletion, 6 pts/pass TD). Skill scoring is reconstructable from nflverse
   if NFL-wide weekly is ever needed; full reconstruction (IDP/HC/ST) is hard.
4. **Advanced metrics are windowed to weeks 1–13** (the fantasy regular season), per season.
5. **Within any one calculation, keep the PPG basis consistent** (don't mix 13-week and full-season).

## PPG basis policy
- **Preferred basis = 13-week fantasy PPG** (weeks 1–13) for **production/value outputs** wherever
  available, because it matches the league's regular season.
- **Caveat:** 13-week weekly is **rostered-only**, so NFL-wide replacement levels / the model
  training pool use **full-season PPG** (both are per-game and close). Pick one basis per calculation.
- **Advanced metrics stay full-season** — a holistic representation of player performance.

## Glossary

### Raw metrics (per player-season unless noted)
- **PPG** — fantasy points per game (in our scoring). Two bases above.
- **EPA** — Expected Points Added. `receiving_epa` / `rushing_epa` / `passing_epa` are EPA **summed**
  over the player's wk 1–13 plays (from play-by-play). *Derived.*
- **target_share** — player targets ÷ team targets (wk 1–13). *Derived.*
- **WOPR** — Weighted Opportunity Rating = 1.5 × target_share + 0.7 × (air-yards share). *Derived.*
- **RACR** — Receiver Air Conversion Ratio = receiving yards ÷ air yards. *Derived.*
- **aDOT / adot** — average Depth Of Target (avg intended air yards), NGS.
- **avg_separation** — avg yards of separation at catch point, NGS.
- **yac_above_expected** — yards-after-catch over NGS expectation.
- **catch_pct** — catch rate on targets, NGS.
- **CPOE / cpoe** — Completion % Over Expected (QB accuracy), NGS.
- **RYOE / ryoe_per_att** — Rush Yards Over Expected per attempt, NGS.
- **time_to_los** — avg time to line of scrimmage on runs, NGS.
- **ybc_att / yac_att** — yards Before / After Contact per rush attempt, PFR.
- **pressure_pct / on_tgt_pct** — % dropbacks pressured / % throws on-target, PFR.
- **snap_pct** — avg offensive snap share over wk 1–13. *Derived.*
- **carries** — rush attempts (wk 1–13). **draft_value** — pick value (OTC chart); **undrafted** = 1 if UDFA.
- **forty / vertical / broad_jump** — combine measurables.

### Team / offense context (derived from the same play-by-play, by `posteam`)
- **team_pass_epa** — mean EPA per dropback for the player's team (offense passing efficiency).
- **team_cpoe** — mean CPOE per dropback for the team.
- **team_pass_rate** — pass plays ÷ (pass + rush) plays.
- **team_rush_epa** — mean EPA per rush for the team.

### Derived value-model stats (the model layer)
- **VOR** — Value Over Replacement: a player's PPG minus the **replacement level** at their position
  (`models/production.py`).
- **replacement level** — PPG of the worst startable player at a position, from filling all 8 teams'
  starting lineups (rules §4) over the player pool.
- **downside deviation** — RMS of weekly shortfalls *below a player's own average* (floor risk);
  used to risk-adjust VOR (`vor_adj = vor − λ·downside`, λ = `value.RISK_LAMBDA`). *Derived.*
- **redistribution rate** — cap units per VOR point: total skill-cap spend ÷ total positive VOR.
- **prod_fair / surplus_prod** — production-anchored fair value & surplus (`= actual − prod_fair`).
- **market_fair / surplus_market** — market-fit model's predicted salary & surplus (secondary lens).
- **expected_ppg** — production model's same-season PPG estimate (`models/production.py`).
- **projected PPG** — next-season / multi-year PPG from the projection model (`models/projection.py`).

*(Newer context/projection columns — `{metric}_baseline/_delta/_z`, `year_type`, dynasty value —
are documented as they land; see [analysis_plan.md](analysis_plan.md).)*
