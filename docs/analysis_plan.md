# Analysis & ML Plan

The core motivation: a **player-value engine** for this dynasty league — model what
drives a player's salary/value (performance, age, position, advanced metrics) to
surface over- and under-valued players and support roster decisions under the cap.

**Last updated:** 2026-05-25

## Confirmed direction (decisions)

| Decision | Choice |
| --- | --- |
| Primary goal | **Fair-value model first**, then performance projection |
| Data sources | **ESPN** (fantasy points in our scoring) **+ nflverse** (`nfl_data_py`) for age, snaps, advanced metrics, and the player-id crosswalk |
| Modeling scope | **All NFL players by position** (learn league-wide patterns, then apply to our rostered players + free agents) |
| Time horizon | **Both** — current-season (2026) cap efficiency *and* forward-looking dynasty value (age curves) |

## What "value" means (targets)

- **Cap efficiency (current season):** fantasy points (in our scoring) per cap unit, and **value over replacement (VOR)** by position vs cap hit.
- **Fair salary:** model-predicted salary from performance + attributes; **residual (actual − predicted) = over/under-valued**. Salary labels exist only for our ~200 rostered players, but features are computed for *all* NFL players so we can also price free agents.
- **Dynasty value:** age-curve-adjusted, discounted multi-year projected production over a player's remaining contract years, net of remaining cap cost.

## Phased plan

### Phase A — Cap-distribution figures (near-term; mostly doable now)
- **A1. Total salary vs cap** per team — CAP USED stacked to the 1500 cap, with CAP SPACE. *(have the data)*
- **A2. Where salary is going** — component breakdown per team: active contracts, rookies, extensions, tags, practice squad, dead cap, admin penalties, trade adjustments. *(have the data via `cap.reconcile`)*
- **A3. Salary by position group** — ✅ done (`viz/cap.plot_salary_by_position`):
  per-team 2026 salary stacked by position + league-wide totals. (Insight: league
  spend is WR ≫ RB ≫ TE ≈ QB; almost nothing on IDP/K/P/HC.)

### Phase B — Player join & data foundation
- **B1. Contract ↔ ESPN join.** ✅ done (`src/data/players.py`): normalized + fuzzy
  matcher with a small alias table and HC handling; **100% of 242 contract players
  matched** (all fuzzy matches verified). Crosswalk carries `position`, `proTeam`,
  `espn_id` → `data/processed/player_crosswalk.csv`. *(Still to do: sheet-nickname
  → ESPN-team-id map.)*
- **B2. ESPN performance pull.** ✅ done (`src/data/performance.py`):
  - *Season pull* — per-player full-season points / PPG / games for the rostered +
    free-agent pool, 2022–2025 (~6,950 player-seasons → `performance.csv`). Broad
    coverage, but spans the full 17-game NFL season.
  - *Weekly pull* — per-player per-week points from box scores over weeks
    1..`reg_season_count` (**13** every year) → `performance_weekly.csv` (~11.9k
    player-weeks). Restricts production to the weeks that count in this league and
    adds consistency (stdev); covers players rostered that week (~300/season).
  - *Deferred:* ESPN projections; fantasy-season points for the full (FA) pool.
- **B3. nflverse integration (`nfl_data_py`).** ✅ started (`src/data/nflverse.py`):
  age (as of the 2026 season) + experience from seasonal rosters, joined on
  `espn_id` — 97% coverage (only HCs missing, as expected). `players.attributes_table`
  merges it onto the crosswalk; `cap.position_salary_tables` now reports avg age.
  *Still to pull: weekly stats, snap counts, and advanced metrics (NGS, PFR).*
- **B4. Unified player dataset.** ✅ started (`src/data/dataset.py`): one row per
  2026 contract player — salary, position group, age, and recent **fantasy
  regular-season** production (2025 fpts/PPG/games/stdev, 2024 PPG), joined on
  `espn_id` (`data/processed/player_dataset_2026.csv`). Falls back to full-season
  points where weekly is unavailable (a `prod_source` flag records which), so all
  156 skill players have a 2025 value (only HCs remain blank). *To add: advanced
  metrics, and a full all-players-by-season version for model training.*

Advanced features to engineer (by position): QB — EPA/play, CPOE, aDOT; RB — snap share, target share, rush yards over expected, YAC; WR/TE — target share, air yards, aDOT, separation (NGS), route participation.

### Phase C — Fair-value model (value first)
- **C1.** Build positional **replacement levels** and VOR across all NFL players.
- **C2.** Train **fair-salary** model (e.g. gradient boosting / regularized regression) on rostered players: `salary ~ performance + age + position + advanced`. Predict fair salary for everyone; **surplus = fair − actual** flags value.
- **C3.** Report **current-season cap efficiency** and a **dynasty value** (C/D combined).

### Phase D — Performance projection (value second)
- Project future PPG over contract years using **age curves by position** (fit on historical nflverse data) + recent performance + advanced metrics. Feeds dynasty value.

### Phase E — Roster optimization (later)
- Combine fair-value + the cap ledger to recommend keep / cut / tag / extend / trade moves that maximize value under the 1500 cap.

## Immediate next steps (when we resume)
1. **A1 + A2 figures** (total-vs-cap and where-salary-goes) — buildable now from `cap.py`.
2. **B1 player join** (contracts ↔ ESPN positions) — unlocks A3 and everything downstream.
3. **B3 nflverse** pull (ages + advanced) once the join exists.

## Open questions for later
- Exact advanced-metric set to prioritize per position.
- How to score deep free agents not in ESPN's pool (apply league scoring to nflverse stats — note custom IDP/ST/HC/win-margin rules make full reconstruction hard).
- Discount rate / aging-curve shape for dynasty value.
- Whether HC and team/DST "players" get their own value treatment.
