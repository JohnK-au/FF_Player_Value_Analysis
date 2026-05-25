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
- **A3. Salary by position group** — needs positions → depends on the player join (B1).

### Phase B — Player join & data foundation
- **B1. Contract ↔ ESPN join.** ✅ done (`src/data/players.py`): normalized + fuzzy
  matcher with a small alias table and HC handling; **100% of 242 contract players
  matched** (all fuzzy matches verified). Crosswalk carries `position`, `proTeam`,
  `espn_id` → `data/processed/player_crosswalk.csv`. *(Still to do: sheet-nickname
  → ESPN-team-id map.)*
- **B2. ESPN performance pull.** Weekly + seasonal fantasy points (our scoring, already computed by ESPN) for the player universe, 2022–2025; also ESPN projections.
- **B3. nflverse integration (`nfl_data_py`).** Player-id crosswalk (`import_ids` has `espn_id` ↔ `gsis_id`), rosters (birthdate/age, draft year, experience), weekly stats, snap counts, and advanced metrics (NGS, PFR). Join to ESPN via the id crosswalk.
- **B4. Unified player-season dataset.** One row per (player, season): performance (PPG, totals, advanced), attributes (age, position, experience), and — for our players — contract (salary, years remaining, cap hit, status).

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
