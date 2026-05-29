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
  ✅ **Advanced metrics done** (`src/data/advanced.py`): 19 metrics — target share/
  WOPR/RACR/EPA/carries from **2025 play-by-play**, NGS (separation/aDOT/RYOE/CPOE),
  PFR (ybc/yac/pressure/on-target), snap share — plus draft capital + combine
  (`nflverse.py`). Windowed to fantasy weeks 1–13, cached to Parquet per season.
- **B4. Unified player dataset.** ✅ started (`src/data/dataset.py`): one row per
  2026 contract player — salary, position group, age, and recent **fantasy
  regular-season** production (2025 fpts/PPG/games/stdev, 2024 PPG), joined on
  `espn_id` (`data/processed/player_dataset_2026.csv`). Falls back to full-season
  points where weekly is unavailable (a `prod_source` flag records which), so all
  156 skill players have a 2025 value (only HCs remain blank). *To add: advanced
  metrics, and a full all-players-by-season version for model training.*

Advanced features to engineer (by position): QB — EPA/play, CPOE, aDOT; RB — snap share, target share, rush yards over expected, YAC; WR/TE — target share, air yards, aDOT, separation (NGS), route participation.

### Phase C — Fair-value model (value first)

**Design decision (2026-05-27): anchor fair value to production, not to prices.**
A `salary ~ features` model only learns the market's *average* pricing, so it cannot
flag *systematic* mispricing — and we believe the market is inefficient. So the
engine now has **two lenses** (`src/models/value.py`):
- ✅ **PRIMARY — production-anchored (`combined_value_table`/`production_value_table`).**
  Value-over-replacement (VOR) in our scoring, replacement levels from the 8-team
  starting lineup (rules §4) over the full NFL pool, priced to cap units by
  redistributing the league's *total* skill-cap spend by VOR. `surplus_prod` flags
  mispricing against on-field value, catching league-wide bias. VOR is
  **downside-risk-adjusted** (`vor_adj = vor − 0.5·downside`) so boom-bust players
  are marked down for *bust* weeks only — see Phase D.
- ✅ **SECONDARY — market-fit.** The original `salary ~ features` HistGBR model
  (advanced lift R² 0.31 → 0.37), kept as a "what the market pays" comparison. Its
  R² is a *diagnostic we do NOT maximize* — a perfect fit would call everyone fair.
- ✅ **C0/C1.** Expanded training set ([`population.py`](../src/data/population.py):
  all NFL skill players × 2022–2025, 2,659 rows) + **production model**
  ([`production.py`](../src/models/production.py): expected PPG, OOF **R² 0.80**)
  + positional **replacement levels / VOR**.
- **C3.** Report **current-season cap efficiency** and a **dynasty value** (C/D combined).

- ✅ **Team/offense efficiency context** added to `ADV_NUM` (`team_pass_epa`/`team_cpoe`/
  `team_rush_epa`/`team_pass_rate`, from the same pbp) → production R² 0.80→0.81, market lens
  +adv 0.37→0.42.

- ✅ **Pricing degeneracy fixed (2026-05-27).** The earlier subtractive risk penalty + $1
  floor floored 130/155 (84%) players to `prod_fair = 1` while Puka absorbed ~1,090. Replaced
  with two changes (see `production_value_table` docstring and `data_dictionary.md` glossary):
  - **Multiplicative consistency factor** (bounded in `[MIN_CONSISTENCY_FACTOR, 1.0]`):
    `prod_adj = ppg_full × max(0.5, 1 − λ · downside / ppg_full)` — volatile-but-startable
    players are penalized but never zeroed.
  - **Deep-baseline pricing**: `deep_baseline = DEEP_FACTOR × starter replacement` (default 0.5).
    `prod_fair` redistributes the cap pool over positive `deep_vor = prod_adj − deep_baseline`;
    sub-baseline players get `prod_fair = 0` (their salary registers as surplus).
  - Results: rate 87→15, top fair 1,090→291, sub-baseline 84%→34%, AJ Brown overpaid by
    144 (fair 56, not 1), Jefferson by 132 (fair 18, not 1).

**Remaining limitations (Phase D/projection fixes these):** values still use *single-season*
(2025) production, so down/injury years distort (e.g. Jefferson still reads overpaid because
the model only sees his 11 PPG year); the replacement baseline isn't risk-adjusted.

### Phase D — Performance projection & risk (value second)
- Project future PPG over contract years using **age curves by position** (fit on historical nflverse data) + recent performance + advanced metrics. Feeds dynasty value.
- ✅ **Consistency / variance (first cut done).** Value a steady player above a
  boom-bust one with the same mean — this is a **weekly head-to-head** league (you
  bank a win, not a season total; a 0 loses the matchup). Implemented as a
  **downside-deviation** penalty on VOR (`production.weekly_consistency` →
  `vor_adj = vor − λ·downside`, λ=`value.RISK_LAMBDA`=0.5). Uses *downside* (shortfalls
  below a player's own average), **not** symmetric stdev, so big ceiling weeks aren't
  punished (boom weeks must not hurt elite RBs like Gibbs/Bijan — user-confirmed).
  Distinct from *sustainability* (fluky efficiency/TD spikes inflating the mean),
  which the usage-based `production.expected_ppg` already dampens.
  *Still to do:* the replacement baseline isn't risk-adjusted (no NFL-wide weekly
  data), and a **weekly win-probability** simulation would be more principled.

### Phase E — Roster optimization (later)
- Combine fair-value + the cap ledger to recommend keep / cut / tag / extend / trade moves that maximize value under the 1500 cap.

## Immediate next steps (when we resume)
An approved plan to build per-player context + projection + dynasty value + a Streamlit
app is **in progress on branch `value-engine-projection-app`** (M1 data dictionary, S1 team
context, and the pricing fix done & pushed). Next, in priority order:
1. **Per-player context** (`src/data/context.py`): baseline/delta/z + `year_type` (down/up/par),
   no-leakage `shift(1).rolling`. Diagnostics now + features for the projection.
2. **Next-season projection** (`src/models/projection.py`) + age curves; then **dynasty value**
   (current + multi-year via `years_2026`) — fixes the single-season distortions still present
   (Jefferson/AJ Brown reading overpaid off their 2025 down years).
3. **Streamlit app**: market/driver explorer, over/under board, value card, roster view
   (drop/extend/tag/keep), auction bid targets, trade evaluator.
4. **Roster optimization** (Phase E): keep/cut/tag/extend/trade under the 1500 cap.

## Open questions for later
- Exact advanced-metric set to prioritize per position.
- How to score deep free agents not in ESPN's pool (apply league scoring to nflverse stats — note custom IDP/ST/HC/win-margin rules make full reconstruction hard).
- Discount rate / aging-curve shape for dynasty value.
- Whether HC and team/DST "players" get their own value treatment.
