# Production Component

> **Status:** Phase 1A complete (extended training frame ready). Phase 1B (WR
> implementation) is next.

## Intent

Score each player in [0, 100] based on **what they have produced on the field**,
using only player-driven signals (usage + skill + per-player history). Age is
owned by the [Age component](age.md); offensive environment is owned by the
[Team component](team.md); this component must not double-count them.

## Locked design choices

| Decision | Choice |
|---|---|
| **Training scope** | Per-position model — each position trains on its own pool of player-seasons (no cross-position pooling); feature sets differ by position |
| **Same-season basis** | One model per position predicts that season's PPG from that season's player-level features |
| **Historical blend** | Recency-weighted: current = 1.0, prior = 0.5, 2-yr-old = 0.25, 3-yr-old = 0.125 (geometric, normalized over available seasons) |
| **Score mapping** | PPG-anchored linear scale within position, per season — `score = clip(0, 100, 100 × (predicted_ppg − floor) / (top − floor))` where `floor` = positional replacement, `top` = season-best |
| **Tier integration** | Draft + combine features stay in the model for all tiers; their *effective* weight decays naturally as production data accumulates |
| **Rollover trigger** | Mid-season — `CURRENT_SEASON` flips to the active NFL year once ~Week 6 of new-season data is in (stable enough to anchor on). Until then prior completed year holds the "current" slot. |

## Feature space per position

### Clearly IN — player usage + skill
| Position | Features |
|---|---|
| **WR / TE** | `target_share`, `wopr`, `racr`, `snap_pct`, `avg_separation`, `yac_above_expected`, `receiving_epa`, `adot`, `catch_pct` |
| **RB** (Phase 2) | `carries`, `snap_pct`, `target_share`, `receptions`, `rushing_epa`, `ryoe_per_att`, `time_to_los`, `yac_att`, `receiving_epa`, `catch_pct`. **Dropped per user**: `ybc_att` (relabelled as O-line proxy → Team component) |
| **QB** | `passing_epa`, `cpoe`, `on_tgt_pct`, `pressure_pct`, `passing_yards/game`, `passing_tds/game`, `interceptions/game`, `carries`, `rushing_epa` (mobile-QB premium), `rushing_yards/game` |

### Clearly OUT
- **Team-owned** (→ Team component): `team_pass_epa`, `team_cpoe`, `team_rush_epa`, `team_pass_rate`
- **Age-owned** (→ Age component): `age`, `years_exp`
- **QB-stat in receiver model** (→ Team component for WR/TE/RB models): `pressure_pct`, `on_tgt_pct`, `passing_epa`, `cpoe` (these belong to the QB throwing to them, not the receiver)

### IN per the spec — weight decays naturally as production accumulates
| Feature | Source |
|---|---|
| `draft_round`, `draft_pick`, `draft_value`, `undrafted` | NFL draft capital ([nflverse.py](../../src/data/nflverse.py)) |
| `forty`, `vertical`, `broad_jump` | Combine athleticism ([nflverse.py](../../src/data/nflverse.py)) |

## Subjective override (planned)

User reserves the right to nudge production for a player based on non-data
signal (scouting, role notes). Default path: an optional `production_value`
column in [`data/research/intangibles_overrides.csv`](../../data/research/intangibles_overrides.csv).
Mechanics TBD — covered in the [intangibles doc](intangibles.md).

## Training data

Per Phase 1A: **5,598 player-seasons × 10 years (2016-2025)** in
[`data/processed/training_frame_extended.csv`](../../data/processed/training_frame_extended.csv):

| Position | Total seasons |
|---|---:|
| QB | ~790 |
| RB | ~1,560 |
| WR | ~2,120 |
| TE | ~1,130 |

PPG basis: nflverse `seasonal_data` + our [2025 scoring rules](../../src/data/scoring.py)
applied uniformly to 2016-2024; ESPN-reported PPG for 2025 (nflverse 2025
seasonal_data not yet published). Reconstruction validated against 2024 ESPN
data (MAE 0.82 PPG; median 0.46) — well within tolerance for training-data
purposes.

## Reused infrastructure

- [`src/data/scoring.py::nflverse_season_production`](../../src/data/scoring.py) — PPG reconstruction from raw stats
- [`src/data/population.py::extended_training_frame`](../../src/data/population.py) — assembled per-season feature frames stacked across the v2 window
- [`src/data/context.py::add_player_context`](../../src/data/context.py) — rolling baselines (no leakage) — used for the historical blend's per-player time series

## Phase plan

- **Phase 1A** ✅ — extend training data to 2016-2025; build scoring reconstruction
- **Phase 1B** ✅ — WR Production model: OOF R² 0.816, MAE 1.89 PPG
- **Phase 2** ✅ — RB Production model: OOF R² **0.829**, MAE **1.83 PPG** (slightly stronger than WR)
- **Phase 3** ✅ — TE Production model: reuses WR feature set unchanged (TEs are receivers)
- **Phase 4** — QB with position-specific feature set
- **Later** — enable subjective production override; tune the recency-decay weights empirically (see TODOs)

## TODOs (revisit after first model output)

- **Time-decay weights** (currently `[1.0, 0.5, 0.25, 0.125]` geometric) need
  empirical tuning. The right falloff likely differs by position (RB careers
  decay faster than WR, etc.) and may be a learned parameter.
- **Production × Team combination band** (Team currently proposed as a
  multiplier in [0.7, 1.3]) needs empirical tuning once both components produce
  real scores.

## Known gaps

- **Dynasty league rookie draft slot data** — not yet pulled. Source TBD
  (likely a new column on the Master Cap Sheet or a separate sheet).
- **Subjective override mechanism** — design deferred; default scoring should
  work without it for v1.
- **NGS / PFR feature coverage** — NGS receiving starts 2016 (full coverage),
  NGS rushing + passing accuracy and PFR pass stats come online 2017-2018. Older
  player-seasons have NaN on those features (HistGBR handles natively).
