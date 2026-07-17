# Production Component

> **Status:** ✅ Live for all four skill positions (WR/RB/TE/QB — Phases 1A–4
> complete). Remaining work is refinement, not implementation: see
> [TODOs](#todos-revisit-after-first-model-output) and [Known gaps](#known-gaps).

## Intent

Score each player in [0, 100] based on **what they have produced on the field**,
using only player-driven signals (usage + skill + per-player history). Age is
owned by the [Age component](age.md); offensive environment is owned by the
[Team component](team.md); this component must not double-count them.

## Locked design choices

| Decision | Choice |
|---|---|
| **Training scope** | Per-position model — each position trains on its own pool of player-seasons (no cross-position pooling); feature sets differ by position |
| **Model** | `HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0, random_state=0)` — see [Why HistGBR](#why-histgbr) |
| **Same-season basis** | One model per position predicts that season's PPG from that season's player-level features |
| **Historical blend** | Recency-weighted: current = 1.0, prior = 0.5, 2-yr-old = 0.25, 3-yr-old = 0.125 (geometric, normalized over available seasons) |
| **Score mapping** | PPG-anchored linear scale within position, per season — `score = clip(0, 100, 100 × (predicted_ppg − floor) / (top − floor))` where **`floor` = 0 PPG** and `top` = that season's best PPG in the position pool |
| **Tier integration** | Draft + combine features stay in the model for all tiers; their *effective* weight decays naturally as production data accumulates |
| **Rollover trigger** | Mid-season — `CURRENT_SEASON` flips to the active NFL year once ~Week 6 of new-season data is in (stable enough to anchor on). Until then prior completed year holds the "current" slot. |

### Why HistGBR

Not an accuracy-driven choice — **NaN-native handling is the binding constraint**,
and it is load-bearing in two independent places:

1. **Structural feature gaps.** NGS receiving starts 2016, but NGS rushing /
   passing accuracy and PFR pass stats only come online 2017–2018, so pre-2018
   player-seasons are NaN on those columns (see [Known gaps](#known-gaps)).
2. **The no-history path.** Rookies / new arrivals have no NFL seasons to blend,
   so [`_score_one_player`](../../src/models/components/production.py) builds a
   synthetic row that is **all-NaN except draft + combine** and predicts on it
   directly.

Any replacement regressor must either be NaN-native (XGBoost / LightGBM /
CatBoost) or gain an imputer step — and note that imputing changes the rookie
path's predictions for reasons unrelated to model quality, which would make a
head-to-head comparison misleading.

### Why `floor = 0` (not replacement level)

The scale is **full-range**: 0 PPG → 0, season-best → 100. An earlier draft of
this doc specified `floor` = positional replacement; the code has always
implemented zero-anchoring
([`_season_anchors`](../../src/models/components/production.py)), and this doc was
the thing that was wrong. Consequence worth knowing: scores are *not* spread
across the full 0–100 range — a replacement-level player scores well above 0, so
the usable band is compressed toward the middle. Replacement level is handled
downstream instead, by the pricing engine's per-position baselines
([pricing.md](pricing.md)).

## Feature space per position

### Clearly IN — player usage + skill
| Position | Features |
|---|---|
| **WR / TE** | `target_share`, `wopr`, `racr`, `snap_pct`, `avg_separation`, `yac_above_expected`, `receiving_epa`, `adot`, `catch_pct` |
| **RB** (Phase 2) | `carries`, `snap_pct`, `target_share`, `rushing_epa`, `ryoe_per_att`, `time_to_los`, `yac_att`, `receiving_epa`, `catch_pct`. **Dropped per user**: `ybc_att` (relabelled as O-line proxy → Team component). **Wanted but unavailable**: `receptions` — see below |
| **QB** (Phase 4) | `passing_epa`, `cpoe`, `on_tgt_pct`, `adot`, `carries`, `rushing_epa`, draft + combine. **Dropped per user**: `pressure_pct` (moved to Team for OL attribution, then dropped from Team too after empirical regression showed it as noise) |

### Wanted but unavailable
| Feature | Position | Status |
|---|---|---|
| `receptions` | RB | **Not in `extended_training_frame()`.** It was declared in `RB_FEATURES` from Phase 2 but the availability filter dropped it silently, so the RB model has *never* trained on it — the recorded RB numbers below are a 16-feature model. Discovered 2026-07-16 and removed from the declared list to keep code and doc honest. Adding it is worthwhile (3-down-back role drives RB dynasty value) but is a **modelling change, not a fix**: it shifts the DV distribution, which silently miscalibrates the pricing baselines + `pool_scale` ([pricing.md](pricing.md)). Needs its own pass with a re-calibration. |

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

| Position | Total seasons | **Actually fit on** |
|---|---:|---:|
| QB | ~790 | **534** |
| RB | ~1,560 | **1,217** |
| WR | ~2,120 | **1,710** |
| TE | ~1,130 | **895** |

The left column is the raw pool; the right is what each model actually trains on
after [`_position_training`](../../src/models/components/production.py) applies
`games >= 4` and `ppg.notna()`. Quote the right-hand column when reasoning about
sample size — the QB model in particular is fit on 534 rows, which is the main
reason its R² trails the other positions.

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
- **Phase 2** ✅ — RB Production model: OOF R² **0.829**, MAE **1.83 PPG** (slightly stronger than WR). Note: 16 features, not 17 — `receptions` was never available (see [Wanted but unavailable](#wanted-but-unavailable)).
- **Phase 3** ✅ — TE Production model: OOF R² **0.776**, MAE **1.44 PPG**. Reuses WR feature set unchanged (TEs are receivers).
- **Phase 4** ✅ — QB Production model: OOF R² **0.688**, MAE 4.28 PPG. Lower than other positions because of fewer samples (534 rows) + higher variance scale (QBs have higher PPG range). `pressure_pct` dropped after user moved it to Team and the empirical regression showed it as unstable noise at the team level too.
- **Later** — enable subjective production override; tune the recency-decay weights empirically (see TODOs)

All four figures re-measured 2026-07-16 via
[`position_oof_r2`](../../src/models/components/production.py) on the live code;
WR/RB/QB matched the values already recorded here, TE had never been recorded.
**Read them with the [CV caveat](#known-gaps) below** — they are comparable
across models, but optimistic as absolute accuracy.
Reproduce with `.venv/bin/python -m src.models.components.production`.

## TODOs (revisit after first model output)

- **Time-decay weights** (currently `[1.0, 0.5, 0.25, 0.125]` geometric) need
  empirical tuning. The right falloff likely differs by position (RB careers
  decay faster than WR, etc.) and may be a learned parameter.
- ~~**Production × Team combination band** (Team proposed as a multiplier in
  [0.7, 1.3])~~ — **done.** Superseded by per-position bands derived from
  empirical residual regression and locked in
  [`combine.py`](../../src/models/components/combine.py): WR [0.875, 1.125],
  RB [0.85, 1.15], TE [0.90, 1.10], QB [0.95, 1.05]. See
  [combination.md](combination.md).
- **`receptions` for the RB model** — wanted, unavailable, and a modelling
  change rather than a fix. See [Wanted but unavailable](#wanted-but-unavailable).
- **Player-leaky CV** — the recorded R² values are optimistic; switching to
  `GroupKFold` on `espn_id` would give honest numbers but re-baselines every
  figure in this doc. See [Known gaps](#known-gaps).

## Known gaps

- **The recorded R² values are player-leaky, therefore optimistic.**
  [`position_oof_r2`](../../src/models/components/production.py) uses
  `KFold(5, shuffle=True)` over *player-seasons*, so the same player's 2022 and
  2023 rows can land on opposite sides of a split — the model has effectively
  already seen that player. The figures are **valid for like-for-like comparison**
  between models evaluated on identical folds (that's what they're for), but they
  should **not** be quoted as out-of-sample accuracy. An honest measure needs
  `GroupKFold(groups=espn_id)`, which would lower every number here.
- **Dynasty league rookie draft slot data** — not yet pulled. Source TBD
  (likely a new column on the Master Cap Sheet or a separate sheet).
- **Subjective override mechanism** — design deferred; default scoring should
  work without it for v1.
- **NGS / PFR feature coverage** — NGS receiving starts 2016 (full coverage),
  NGS rushing + passing accuracy and PFR pass stats come online 2017-2018. Older
  player-seasons have NaN on those features (HistGBR handles natively — see
  [Why HistGBR](#why-histgbr)).
- **No regression tests pin these numbers.** `position_oof_r2` is a manual
  diagnostic; nothing fails if a change moves the model. Re-run
  `.venv/bin/python -m src.models.components.production` and compare by hand.

## Version history

| Date | Change | Notes |
|---|---|---|
| 2026-07-17 | **Doc reconciled to code; no model change.** | Documentation-truth pass, verified behaviour-preserving (all four OOF R²/MAE identical before and after). Corrected: `floor` documented as positional replacement but implemented as **0** since Phase 1B — the doc was wrong, not the code. Named the model (HistGBR) and recorded the **NaN-native** rationale, which was previously implicit in a code comment despite being load-bearing for the pre-2018 gaps and the rookie no-history path. Removed `receptions` from the declared RB feature list — it never existed in the training frame and was being dropped silently; the availability filter now warns. Added TE's OOF R² (0.776 / 1.44), never previously recorded. Added actual per-position fit sizes (QB 534 / RB 1217 / WR 1710 / TE 895) alongside the raw pool counts. Flagged the player-leaky CV. |
| 2026-06 | Phases 1A–4 shipped (WR → RB → TE → QB). | Per-position HistGBR on the 2016–2025 extended frame; recency-weighted blend of per-season scores. See [Phase plan](#phase-plan). |

*(Append new rows as the methodology evolves.)*
