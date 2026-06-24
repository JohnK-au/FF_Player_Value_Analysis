# WR Weekly Archetypes — First-Pass Findings

**Date:** 2026-06-24 · **Scope:** rostered WRs, 2022–2025, exploration only
(not wired into the app). **Branch:** `wr-weekly-archetypes`.

## Hypothesis

Season aggregates wash out *conditional* signal (matchup, QB context, game
environment). A week-level model should be able to surface "archetype" rules of the
form *"WR with prior-4-wk target_share ≥ X, vs defense allowing ≥ Y pass EPA/play,
in a game with implied team total ≥ Z → expects N PPG."*

## Setup

- **N**: 2,905 WR-weeks across 4 seasons (146 unique WRs).
- **Target**: weekly fantasy points in our scoring (`performance_weekly.csv`).
- **Features (29)** — pre-game only, no leakage:
  - Player static (age, draft, combine): 9
  - Game environment (Vegas spread / total / implied / home-away): 5
  - Opponent defense, weighted rolling prior weeks (last 4 × 2.0, earlier × 1.0): 5
  - Per-player rolling 4-week history (`shift(1).rolling(4).mean()`): 10
- **Drop-list** (would leak the target): same-week targets, receptions, air_yards,
  rec_epa, target_share, team CPOE, NGS separation, etc. Their *rolling* versions
  are fair game.

## Results

**Predictive (OOF, 5-fold KFold)**

| Model | R² | MAE (PPG) |
|---|---:|---:|
| **HistGradientBoosting** (depth 5) | **+0.024** | **9.05** |
| Baseline: constant mean | +0.000 | 9.23 |
| Baseline: trailing-4-wk PPG | −0.084 | 9.44 |

**Per-week fantasy points are genuinely hard to predict.** The std of weekly fantasy
points is 11.7 on a mean of 12.4 — single-game variance dominates. Our model beats
both trivial baselines but only by ~0.2 PPG on MAE; the R² lift is real but tiny.

The trailing-4-wk PPG baseline being *negative-R²* is its own finding: a player's
recent 4 weeks are barely informative about his next week beyond his career baseline.
Week-to-week is mean-reverting and matchup-dominated.

**Top features by permutation importance**

```
target_share_roll4              0.136
avg_separation_roll4            0.082
fantasy_points_roll4            0.074
def_pass_epa_play_roll          0.068
team_implied_total              0.064
broad_jump                      0.061
rec_epa_roll4                   0.060
def_completion_pct_allowed_roll 0.059
team_pass_epa_play_roll4        0.058
def_yards_allowed_roll          0.058
```

Sensible: **recent role > recent results > defense matchup > game environment > pedigree**.

## The archetype tree (the real deliverable)

A depth-4 decision tree with min-leaf 80 surfaces interpretable rules. Despite the
low predictive R², the splits are coherent:

```
target_share_roll4 ≤ 22% ─── recent role is below average
├── fantasy_points_roll4 ≤ 7.14 ── struggling
│   ├── target_share_roll4 ≤ 11%, low air-yards share         →   4.1 PPG  (deep bench)
│   └── target_share_roll4 in [11–22%]
│       ├── total ≤ 45.25                                     →   7.9 PPG  (low-scoring game)
│       └── total > 45.25                                     →  12.7 PPG  (shootout helps)
└── fantasy_points_roll4 > 7.14 ── decent recent form
    ├── draft_value ≤ 1632 + good rec_epa                     →  14.3 PPG
    └── draft_value > 1632 + team_cpoe_roll4 > 1.75            →  17.3 PPG  ← high pedigree + accurate QB
target_share_roll4 > 22% ─── recent role above average
├── team_implied_total ≤ 24.62
│   ├── separation_roll4 > 3.38                               →  19.3 PPG  ← elite separator
│   └── low separation + low team total                       →  10.7 PPG
└── team_implied_total > 24.62 ── shootout
    ├── targets_roll4 ≤ 9.71 + leaky defense (TDs ≥ 1.41)     →  19.6 PPG
    └── targets_roll4 > 9.71                                  →  23.9 PPG ← WR1 in shootout
```

**Plain-English summary of the archetypes:**

| Tier | Mean PPG | Rule |
|---|---:|---|
| **Elite WR1 in shootout** | 23.9 | recent target_share > 22% AND team total > 24.6 AND targets > 9.7 |
| **High target share + great separator** | 19.3 | target_share > 22% AND separation > 3.4 yds (even with modest total) |
| **High volume + leaky defense** | 19.6 | high targets + high team total + opp allowing TDs |
| **High pedigree + accurate QB + decent form** | 17.3 | top-draft WR + team CPOE > 1.75 + rolling PPG > 7.1 |
| **Decent role + good total** | 12.7 | 11–22% target share + game total > 45 |
| **Decent role + good results, mid-pedigree** | 14.3 | recent rec_epa > 4.3 |
| **Stuck on low-scoring offense** | 10.7 | high target share but low team total + below-avg separation |
| **Marginal usage** | 7.9 | 11–22% target share, low-scoring game |
| **Deep bench** | 4.1 | <11% target share, no air-yards role |

**Conditional heat — target_share_roll4 × separation_roll4 (mean PPG by quartile bin):**

```
              Sep Q1   Sep Q2   Sep Q3   Sep Q4
TgtShr Q1      8.46    10.67    10.34     9.56   (n≈636)
TgtShr Q2     11.70    11.49    12.23    12.80   (n≈635)
TgtShr Q3     13.30    12.86    12.15    15.27   (n≈635)
TgtShr Q4     15.46    16.10    16.66    19.17   (n≈636)
```

Monotonic in target share, broadly monotonic in separation, with a strong **Q4 × Q4
combo** (19.2 PPG). Two players with the same season PPG can differ by ~10 PPG/week
depending on which quadrant they live in.

## What this tells us

1. **Per-week prediction is hard.** R² 0.02 is honest — the format-imposed noise
   (one-game samples, game flow, target distribution decisions made in real time)
   genuinely dominates pre-game signal. *Don't expect to crack 0.10+ R² on weekly
   prediction without play-call / drive-level features we don't have.*
2. **Conditional structure IS real and exploitable.** The tree leaves and the
   heatmap are highly informative even though the model's MAE is barely better
   than a baseline. We're learning *real patterns in averaged outcomes by archetype*,
   we're just not great at predicting any individual week.
3. **The interesting cut is descriptive, not predictive.** "WRs in archetype X
   average Y PPG" is a meaningful and actionable statement; "the model predicts
   13.4 ± 9.0 PPG for this week" is not.
4. **For roster construction / valuation**, this *is* useful — knowing that an
   "elite WR1 in shootout" averages 23.9 PPG vs 7.9 for a marginal-role player
   matters for what you should pay.

## Recommended next moves

1. **Aggregate to season**: sum the per-week predictions across a season and
   compare against the existing season-level projection model (OOF R² 0.48). If
   week-aggregated beats season-fitted, the week-level model earns a place even
   despite low per-week R².
2. **Try a *descriptive* (leaky) variant**: include same-week features (separation,
   target_share, etc.) to see *upper-bound* predictability. That gives a ceiling
   for "if we knew the player's role in the game, how predictable would his points
   be?" — separates role uncertainty from outcome uncertainty.
3. **Add snap-share rolling** (not yet pulled per-week). A WR with declining snaps
   is a real warning the rolling-target_share signal misses early.
4. **Extend to RB, TE, QB** with appropriate features (carries rolling, rush EPA,
   team_rush_epa, etc.) — same architecture, position-specific feature sets.
5. **Reframe as classification** ("startable week" vs "bust week") — a coarser
   target (e.g. ≥15 PPG yes/no) may be more learnable than the continuous
   regression and more decision-useful for lineup choices.

The conditional-archetype framing is the win here, not the OOF R². The user's
intuition was right — week-level surfaces structure that season aggregates hide —
but the structure is best presented as **descriptive archetypes**, not as a
high-accuracy predictor.
