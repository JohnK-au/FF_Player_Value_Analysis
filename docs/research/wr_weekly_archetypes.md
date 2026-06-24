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

---

## Second-pass results (2026-06-24, same day)

### Stable-trait pair exploration (the user's question)

Conditional mean fantasy points by **quartile × quartile** of rolling stable traits:

**Volume × accurate QB** — biggest signal of all the pairs tested:
```
                            CPOE_Q1   CPOE_Q2   CPOE_Q3   CPOE_Q4
target_share Q1 (low)         7.4      9.6       9.4       8.2
target_share Q2              10.7     12.4      11.4      12.3
target_share Q3              11.6     11.9      15.2      14.2
target_share Q4 (high)       14.9     15.4      17.4     18.3
```
*Q4×Q4 = 18.3 PPG vs Q1×Q1 = 7.4 PPG — biggest spread of any pair.*

**aDOT × accurate QB (the user's specific question)** — **counterintuitive**:
```
                            CPOE_Q1   CPOE_Q2   CPOE_Q3   CPOE_Q4
aDOT Q1 (short)              13.4     12.8      13.8      14.1
aDOT Q2                      12.2     12.7      14.3     15.4   ← peak
aDOT Q3                      11.3     12.8      13.4      15.0
aDOT Q4 (deep)                9.5     12.2      13.0     11.3
```
*Deep specialists (Q4-aDOT) DO NOT benefit more from accurate QBs. Peak is at
Q2-aDOT (intermediate routes) × Q4-CPOE — converting intermediate routes
matters more than completing deep ones.*

**Single-feature marginal effects (Q1 → Q4 PPG spread):**
- `target_share` — **+7.8** (dominant single predictor)
- `team_cpoe` — +2.0 (plateaus at Q3)
- `separation` — +2.0 (monotonic)
- `YAC over expected` — +1.7
- `aDOT` — **−2.1** (slightly negative — deep-threat penalty)

**Lesson for value:** the dominant archetype isn't "deep-threat with accurate QB" —
it's **"volume + accurate QB + good YAC ability"**. Deep specialists actually
underperform a typical WR for fantasy because deep balls don't convert. This is a
counterintuitive but consistent finding across the 4-year dataset.

### Option 3 — snap-share rolling (added)

`snap_pct` joined per (week, pfr_id) from `nfl.import_snap_counts`, rolled 4 weeks.
Coverage 96%. Result: `snap_pct_roll4` becomes the #2 feature by permutation
importance (0.074), but model R² stays 0.024 and MAE only moves 9.05 → 9.03.
**Snap share is mostly redundant with target share for WRs** — both capture "is he
on the field and getting looked at." Likely a bigger lift when this approach is
extended to RBs (where snap share varies more independently of touches).

### Option 2 — descriptive (leaky) variant ⇒ **the ceiling**

Re-fit with same-week role + efficiency features (`target_share`, `snap_pct`, NGS
`separation`/`catch_pct`/`aDOT`/`YAC_above_expectation`, team `pass_epa`/`cpoe`,
env, static) — but NOT trivially-score-correlated stats (no targets/receptions/
air_yards/rec_epa, since those are basically the box score):

```
n = 2,905 WR-weeks
OOF R² = 0.684    MAE = 4.90 PPG
```

**The 0.66 R² gap (0.684 descriptive vs 0.024 predictive) is the cost of not
knowing pre-game how the WR will be USED + how EFFICIENT he'll be.**

Top descriptive features by permutation importance:
```
catch_percentage              1.07  ← single biggest
avg_yac_above_expectation     0.19
target_share                  0.16
avg_intended_air_yards        0.13
team_pass_epa_play            0.11
air_yards_share               0.08
team_pass_rate                0.07
```

In plain English: **if we knew a WR's actual catch rate, YAC, target share, and
his team's QB efficiency that game, we'd predict his fantasy points within ~5
PPG**. Almost all pre-game unpredictability is uncertainty about role and
efficiency that game — *not* about the player's profile or environment.

This is the right interpretation of the low predictive R²: the engine isn't
*broken*, the format itself is high-variance and most of the variance is the
in-game realization of role and efficiency, which is fundamentally a future event.

### Option 1 — season aggregation

Stack the weekly OOF predictions per (season, espn_id), filter to ≥6 weeks
(253 player-seasons):

```
Season-total points:  R² = 0.770   MAE = 25.6 pts
Season PPG:           R² = 0.680   MAE = 2.58 PPG
```

The existing season-level production model is **R² = 0.48** (`models/production.py`).
By that comparison the weekly-aggregated model wins handily (0.68 vs 0.48 on PPG).

**Important caveat:** the comparison isn't strictly apples-to-apples. The weekly
rolling features at week W use weeks 1..(W−1) of the **same season** — so this is
an **in-season** projection (each prediction benefits from progressive in-season
information). The season-level R² 0.48 is for **next-season** prediction from
prior-season features — a strictly harder problem.

**Implication:** the weekly model is a strong **in-season projection** tool
(useful for in-season trade evaluation, weekly start/sit, mid-season cap
decisions). It does *not* replace the pre-season projection used by the value
engine. Both have a place, and a future iteration could fit a "pre-season-only"
weekly model (restrict to prior-season rolling) to make the comparison fair.

## Updated recommendations

1. The **archetype tree + conditional heat** is the strongest deliverable —
   surface it in the Streamlit app's Market/Driver Explorer as a position-
   specific "WR archetype" panel (rules + heatmap). Already actionable.
2. The **stable-trait hierarchy** (target_share ≫ team_cpoe ≈ separation ≈ YAC,
   aDOT slightly negative) should inform the value engine's per-player features
   when ranking long-term WR upside. Deep-threat specialists deserve a discount,
   not a premium.
3. The **descriptive R² 0.68** is a *useful* upper bound for any future weekly
   tool — it tells us we shouldn't expect more than ~5 PPG MAE on weekly
   projections even with great features.
4. The **in-season weekly aggregator** (R² 0.68 PPG) could power a *separate*
   in-season tool: "the season-to-date data says player X is on pace for Y PPG."
   Distinct from pre-season valuation.
5. Apply the same architecture to **RB / TE / QB** next — for RBs especially,
   snap_pct will likely add more lift than it did for WRs.
