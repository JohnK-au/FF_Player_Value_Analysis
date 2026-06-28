# Injury Component

> **Status:** Phase 1E complete for WR (this commit). RB / QB / TE land in
> Phases 2-4 with the same formula (parameters may re-tune per position).

## Intent

Score each player in [0, 100] based on **durability / floor risk** — how
likely they are to be available *and* productive going forward. Combines a
chronic-availability read with an acute recovery-penalty read.

## Locked WR design

Two pieces:

### 1. Chronic availability (last 3 seasons, recency-weighted)

```
availability_weights   = [1.0, 0.5, 0.25]    # last 3 seasons, most-recent first
availability_score     = 100 × Σ(games_i × w_i) / Σ(max_games_i × w_i)
```

Where `max_games_i = 17` for 2021+, `16` for 2020 and earlier. Players
without NFL data in a given window (pre-rookie) skip that season — they
don't get a 0 for a year they weren't in the league.

### 2. Acute recovery penalty (latest season only)

```
games_missed_last_season = max_games_last_season − games_played_last_season
recovery_penalty         = max(0, RECOVERY_K × (games_missed_last_season − RECOVERY_THRESHOLD))
```

With `RECOVERY_THRESHOLD = 4` (no penalty for ≤4 games missed — that's
normal NFL wear-and-tear) and `RECOVERY_K = 3` (each excess game missed
docks 3 injury_value points).

### Combined

```
injury_value = clip(0, 100, availability_score − recovery_penalty)
```

## Sanity-check archetypes

| Archetype | Games (last 3) | Last yr missed | Availability | Penalty | **injury_value** |
|---|---|---:|---:|---:|---:|
| Iron-man | 17, 17, 17 | 0 | 100 | 0 | **100** |
| Couple games miss | 17, 17, 14 | 3 | 90 | 0 | **90** |
| Recent shoulder | 17, 17, 8 | 9 | 70 | 15 | **55** |
| Major recent injury | 17, 17, 3 | 14 | 53 | 30 | **23** |
| Out all year | 17, 17, 0 | 17 | 43 | 39 | **4** |
| Chronic risk | 12, 14, 11 | 6 | 71 | 6 | **65** |
| **Injury 2 yrs ago, healthy now** | 17, 8, 17 | 0 | **85** | 0 | **85** |

The last row was the user's explicit ask: missing 8 games two seasons ago
shouldn't hurt much now → injury_value 85 (in the 80s-90s).

## Empirical validation

Two pieces of supporting evidence:

### (a) Recovery penalty slope (empirically derived)

From the residual-vs-games-missed analysis on 1,000 WR transition pairs:

| Games missed last season | n | Mean delta vs age-cohort PPG |
|---:|---:|---:|
| 0 | 232 | +3.98 |
| 1-2 | 260 | +2.60 |
| 3-4 | 131 | +0.82 |
| 5-6 | 111 | −0.17 |
| 7-8 | 96 | −1.40 |
| 11-12 | 41 | −2.57 |
| 13-14 | 41 | −3.53 |

Monotonic decline. Linear fit: −0.53 PPG per game missed (R² = 0.16). The
4-game threshold approximately matches where the empirical penalty becomes
negative; K=3 chosen as a reasonable injury-value penalty per excess game.

### (b) Full formula validation (residual regression)

Tested whether `injury_value` predicts next-season production residual after
controlling for player features (1,099 transition pairs):

| Metric | Result |
|---|---|
| z-coef (injury_value alone) | **+0.160** |
| OOF R² (injury alone) | 0.55% |
| Pearson correlation | +0.089 |
| Coef stability (5-fold std) | 0.017 (very stable, mean 0.16) |

Sign is correct (high injury_value → over-perform expectation), coefficient
is small but stable. The Production model has already captured most
predictable variance via player features; injury adds incremental signal on
top.

## Known limitations

- **Games played is a proxy for injury, not a direct measure.** It conflates
  injury absences with role changes (e.g., Travis Hunter as a two-way
  player), benching, and limited rookie reps. Real injury data
  (nflverse `import_injuries()` — IR placements, severity, recovery time)
  is the v2 follow-up.
- **Within-season game-by-game injuries not captured** — a player who played
  17 games but lost half due to in-game injury looks healthy to this proxy.
- **Recovery penalty is symmetric across injury types** — an ACL recovery
  is structurally different from a hamstring tweak but both count as
  "games missed."

## Reused infrastructure

- [`src/data/population.py::extended_training_frame`](../../src/data/population.py) — provides per-(player, season) games
- LRU-cached `_player_games()` lookup table for O(1) per-player access

## Phase plan

- **Phase 1E** ✅ — WR Injury implemented + validated
- **Phase 2** ✅ — RB uses **same formula and parameters as WR** (THRESHOLD=4, K=3, weights [1.0, 0.5, 0.25]). May need to tune for RB's higher attrition profile in a later iteration; v1 keeps it position-agnostic for simplicity.
- **Phase 3** ✅ — TE uses same formula + threshold, but **gentler K=2.5** (vs 3.0 for WR/RB) per user judgment that injury impact for TEs is slightly less than receivers/runners. Implementation lookup: `RECOVERY_K_BY_POSITION` dict in `src/models/components/injury.py`.
- **Phase 4** ✅ — QB uses same formula + threshold + K=3.0 (matching WR/RB). QB injury impacts (concussions, throwing-shoulder, etc.) feel comparable in severity to receivers/runners; no reason to diverge.
- **v2** — real nflverse injury data; severity-weighted recovery penalty

## TODOs

- Re-tune `RECOVERY_THRESHOLD` and `RECOVERY_K` empirically once more validation runs are available
- Pull `nfl.import_injuries()` data and replace the games-played proxy
- Consider position-specific weights (RB injury risk is structurally higher)
