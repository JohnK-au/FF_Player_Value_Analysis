# Position Component

> **Status:** Phase 4.5 v1 live. Per-position constants computed via
> 4-sub-metric composite, equal-weighted, min-max normalised to [0, 100].
> Re-derive when the player pool meaningfully changes (post-FA sweep,
> post-draft, new season).

## Intent

`position_value` is a **constant per position group** (all QBs share one
value, all RBs share another, etc.) in [0, 100]. It captures **positional
importance for team success** — how much a unit at that position contributes
to a team's competitive edge in our specific 8-team dynasty league.

This is **NOT cap pricing**. Cap-based fair-value computation is a Phase 5+
task; Position v1 is about positional importance only.

Within-position differentiation lives in Production / Team / On-Field Value;
Position only differentiates *across* positions.

## Locked v1 scores

Computed 2026-06-28 from the master CSV (490 priced players: 155 rostered
+ 335 dynasty-league FAs):

| Position | `position_value` |
|---|---:|
| **RB** | **100.0** |
| WR | 35.2 |
| TE | 21.2 |
| **QB** | **0.0** |

**Reading**: in a 1-QB league with deep QB pool, the per-team competitive
advantage from "winning QB" is smallest (elite QB only ~8 OFV above
replacement, with just 1 slot per team). RB combines the biggest elite-vs-
replacement gap (~19 OFV) with 2.5 effective slots per team → highest total
positional importance.

## Methodology — 4 sub-metrics, equal-weighted composite

For each position `p` in {QB, RB, WR, TE}:

### Sub-metric 1: Slot Count `S_p`
Hardcoded from league rules §4 + estimated flex distribution:
- QB = 1.0
- RB = 2.5
- WR = 3.0
- TE = 1.5
- Sum = 8 (= total skill starters per team)

Flex distribution assumptions:
- WR/TE flex ≈ 62% WR + 38% TE
- RB/WR/TE flex ≈ 38% RB + 50% WR + 12% TE

### Sub-metric 2: Marginal Gap `M_p` (per-slot premium of elite vs replacement)
From master CSV's `on_field_value` column:
- **Elite** = top `(8 × S_p)` players at position p by OFV (one per team starter slot)
- **Replacement** = OFV at rank `(8 × S_p + 1)` (bench cutoff)
- `M_p = mean(elite OFV) − replacement OFV`

### Sub-metric 3: Total Impact `T_p`
- `T_p = M_p × S_p`
- "If a team wins all slots at this position, how much PPG-advantage do they get?"

### Sub-metric 4: Supply-Demand Scarcity `D_p`
- **Bench (idealized, proportional to S_p)**: `bench_p = 15 × S_p / 8` (15 = 23 skill slots − 8 starters)
- **League demand**: `8 × (S_p + bench_p)`
- `D_p = league_demand / pool_size_p` where `pool_size_p` from master CSV (rostered + FAs)

### Combination
1. Z-score each sub-metric across the 4 positions
2. Equal weights (v1): composite_z = (M_z + T_z + D_z + S_z) / 4
3. Min-max normalize composite_z across positions to [0, 100]

## Sub-metric values (2026-06-28 snapshot)

| Pos | S | n_elite | Elite OFV | Replacement OFV | M | T | Pool | League demand | D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QB | 1.0 | 8 | 71.61 | 63.40 | **8.21** | 8.21 | 53 | 23.0 | 0.434 |
| RB | 2.5 | 20 | 65.48 | 46.64 | **18.84** | **47.11** | 133 | 57.5 | 0.432 |
| WR | 3.0 | 24 | 58.77 | 48.55 | 10.22 | 30.67 | 204 | 69.0 | 0.338 |
| TE | 1.5 | 12 | 70.85 | 55.29 | 15.56 | 23.35 | 100 | 34.5 | 0.345 |

Z-scored composite:
- RB: +1.09 → 100.0
- WR: −0.07 → 35.2
- TE: −0.32 → 21.2
- QB: −0.70 → 0.0

## Cross-position impact on dynasty value

With Position weighted 0.05 in the OFV-weighted combine:
- Max swing per player = 100 × 0.05 = 5 dynasty-value points
- RB players: +5.0 (vs neutral baseline +2.5) → net +2.5 vs prior stub
- QB players: 0.0 (vs neutral baseline +2.5) → net −2.5 vs prior stub
- WR/TE: between, scaled by their position score

Observed shift after activation:
- Top RBs climbed ~2.5 (Gibbs 85.8 → 88.3, Bijan 84.2 → 86.7)
- Top TEs dropped ~1.4 (McBride 79.6 → 78.1)
- Top WRs dropped ~0.7 (JSN 78.8 → 78.0)
- Top QBs dropped ~2.5 (Drake Maye 78.0 → 75.5; out of top-5 cross-position)

This matches the expected math; cross-position ranking shifted but didn't
dramatically reorder within positions (which it shouldn't, since within-position
Position is constant).

## Reused infrastructure

- Master CSV (`data/processed/player_value_v2_2026.csv`) for OFV + pool size
- `src/data/cap.py::position_salary_tables` — useful diagnostic for current
  league-cap distribution (sanity-check side analysis, not used in composite)

## Phase plan

- **Phase 4.5 v1** ✅ — 4-sub-metric composite, equal weights, locked
- **Weight tuning** (future): the equal-weight v1 may emphasize the wrong sub-metrics.
  Open candidates:
  - Heavy on M (Marginal Gap) — the "winning the position" intuition
  - Heavy on T (Total Impact) — accounts for slot count
  - Empirically: derive weights from auction-outcome data once we have it
- **Cap pricing extension** (Phase 5+): the original framing the user proposed
  was a cap-equilibrium calc. Position v1 is positional importance only;
  cap fair-value comes later.

## TODOs

- Re-derive POSITION_SCORES when the player pool meaningfully changes (FA
  sweep, draft, season start). The `_scratch_position_calc.py`-style analysis
  is the canonical re-computation path.
- Consider empirically deriving flex-distribution percentages (WR/TE 62/38,
  RB/WR/TE 38/50/12) once we have historical lineup data.
- Weight tuning workshop after v1 results have been digested.

## Known gaps

- **Equal-weighted composite is a placeholder.** No reason to believe each
  sub-metric should contribute equally; needs empirical tuning.
- **Idealized bench distribution** (proportional to S_p) may differ from
  actual team behavior. Real rosters likely over-weight RB depth (injury
  attrition) and under-weight TE depth (most teams stream).
- **Static slot counts.** A team could play a 0-RB strategy or stream TEs;
  S_p reflects "typical" starter usage, not adversarial deviation.
