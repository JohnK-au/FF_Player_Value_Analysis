# Position Component

> **Status:** Phase 4.5 v2 live (VORP-Deep Total Impact). Per-position constants
> derived from absolute PPG (not OFV). Re-derive when the player pool meaningfully
> changes (post-FA sweep, post-draft, new season).

## Intent

`position_value` is a **constant per position group** (all QBs share one
value, all RBs share another, etc.) in [0, 100]. It captures **positional
importance for team success** — how much a unit at that position contributes
to a team's competitive edge in our specific 8-team dynasty league.

This is **NOT cap pricing**. Cap-based fair-value computation is a Phase 5+
task; Position v2 is about positional importance only.

Within-position differentiation lives in Production / Team / On-Field Value;
Position only differentiates *across* positions.

## Locked v2 scores

Computed 2026-06-28 from the master CSV (490 priced players: 155 rostered
+ 335 dynasty-league FAs):

| Position | `position_value` |
|---|---:|
| **RB** | **100.0** |
| WR | 93.1 |
| TE | 8.6 |
| **QB** | **0.0** |

**Reading**: RB and WR are nearly tied at the top because both combine
meaningful elite-vs-deep-FA PPG gaps with multiple roster slots per team.
QB sits at 0 — in a 1-QB league with a deep QB pool, the per-team
positional advantage from elite QBs is smallest. TE is genuinely low —
its PPG distribution is flatter across the pool, so elite TEs don't have
the same absolute PPG advantage as elite RB/WR.

## Methodology — VORP-Deep Total Impact (T-only)

For each position `p` in {QB, RB, WR, TE}:

### Step 1 — Slot Count `S_p` (effective starters per team)
Hardcoded from rules §4 + flex distribution:
- QB = 1.0, RB = 2.5, WR = 3.0, TE = 1.5 (sum = 8 = skill starters/team)

### Step 2 — Elite and Replacement PPG
From 2025 PPG (games ≥ 4) over the master CSV pool:
- `N = 8 × S_p` (league-wide starter slots, one per team)
- **Elite** = mean PPG of top `N` players by 2025 PPG
- **Replacement (deep FA)** = mean PPG of ranks `3N+1` to `4N`
  - QB: ranks 25-32 (the deep FA tier — realistic worst-case raid)
  - RB: ranks 61-80
  - WR: ranks 73-96
  - TE: ranks 37-48

### Step 3 — Marginal Gap and Total Impact (PPG units)
- `M_p = elite_avg − replacement_avg` (per-slot PPG gap vs deep FA)
- `T_p = M_p × S_p` (total team PPG advantage if a team wins all slots at this position)

### Step 4 — Composite + Normalisation
- Z-score `T_p` across the 4 positions
- Min-max normalise to [0, 100]
- (Equivalently: only the T sub-metric is weighted; M, D, S sub-metrics are dropped in v2.)

## Sub-metric snapshot (2026-06-28)

| Position | S | N (starters) | Elite avg PPG | Deep-FA replacement PPG | M_ppg | **T_ppg** |
|---|---:|---:|---:|---:|---:|---:|
| QB | 1.0 | 8 | 25.07 | 12.25 | 12.82 | **12.82** |
| RB | 2.5 | 20 | 18.35 | 4.85 | 13.49 | **33.74** |
| WR | 3.0 | 24 | 16.84 | 6.08 | 10.77 | **32.30** |
| TE | 1.5 | 12 | 14.42 | 4.68 | 9.74 | **14.61** |

## How v2 was chosen (over v1 + other variants)

**v1** (OFV-based composite with equal weights) had a methodological bug:
OFV is normalized **within position** (top WR ≈ top TE ≈ 100 by construction),
so comparing M_p in OFV units across positions doesn't preserve absolute
PPG-gap differences. Result: TE looked artificially scarce (M_TE = 15.6 in
OFV) even though its absolute PPG gap is small (~3 PPG strict, ~10 PPG deep).

**v2 fix**: re-derive M using **absolute 2025 PPG** instead of OFV. Now M_p
is comparable across positions. The "drop-off from Chase to next-tier WR is
bigger than Kelce to next-tier TE in absolute PPG terms" — user's intuition —
is correctly captured.

**Replacement-tier choice**: explored four replacement definitions in PPG units:

| Tier | QB replacement | RB | WR | TE | Reads as |
|---|---:|---:|---:|---:|---|
| Strict (rank N+1) | 22.04 | 13.77 | 13.03 | 11.61 | Bench cutoff; gap is small |
| Backup (avg N+1 to 2N) | 20.02 | 10.55 | 11.62 | 9.98 | First-bench tier |
| FA (avg 2N+1 to 3N) | 15.43 | 6.96 | 9.01 | 6.78 | Realistic FA tier (QB ranks 17-24) |
| **Deep (avg 3N+1 to 4N)** | **12.25** | **4.85** | **6.08** | **4.68** | Deep FA / "if you're desperate" |

Chose **Deep** because:
- It captures the realistic worst-case replacement when no top players are available
- It produces the most differentiated cross-position scores
- It correctly elevates WR (whose deep FA pool is genuinely weak) close to RB
- It matches user's intuition that "elite WR Chase vs deep-FA WR is a huge gap"

**Weighting choice**: T-only (Total Impact only). Equivalent to v3 in the
variant exploration. Reasoning:
- M_p captures per-slot value; T = M × S adds slot count
- T is the cleanest single answer to "how much does winning this position
  contribute to team PPG advantage"
- D (demand/supply) was deprioritized because it doesn't reflect on-field impact
- S alone is just slot count, no production info

## Cross-position impact on dynasty value

With Position weighted 0.05 in the OFV-weighted combine:
- Max swing per player = 100 × 0.05 = 5 dynasty-value points
- RB players: +5.0 (vs neutral baseline +2.5) → net +2.5 vs prior stub
- WR players: +93.1 × 0.05 = +4.66 (close to RB; was +1.76 in v1)
- TE players: +8.6 × 0.05 = +0.43 (was +1.06 in v1; dropped)
- QB players: 0.0 (vs neutral baseline +2.5) → net −2.5 vs prior stub

Cross-position rank shifts vs v1: WR climbs significantly, TE drops, QB and RB
stay at their extremes.

## Reused infrastructure

- Master CSV (`data/processed/player_value_v2_2026.csv`) for pool size
- Extended training frame (`src/data/population.py::extended_training_frame`)
  for 2025 PPG by player

## TODOs

- **Multi-tier concentration overlay** (deferred, NOT core methodology — per user 2026-06-28):
  expose elite / startable / bench / reserve tier breakdowns as a *secondary
  analytical layer* on top of the locked `position_value`, used for filtering
  and sorting results (e.g., "show me the elite-tier WRs", "which TEs are
  truly in the concentrated elite vs the long tail"). The single-tier
  VORP-Deep score stays as the cross-position importance constant; multi-tier
  views are an overlay on top of the existing rankings, not a recomputation
  of the score itself. See [[multi-tier-position-overlay]].
- **Re-derivation cadence**: re-run when the player pool meaningfully changes
  (post-FA sweep, post-draft, new season). Helper: `_scratch_vorp_tiers.py`-style
  analysis from a fresh master CSV.
- **Empirically tune flex distribution** (WR/TE 62/38, RB/WR/TE 38/50/12)
  once we have historical lineup data.

## Known gaps

- **Single-tier replacement** (deep FA only) ignores within-position
  bifurcation. The "TE elite tier is concentrated" intuition isn't captured
  cleanly — TE drops to 8.6 instead of getting a bifurcation premium.
- **Idealized bench distribution** (proportional to S_p) may differ from
  actual team behavior. Used in computing D_p (now unused in v2's T-only
  combine, but still relevant if we ever re-weight).
- **Static slot counts.** A team could play a 0-RB strategy or stream TEs;
  S_p reflects typical lineup behavior, not adversarial deviation.

## Version history

| Version | Date | Method | Scores (QB/RB/WR/TE) |
|---|---|---|---|
| v1 | 2026-06-28 | OFV-based 4-sub-metric composite, equal weights, min-max | 0 / 100 / 35.2 / 21.2 |
| **v2** | 2026-06-28 | **PPG-based VORP-Deep Total Impact (T-only)** | **0 / 100 / 93.1 / 8.6** |
