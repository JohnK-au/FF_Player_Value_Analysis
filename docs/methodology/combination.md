# Combination Function

> **Status:** Phase 1D — **On-Field Value** (Production × Team) locked for WR.
> Final 6-component **Dynasty Value** combination method workshopped in
> Phase 5 (after all 4 positions have their full component scores).

## Intent

Combine component scores into intermediate + final headline values:

- **On-Field Value** = Production × Team multiplier. Captures "what the
  player is delivering on the field given their environment." Phase 1D.
- **Dynasty Value** = combine of all 6 component scores. The headline metric
  for player worth. Phase 5 locks the method.

## Interfaces

### Final combine — 6 components → Dynasty Value (Phase 5 will lock)

```python
def combine(components: pd.DataFrame, method: str = "uniform_weighted_sum") -> pd.Series:
    """Combine the 6 component columns into a Dynasty Value Series.

    Required columns: production_value, age_value, team_value, injury_value,
    position_value, intangibles_value
    """
```

Pluggable via `method` keyword so swapping strategies doesn't touch callers.

### On-Field Value — Production × Team (Phase 1D, WR)

Per-player intermediate combine; written to the master CSV as the
``on_field_value`` column.

```
multiplier      = MULT_LO + (team_value / 100) * (MULT_HI - MULT_LO)
on_field_value  = production_value * multiplier
```

Implementation: [`src/models/components/combine.py::on_field_value`](../../src/models/components/combine.py).
Multiplier bands are position-keyed via ``MULTIPLIER_BANDS``; non-WR
positions currently default to ``(1.0, 1.0)`` (pass-through) until their
components land in Phases 2-4.

## Locked: Production × Team multiplier band (WR)

| Parameter | Value | Rationale |
|---|---|---|
| `MULT_LO` | **0.875** | User-chosen (see note below) |
| `MULT_HI` | **1.125** | User-chosen — Team can swing Production ±12.5% |

**Note: user override of data-driven recommendation.** The residual
regression in [team.md](team.md) showed team context explains only ~1.2% of
WR Production residuals, with a max worst-to-best PPG swing of ~1 PPG
(≈8% of a 13-PPG baseline). My initial empirical recommendation was a
**tighter band of [0.92, 1.08] or even [0.95, 1.05]**. The user reviewed
the data and chose to **widen to ±12.5%** as a subjective design decision —
giving Team component more voice than the regression alone would support,
reflecting belief that team context matters more than 1.2% of residual
variance suggests.

This decision is intentionally documented so future-us understands the band
isn't data-fit; it's a deliberate trade between empirical conservatism and
design preference.

## Naming

| Term | What it is |
|---|---|
| **Production** | Per-position component score [0, 100] — past on-field production tiered + recency-blended |
| **Team** | Per-position component score [0, 100] — offensive-environment quality |
| **On-Field Value** | Production × Team multiplier = "what they deliver on the field given their environment" |
| Age / Injury / Position / Intangibles | Per-position component scores [0, 100] — off-field dimensions |
| **Dynasty Value** | Final combine of all 6 component scores — the headline metric |
| Contract Value | Dynasty Value / max(years_2026, 1) — generic per-year derivation |

## Supported full-combine methods (Phase 5 decides)

| Method | Status | Description |
|---|---|---|
| `ofv_weighted_sum` | **current default** | Weighted sum using **On-Field Value** (which already encodes Production × Team) plus the 4 off-field components. Default weights: OFV 0.55, Age 0.20, Injury 0.15, Position 0.05, Intangibles 0.05. User-driven: chosen after the uniform-1/6 placeholder was over-diluting OFV. |
| `uniform_weighted_sum` | legacy | Each of the 6 raw components weighted 1/6. Was the Phase 0 placeholder; kept for back-compat / comparison. |
| `weighted_sum` | planned | Human-set or learned per-raw-component weights summing to 1 |
| `multiplicative` | planned | OFV base × multiplier in [low, high] from each off-field component |
| `learned` | planned | Weights learned from an objective (user rankings / historical fantasy outcomes / teacher signal). User leans toward this for the final method. |
| `hybrid` | planned | Mixed: OFV + Age weighted sum forms the base; Injury / Position / Intangibles emit multipliers |

## Decision criteria (for Phase 5 workshop)

Once Phases 1-4 produce real component scores for all 4 positions:

1. **Interpretability** — can the user explain a player's Dynasty Value in plain English?
2. **Sensible disagreements with the legacy engine** — where v2 differs, are the differences defensible?
3. **Robustness** — small weight changes → small ranking changes? (sensitivity sweep)
4. **Fit to user gut take** — for a basket of named players, does v2 generally match intuitive rankings?

## Version history

| Date | Method | Notes |
|---|---|---|
| 2026-06-27 | `uniform_weighted_sum` | Phase 0 placeholder; all components neutral 50 so trivially returns 50 |
| 2026-06-28 | On-Field Value (= Production × Team) multiplier band locked at [0.875, 1.125] for WR | User override of data-driven recommendation [0.92, 1.08] — see note above |
| 2026-06-28 | Name locked: "On-Field Value" for the Production × Team sub-value | Reads cleanly against Age/Injury/Position/Intangibles (off-field dimensions); appears as `on_field_value` column in the master CSV |
| 2026-06-28 | Dynasty Value combine swapped from `uniform_weighted_sum` to `ofv_weighted_sum` (OFV 0.55 / Age 0.20 / Injury 0.15 / Position 0.05 / Intangibles 0.05) | User feedback during Phase 1F WR-table review: uniform 1/6 was over-diluting OFV; players with low Production but max Age + Injury + Team boosts (e.g. Xavier Worthy) were ranking too high. New default makes OFV the dominant signal. Weights are placeholder defaults; final tuning is a Phase 5 task once all 4 positions have full components. |
| 2026-06-28 | RB multiplier band locked at [0.85, 1.15] (±15%) | Phase 2: data-driven from RB Team residual regression. Max worst-to-best PPG swing for RB Team is ~1.85 PPG (vs ~1 PPG for WR), so band set wider than WR's [0.875, 1.125] to reflect stronger empirical effect. |
| 2026-06-28 | TE multiplier band locked at [0.90, 1.10] (±10%) | Phase 3: TE Team residual regression R² only 0.23% (5x weaker than WR's 1.18%). Per user, band set TIGHTER than WR/RB because the SIGNAL strength is weaker, even though the relative PPG effect magnitude (~25% on 5-7 PPG TE) is comparable to RB. |
| 2026-06-28 | QB multiplier band locked at [0.95, 1.05] (±5%) | Phase 4: TIGHTEST of all positions. QB Team residual regression OOF R² was NEGATIVE — team features genuinely don't add signal beyond the QB's own stats (cpoe/passing_epa absorb the supporting-cast effects). Path A "minimal QB Team" picked over multi-feature variants to avoid manufacturing noise. Only feature kept: `team_pass_rate` (sole one with non-trivial standalone signal). |

(Append new rows as the methodology evolves.)
