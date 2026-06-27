# Combination Function

> **Status:** Phase 1D — Production × Team sub-value multiplier locked for WR.
> Final 6-component combination method workshopped in Phase 5 (after all 4
> positions have their full component scores).

## Intent

Combine the 6 component scores (each in [0, 100]) into a single **Dynasty
Value** — the headline metric for player worth. Plus, for staged validation,
combine specific component pairs into intermediate sub-values.

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

### Sub-value combine — Production × Team (Phase 1D, WR)

A diagnostic / staged-validation combine. Not currently a column in the
master CSV; computed on demand for inspection.

```
multiplier = MULT_LO + (team_value / 100) * (MULT_HI - MULT_LO)
sub_value  = production_value * multiplier
```

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

## Supported full-combine methods (Phase 5 decides)

| Method | Status | Description |
|---|---|---|
| `uniform_weighted_sum` | **v1 default** | Each component weight = 1/6; output = average. Interpretable placeholder. |
| `weighted_sum` | planned | Human-set or learned per-component weights summing to 1 |
| `multiplicative` | planned | Production base × multiplier in [low, high] from each other component |
| `learned` | planned | Weights learned from an objective (user rankings / historical fantasy outcomes / teacher signal). User leans toward this for the final method. |
| `hybrid` | planned | Mixed: Production + Age weighted sum forms the base; Team / Injury / Position / Intangibles emit multipliers |

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
| 2026-06-28 | Production × Team multiplier band locked at [0.875, 1.125] for WR | User override of data-driven recommendation [0.92, 1.08] — see note above |

(Append new rows as the methodology evolves.)
