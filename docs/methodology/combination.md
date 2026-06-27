# Combination Function

> **Status:** Phase 0 (foundation). v1 placeholder = uniform weighted sum.
> Final combination method will be workshopped at end of Phase 4 (after all
> positions have component scores).

## Intent

Combine the 6 component scores (each in [0, 100]) into a single **Dynasty
Value** — the headline metric for player worth.

## Interface

```python
def combine(components: pd.DataFrame, method: str = "uniform_weighted_sum") -> pd.Series:
    """Combine the 6 component columns into a Dynasty Value Series.

    Required columns in `components`:
        production_value, age_value, team_value, injury_value,
        position_value, intangibles_value
    """
```

The function is **pluggable**: a single keyword arg picks the combination
method so we can change strategies without touching callers.

## Supported methods

| Method | Status | Description |
|---|---|---|
| `uniform_weighted_sum` | **v1 default** | Each component weight = 1/6; output = average of the 6 scores. Interpretable; useful while real components are being filled in. |
| `weighted_sum` | planned | Each component gets a fixed weight; weights sum to 1. Human-set weights based on judgment. |
| `multiplicative` | planned | Each component scaled to [0.5, 1.5] multiplier; output = product × Production base. Production-centric. |
| `learned` | planned | Weights learned to fit an objective. Objective TBD (actual production / user rankings / league outcomes / teacher signal). User leans toward this for the final method. |
| `hybrid` | planned | Production + Age (weighted sum) form the base; Team / Injury / Position / Intangibles emit multipliers. |

## Decision criteria (for Phase 5 workshop)

Once Phases 1–4 produce real component scores, we'll evaluate the candidate
methods on:

1. **Interpretability** — can the user explain a player's Dynasty Value to
   themselves in plain English?
2. **Sensible disagreements with the old engine** — where v2 differs from
   the legacy engine, are the differences defensible?
3. **Robustness** — do small weight changes produce small ranking changes?
   (sensitivity sweep — same tooling as
   [`src/research/value_sensitivity.py`](../../src/research/value_sensitivity.py))
4. **Fit to user gut take** — for a basket of named players, does the v2
   ranking generally match the user's intuitive ranking?

## Version history

| Date | Method | Notes |
|---|---|---|
| 2026-06-27 | `uniform_weighted_sum` | Phase 0 placeholder; all components currently neutral 50 so this trivially returns 50 for everyone |

(Append new rows here as the method evolves.)
