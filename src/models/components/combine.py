"""Combination functions — fold component scores into intermediate + final values.

Two distinct combine operations:

1. **On-Field Value** (Phase 1D): Production x Team multiplicative sub-value.
   "What the player is delivering on the field given their environment."
   Position-aware multiplier band; WR locked at [0.875, 1.125] (see
   docs/methodology/combination.md for the rationale + user override).

2. **Dynasty Value** (Phase 5 final): folds all 6 component scores. Pluggable
   interface; v1 default = uniform_weighted_sum. Final method workshopped in
   Phase 5 after all 4 positions have full component scores.
"""
from __future__ import annotations

import pandas as pd

COMPONENT_COLS = (
    "production_value",
    "age_value",
    "team_value",
    "injury_value",
    "position_value",
    "intangibles_value",
)

# Position-aware Production x Team multiplier bands.
# Each is (multiplier_lo, multiplier_hi) applied as:
#   multiplier = lo + (team_value / 100) * (hi - lo)
# Only WR has a tuned band (Phase 1D). Other positions default to (1.0, 1.0) =
# pass-through (no team effect) until their components land in Phases 2-4.
MULTIPLIER_BANDS: dict[str, tuple[float, float]] = {
    "WR": (0.875, 1.125),  # USER OVERRIDE of data-driven [0.92, 1.08] -- see combination.md
    "RB": (1.0, 1.0),      # Phase 2
    "QB": (1.0, 1.0),      # Phase 3
    "TE": (1.0, 1.0),      # Phase 4
}


def _team_multiplier(team_value: float, position: str) -> float:
    lo, hi = MULTIPLIER_BANDS.get(position, (1.0, 1.0))
    if hi == lo:
        return lo
    return lo + (team_value / 100.0) * (hi - lo)


def on_field_value(
    production_value: pd.Series,
    team_value: pd.Series,
    position: pd.Series,
) -> pd.Series:
    """Per-player On-Field Value = Production x Team multiplier.

    Captures "what the player delivers on the field given their environment."
    Phase 1D for WR; other positions pass through (multiplier = 1.0) until
    Phases 2-4 add their bands.
    """
    multiplier = pd.Series(
        [_team_multiplier(tv, pos) for tv, pos in zip(team_value, position)],
        index=production_value.index,
    )
    return (production_value * multiplier).round(2)


def uniform_weighted_sum(components: pd.DataFrame) -> pd.Series:
    """Average of the 6 component scores. Each weight = 1/6."""
    return components[list(COMPONENT_COLS)].mean(axis=1)


_METHODS = {
    "uniform_weighted_sum": uniform_weighted_sum,
}


def combine(components: pd.DataFrame, method: str = "uniform_weighted_sum") -> pd.Series:
    """Combine the 6 component columns into a Dynasty Value Series in [0, 100]."""
    missing = [c for c in COMPONENT_COLS if c not in components.columns]
    if missing:
        raise ValueError(f"combine() missing component columns: {missing}")
    if method not in _METHODS:
        raise ValueError(f"unknown combination method {method!r}; known: {list(_METHODS)}")
    return _METHODS[method](components)
