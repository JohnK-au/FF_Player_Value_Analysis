"""Combination functions — fold component scores into intermediate + final values.

Two distinct combine operations:

1. **On-Field Value** (Phase 1D): Production x Team multiplicative sub-value.
   "What the player is delivering on the field given their environment."
   Position-aware multiplier band; WR locked at [0.875, 1.125] (see
   docs/methodology/combination.md for the rationale + user override).

2. **Dynasty Value** (Phase 5 final): folds On-Field Value + the 4 off-field
   components (Age / Injury / Position / Intangibles). Pluggable interface.
   Current default: `ofv_weighted_sum` — weighted sum prioritising
   On-Field Value. Set after user feedback that the prior uniform-1/6
   combine was over-diluting OFV.
"""
from __future__ import annotations

import pandas as pd

# Original 6 raw component cols (kept for back-compat with uniform_weighted_sum).
COMPONENT_COLS = (
    "production_value",
    "age_value",
    "team_value",
    "injury_value",
    "position_value",
    "intangibles_value",
)

# OFV-centric combine inputs: On-Field Value REPLACES the raw Production +
# Team columns (it already encodes their multiplicative combination via the
# Phase 1D multiplier).
OFV_COMBINE_COLS = (
    "on_field_value",
    "age_value",
    "injury_value",
    "position_value",
    "intangibles_value",
)

# Default OFV-weighted-sum weights. TODO Phase 5: empirically tune.
DEFAULT_OFV_WEIGHTS: dict[str, float] = {
    "on_field_value": 0.55,
    "age_value": 0.20,
    "injury_value": 0.15,
    "position_value": 0.05,
    "intangibles_value": 0.05,
}

# Position-aware Production x Team multiplier bands (for on_field_value).
# WR locked Phase 1D; other positions default to pass-through.
MULTIPLIER_BANDS: dict[str, tuple[float, float]] = {
    "WR": (0.875, 1.125),  # USER OVERRIDE of data-driven [0.92, 1.08] -- see combination.md
    # RB Phase 2: data-driven empirical effect ~22% (max worst-best PPG swing ~1.85 PPG
    # on ~8.5 PPG mean RB). Set wider than WR to reflect the stronger empirical effect.
    "RB": (0.85, 1.15),
    # TE Phase 3: TIGHTER than WR per user. Residual regression OOF R^2 only
    # 0.23% -- 5x weaker than WR's 1.18%. The relative PPG effect is comparable
    # (~25% on a 5-7 PPG TE) but the SIGNAL is much weaker, so band reflects
    # signal strength rather than effect magnitude.
    "TE": (0.90, 1.10),
    "QB": (1.0, 1.0),      # Phase 4
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
    """Per-player On-Field Value = Production x Team multiplier."""
    multiplier = pd.Series(
        [_team_multiplier(tv, pos) for tv, pos in zip(team_value, position)],
        index=production_value.index,
    )
    return (production_value * multiplier).round(2)


def uniform_weighted_sum(components: pd.DataFrame) -> pd.Series:
    """Average of the 6 raw component scores. Each weight = 1/6. Legacy."""
    return components[list(COMPONENT_COLS)].mean(axis=1)


def ofv_weighted_sum(
    components: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """OFV-centric weighted sum (default).

    Uses On-Field Value (which already encodes Production x Team) plus the 4
    off-field components (Age, Injury, Position, Intangibles). Default weights
    prioritise OFV (0.55) so it isn't diluted by stub / lower-signal components.
    """
    w = weights or DEFAULT_OFV_WEIGHTS
    total = sum(w.values())
    if abs(total - 1.0) > 1e-6:
        # Auto-normalise so weights always sum to 1
        w = {k: v / total for k, v in w.items()}
    return sum(components[col] * weight for col, weight in w.items())


_METHODS = {
    "uniform_weighted_sum": (uniform_weighted_sum, COMPONENT_COLS),
    "ofv_weighted_sum": (ofv_weighted_sum, OFV_COMBINE_COLS),
}

DEFAULT_METHOD = "ofv_weighted_sum"


def combine(components: pd.DataFrame, method: str = DEFAULT_METHOD) -> pd.Series:
    """Combine component columns into a Dynasty Value Series in [0, 100]."""
    if method not in _METHODS:
        raise ValueError(f"unknown combination method {method!r}; known: {list(_METHODS)}")
    fn, required = _METHODS[method]
    missing = [c for c in required if c not in components.columns]
    if missing:
        raise ValueError(f"combine() method={method!r} missing component columns: {missing}")
    return fn(components)
