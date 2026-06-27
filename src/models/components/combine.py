"""Combination function — fold 6 component scores into Dynasty Value.

Pluggable interface; v1 default = uniform weighted sum. See
docs/methodology/combination.md for the version history and the candidate
methods we'll workshop in Phase 5.
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
