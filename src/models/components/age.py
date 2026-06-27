"""Age component — dynasty-horizon scoring in [0, 100].

Captures **years of productive runway ahead** for a multi-year contract, not
where the player sits on the typical performance curve. A 22-year-old beats a
29-year-old on Age even when the 29-year-old is in their prime — because the
22-year-old has more remaining productive seasons under a multi-year deal.
Current performance level is captured separately by the Production component.

WR scoring: logistic decay centered on the typical decline age. Smooth
monotonic in age (younger = higher).
"""
from __future__ import annotations

import math

import pandas as pd

NEUTRAL = 50.0

# Sigmoid parameters per position: 100 / (1 + exp((age - center) / steepness))
# center = age at which age_value = 50 (decline is 50/50)
# steepness = how sharply the curve drops
#
# WR locked v1: center 28, steepness 2. Hand-parameterized per dynasty
# intuition; survivorship bias in WR median PPG curve makes the naive
# empirical curve uninformative (31+ year-old WRs in the data are
# selection-biased toward elites). Re-tune empirically once we have a
# cleaner age-survival analysis.
AGE_PARAMS: dict[str, dict[str, float]] = {
    "WR": {"center": 28.0, "steepness": 2.0},
    # RB / QB / TE land in Phases 2-4 with their own tuned parameters.
}


def _logistic_age_value(age: float, center: float, steepness: float) -> float:
    """Logistic decay: ~100 at young ages, 50 at center, ~0 at old ages."""
    if pd.isna(age):
        return NEUTRAL
    return 100.0 / (1.0 + math.exp((float(age) - center) / steepness))


def _score_wr(players: pd.DataFrame) -> pd.DataFrame:
    """WR Age scoring: sigmoid centered at age 28."""
    out = players.copy()
    params = AGE_PARAMS["WR"]
    out["age_value"] = [
        _logistic_age_value(a, params["center"], params["steepness"])
        for a in out["age"]
    ]
    return out


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Age score per player in [0, 100]."""
    if position == "WR":
        return _score_wr(players)
    out = players.copy()
    out["age_value"] = NEUTRAL  # RB/QB/TE in Phases 2-4
    return out


if __name__ == "__main__":
    # Smoke test: print the age curve
    params = AGE_PARAMS["WR"]
    print("WR Age curve (sigmoid center=28, steepness=2):")
    for a in range(20, 38):
        v = _logistic_age_value(a, params["center"], params["steepness"])
        bar = "#" * int(v / 2)
        print(f"  age {a}: {v:6.2f}  {bar}")
