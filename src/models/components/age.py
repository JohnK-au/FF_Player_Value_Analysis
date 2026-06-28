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
    # RB Phase 2: empirically derived from 3-yr forward cumulative PPG.
    # RB cliff is sharper and earlier than WR: median future-3yr drops from
    # 16 (age 22) -> 4 (age 26) -> 0 (age 30+). Grid-search best fit was
    # center=25, steepness=2.5. Validation: Pearson +0.32, OOF R^2 0.10
    # (3x WR's age signal). See docs/methodology/age.md.
    "RB": {"center": 25.0, "steepness": 2.5},
    # TE Phase 3: hand-tuned. The empirical median-PPG-by-age curve has a
    # U-shape (high young -> valley at 24 -> recovery 26-27 -> decline 28+)
    # which a monotonic sigmoid can't fit cleanly; user judged the dip a
    # coincidence and chose hand-tuned smoother decline reflecting TE prime
    # extending to ~28 (Kelce-era TEs, elite-aging tail).
    "TE": {"center": 27.0, "steepness": 3.0},
    # QB Phase 4: hand-tuned. QB careers extend latest of any position
    # (Brady 45, Brees 41, Manning 39, Rodgers 41). The empirical age curve
    # has wild noise from elite-aging extremes; user chose smoother decline
    # centered later. Steepness=4 gives a gentle cliff (longer prime than
    # WR/RB/TE).
    "QB": {"center": 33.0, "steepness": 4.0},
}


def _logistic_age_value(age: float, center: float, steepness: float) -> float:
    """Logistic decay: ~100 at young ages, 50 at center, ~0 at old ages."""
    if pd.isna(age):
        return NEUTRAL
    return 100.0 / (1.0 + math.exp((float(age) - center) / steepness))


def _score_position(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Age scoring for one position using its tuned sigmoid."""
    out = players.copy()
    params = AGE_PARAMS[position]
    out["age_value"] = [
        _logistic_age_value(a, params["center"], params["steepness"])
        for a in out["age"]
    ]
    return out


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Age score per player in [0, 100]."""
    if position in AGE_PARAMS:
        return _score_position(players, position)
    out = players.copy()
    out["age_value"] = NEUTRAL  # QB / TE in Phases 3-4
    return out


if __name__ == "__main__":
    for position, params in AGE_PARAMS.items():
        print(f"\n{position} Age curve (sigmoid center={params['center']}, steepness={params['steepness']}):")
        for a in range(20, 38):
            v = _logistic_age_value(a, params["center"], params["steepness"])
            bar = "#" * int(v / 2)
            print(f"  age {a}: {v:6.2f}  {bar}")
