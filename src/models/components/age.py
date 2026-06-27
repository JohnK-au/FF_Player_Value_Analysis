"""Age component — positional age curve scoring in [0, 100].

Per the methodology (docs/methodology/age.md), age value is derived from a
**position-specific** empirical age curve fit on historical NFL production
data. The curve identifies:

  * **Prime onset** — the age range a player typically reaches their best
  * **Prime length** — how long that peak typically lasts at this position
  * **Regression onset** — the age decline typically begins
  * **Elite-aging tail** — pattern detection for outliers (Brady, Rice, Hopkins)
    whose primes last longer than the positional median

Scoring intent: a player in mid-prime gets ~100; clearly past their typical
decline gets low; rookies near the typical prime-onset age get a moderate
default; the elite-aging detector lifts the score for established elites at
ages where the positional curve says regression should have started.

Phase 0 (foundation): returns neutral 50. Phase 1 (WR) fits the WR age curve
from historical nflverse data and computes real scores.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Age score per player in [0, 100]. Phase 0: neutral stub."""
    out = players.copy()
    out["age_value"] = NEUTRAL
    return out
