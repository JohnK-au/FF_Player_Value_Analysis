"""Position component — positional scarcity in [0, 100].

Per the methodology (docs/methodology/position.md), position value captures
how much a player's *position* contributes to their dynasty value, driven by:

  * **Replacement gap** — how big is the drop-off from a starter at this
    position to a replacement (uses `replacement_levels` from
    `src/models/production.py`)
  * **Elite-tier concentration** — when a position has very few elite
    producers (3-4) vs. the field, those elites are *more* valuable because
    you must start someone at that position. Formula TBD (likely a
    coefficient of variation of the top-N production distribution).
  * **Roster-slot demand** — how many starters of this position the league
    lineup requires (rules §4: 1 QB, 2 RB, 2 WR + WR/TE flex, 1 TE,
    RB/WR/TE flex). Higher required starters → more competition for the
    elite tier → higher value at that position.

Phase 0 (foundation): returns neutral 50. Phase 1 (WR) computes real WR
positional scarcity using replacement + elite-tier + roster demand.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Position score per player in [0, 100]. Phase 0: neutral stub."""
    out = players.copy()
    out["position_value"] = NEUTRAL
    return out
