"""Injury component — durability / floor risk in [0, 100].

Per the methodology (docs/methodology/injury.md), this captures the risk that
a player misses games or underperforms due to durability issues.

**v1 (this phase, deferred to Phase 1):** use the existing
`weekly_consistency` (downside deviation of weekly fantasy points below the
player's own mean — from `src/models/production.py`) as a proxy. Players whose
weekly distribution has high downside (frequent zero-or-negative weeks) get a
lower injury score. Limitation: this is a *consistency* signal, not an
*injury history* signal — it picks up bust weeks for any reason, not strictly
injury.

**Later:** integrate a real injury data source (nflverse `import_injuries`
exposes IR placements + designations) for a per-player games-missed history
and injury-severity profile. The score interface is unchanged; the inputs
get richer.

Phase 0 (foundation): returns neutral 50.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Injury score per player in [0, 100]. Phase 0: neutral stub."""
    out = players.copy()
    out["injury_value"] = NEUTRAL
    return out
