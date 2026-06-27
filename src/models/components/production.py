"""Production component — past on-field production scored to [0, 100].

Per the methodology (docs/methodology/production.md), production scoring is
**tiered by years of experience**:

  * **Rookies (years_exp == 0)**: NFL draft + combine + (deferred) dynasty
    league draft slot. No prior NFL production to draw on.
  * **Years 1–3 (years_exp in 1..3)**: blend of NFL draft + combine + accumulating
    NFL production history. Draft/combine weight decays as the player accumulates
    real production data.
  * **Years 4+ (years_exp >= 4)**: production history only (typically the
    projected_ppg from the projection model, with rolling history). Draft and
    combine no longer carry signal at this point in a career.

A subjective production-rating override (user-supplied) is supported via the
intangibles_overrides.csv pathway — see docs/methodology/production.md.

Phase 0 (foundation): this module returns a neutral 50 for every player so
the framework runs end-to-end. Phase 1 (WR) and Phases 2–4 (RB/QB/TE) fill
in the real position-specific scoring.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Production score per player in [0, 100]. Phase 0: neutral stub."""
    out = players.copy()
    out["production_value"] = NEUTRAL
    return out
