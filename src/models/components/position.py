"""Position component — cross-position scarcity / importance in [0, 100].

Each player's ``position_value`` is a **constant per position group**, shared
across all players at that position. The constant captures how scarce / valuable
that position is in our specific 8-team dynasty league context. Within-position
differentiation lives in Production / Team / On-Field Value.

Phase 4.5 v1: composite of four sub-metrics, equal-weighted then min-max
normalised across the 4 positions to [0, 100]. See docs/methodology/position.md
for the full derivation. Re-derive via _scratch_position_calc.py-style analysis
whenever the master CSV's player pool meaningfully changes (e.g., post-FA
sweep, post-draft, new season).

This is NOT cap pricing -- the values reflect positional importance for team
success, not market price. Cap-based valuation is a Phase 5+ task.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0

# Hardcoded per-position slot counts. Source: league rules §4 starting lineup
# (1 QB, 2 RB, 2 WR, 1 WR/TE flex, 1 TE, 1 RB/WR/TE flex) plus the estimated
# flex distribution:
#   WR/TE flex   ≈ 62% WR + 38% TE
#   RB/WR/TE flex ≈ 38% RB + 50% WR + 12% TE
# These S_p values sum to 8 = total skill starters per team.
SLOT_COUNTS: dict[str, float] = {
    "QB": 1.0,
    "RB": 2.5,
    "WR": 3.0,
    "TE": 1.5,
}

# Phase 4.5 v1 position scores. Computed via the 4-sub-metric composite
# (Slot Count, Marginal Gap, Total Impact, Supply-Demand Scarcity), z-scored
# across positions with equal weights, min-max normalised to [0, 100].
# Re-derived 2026-06-28 from master CSV with 490 priced players (155 rostered
# + 335 dynasty-league FAs).
#
# Reads: in a 1-QB league with deep pool, QB has the SMALLEST marginal gap
# (elite QB only ~8 OFV above replacement) and just 1 slot/team -> lowest
# positional advantage. RB combines big elite gap (~19 OFV) with 2.5 slots/team
# -> highest positional advantage. WR/TE land in the middle.
POSITION_SCORES: dict[str, float] = {
    "QB": 0.0,
    "RB": 100.0,
    "WR": 35.2,
    "TE": 21.2,
}


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Position score per player in [0, 100]. Constant per position."""
    out = players.copy()
    out["position_value"] = POSITION_SCORES.get(position, NEUTRAL)
    return out
