"""Position component — cross-position scarcity / importance in [0, 100].

Each player's ``position_value`` is a **constant per position group**, shared
across all players at that position. The constant captures how scarce / valuable
that position is in our specific 8-team dynasty league context. Within-position
differentiation lives in Production / Team / On-Field Value.

Phase 4.5 v2 (current): VORP-Deep Total Impact -- T = M x S where M is the
PPG gap between elite (top N=8*S_p players) and the average of a "deep FA tier"
(ranks 3N+1 to 4N). Z-scored across positions, min-max normalised to [0, 100].
See docs/methodology/position.md for the derivation + comparison vs other
variants. Re-derive via _scratch_vorp_tiers.py-style analysis whenever the
master CSV's player pool meaningfully changes (e.g., post-FA sweep,
post-draft, new season).

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

# Phase 4.5 v2 position scores. Computed via VORP-Deep Total Impact:
#   For each position p:
#     elite_avg_PPG  = mean PPG of top (8 * S_p) players (1 starter slot per team)
#     replacement   = mean PPG of ranks (3 * 8 * S_p + 1) to (4 * 8 * S_p)
#                     (i.e. the "deep FA tier" -- realistic worst-case replacement
#                      when raiding the FA pool)
#     M_ppg         = elite_avg_PPG - replacement
#     T_ppg         = M_ppg * S_p     (total team-success PPG advantage)
#     z-score T across positions, min-max normalize to [0, 100]
#
# Re-derived 2026-06-28 from 2025 PPG (games>=4) across the 490 priced players
# in master CSV. Replaces v1 (OFV-based equal-weighted composite) which was
# normalized within position, hiding cross-position absolute PPG differences.
#
# Reads: in a 1-QB league with deep pool, QB has the SMALLEST PPG gap and only
# 1 slot/team -> lowest positional advantage. RB and WR are nearly tied at the
# top (deep-FA T values of 33.7 and 32.3 PPG/week respectively) because both
# combine meaningful elite-vs-deep-FA gaps with multiple slot counts. TE
# stays low (T_deep = 14.6) because TE production is genuinely flatter
# across the position pool.
POSITION_SCORES: dict[str, float] = {
    "QB": 0.0,
    "RB": 100.0,
    "WR": 93.1,
    "TE": 8.6,
}


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Position score per player in [0, 100]. Constant per position."""
    out = players.copy()
    out["position_value"] = POSITION_SCORES.get(position, NEUTRAL)
    return out
