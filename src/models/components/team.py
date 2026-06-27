"""Team component — offensive environment scoring in [0, 100].

Per the methodology (docs/methodology/team.md), team value captures the
**quality of the offense the player operates within**, weighted differently
by position:

  * **WR / TE**: QB efficiency (team_pass_epa) + accuracy (team_cpoe) + pass rate
  * **RB**: rushing-game efficiency (team_rush_epa) + offensive line proxy +
    game script (pass rate, point total)
  * **QB**: supporting cast quality (receiver separation, O-line pressure
    allowed, run-game efficiency)

A great player on a bad offense is limited; a mediocre player on a great
offense gets a boost.

Phase 0 (foundation): returns neutral 50. Phase 1 (WR) wires the
team_pass_epa / team_cpoe weighting and produces real scores. Future
enhancement: standardised subjective overrides (e.g. "Packers play from the
lead and primarily run") — deferred.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Team score per player in [0, 100]. Phase 0: neutral stub."""
    out = players.copy()
    out["team_value"] = NEUTRAL
    return out
