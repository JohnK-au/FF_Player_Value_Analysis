"""Injury component — durability / floor risk in [0, 100].

Two pieces:

1. **Chronic availability** — weighted average of games-played-as-fraction-of-
   season over the last 3 seasons, with most-recent weighted heaviest (decay
   weights [1.0, 0.5, 0.25]). High availability → high score.

2. **Acute recovery penalty** — for major absences specifically in the latest
   completed season. Threshold-based: no penalty for games_missed_last <= 4
   (normal NFL wear-and-tear), then 3 points-of-injury-score per excess game
   missed. Empirically supported by ~ −0.53 PPG-per-game-missed slope on
   next-season production from 1,000 WR transition pairs.

`injury_value = clip(0, 100, availability − recovery_penalty)`.

Rookies and new arrivals (no NFL history within the window) → neutral 50.
Veterans who missed an ENTIRE season are treated as games=0 for that season
(injury assumption) so they get hit by both the availability drop and the
recovery penalty.

Real injury data (nflverse `import_injuries()` — IR placements + severity) is
a v2 follow-up; this games-played proxy is the v1 read.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pandas as pd

from src.data.population import extended_training_frame

NEUTRAL = 50.0

# Mirror production/team CURRENT_SEASON — latest fully-completed NFL season.
LATEST_SEASON = 2025

# Most-recent-first weights applied across the last 3 seasons of NFL history.
AVAILABILITY_WEIGHTS = (1.0, 0.5, 0.25)

# Recovery penalty (TODO: re-tune empirically later).
# Threshold is shared across positions (4 games = normal NFL wear-and-tear);
# K varies per user judgment that some positions are more injury-tolerant.
RECOVERY_THRESHOLD = 4  # games_missed_last_season this many or fewer -> no penalty
RECOVERY_K_BY_POSITION: dict[str, float] = {
    "WR": 3.0,
    "RB": 3.0,
    "TE": 2.5,  # user: TE injury impact slightly less than WR/RB
    "QB": 3.0,  # Phase 4 placeholder
}
RECOVERY_K = 3.0  # legacy default (back-compat); position-specific lookup preferred


def _max_games(season: int) -> int:
    """17 from 2021+, 16 prior."""
    return 16 if season <= 2020 else 17


@lru_cache(maxsize=1)
def _player_games() -> tuple[dict, dict]:
    """Cached: ({(espn_id, season): games}, {espn_id: set[season]}).

    Built once per Python process from the extended training frame. All
    subsequent lookups are O(1) dict access.
    """
    ext = extended_training_frame()
    sub = ext.dropna(subset=["espn_id", "games"])
    games_lookup: dict[tuple[int, int], int] = {}
    seasons_per_player: dict[int, set[int]] = defaultdict(set)
    for _, r in sub.iterrows():
        eid = int(r["espn_id"])
        s = int(r["season"])
        games_lookup[(eid, s)] = int(r["games"])
        seasons_per_player[eid].add(s)
    return games_lookup, dict(seasons_per_player)


def _lookup_injury_value(espn_id, position: str = "WR",
                         latest_season: int = LATEST_SEASON) -> float:
    """Per-player injury_value [0, 100]; NEUTRAL when player has no NFL history.

    ``position`` controls the recovery-penalty K (RECOVERY_K_BY_POSITION).
    """
    if pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)
    games_lookup, seasons_per_player = _player_games()

    player_seasons = seasons_per_player.get(espn_id, set())
    if not player_seasons:
        return NEUTRAL  # no NFL history at all -> rookie / never played
    first_nfl_season = min(player_seasons)
    if first_nfl_season > latest_season:
        return NEUTRAL

    seasons_data: list[tuple[int, int, float]] = []
    for offset, weight in enumerate(AVAILABILITY_WEIGHTS):
        s = latest_season - offset
        if s < first_nfl_season:
            break
        games = games_lookup.get((espn_id, s), 0)
        seasons_data.append((s, games, weight))

    if not seasons_data:
        return NEUTRAL

    weighted_games = sum(g * w for _, g, w in seasons_data)
    weighted_max = sum(_max_games(s) * w for s, _, w in seasons_data)
    if weighted_max == 0:
        return NEUTRAL
    availability = 100.0 * weighted_games / weighted_max

    latest_entry = next(((s, g) for s, g, _ in seasons_data if s == latest_season), None)
    k = RECOVERY_K_BY_POSITION.get(position, RECOVERY_K)
    if latest_entry is None:
        recovery_penalty = 0.0
    else:
        s, latest_games = latest_entry
        games_missed = max(0, _max_games(s) - latest_games)
        recovery_penalty = max(0.0, k * (games_missed - RECOVERY_THRESHOLD))

    return float(max(0.0, min(100.0, availability - recovery_penalty)))


SCORED_POSITIONS: tuple[str, ...] = ("WR", "RB", "TE")


def _score_position(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Injury scoring -- same formula across positions, position-specific K."""
    out = players.copy()
    out["injury_value"] = [
        _lookup_injury_value(eid, position, LATEST_SEASON)
        for eid in out["espn_id"]
    ]
    return out


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Injury score per player in [0, 100]."""
    if position in SCORED_POSITIONS:
        return _score_position(players, position)
    out = players.copy()
    out["injury_value"] = NEUTRAL  # QB lands in Phase 4
    return out


if __name__ == "__main__":
    # Smoke test: a handful of known WR cases
    from src.data.cap import player_salaries_2026
    sal = player_salaries_2026()
    wr = sal[sal["position_group"] == "WR"].copy()
    wr["injury_value"] = [_lookup_injury_value(eid, LATEST_SEASON) for eid in wr["espn_id"]]
    wr = wr.sort_values("injury_value", ascending=False)
    print(wr[["player", "team", "age", "injury_value"]].to_string(index=False))
