"""Pull per-player season fantasy points (in our league's scoring) from ESPN.

For each season we read the whole player pool (rostered + free agents) and take
ESPN's already-scored season total and per-game average — so points reflect this
league's custom scoring. Output is one row per (season, player), cached to
``data/processed/performance.csv`` for the value model.

Weekly granularity (box scores, for consistency metrics) is deferred.
"""
from __future__ import annotations

import pandas as pd

from .contracts import PROCESSED_DIR
from .espn import get_league

PERFORMANCE_SEASONS = [2022, 2023, 2024, 2025]


def season_player_points(year: int, fa_size: int = 1500) -> pd.DataFrame:
    """One row per player for ``year``: season points, PPG, games, position."""
    lg = get_league(year)
    seen: set = set()
    rows: list[dict] = []

    def add(p) -> None:
        if p.playerId in seen:
            return
        seen.add(p.playerId)
        pts = getattr(p, "total_points", None)
        ppg = getattr(p, "avg_points", None)
        games = int(round(pts / ppg)) if (pts and ppg and ppg > 0) else 0
        rows.append(
            {
                "season": year,
                "espn_id": p.playerId,
                "name": p.name,
                "position": getattr(p, "position", None),
                "pro_team": getattr(p, "proTeam", None),
                "points": pts,
                "ppg": ppg,
                "games": games,
            }
        )

    for team in lg.teams:
        for p in team.roster:
            add(p)
    try:
        for p in lg.free_agents(size=fa_size):
            add(p)
    except Exception as e:  # free-agent pull is best-effort
        print(f"(free-agent pull failed for {year}: {type(e).__name__}: {str(e)[:80]})")

    return pd.DataFrame(rows)


def weekly_player_points(year: int, weeks: int | None = None) -> pd.DataFrame:
    """Per-player per-week points from box scores, over the league's regular season.

    Defaults to weeks 1..``reg_season_count`` (the fantasy regular season — ~13
    weeks here), so production excludes NFL weeks after the league ends. Covers
    players who were on a roster (starter or bench) that week.
    """
    lg = get_league(year)
    n = weeks or lg.settings.reg_season_count
    rows: list[dict] = []
    for wk in range(1, n + 1):
        for matchup in lg.box_scores(wk):
            for bp in list(matchup.home_lineup) + list(matchup.away_lineup):
                slot = getattr(bp, "slot_position", None)
                rows.append(
                    {
                        "season": year,
                        "week": wk,
                        "espn_id": bp.playerId,
                        "name": bp.name,
                        "position": getattr(bp, "position", None),
                        "slot": slot,
                        "points": bp.points,
                        "started": slot not in ("BE", "IR", None),
                    }
                )
    return pd.DataFrame(rows)


def fantasy_season_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly points to per-player fantasy-season stats (total, PPG,
    games, consistency)."""
    played = weekly[weekly["points"].notna()].copy()
    played["is_game"] = played["points"] != 0
    g = played.groupby(["season", "espn_id", "name"]).agg(
        fpts=("points", "sum"),
        games=("is_game", "sum"),
        weeks=("week", "nunique"),
        ppg=("points", lambda s: s[s != 0].mean()),
        stdev=("points", lambda s: s[s != 0].std()),
    ).reset_index()
    g["ppg"] = g["ppg"].round(2)
    g["stdev"] = g["stdev"].round(2)
    return g


def pull_performance(seasons: list[int] | None = None) -> pd.DataFrame:
    """Pull and concatenate per-player season points across ``seasons``."""
    seasons = seasons or PERFORMANCE_SEASONS
    frames = [season_player_points(y) for y in seasons]
    return pd.concat(frames, ignore_index=True)


def save_performance(df: pd.DataFrame | None = None) -> "pd.Path":
    df = pull_performance() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "performance.csv"
    df.to_csv(out, index=False)
    return out


def pull_weekly_performance(seasons: list[int] | None = None) -> pd.DataFrame:
    """Per-player per-week points over each season's regular season."""
    seasons = seasons or PERFORMANCE_SEASONS
    return pd.concat([weekly_player_points(y) for y in seasons], ignore_index=True)


def save_weekly_performance(df: pd.DataFrame | None = None) -> "pd.Path":
    df = pull_weekly_performance() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "performance_weekly.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    perf = pull_performance()
    print(f"Pulled {len(perf)} player-seasons across {perf['season'].nunique()} seasons.")
    print("\nrows per season:")
    print(perf.groupby("season").agg(players=("espn_id", "nunique"),
                                      scored=("points", lambda s: (s > 0).sum())).to_string())
    print("\nTop scorers 2025:")
    top = perf[perf.season == 2025].nlargest(8, "points")
    print(top[["name", "position", "points", "ppg", "games"]].to_string(index=False))
    path = save_performance(perf)
    print(f"\nSaved -> {path}")

    print("\nPulling weekly box scores (regular season)…")
    wk = pull_weekly_performance()
    print(f"weekly rows: {len(wk)} across seasons {sorted(wk['season'].unique())}; "
          f"{wk['espn_id'].nunique()} players")
    wpath = save_weekly_performance(wk)
    print(f"Saved -> {wpath}")
