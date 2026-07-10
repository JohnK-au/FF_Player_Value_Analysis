"""Fantasy point scoring for our ESPN dynasty league.

Canonical 2025 rules, regression-derived from full 2025 regular-season box
scores (R^2 >= 0.993 across 1,849 player-weeks). Used to reconstruct fantasy
points for seasons where our ESPN league either didn't exist or we weren't in
it (pre-2022), enabling the extended training frame for the v2 component
framework.

Design choice (confirmed with the manager): **2025 rules apply uniformly across
all seasons** — past raw stats are re-scored as if our current scoring had
always been in place. The model is projecting under today's rules; consistency
of target is more valuable than reconstructing historical league results.

Three yardage rates (receiving_yards, rushing_yards, targets) carry a small
residual variance (~0.7% of variance unexplained); likely a hidden sub-stat
bundled into the receivingYards points entry (possibly YAC). Acceptable for
training-data purposes — worst-case per-player error ~5%, washes out across
seasonal aggregations.

Bonuses (40+/50+ yd TDs, 200+ yd rushing game, 400+ yd passing game) and the
per-game team-win/team-loss penalty are NOT included in v1. They contribute
0-3 pts per player per season for skill players; if validation MAE is too
high we'll add them by pulling pbp data.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

# Canonical 2025 league scoring rules. Stat names match nflverse seasonal-data
# columns where they exist (passing_yards, rushing_yards, receiving_yards,
# completions, attempts, carries, targets, receptions, sacks, etc.).
LEAGUE_SCORING_RULES_2025: dict[str, float] = {
    # passing (QB)
    "passing_yards": 0.18,
    "passing_tds": 6.0,
    "passing_interceptions": -2.0,
    "completions": 0.35,
    "attempts": -1.0,
    "passing_incompletions": -0.66,
    "sacks": -1.0,
    "passing_2pt_conversions": 2.0,
    # rushing
    "rushing_yards": 0.329,
    "rushing_tds": 6.0,
    "carries": -1.0,
    "rushing_2pt_conversions": 2.0,
    # receiving
    "receiving_yards": 0.236,
    "receiving_tds": 6.0,
    "receptions": 2.0,
    "targets": -1.67,
    "receiving_2pt_conversions": 2.0,
    # turnovers
    "fumbles": -1.0,
    "fumbles_lost": -1.0,  # stacks with fumbles
}


def compute_fantasy_points(
    stats: Mapping[str, float] | pd.Series,
    rules: Mapping[str, float] = LEAGUE_SCORING_RULES_2025,
) -> float:
    """Compute total fantasy points for one player-season (or any aggregation)."""
    total = 0.0
    for stat, rate in rules.items():
        val = stats.get(stat, 0) if hasattr(stats, "get") else 0
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        total += rate * float(val)
    return total


def compute_fantasy_points_frame(
    df: pd.DataFrame,
    rules: Mapping[str, float] = LEAGUE_SCORING_RULES_2025,
) -> pd.Series:
    """Vectorised: returns a Series of fantasy points, one per row.

    Missing columns are treated as 0 (no points contribution).
    """
    pts = pd.Series(0.0, index=df.index)
    for stat, rate in rules.items():
        if stat in df.columns:
            col = pd.to_numeric(df[stat], errors="coerce").fillna(0.0)
            pts = pts + rate * col
    return pts


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def _harmonize_nflverse_seasonal(sd: pd.DataFrame) -> pd.DataFrame:
    """Map nflverse seasonal_data columns to our compute_fantasy_points schema.

    Handles: pass-incompletions derivation, the `interceptions` -> `passing_interceptions`
    rename, and rolling up split fumble columns into total + lost.
    """
    sd = sd.copy()
    if "passing_incompletions" not in sd.columns:
        sd["passing_incompletions"] = sd.get("attempts", 0) - sd.get("completions", 0)
    if "passing_interceptions" not in sd.columns and "interceptions" in sd.columns:
        sd["passing_interceptions"] = sd["interceptions"]
    fumble_cols = [c for c in ("sack_fumbles", "rushing_fumbles", "receiving_fumbles") if c in sd.columns]
    if fumble_cols and "fumbles" not in sd.columns:
        sd["fumbles"] = sd[fumble_cols].fillna(0).sum(axis=1)
    fumble_lost_cols = [c for c in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost") if c in sd.columns]
    if fumble_lost_cols and "fumbles_lost" not in sd.columns:
        sd["fumbles_lost"] = sd[fumble_lost_cols].fillna(0).sum(axis=1)
    return sd


def nflverse_season_production(season: int) -> pd.DataFrame:
    """NFL-wide per-player PPG for ``season`` reconstructed from nflverse stats + 2025 rules.

    The production basis for the v2 extended training frame. Same schema as the
    existing ``population.season_production`` so downstream code (advanced merge,
    draft/combine join, etc.) is unchanged: columns [espn_id, name,
    position_group, points, ppg, games].

    Restricted to skill positions (QB/RB/WR/TE). Only players with games > 0 and
    a usable espn_id are returned (drops UDFAs without espn coverage and roster-
    only entries who didn't appear in a game).
    """
    import nfl_data_py as nfl

    sd = _harmonize_nflverse_seasonal(nfl.import_seasonal_data([season]))
    sd["points"] = compute_fantasy_points_frame(sd)
    sd["games"] = pd.to_numeric(sd.get("games", 0), errors="coerce").fillna(0)
    sd = sd[sd["games"] > 0].copy()
    sd["ppg"] = (sd["points"] / sd["games"]).round(2)
    sd["points"] = sd["points"].round(2)

    rost = nfl.import_seasonal_rosters([season])[["player_id", "espn_id", "position", "player_name"]]
    rost["espn_id"] = pd.to_numeric(rost["espn_id"], errors="coerce").astype("Int64")
    rost = rost.dropna(subset=["espn_id"]).drop_duplicates("player_id")
    sd = sd.merge(rost, on="player_id", how="left")
    sd = sd.dropna(subset=["espn_id"]).drop_duplicates("espn_id")
    sd = sd[sd["position"].isin(SKILL_POSITIONS)].copy()
    sd["position_group"] = sd["position"]
    return sd[["espn_id", "player_name", "position_group", "points", "ppg", "games"]].rename(
        columns={"player_name": "name"}
    ).reset_index(drop=True)


def _validate(season: int = 2024) -> None:
    """Compute PPG for ``season`` from nflverse + 2025 rules; compare to ESPN-reported.

    Uses ``season`` = 2024 because nflverse hasn't published 2025 seasonal_data
    as of June 2026. The reconstruction function is what we use for historical
    (2016-2021) years.

    Pass criteria (rough): MAE < ~2 PPG, median abs error < 1 PPG, no systemic
    position bias > 1 PPG.
    """
    print(f"Reconstructing {season} PPG from nflverse seasonal_data + 2025 rules...")
    sd = nflverse_season_production(season)
    sd = sd.rename(columns={"ppg": "computed_ppg", "points": "computed_pts"})

    espn = pd.read_csv("data/processed/performance.csv")
    espn = espn[espn["season"] == season].copy()
    espn["espn_id"] = pd.to_numeric(espn["espn_id"], errors="coerce").astype("Int64")
    espn["espn_ppg"] = pd.to_numeric(espn["ppg"], errors="coerce")
    espn = espn.dropna(subset=["espn_id", "espn_ppg"])[["espn_id", "espn_ppg"]]

    m = sd.merge(espn, on="espn_id", how="inner")
    m["abs_err"] = (m["computed_ppg"] - m["espn_ppg"]).abs()

    print(f"\nMatched {len(m)} skill players for {season}.")
    print(f"  Mean computed PPG: {m['computed_ppg'].mean():.2f}")
    print(f"  Mean ESPN PPG:     {m['espn_ppg'].mean():.2f}")
    print(f"  MAE (abs err):     {m['abs_err'].mean():.2f}")
    print(f"  Median abs err:    {m['abs_err'].median():.2f}")
    print(f"  P90 abs err:       {m['abs_err'].quantile(0.9):.2f}")

    print("\nBy position:")
    by_pos = m.groupby("position_group").agg(
        n=("espn_id", "count"),
        mae=("abs_err", "mean"),
        med_err=("abs_err", "median"),
        mean_computed=("computed_ppg", "mean"),
        mean_espn=("espn_ppg", "mean"),
    ).round(2)
    print(by_pos)


if __name__ == "__main__":
    _validate(2024)
