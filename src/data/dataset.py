"""Unified player dataset: 2026 contract + position + age + recent performance.

Joins each 2026 contract player (salary, position group, age) to their ESPN
fantasy production via ``espn_id``. This is the table the fair-value model
(Phase C) will train on; for now it also supports simple efficiency views.
"""
from __future__ import annotations

import pandas as pd

from .contracts import PROCESSED_DIR


def load_weekly(use_cache: bool = True) -> pd.DataFrame:
    """Load cached per-player per-week points, pulling + caching if missing."""
    path = PROCESSED_DIR / "performance_weekly.csv"
    if use_cache and path.exists():
        return pd.read_csv(path)
    from .performance import pull_weekly_performance, save_weekly_performance

    df = pull_weekly_performance()
    save_weekly_performance(df)
    return df


def build_player_dataset() -> pd.DataFrame:
    """One row per 2026 contract player: salary/age/position + recent production.

    Production is the **fantasy regular season** (weeks 1..reg_season_count), so
    it reflects the weeks that count in this league — not the full NFL season.
    """
    from .cap import player_salaries_2026
    from .performance import fantasy_season_summary

    sal = player_salaries_2026().copy()
    sal["espn_id"] = pd.to_numeric(sal["espn_id"], errors="coerce").astype("Int64")

    summ = fantasy_season_summary(load_weekly())
    summ["espn_id"] = pd.to_numeric(summ["espn_id"], errors="coerce").astype("Int64")

    def season_slice(year: int, cols: list[str]) -> pd.DataFrame:
        s = summ.loc[summ["season"] == year, ["espn_id"] + cols].copy()
        return s.rename(columns={c: f"{c}_{year}" for c in cols})

    df = sal.merge(season_slice(2025, ["fpts", "ppg", "games", "stdev"]), on="espn_id", how="left")
    df = df.merge(season_slice(2024, ["ppg"]), on="espn_id", how="left")

    # Simple cap efficiency: PPG per 100 cap units (proper fair-value is Phase C).
    df["ppg_per_100cap"] = (100 * df["ppg_2025"] / df["salary_2026"]).round(2)
    return df


def save_player_dataset(df: pd.DataFrame | None = None) -> "pd.Path":
    df = build_player_dataset() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "player_dataset_2026.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = build_player_dataset()
    skill = df[df["position_group"].isin(["QB", "RB", "WR", "TE"])]
    have_perf = skill["ppg_2025"].notna().sum()
    print(f"{len(df)} contract players; {len(skill)} skill-position; "
          f"{have_perf} have 2025 production "
          f"(rest are rookies / didn't play).")

    regulars = skill[(skill["games_2025"] >= 8) & (skill["salary_2026"] >= 1)]
    print("\n=== Best cap value: most PPG per 100 cap units (>=8 games) ===")
    cols = ["player", "team", "position_group", "salary_2026", "ppg_2025", "age", "ppg_per_100cap"]
    print(regulars.nlargest(10, "ppg_per_100cap")[cols].to_string(index=False))

    print("\n=== Priciest underperformers: high salary, low PPG (>=8 games) ===")
    pricey = regulars[regulars["salary_2026"] >= 80]
    print(pricey.nsmallest(10, "ppg_per_100cap")[cols].to_string(index=False))

    path = save_player_dataset(df)
    print(f"\nSaved -> {path}")
