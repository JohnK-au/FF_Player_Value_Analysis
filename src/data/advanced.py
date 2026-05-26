"""Advanced player metrics from nflverse (selected value-model features).

Windows weekly/Next-Gen-Stats sources to the fantasy regular season
(weeks 1..FANTASY_WEEKS) and joins to ``espn_id`` via the seasonal-roster id
crosswalk. This module covers the gsis-keyed metrics (weekly_data + NGS); PFR /
snaps and draft / combine arrive in companion functions.
"""
from __future__ import annotations

import pandas as pd

FANTASY_WEEKS = 13  # league regular season (reg_season_count)

# Latest season with ALL advanced sources available. nflverse publishes NGS / PFR
# / snaps for 2025, but weekly player stats only through 2024 (in this mirror), so
# we standardise advanced metrics on 2024 and bump this once 2025 weekly lands.
ADV_SEASON = 2024


def _id_crosswalk(season: int) -> pd.DataFrame:
    """gsis_id ↔ espn_id ↔ pfr_id for the season's players."""
    import nfl_data_py as nfl

    r = nfl.import_seasonal_rosters([season])[["player_id", "espn_id", "pfr_id"]].copy()
    r["espn_id"] = pd.to_numeric(r["espn_id"], errors="coerce").astype("Int64")
    return (
        r.dropna(subset=["player_id"])
        .drop_duplicates("player_id")
        .rename(columns={"player_id": "gsis_id"})
    )


def _weekly(season: int, weeks: int) -> pd.DataFrame:
    """Weekly-derived metrics over weeks 1..weeks (shares/rates averaged, EPA summed)."""
    import nfl_data_py as nfl

    w = nfl.import_weekly_data([season])
    w = w[(w["week"] >= 1) & (w["week"] <= weeks)]
    return (
        w.groupby("player_id")
        .agg(
            carries=("carries", "sum"),
            target_share=("target_share", "mean"),
            wopr=("wopr", "mean"),
            racr=("racr", "mean"),
            passing_epa=("passing_epa", "sum"),
            rushing_epa=("rushing_epa", "sum"),
            receiving_epa=("receiving_epa", "sum"),
        )
        .reset_index()
        .rename(columns={"player_id": "gsis_id"})
    )


def _ngs(season: int, weeks: int) -> pd.DataFrame:
    """Next Gen Stats (receiving / rushing / passing) averaged over weeks 1..weeks."""
    import nfl_data_py as nfl

    def win(df):
        return df[(df["week"] >= 1) & (df["week"] <= weeks)]

    rec = win(nfl.import_ngs_data("receiving", [season])).groupby("player_gsis_id").agg(
        avg_separation=("avg_separation", "mean"),
        yac_above_expected=("avg_yac_above_expectation", "mean"),
        adot=("avg_intended_air_yards", "mean"),
        catch_pct=("catch_percentage", "mean"),
    )
    rush = win(nfl.import_ngs_data("rushing", [season])).groupby("player_gsis_id").agg(
        ryoe_per_att=("rush_yards_over_expected_per_att", "mean"),
        time_to_los=("avg_time_to_los", "mean"),
    )
    pas = win(nfl.import_ngs_data("passing", [season])).groupby("player_gsis_id").agg(
        cpoe=("completion_percentage_above_expectation", "mean"),
    )
    return (
        rec.join([rush, pas], how="outer")
        .reset_index()
        .rename(columns={"player_gsis_id": "gsis_id"})
    )


def advanced_gsis(season: int, weeks: int = FANTASY_WEEKS) -> pd.DataFrame:
    """Per-player gsis-keyed advanced metrics for a season, keyed by espn_id."""
    feats = _weekly(season, weeks).merge(_ngs(season, weeks), on="gsis_id", how="outer")
    out = _id_crosswalk(season).merge(feats, on="gsis_id", how="right")
    return out[out["espn_id"].notna()].reset_index(drop=True)


if __name__ == "__main__":
    df = advanced_gsis(ADV_SEASON)
    metric_cols = [c for c in df.columns if c not in ("gsis_id", "espn_id", "pfr_id")]
    print(f"{len(df)} players with advanced (gsis) metrics; "
          f"{int(df['espn_id'].notna().sum())} mapped to espn_id\n")
    print(f"metric columns ({len(metric_cols)}): {metric_cols}\n")
    cols = ["target_share", "wopr", "carries", "ryoe_per_att", "avg_separation", "adot", "cpoe"]
    print(df.dropna(subset=["target_share"]).nlargest(6, "wopr")[["espn_id"] + cols].to_string(index=False))
