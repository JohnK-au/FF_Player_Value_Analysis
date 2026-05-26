"""Advanced player metrics from nflverse (selected value-model features).

Windows weekly/Next-Gen-Stats sources to the fantasy regular season
(weeks 1..FANTASY_WEEKS) and joins to ``espn_id`` via the seasonal-roster id
crosswalk. Covers the gsis-keyed metrics (pbp-derived opportunity/EPA + NGS) and
the pfr-keyed metrics (PFR advanced + snap share). Draft / combine arrive next.
"""
from __future__ import annotations

import pandas as pd

FANTASY_WEEKS = 13  # league regular season (reg_season_count)

# All advanced sources cover 2025: NGS / PFR / snaps via nfl_data_py, and the
# weekly opportunity/EPA metrics derived from 2025 play-by-play (nflverse's
# pre-aggregated weekly file lags a season, so we compute them from pbp instead).
ADV_SEASON = 2025

PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)


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


def _pbp_weekly(season: int, weeks: int) -> pd.DataFrame:
    """Opportunity / EPA metrics derived from play-by-play (regular season wk 1..weeks).

    Computed from pbp because nflverse's pre-aggregated weekly file lags a season.
    Player ids in pbp are gsis ids. target/air-yards shares are vs the player's team.
    """
    cols = ["week", "season_type", "posteam", "passer_player_id", "rusher_player_id",
            "receiver_player_id", "complete_pass", "air_yards", "yards_gained", "epa", "rush"]
    p = pd.read_parquet(PBP_URL.format(season=season), columns=cols)
    p = p[(p["season_type"] == "REG") & (p["week"] >= 1) & (p["week"] <= weeks)]

    tgt = p[p["receiver_player_id"].notna()]
    rec = tgt.groupby("receiver_player_id").agg(
        targets=("posteam", "size"), air=("air_yards", "sum"), receiving_epa=("epa", "sum"),
    )
    rec["rec_yards"] = tgt[tgt["complete_pass"] == 1].groupby("receiver_player_id")["yards_gained"].sum()
    rec["team"] = tgt.groupby("receiver_player_id")["posteam"].agg(lambda s: s.mode().iat[0])
    team = tgt.groupby("posteam").agg(team_tgt=("posteam", "size"), team_air=("air_yards", "sum"))
    rec = rec.join(team, on="team")
    rec["target_share"] = rec["targets"] / rec["team_tgt"]
    air_share = rec["air"] / rec["team_air"]
    rec["wopr"] = 1.5 * rec["target_share"] + 0.7 * air_share
    rec["racr"] = rec["rec_yards"] / rec["air"].where(rec["air"] > 0)
    rec = rec[["target_share", "wopr", "racr", "receiving_epa"]]

    car = p[p["rush"] == 1].groupby("rusher_player_id").agg(
        carries=("posteam", "size"), rushing_epa=("epa", "sum"),
    )
    pas = p[p["passer_player_id"].notna()].groupby("passer_player_id").agg(
        passing_epa=("epa", "sum"),
    )
    out = rec.join([car, pas], how="outer")
    out.index.name = "gsis_id"
    return out.reset_index()


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


def _pfr(season: int) -> pd.DataFrame:
    """PFR advanced rate stats (full season), keyed by pfr_id."""
    import nfl_data_py as nfl

    rush = nfl.import_seasonal_pfr("rush", [season])[["pfr_id", "ybc_att", "yac_att"]]
    pas = nfl.import_seasonal_pfr("pass", [season])[["pfr_id", "pressure_pct", "on_tgt_pct"]]
    return rush.merge(pas, on="pfr_id", how="outer")


def _snaps(season: int, weeks: int) -> pd.DataFrame:
    """Average offensive snap share over the regular season (wk 1..weeks), by pfr_id."""
    import nfl_data_py as nfl

    s = nfl.import_snap_counts([season])
    s = s[(s["game_type"] == "REG") & (s["week"] >= 1) & (s["week"] <= weeks)]
    return (
        s.groupby("pfr_player_id")
        .agg(snap_pct=("offense_pct", "mean"))
        .reset_index()
        .rename(columns={"pfr_player_id": "pfr_id"})
    )


def advanced_features(season: int = ADV_SEASON, weeks: int = FANTASY_WEEKS) -> pd.DataFrame:
    """All selected advanced metrics for a season, keyed by espn_id.

    gsis-keyed (pbp + NGS) and pfr-keyed (PFR + snaps) sources joined to the
    roster crosswalk; one row per rostered player (NaN where a metric doesn't apply).
    """
    xwalk = _id_crosswalk(season)  # gsis_id, espn_id, pfr_id
    gsis_feats = _pbp_weekly(season, weeks).merge(_ngs(season, weeks), on="gsis_id", how="outer")
    pfr_feats = _pfr(season).merge(_snaps(season, weeks), on="pfr_id", how="outer")
    out = (
        xwalk.merge(gsis_feats, on="gsis_id", how="left")
        .merge(pfr_feats, on="pfr_id", how="left")
    )
    return out[out["espn_id"].notna()].reset_index(drop=True)


if __name__ == "__main__":
    df = advanced_features(ADV_SEASON)
    metric_cols = [c for c in df.columns if c not in ("gsis_id", "espn_id", "pfr_id")]
    print(f"{len(df)} rostered players; {len(metric_cols)} advanced metrics\n")
    print(f"metrics: {metric_cols}\n")
    print("coverage:")
    for c in metric_cols:
        print(f"  {c}: {int(df[c].notna().sum())}")
