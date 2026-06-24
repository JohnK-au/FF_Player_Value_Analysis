"""Per-week WR feature builder — research module for archetype discovery.

Goal: shift from season-aggregate to week-level modeling so the model can learn
**conditional** signal (matchup, QB context, game environment) that season averages
wash out. Scope per user (2026-06-24 session): rostered WRs only, exploration only —
do NOT wire into the app yet.

One row per ``(season, week, espn_id)`` for WRs who appear in `performance_weekly.csv`.
Features cover:

- **Per-week usage** (from pbp by receiver_player_id): targets, receptions, air_yards,
  yards-after-catch, receiving EPA, target_share (vs his team that week), air-yards share.
- **Per-week NGS** (`import_ngs_data('receiving')`): avg separation, cushion, intended
  air yards, catch %, YAC above expected.
- **QB / team-passing context this week** (pbp by `posteam`): team pass EPA/play,
  team CPOE, completion %, pass rate, pass volume.
- **Opponent defense** (pbp by `defteam`, **rolling**): weighted mean across prior
  season-to-date weeks, weight 2.0 for the last 4 weeks vs 1.0 for earlier ones. No
  look-ahead leakage.
- **Game environment** (`import_schedules`): is_home, team_spread (positive = favored),
  team_implied_total, opponent_implied_total, point total.
- **Player-static**: age (per season), draft_value, undrafted, forty, vertical, broad_jump.

Target = `points` from `performance_weekly.csv` (our league's scoring, rostered).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR  # noqa: E402
from src.data.advanced import FANTASY_WEEKS, PBP_URL  # noqa: E402
from src.data.cache import cached_parquet  # noqa: E402
from src.data.nflverse import combine_athleticism, draft_capital  # noqa: E402

RESEARCH_DIR = PROCESSED_DIR / "research"
PBP_CACHE_DIR = RESEARCH_DIR / "pbp"
SEASONS = [2022, 2023, 2024, 2025]

# pbp columns we need (much larger than advanced.py's slim set)
PBP_COLS = [
    "season", "week", "season_type", "game_id", "posteam", "defteam",
    "passer_player_id", "rusher_player_id", "receiver_player_id",
    "complete_pass", "air_yards", "yards_gained", "yards_after_catch",
    "epa", "cpoe", "qb_epa", "rush", "pass", "pass_touchdown",
]


# --- 1. Cached raw data loaders -----------------------------------------------
def load_pbp(season: int) -> pd.DataFrame:
    """Regular-season pbp for one season, cached to local parquet."""
    def _build():
        df = pd.read_parquet(PBP_URL.format(season=season), columns=PBP_COLS)
        return df[df["season_type"] == "REG"].reset_index(drop=True)
    return cached_parquet(PBP_CACHE_DIR / f"pbp_{season}.parquet", _build)


def load_schedule(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Schedule + Vegas lines per game (cached)."""
    out = RESEARCH_DIR / "schedule.parquet"

    def _build():
        import nfl_data_py as nfl

        s = nfl.import_schedules(list(seasons))
        keep = ["season", "week", "game_id", "home_team", "away_team",
                "spread_line", "total_line"]
        return s[keep]
    return cached_parquet(out, _build)


def load_ngs_receiving(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Per-week receiving NGS, restricted to the fantasy regular season."""
    out = RESEARCH_DIR / "ngs_receiving_weekly.parquet"

    def _build():
        import nfl_data_py as nfl

        ngs = nfl.import_ngs_data("receiving", list(seasons))
        ngs = ngs[(ngs["season_type"] == "REG")
                  & ngs["week"].between(1, FANTASY_WEEKS)].copy()
        keep = ["season", "week", "player_gsis_id",
                "avg_cushion", "avg_separation", "avg_intended_air_yards",
                "catch_percentage", "avg_yac_above_expectation"]
        return ngs[keep].rename(columns={"player_gsis_id": "gsis_id"})
    return cached_parquet(out, _build)


def load_id_crosswalk(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Per-season gsis_id ↔ espn_id ↔ pfr_id ↔ age/years_exp from seasonal rosters."""
    out = RESEARCH_DIR / "id_crosswalk.parquet"

    def _build():
        import nfl_data_py as nfl

        frames = []
        for s in seasons:
            r = nfl.import_seasonal_rosters([s])[
                ["season", "player_id", "espn_id", "pfr_id", "position", "age", "years_exp"]
            ].copy()
            r["espn_id"] = pd.to_numeric(r["espn_id"], errors="coerce").astype("Int64")
            r = r.rename(columns={"player_id": "gsis_id"})
            frames.append(r)
        return pd.concat(frames, ignore_index=True).drop_duplicates(["season", "gsis_id"])
    return cached_parquet(out, _build)


def load_snaps_weekly(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Per (season, week, pfr_id) offensive snap share over wk 1..FANTASY_WEEKS."""
    out = RESEARCH_DIR / "snaps_weekly.parquet"

    def _build():
        import nfl_data_py as nfl

        s = nfl.import_snap_counts(list(seasons))
        s = s[(s["game_type"] == "REG") & s["week"].between(1, FANTASY_WEEKS)].copy()
        keep = ["season", "week", "pfr_player_id", "offense_pct", "offense_snaps"]
        return s[keep].rename(columns={"pfr_player_id": "pfr_id",
                                       "offense_pct": "snap_pct",
                                       "offense_snaps": "snap_count"})
    return cached_parquet(out, _build)


# --- 2. Per-week aggregations from pbp ----------------------------------------
def receiver_week(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (week, receiver_gsis_id): usage + efficiency from pbp pass plays."""
    tgt = pbp[pbp["receiver_player_id"].notna()].copy()
    g = (
        tgt.groupby(["week", "receiver_player_id"])
        .agg(
            posteam=("posteam", lambda s: s.mode().iat[0]),
            targets=("posteam", "size"),
            receptions=("complete_pass", "sum"),
            air_yards=("air_yards", "sum"),
            rec_yards_gained=("yards_gained", "sum"),
            yac=("yards_after_catch", "sum"),
            rec_tds=("pass_touchdown", "sum"),
            rec_epa=("epa", "sum"),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "gsis_id", "posteam": "team"})
    )
    # 'rec_yards_gained' on pass plays includes incompletes (which are 0), so it's
    # the receiver's actual receiving yards on this set. Keep as `rec_yards`.
    g = g.rename(columns={"rec_yards_gained": "rec_yards"})
    return g


def team_pass_week(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (week, posteam): team-passing profile this week (QB context for WRs)."""
    db = pbp[pbp["passer_player_id"].notna()].copy()
    g = (
        db.groupby(["week", "posteam"])
        .agg(
            team_pass_plays=("posteam", "size"),
            team_pass_epa_play=("epa", "mean"),
            team_cpoe=("cpoe", "mean"),
            team_completion_pct=("complete_pass", "mean"),
            team_air_yards=("air_yards", "sum"),
            team_pass_tds=("pass_touchdown", "sum"),
            team_targets=("posteam", "size"),  # same as plays for the WR-share denom
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    rush = (
        pbp[pbp["rush"] == 1].groupby(["week", "posteam"]).size()
        .rename("team_rush_plays").reset_index().rename(columns={"posteam": "team"})
    )
    g = g.merge(rush, on=["week", "team"], how="left")
    g["team_rush_plays"] = g["team_rush_plays"].fillna(0)
    g["team_pass_rate"] = g["team_pass_plays"] / (g["team_pass_plays"] + g["team_rush_plays"])
    return g


def defense_week_raw(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (week, defteam): defense-passing profile this week (NOT rolled yet)."""
    db = pbp[pbp["passer_player_id"].notna()].copy()
    g = (
        db.groupby(["week", "defteam"])
        .agg(
            def_pass_plays=("defteam", "size"),
            def_pass_epa_play=("epa", "mean"),
            def_cpoe_allowed=("cpoe", "mean"),
            def_completion_pct_allowed=("complete_pass", "mean"),
            def_pass_tds_allowed=("pass_touchdown", "sum"),
            def_yards_allowed=("yards_gained", "sum"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return g


DEF_ROLL_COLS = [
    "def_pass_epa_play", "def_cpoe_allowed", "def_completion_pct_allowed",
    "def_pass_tds_allowed", "def_yards_allowed",
]


def defense_rolling(def_raw: pd.DataFrame, recent_weeks: int = 4,
                    recent_weight: float = 2.0, prior_weight: float = 1.0) -> pd.DataFrame:
    """For each (week, defteam), weighted-rolling defense metrics from PRIOR weeks only.

    Weights: ``recent_weight`` for weeks in [w − recent_weeks, w − 1], ``prior_weight``
    for earlier season-to-date weeks. Week 1 → NaN (no prior data).
    """
    out = []
    for team, sub in def_raw.groupby("team", sort=False):
        sub = sub.sort_values("week").reset_index(drop=True)
        for _, row in sub.iterrows():
            w = int(row["week"])
            prior = sub[sub["week"] < w]
            rolled = {"team": team, "week": w}
            if len(prior) == 0:
                for c in DEF_ROLL_COLS:
                    rolled[f"{c}_roll"] = np.nan
            else:
                weights = np.where(prior["week"] >= (w - recent_weeks),
                                   recent_weight, prior_weight)
                for c in DEF_ROLL_COLS:
                    vals = prior[c].values.astype(float)
                    mask = ~np.isnan(vals)
                    rolled[f"{c}_roll"] = (
                        float(np.average(vals[mask], weights=weights[mask]))
                        if mask.any() else np.nan
                    )
            out.append(rolled)
    return pd.DataFrame(out)


# --- 3. Schedule → per-team-week (Vegas + home/away + opponent) ---------------
def schedule_team_week(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with opponent, is_home, Vegas lines."""
    home = schedule.assign(
        team=schedule["home_team"], opponent=schedule["away_team"], is_home=1,
    )
    away = schedule.assign(
        team=schedule["away_team"], opponent=schedule["home_team"], is_home=0,
    )
    both = pd.concat([home, away], ignore_index=True)
    # spread_line is the home team's spread (negative when home is underdog).
    # team_spread from this team's perspective: positive = favored.
    both["team_spread"] = np.where(both["is_home"] == 1,
                                   both["spread_line"], -both["spread_line"])
    both["team_implied_total"] = np.where(
        both["is_home"] == 1,
        (both["total_line"] + both["spread_line"]) / 2,
        (both["total_line"] - both["spread_line"]) / 2,
    )
    both["opp_implied_total"] = both["total_line"] - both["team_implied_total"]
    return both[["season", "week", "game_id", "team", "opponent",
                 "is_home", "spread_line", "total_line",
                 "team_spread", "team_implied_total", "opp_implied_total"]]


# --- 4. Player-static (one row per espn_id) -----------------------------------
def player_static() -> pd.DataFrame:
    """Stable player attributes by espn_id (draft + combine)."""
    draft = draft_capital().copy()
    comb = combine_athleticism().copy()
    for df in (draft, comb):
        df["espn_id"] = pd.to_numeric(df["espn_id"], errors="coerce").astype("Int64")
    keep_draft = ["espn_id", "draft_round", "draft_pick", "draft_value", "undrafted"]
    keep_comb = ["espn_id", "forty", "vertical", "broad_jump"]
    return draft[keep_draft].merge(comb[keep_comb], on="espn_id", how="outer")


# --- 5. The season-level WR feature build -------------------------------------
def build_wr_weekly_features_season(season: int) -> pd.DataFrame:
    """Per-week WR features for one season, joined to the league scoring target."""
    pbp = load_pbp(season)
    pbp = pbp[pbp["week"].between(1, FANTASY_WEEKS)].copy()

    rec_w = receiver_week(pbp)
    team_w = team_pass_week(pbp)
    def_raw = defense_week_raw(pbp)
    def_roll = defense_rolling(def_raw)

    # join team-passing context to the WR's row by his team that week
    rec_w = rec_w.merge(team_w, on=["week", "team"], how="left")
    # add usage shares
    rec_w["target_share"] = rec_w["targets"] / rec_w["team_pass_plays"]
    rec_w["air_yards_share"] = rec_w["air_yards"] / rec_w["team_air_yards"].where(
        rec_w["team_air_yards"] > 0
    )

    # join schedule to know opponent + Vegas + home/away
    sched = load_schedule()
    stw = schedule_team_week(sched)
    stw_s = stw[stw["season"] == season]
    rec_w["season"] = season
    rec_w = rec_w.merge(
        stw_s[["week", "team", "opponent", "is_home",
               "team_spread", "team_implied_total", "opp_implied_total", "total_line"]],
        on=["week", "team"], how="left",
    )

    # join opponent's rolling defense (renaming team→opponent)
    def_roll_opp = def_roll.rename(columns={"team": "opponent"})
    rec_w = rec_w.merge(def_roll_opp, on=["week", "opponent"], how="left")

    # join NGS receiving (per-week separation, catch%, etc.)
    ngs = load_ngs_receiving()
    ngs = ngs[ngs["season"] == season]
    rec_w = rec_w.merge(ngs.drop(columns=["season"]), on=["week", "gsis_id"], how="left")

    # join espn_id + pfr_id + age via the crosswalk
    xw = load_id_crosswalk()
    xw_s = xw[(xw["season"] == season) & xw["position"].eq("WR")]
    rec_w = rec_w.merge(
        xw_s[["gsis_id", "espn_id", "pfr_id", "age", "years_exp"]],
        on="gsis_id", how="inner",  # inner → only keep actual WRs with an espn_id
    )

    # join player-static (draft + combine)
    rec_w = rec_w.merge(player_static(), on="espn_id", how="left")

    # join weekly snap share (via pfr_id)
    snaps = load_snaps_weekly()
    snaps_s = snaps[snaps["season"] == season].drop(columns=["season"])
    rec_w = rec_w.merge(snaps_s, on=["week", "pfr_id"], how="left")

    # join target: league-scored weekly points for the rostered universe
    from src.data.dataset import load_weekly
    weekly = load_weekly()
    weekly = weekly[(weekly["season"] == season) & weekly["position"].eq("WR")].copy()
    weekly["espn_id"] = pd.to_numeric(weekly["espn_id"], errors="coerce").astype("Int64")
    weekly = weekly.dropna(subset=["espn_id"]).drop_duplicates(["week", "espn_id"])
    rec_w = rec_w.merge(
        weekly[["week", "espn_id", "name", "points", "started"]],
        on=["week", "espn_id"], how="inner",  # inner = rostered-only universe
    )
    rec_w = rec_w.rename(columns={"points": "fantasy_points"})
    return rec_w


def build_wr_weekly_features(seasons: list[int] = SEASONS) -> pd.DataFrame:
    out = pd.concat([build_wr_weekly_features_season(s) for s in seasons], ignore_index=True)
    cols_first = ["season", "week", "espn_id", "gsis_id", "name", "team", "opponent",
                  "is_home", "fantasy_points"]
    rest = [c for c in out.columns if c not in cols_first]
    return out[cols_first + rest]


def save(df: pd.DataFrame | None = None) -> Path:
    df = build_wr_weekly_features() if df is None else df
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out = RESEARCH_DIR / "wr_weekly_features.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = build_wr_weekly_features()
    print(f"{len(df)} WR-weeks across {df['season'].nunique()} seasons "
          f"({df['espn_id'].nunique()} unique WRs)")
    print("\nRows per season:")
    print(df.groupby("season").size().to_string())
    print("\nTarget (fantasy_points) summary:")
    print(df["fantasy_points"].describe().round(2).to_string())
    print("\nColumns:")
    for c in df.columns:
        nn = df[c].notna().sum()
        pct = 100 * nn / len(df)
        print(f"  {c:30}  notna={nn:5}  ({pct:.0f}%)")
    out = save(df)
    print(f"\nSaved -> {out}")
