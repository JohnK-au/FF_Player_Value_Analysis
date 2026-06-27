"""League-wide training frame: every NFL skill player × season.

One row per (season, player) for QB/RB/WR/TE across multiple seasons, joining
ESPN fantasy production (our scoring) to nflverse advanced metrics, draft
capital, combine athleticism, and age/experience. This is the broad base the
production model trains on and the pool from which positional replacement levels
are drawn — i.e. the data the fair-value engine reasons about *beyond* our own
~156 rostered players.

Unlike ``dataset.build_player_dataset`` (our contract players only, 13-week
fantasy production), this uses the full-season ESPN pull so every rostered/FA
player in the league's scoring is present and comparable across positions.
"""
from __future__ import annotations

import pandas as pd

from ..config import PROCESSED_DIR
from .advanced import advanced_features
from .nflverse import combine_athleticism, draft_capital
from .scoring import nflverse_season_production

SKILL = ["QB", "RB", "WR", "TE"]
TRAIN_SEASONS = [2022, 2023, 2024, 2025]
# v2 extended window (2016-2025) for the per-position component models. Uses
# nflverse seasonal_data + our 2025 scoring rules to reconstruct PPG for years
# the league didn't exist (2016-2021), and ESPN-reported PPG for 2025 (nflverse
# seasonal_data not yet published for 2025 as of June 2026). See docs/methodology.
EXTENDED_TRAIN_SEASONS = list(range(2016, 2026))


def _roster_attrs(season: int) -> pd.DataFrame:
    """espn_id, age (during ``season``), years_exp from the seasonal roster."""
    import nfl_data_py as nfl

    r = nfl.import_seasonal_rosters([season])[["espn_id", "age", "years_exp"]].copy()
    r["espn_id"] = pd.to_numeric(r["espn_id"], errors="coerce").astype("Int64")
    return r.dropna(subset=["espn_id"]).drop_duplicates("espn_id")


def season_production(season: int) -> pd.DataFrame:
    """NFL-wide skill-player season production for ``season`` (our scoring).

    Pulls from the cached full-season ESPN performance table. ``position_group``
    mirrors the ESPN position (already QB/RB/WR/TE for skill players).
    """
    from .dataset import load_season_performance

    perf = load_season_performance()
    p = perf[(perf["season"] == season) & (perf["position"].isin(SKILL))].copy()
    p["espn_id"] = pd.to_numeric(p["espn_id"], errors="coerce").astype("Int64")
    p = p.dropna(subset=["espn_id"]).drop_duplicates("espn_id")
    p["position_group"] = p["position"]
    return p[["espn_id", "name", "position_group", "points", "ppg", "games"]]


def season_frame(season: int) -> pd.DataFrame:
    """All features for one season's skill players (production + advanced + context)."""
    prod = season_production(season)
    attrs = _roster_attrs(season)
    adv = advanced_features(season).drop(columns=["gsis_id", "pfr_id"], errors="ignore")
    draft = draft_capital()
    comb = combine_athleticism()
    for t in (adv, draft, comb):
        t["espn_id"] = pd.to_numeric(t["espn_id"], errors="coerce").astype("Int64")

    df = prod.merge(attrs, on="espn_id", how="left")
    for t in (adv, draft, comb):
        df = df.merge(t, on="espn_id", how="left")
    df.insert(0, "season", season)
    return df


def training_frame(seasons: list[int] | None = None) -> pd.DataFrame:
    """Stacked per-season skill-player frames across ``seasons``."""
    seasons = seasons or TRAIN_SEASONS
    return pd.concat([season_frame(s) for s in seasons], ignore_index=True)


def save_training_frame(df: pd.DataFrame | None = None) -> "pd.Path":
    df = training_frame() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "training_frame.csv"
    df.to_csv(out, index=False)
    return out


# --- v2 extended training frame (2016-2025) -----------------------------------
def extended_season_production(season: int) -> pd.DataFrame:
    """Per-player PPG for ``season`` using a uniform scoring basis.

    Source per year:
      - **2025**: ESPN-reported PPG (our league data; nflverse seasonal_data not
        yet published for 2025 as of June 2026; ESPN values include yardage
        bonuses + team-W/L which our reconstruction omits — ~0.17 PPG higher on
        average than reconstructed would be).
      - **2016-2024**: reconstructed via ``nflverse_season_production`` (nflverse
        seasonal_data + our 2025 scoring rules), giving a uniform per-position
        training target across all historical years.

    Same schema as ``season_production`` — drop-in replacement for downstream
    joins in ``extended_season_frame``.
    """
    if season == 2025:
        return season_production(season)
    return nflverse_season_production(season)


def extended_season_frame(season: int) -> pd.DataFrame:
    """All features for one season's skill players using the v2 PPG basis."""
    prod = extended_season_production(season)
    attrs = _roster_attrs(season)
    adv = advanced_features(season).drop(columns=["gsis_id", "pfr_id"], errors="ignore")
    draft = draft_capital()
    comb = combine_athleticism()
    for t in (adv, draft, comb):
        t["espn_id"] = pd.to_numeric(t["espn_id"], errors="coerce").astype("Int64")

    df = prod.merge(attrs, on="espn_id", how="left")
    for t in (adv, draft, comb):
        df = df.merge(t, on="espn_id", how="left")
    df.insert(0, "season", season)
    return df


def extended_training_frame(seasons: list[int] | None = None) -> pd.DataFrame:
    """Stacked per-season skill-player frames across the v2 extended window."""
    seasons = seasons or EXTENDED_TRAIN_SEASONS
    return pd.concat([extended_season_frame(s) for s in seasons], ignore_index=True)


def save_extended_training_frame(df: pd.DataFrame | None = None) -> "pd.Path":
    df = extended_training_frame() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "training_frame_extended.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = training_frame()
    print(f"{len(df)} skill player-seasons across {df['season'].nunique()} seasons")
    print("\nrows by season × position:")
    print(df.groupby(["season", "position_group"]).size().unstack(fill_value=0).to_string())
    scored = df["ppg"].notna().sum()
    print(f"\nwith a season PPG (a usable target): {scored} "
          f"({scored / len(df):.0%})")
    path = save_training_frame(df)
    print(f"\nSaved -> {path}")
