"""Pull player attributes (age, experience) from nflverse via ``nfl_data_py``.

Joined to our players on ``espn_id`` (already in the contract↔ESPN crosswalk).
Age is computed as of the upcoming season for dynasty relevance.
"""
from __future__ import annotations

import pandas as pd

from ..config import PROCESSED_DIR
from .cache import cached_parquet

DEFAULT_SEASON = 2025
SEASON_START = "2026-09-01"  # reference date for "age entering the 2026 season"
ATTR_CACHE_DIR = PROCESSED_DIR / "attributes"
CONTEXT_DIR = PROCESSED_DIR / "context"
DRAFT_YEARS = list(range(2005, DEFAULT_SEASON + 1))  # cover current players' draft years


def _crosswalk(season: int = DEFAULT_SEASON) -> pd.DataFrame:
    """gsis_id ↔ espn_id ↔ pfr_id for the season's players."""
    import nfl_data_py as nfl

    r = nfl.import_seasonal_rosters([season])[["player_id", "espn_id", "pfr_id"]].copy()
    r["espn_id"] = pd.to_numeric(r["espn_id"], errors="coerce").astype("Int64")
    return (
        r.dropna(subset=["player_id"])
        .drop_duplicates("player_id")
        .rename(columns={"player_id": "gsis_id"})
    )


def player_attributes(
    season: int = DEFAULT_SEASON, as_of: str = SEASON_START, refresh: bool = False
) -> pd.DataFrame:
    """Per-player age (as of ``as_of``) and experience, keyed by ``espn_id``.

    Cached to Parquet per season (completed seasons are immutable)."""
    return cached_parquet(
        ATTR_CACHE_DIR / f"ages_{season}.parquet",
        lambda: _build_attributes(season, as_of),
        refresh=refresh,
    )


def _build_attributes(season: int, as_of: str) -> pd.DataFrame:
    import nfl_data_py as nfl

    r = nfl.import_seasonal_rosters([season])
    r = r[["espn_id", "player_name", "position", "birth_date", "age", "years_exp"]].copy()
    r["espn_id"] = pd.to_numeric(r["espn_id"], errors="coerce").astype("Int64")
    r = r.dropna(subset=["espn_id"]).drop_duplicates("espn_id")

    ref = pd.Timestamp(as_of)
    roster_age = pd.to_numeric(r["age"], errors="coerce")  # age during the 2025 season
    birth = pd.to_datetime(r["birth_date"], errors="coerce")
    computed = (ref - birth).dt.days / 365.25
    # Prefer birth-date-derived age; else fall back to roster age + 1 (≈ next season).
    r["age"] = computed.fillna(roster_age + 1).round(1)

    return r[["espn_id", "player_name", "age", "years_exp"]].reset_index(drop=True)


# --- Player headshots (for the Streamlit Compare page) ------------------------

def player_headshots(season: int = DEFAULT_SEASON, refresh: bool = False) -> pd.DataFrame:
    """Per-player headshot URL, keyed by espn_id. Cached to CSV under processed/.

    Pulled once from ``nfl.import_seasonal_rosters([season])``. Enables the
    Compare page to display player photos via ``st.image(url)`` without an
    extra API round-trip per render.
    """
    out_path = PROCESSED_DIR / "player_headshots.csv"
    if out_path.exists() and not refresh:
        return pd.read_csv(out_path)

    import nfl_data_py as nfl
    r = nfl.import_seasonal_rosters([season])[["espn_id", "headshot_url"]].copy()
    r["espn_id"] = pd.to_numeric(r["espn_id"], errors="coerce").astype("Int64")
    r = r.dropna(subset=["espn_id", "headshot_url"]).drop_duplicates("espn_id").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(out_path, index=False)
    return r


# --- Draft capital & combine (player-level, immutable → cached once) ----------

def draft_capital(value_chart: str = "otc", refresh: bool = False) -> pd.DataFrame:
    """Per-player draft round/pick/value (+ undrafted flag), keyed by espn_id."""
    return cached_parquet(
        CONTEXT_DIR / "draft_capital.parquet",
        lambda: _build_draft(value_chart),
        refresh=refresh,
    )


def _build_draft(value_chart: str) -> pd.DataFrame:
    import nfl_data_py as nfl

    dp = nfl.import_draft_picks(DRAFT_YEARS)[["round", "pick", "gsis_id"]].dropna(subset=["gsis_id"])
    dp = dp.rename(columns={"round": "draft_round", "pick": "draft_pick"}).drop_duplicates("gsis_id")
    dv = nfl.import_draft_values()[["pick", value_chart]].rename(
        columns={"pick": "draft_pick", value_chart: "draft_value"}
    )
    dp = dp.merge(dv, on="draft_pick", how="left")
    out = _crosswalk().merge(dp, on="gsis_id", how="left")
    out["undrafted"] = out["draft_pick"].isna().astype(int)
    out["draft_value"] = out["draft_value"].fillna(0.0)         # UDFA -> 0
    out["draft_round"] = out["draft_round"].fillna(8)           # UDFA tier
    out = out.dropna(subset=["espn_id"]).drop_duplicates("espn_id")
    return out[["espn_id", "draft_round", "draft_pick", "draft_value", "undrafted"]]


def combine_athleticism(refresh: bool = False) -> pd.DataFrame:
    """Per-player combine measurables (40-yd, vertical, broad jump), keyed by espn_id."""
    return cached_parquet(
        CONTEXT_DIR / "combine.parquet", _build_combine, refresh=refresh
    )


def _build_combine() -> pd.DataFrame:
    import nfl_data_py as nfl

    cd = (
        nfl.import_combine_data(DRAFT_YEARS)[["pfr_id", "forty", "vertical", "broad_jump"]]
        .dropna(subset=["pfr_id"])
        .drop_duplicates("pfr_id")
    )
    out = _crosswalk()[["espn_id", "pfr_id"]].merge(cd, on="pfr_id", how="left")
    out = out.dropna(subset=["espn_id"]).drop_duplicates("espn_id")
    return out[["espn_id", "forty", "vertical", "broad_jump"]]
