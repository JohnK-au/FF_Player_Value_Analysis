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
