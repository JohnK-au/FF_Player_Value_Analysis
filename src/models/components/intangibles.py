"""Intangibles component — user-supplied subjective overrides in [0, 100].

Per the methodology (docs/methodology/intangibles.md), every player starts at
a **neutral baseline of 50** (mid-scale). The user can override any player
with a numeric score that captures information outside the data — e.g.
"Rashee Rice spent 30 days in prison for a car accident", "this player wants
to be traded", "rumored coaching change favors his role".

Overrides live in an editable CSV at
``data/research/intangibles_overrides.csv`` with the schema:

    espn_id,player_name,intangibles_value,note,updated_at

The loader uses `espn_id` as the join key (player_name is for human review
only). Any value not in [0, 100] is clipped. Missing entries → neutral.

Phase 0 (foundation): a minimal but functional implementation since the
file format is stable. Reading the CSV file (if it exists) and merging
overrides onto the players frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NEUTRAL = 50.0
OVERRIDES_PATH = Path("data/research/intangibles_overrides.csv")
LO, HI = 0.0, 100.0


def _load_overrides() -> pd.DataFrame:
    """Read the overrides CSV (if it exists), return empty frame otherwise."""
    if not OVERRIDES_PATH.exists():
        return pd.DataFrame(columns=["espn_id", "intangibles_value"])
    df = pd.read_csv(OVERRIDES_PATH)
    df["espn_id"] = pd.to_numeric(df["espn_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["espn_id"]).drop_duplicates("espn_id", keep="last")
    df["intangibles_value"] = pd.to_numeric(df["intangibles_value"], errors="coerce")
    df["intangibles_value"] = df["intangibles_value"].clip(lower=LO, upper=HI)
    return df[["espn_id", "intangibles_value"]]


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Intangibles score per player in [0, 100], from the overrides CSV."""
    out = players.copy()
    overrides = _load_overrides()
    if "espn_id" in out.columns and len(overrides):
        out = out.merge(overrides, on="espn_id", how="left")
        out["intangibles_value"] = out["intangibles_value"].fillna(NEUTRAL)
    else:
        out["intangibles_value"] = NEUTRAL
    return out
