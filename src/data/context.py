"""Per-player longitudinal context: baseline / delta / z-score vs each player's own history.

Adds, per ``(espn_id, season)``, prior-year value + trailing baseline (over the prior
``window`` seasons, **no-leakage**) + delta + z-score for each chosen metric. Plus four
diagnostic rollups for human-glance + downstream features:

- ``usage_trend``   = mean z-score across the *sticky* role metrics (target_share, wopr,
  snap_pct, carries, adot). Positive = role held/grew.
- ``results_trend`` = mean z-score across the *volatile* efficiency metrics (EPA, catch%,
  RACR, ryoe/att, etc.). Positive = efficiency held; sharply negative with stable usage =
  classic rebound candidate (e.g. Jefferson 2025).
- ``qb_context``    = the team passing (WR/TE/QB) or rushing (RB) EPA z-score — captures
  offense-environment driven up/down years.
- ``year_type``     ∈ {up, par, down, rookie, partial} — a glance label off ``ppg_z`` with
  rookie/partial overrides.

The baseline transform is **no-leakage** by construction:
``groupby(espn_id)[m].shift(1).rolling(window, min_periods=min_seasons).mean()`` — the
``shift(1)`` guarantees the baseline for season *t* uses only seasons *< t*. Seasons with
``games < min_games`` are **masked to NaN before** the rolling mean so injury-shortened
years don't pollute the trailing average.

Public entry points:
- ``add_player_context(frame, metrics)`` — generic stacked-frame transform.
- ``add_diagnostic_rollups(df)`` — adds usage/results/qb_context/year_type.
- ``training_frame_with_context()`` — convenience: ``training_frame()`` + both transforms.
- ``build_context_2026()`` — slim per-espn_id 2025 context for the 2026 valuation path.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR
from .population import training_frame

# --- Metric groups (from the approved Phase-3 design) -------------------------
HEADLINE = "ppg"
# Sticky role/opportunity signals. Baseline + z answers "is the role intact".
USAGE_METRICS = ["target_share", "wopr", "snap_pct", "carries", "adot"]
# Volatile efficiency outcomes. Cratering here while usage holds = fluke/rebound.
RESULTS_METRICS = [
    "receiving_epa", "catch_pct", "racr", "yac_above_expected",
    "rushing_epa", "ryoe_per_att", "cpoe", "on_tgt_pct",
]
# Team/offense environment (from src/data/advanced.py — efficiency-driven scoring).
TEAM_METRICS = ["team_pass_epa", "team_cpoe", "team_rush_epa", "team_pass_rate"]

DEFAULT_METRICS: list[str] = [HEADLINE] + USAGE_METRICS + RESULTS_METRICS + TEAM_METRICS

# --- Knobs --------------------------------------------------------------------
DEFAULT_WINDOW = 3       # trailing seasons in the rolling baseline
DEFAULT_MIN_SEASONS = 2  # need at least this many prior seasons for a baseline
DEFAULT_MIN_GAMES = 6    # below this, mask the season's metric (don't pollute baselines)
UP_Z = 0.75              # ppg_z > UP_Z → year_type "up"
DOWN_Z = -0.75           # ppg_z < DOWN_Z → year_type "down"


# --- Core transform -----------------------------------------------------------
def add_player_context(
    frame: pd.DataFrame,
    metrics: Iterable[str] | None = None,
    window: int = DEFAULT_WINDOW,
    min_seasons: int = DEFAULT_MIN_SEASONS,
    min_games: int = DEFAULT_MIN_GAMES,
) -> pd.DataFrame:
    """Add `{m}_prior`, `{m}_baseline`, `{m}_baseline_sd`, `{m}_delta`, `{m}_z` per metric.

    Operates on a one-row-per-`(espn_id, season)` frame. Masks `games < min_games` BEFORE
    the rolling baseline (so injury years don't pollute the trailing mean) but leaves the
    raw current-season value and the raw ``{m}_prior`` (last season, even if partial) alone.
    """
    metrics = list(metrics) if metrics is not None else DEFAULT_METRICS
    metrics = [m for m in metrics if m in frame.columns]
    if not metrics:
        return frame.copy()

    dup = int(frame.duplicated(subset=["espn_id", "season"]).sum())
    if dup:
        print(f"[context] dropping {dup} duplicate (espn_id, season) rows before rolling "
              "(keep first; usually merge artifacts from advanced/attrs joins).")
    df = (
        frame.sort_values(["espn_id", "season"])
        .drop_duplicates(subset=["espn_id", "season"], keep="first")
        .copy()
        .reset_index(drop=True)
    )
    games_ok = df["games"].fillna(0) >= min_games
    eid = df["espn_id"]

    for m in metrics:
        # Raw "last season" — unmasked (it's "what they did most recently" by definition).
        df[f"{m}_prior"] = df.groupby(eid, sort=False)[m].shift(1)
        # Baseline = mean of <window> trailing PRIOR seasons (shift(1) first → no leakage),
        # ignoring seasons masked to NaN by the games filter.
        masked = df[m].where(games_ok)
        shifted = masked.groupby(eid, sort=False).shift(1)
        roll = shifted.groupby(eid, sort=False).rolling(window=window, min_periods=min_seasons)
        baseline = roll.mean().reset_index(level=0, drop=True)
        baseline_sd = roll.std().reset_index(level=0, drop=True)
        df[f"{m}_baseline"] = baseline.round(3)
        df[f"{m}_baseline_sd"] = baseline_sd.round(3)
        df[f"{m}_delta"] = (df[m] - baseline).round(3)
        df[f"{m}_z"] = ((df[m] - baseline) / baseline_sd.where(baseline_sd > 0)).round(3)
    return df


# --- Diagnostic rollups -------------------------------------------------------
def add_diagnostic_rollups(
    df: pd.DataFrame,
    usage: list[str] | None = None,
    results: list[str] | None = None,
    min_games: int = DEFAULT_MIN_GAMES,
) -> pd.DataFrame:
    """Add `usage_trend`, `results_trend`, `qb_context`, `year_type`."""
    usage = usage if usage is not None else USAGE_METRICS
    results = results if results is not None else RESULTS_METRICS
    out = df.copy()

    def avg_z(cols: list[str]) -> pd.Series:
        present = [f"{c}_z" for c in cols if f"{c}_z" in out.columns]
        if not present:
            return pd.Series(np.nan, index=out.index)
        return out[present].mean(axis=1, skipna=True)

    out["usage_trend"] = avg_z(usage).round(3)
    out["results_trend"] = avg_z(results).round(3)

    # qb_context = team-passing z for WR/TE/QB; team-rushing z for RB.
    pass_z = out.get("team_pass_epa_z")
    rush_z = out.get("team_rush_epa_z")
    if pass_z is None:
        pass_z = pd.Series(np.nan, index=out.index)
    if rush_z is None:
        rush_z = pd.Series(np.nan, index=out.index)
    is_rb = out["position_group"].eq("RB")
    out["qb_context"] = np.where(is_rb, rush_z.values, pass_z.values)
    out["qb_context"] = pd.Series(out["qb_context"], index=out.index).round(3)

    # year_type: par by default; up/down off ppg_z; partial / rookie override.
    n = len(out)
    yt = np.full(n, "par", dtype=object)
    ppg_z = out.get("ppg_z", pd.Series(np.nan, index=out.index))
    ppg_baseline = out.get("ppg_baseline", pd.Series(np.nan, index=out.index))
    games = out["games"].fillna(0)
    yt[(ppg_z > UP_Z).fillna(False).values] = "up"
    yt[(ppg_z < DOWN_Z).fillna(False).values] = "down"
    yt[(games < min_games).values] = "partial"   # partial overrides up/down
    yt[ppg_baseline.isna().values] = "rookie"    # rookie overrides everything else
    out["year_type"] = yt
    return out


# --- Convenience entry points -------------------------------------------------
def training_frame_with_context(seasons: list[int] | None = None) -> pd.DataFrame:
    """training_frame() + add_player_context() + add_diagnostic_rollups()."""
    base = training_frame(seasons)
    return add_diagnostic_rollups(add_player_context(base))


def build_context_2026() -> pd.DataFrame:
    """Per-player 2025 context for the 2026 valuation path (one row per espn_id)."""
    full = training_frame_with_context()
    return full[full["season"] == 2025].copy().reset_index(drop=True)


def save_context_frame(df: pd.DataFrame | None = None) -> "pd.Path":
    """Cache the context-augmented training frame to Parquet for downstream use."""
    df = training_frame_with_context() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "training_frame_context.parquet"
    df.to_parquet(out, index=False)
    return out


# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    df = training_frame_with_context()
    print(f"{len(df)} player-seasons; window={DEFAULT_WINDOW}, "
          f"min_seasons={DEFAULT_MIN_SEASONS}, min_games={DEFAULT_MIN_GAMES}")
    print("\nyear_type distribution by season:")
    print(df.groupby(["season", "year_type"]).size().unstack(fill_value=0))

    # Spot-check headline players for 2025 (the year that feeds 2026 valuation).
    s25 = df[df["season"] == 2025].copy()
    cols = ["player_name", "position_group", "ppg", "ppg_baseline", "ppg_z",
            "usage_trend", "results_trend", "qb_context", "year_type"]
    # 'player_name' isn't always populated; fall back to 'name' from training_frame
    name_col = "name" if "name" in df.columns else "player_name"
    cols[0] = name_col

    notable = [
        "Justin Jefferson", "AJ Brown", "Puka Nacua", "Bijan Robinson",
        "Jahmyr Gibbs", "Patrick Mahomes", "Marvin Harrison Jr.",
        "Ashton Jeanty", "Drake Maye", "Brock Bowers",
    ]
    pick = s25[s25[name_col].isin(notable)].copy()
    if len(pick):
        pick["_p"] = pick["position_group"].map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
        pick = pick.sort_values(["_p", name_col])
        print("\nNotable 2025 player context (down/up/par diagnosis):")
        print(pick[cols].to_string(index=False))

    out = save_context_frame(df)
    print(f"\nSaved -> {out}")
