"""Team component — offensive environment scoring in [0, 100].

Position-specific feature sets and weights. Both WR and RB are empirically
tuned via residual regression: fit Production model on player features only,
compute residuals, regress on team features. See docs/methodology/team.md.

WR (Phase 1C):
    +0.38 * team_pass_rate_z         (passing volume)
    +0.37 * team_pass_epa_z          (passing efficiency / QB quality)
    -0.25 * top_2_target_share_excl_self_z   (competition from other team WRs)

RB (Phase 2):
    +1.00 * team_ybc_att_z           (O-line: carry-weighted avg ybc_att
                                       across team's RBs; sole signal that
                                       added residual variance per regression)

Dropped in the regression:
- WR: team_cpoe (wrong sign, noise)
- RB: team_rush_epa (multicollinear with RB's own rushing_epa, came out
  with WRONG sign), team_pass_rate (noise), top_rb_other_share (backfield
  competition surprisingly added no signal beyond carries usage)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from src.data.population import extended_training_frame

NEUTRAL = 50.0
CURRENT_SEASON = 2025

# --- WR weights / features --------------------------------------------------
WR_W_PASS_RATE = 0.38
WR_W_PASS_EPA = 0.37
WR_W_TOP_2_EXCL_SELF = -0.25

# --- RB weights / features --------------------------------------------------
# Single feature (team_ybc_att). Could have weight 1.0 since it's the only
# signal; here for symmetry with WR composite structure.
RB_W_YBC_ATT = 1.0

# --- TE weights / features --------------------------------------------------
# Normalized empirical weights from TE residual regression. Dropped:
# team_pass_epa (near-zero contribution, unstable) and top_2_excl_self
# (near-zero R^2, unstable). team_cpoe and team_pass_rate are the two
# survivors with stable + positive coefs.
TE_W_PASS_RATE = 0.52
TE_W_CPOE = 0.48

# --- QB weights / features --------------------------------------------------
# Path A "minimal QB Team": single feature (team_pass_rate). User-chosen
# after empirical residual regression showed that no team feature has
# meaningful signal for QB residual (OOF R^2 was negative). team_pass_rate
# was the only feature with non-trivial standalone signal (R^2 1.14%) and
# stable sign. Other candidates (team_avg_separation, team_avg_yac,
# team_sack_rate, team_pressure_pct) were noise.
QB_W_PASS_RATE = 1.0


def _top2_target_share_excl_self(ext: pd.DataFrame) -> dict[tuple, float]:
    """Per-(team, season, espn_id): sum of top 2 target shares from OTHER team-mates."""
    target_getters = ext[ext["target_share"].notna() & (ext["target_share"] > 0)]
    out: dict[tuple, float] = {}
    for (team, season), grp in target_getters.groupby(["team", "season"]):
        sorted_shares = grp.sort_values("target_share", ascending=False)
        ids = sorted_shares["espn_id"].astype("Int64").tolist()
        shares = sorted_shares["target_share"].tolist()
        for idx, espn_id in enumerate(ids):
            if pd.isna(espn_id):
                continue
            others = shares[:idx] + shares[idx + 1:]
            out[(team, int(season), int(espn_id))] = float(sum(others[:2]))
    return out


def _team_ybc_att_lookup(ext: pd.DataFrame) -> dict[tuple, float]:
    """Per-(team, season): carry-weighted avg ybc_att across the team's RBs.

    Pure O-line proxy -- the player's own ybc_att captures both O-line + RB
    vision; aggregating across the team's backfield isolates the O-line.
    """
    rb_data = ext[
        (ext["position_group"] == "RB")
        & ext["ybc_att"].notna()
        & ext["carries"].notna()
        & (ext["carries"] > 0)
    ]
    out: dict[tuple, float] = {}
    for (team, season), grp in rb_data.groupby(["team", "season"]):
        w = grp["carries"].sum()
        if w > 0:
            out[(team, int(season))] = float((grp["ybc_att"] * grp["carries"]).sum() / w)
    return out


# --- WR artifacts -----------------------------------------------------------
@lru_cache(maxsize=1)
def _wr_team_artifacts() -> dict:
    """Cached WR artifacts (Phase 1C)."""
    ext = extended_training_frame()
    wr = ext[ext["position_group"] == "WR"]

    player_team_ctx: dict[tuple, dict] = {}
    for _, r in wr.iterrows():
        if pd.isna(r.get("espn_id")) or pd.isna(r.get("team")):
            continue
        if pd.isna(r.get("team_pass_rate")) or pd.isna(r.get("team_pass_epa")):
            continue
        player_team_ctx[(int(r["espn_id"]), int(r["season"]))] = {
            "team": r["team"],
            "team_pass_rate": float(r["team_pass_rate"]),
            "team_pass_epa": float(r["team_pass_epa"]),
        }

    top2 = _top2_target_share_excl_self(ext)

    rows = []
    for (espn_id, season), ctx in player_team_ctx.items():
        t2 = top2.get((ctx["team"], season, espn_id))
        if t2 is None or pd.isna(t2):
            continue
        rows.append({
            "season": season,
            "team_pass_rate": ctx["team_pass_rate"],
            "team_pass_epa": ctx["team_pass_epa"],
            "top_2_excl_self": t2,
        })
    feat_df = pd.DataFrame(rows)

    cols = ("team_pass_rate", "team_pass_epa", "top_2_excl_self")
    z_stats = {c: {"mean": float(feat_df[c].mean()), "std": float(feat_df[c].std())} for c in cols}

    def _composite(rate, epa, t2):
        pr_z = (rate - z_stats["team_pass_rate"]["mean"]) / z_stats["team_pass_rate"]["std"]
        pe_z = (epa - z_stats["team_pass_epa"]["mean"]) / z_stats["team_pass_epa"]["std"]
        t2_z = (t2 - z_stats["top_2_excl_self"]["mean"]) / z_stats["top_2_excl_self"]["std"]
        return (WR_W_PASS_RATE * pr_z + WR_W_PASS_EPA * pe_z
                + WR_W_TOP_2_EXCL_SELF * t2_z)

    feat_df["composite"] = feat_df.apply(
        lambda r: _composite(r["team_pass_rate"], r["team_pass_epa"], r["top_2_excl_self"]), axis=1)
    season_bounds = {
        int(s): {"min": float(g["composite"].min()), "max": float(g["composite"].max())}
        for s, g in feat_df.groupby("season")
    }

    return {
        "player_team_ctx": player_team_ctx,
        "top2": top2,
        "z_stats": z_stats,
        "season_bounds": season_bounds,
        "composite_fn": _composite,
    }


# --- RB artifacts -----------------------------------------------------------
@lru_cache(maxsize=1)
def _rb_team_artifacts() -> dict:
    """Cached RB artifacts (Phase 2)."""
    ext = extended_training_frame()
    team_ybc = _team_ybc_att_lookup(ext)

    # Per-(espn_id, season) team lookup for RBs
    rb = ext[ext["position_group"] == "RB"]
    player_team: dict[tuple, str] = {}
    for _, r in rb.iterrows():
        if pd.isna(r.get("espn_id")) or pd.isna(r.get("team")):
            continue
        player_team[(int(r["espn_id"]), int(r["season"]))] = r["team"]

    # Pooled z-stats for team_ybc_att across all (team, season) values
    ybc_vals = pd.Series(list(team_ybc.values()))
    z_stats = {"team_ybc_att": {"mean": float(ybc_vals.mean()), "std": float(ybc_vals.std())}}

    def _composite(ybc):
        ybc_z = (ybc - z_stats["team_ybc_att"]["mean"]) / z_stats["team_ybc_att"]["std"]
        return RB_W_YBC_ATT * ybc_z

    # Per-season min/max for [0, 100] mapping using all RB-team-seasons
    season_records: dict[int, list[float]] = {}
    for (team, season), ybc in team_ybc.items():
        comp = _composite(ybc)
        season_records.setdefault(season, []).append(comp)
    season_bounds = {
        s: {"min": min(vs), "max": max(vs)} for s, vs in season_records.items()
    }

    return {
        "team_ybc": team_ybc,
        "player_team": player_team,
        "z_stats": z_stats,
        "season_bounds": season_bounds,
        "composite_fn": _composite,
    }


# --- Per-position scoring ---------------------------------------------------
def _lookup_wr_team_value(espn_id, season: int = CURRENT_SEASON) -> float:
    art = _wr_team_artifacts()
    if pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)
    ctx = art["player_team_ctx"].get((espn_id, season))
    if ctx is None:
        return NEUTRAL
    t2 = art["top2"].get((ctx["team"], season, espn_id))
    if t2 is None or pd.isna(t2):
        t2 = art["z_stats"]["top_2_excl_self"]["mean"]
    composite = art["composite_fn"](ctx["team_pass_rate"], ctx["team_pass_epa"], float(t2))
    bounds = art["season_bounds"].get(season)
    if not bounds or bounds["max"] == bounds["min"]:
        return NEUTRAL
    val = 100.0 * (composite - bounds["min"]) / (bounds["max"] - bounds["min"])
    return float(max(0.0, min(100.0, val)))


def _lookup_rb_team_value(espn_id, season: int = CURRENT_SEASON) -> float:
    art = _rb_team_artifacts()
    if pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)
    team = art["player_team"].get((espn_id, season))
    if team is None:
        return NEUTRAL
    ybc = art["team_ybc"].get((team, season))
    if ybc is None or pd.isna(ybc):
        return NEUTRAL
    composite = art["composite_fn"](ybc)
    bounds = art["season_bounds"].get(season)
    if not bounds or bounds["max"] == bounds["min"]:
        return NEUTRAL
    val = 100.0 * (composite - bounds["min"]) / (bounds["max"] - bounds["min"])
    return float(max(0.0, min(100.0, val)))


@lru_cache(maxsize=1)
def _te_team_artifacts() -> dict:
    """Cached TE artifacts (Phase 3). Uses team_pass_rate + team_cpoe only."""
    ext = extended_training_frame()
    te = ext[ext["position_group"] == "TE"]

    player_team_ctx: dict[tuple, dict] = {}
    for _, r in te.iterrows():
        if pd.isna(r.get("espn_id")) or pd.isna(r.get("team")):
            continue
        if pd.isna(r.get("team_pass_rate")) or pd.isna(r.get("team_cpoe")):
            continue
        player_team_ctx[(int(r["espn_id"]), int(r["season"]))] = {
            "team": r["team"],
            "team_pass_rate": float(r["team_pass_rate"]),
            "team_cpoe": float(r["team_cpoe"]),
        }

    rows = []
    for (espn_id, season), ctx in player_team_ctx.items():
        rows.append({
            "season": season,
            "team_pass_rate": ctx["team_pass_rate"],
            "team_cpoe": ctx["team_cpoe"],
        })
    feat_df = pd.DataFrame(rows)
    cols = ("team_pass_rate", "team_cpoe")
    z_stats = {c: {"mean": float(feat_df[c].mean()), "std": float(feat_df[c].std())} for c in cols}

    def _composite(rate, cpoe):
        pr_z = (rate - z_stats["team_pass_rate"]["mean"]) / z_stats["team_pass_rate"]["std"]
        co_z = (cpoe - z_stats["team_cpoe"]["mean"]) / z_stats["team_cpoe"]["std"]
        return TE_W_PASS_RATE * pr_z + TE_W_CPOE * co_z

    feat_df["composite"] = feat_df.apply(
        lambda r: _composite(r["team_pass_rate"], r["team_cpoe"]), axis=1)
    season_bounds = {
        int(s): {"min": float(g["composite"].min()), "max": float(g["composite"].max())}
        for s, g in feat_df.groupby("season")
    }
    return {
        "player_team_ctx": player_team_ctx,
        "z_stats": z_stats,
        "season_bounds": season_bounds,
        "composite_fn": _composite,
    }


def _lookup_te_team_value(espn_id, season: int = CURRENT_SEASON) -> float:
    art = _te_team_artifacts()
    if pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)
    ctx = art["player_team_ctx"].get((espn_id, season))
    if ctx is None:
        return NEUTRAL
    composite = art["composite_fn"](ctx["team_pass_rate"], ctx["team_cpoe"])
    bounds = art["season_bounds"].get(season)
    if not bounds or bounds["max"] == bounds["min"]:
        return NEUTRAL
    val = 100.0 * (composite - bounds["min"]) / (bounds["max"] - bounds["min"])
    return float(max(0.0, min(100.0, val)))


@lru_cache(maxsize=1)
def _qb_team_artifacts() -> dict:
    """Cached QB artifacts (Phase 4). Path A minimal: team_pass_rate only."""
    ext = extended_training_frame()
    qb = ext[ext["position_group"] == "QB"]

    # Per-(espn_id, season) team_pass_rate lookup
    player_team_ctx: dict[tuple, dict] = {}
    for _, r in qb.iterrows():
        if pd.isna(r.get("espn_id")) or pd.isna(r.get("team")):
            continue
        if pd.isna(r.get("team_pass_rate")):
            continue
        player_team_ctx[(int(r["espn_id"]), int(r["season"]))] = {
            "team": r["team"],
            "team_pass_rate": float(r["team_pass_rate"]),
        }

    # Per-season min/max for [0, 100] mapping
    season_records: dict[int, list[float]] = {}
    for (espn_id, season), ctx in player_team_ctx.items():
        season_records.setdefault(season, []).append(ctx["team_pass_rate"])
    season_bounds = {
        s: {"min": min(vs), "max": max(vs)} for s, vs in season_records.items()
    }
    return {
        "player_team_ctx": player_team_ctx,
        "season_bounds": season_bounds,
    }


def _lookup_qb_team_value(espn_id, season: int = CURRENT_SEASON) -> float:
    """QB team_value = season-normalised team_pass_rate (single feature)."""
    art = _qb_team_artifacts()
    if pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)
    ctx = art["player_team_ctx"].get((espn_id, season))
    if ctx is None:
        return NEUTRAL
    bounds = art["season_bounds"].get(season)
    if not bounds or bounds["max"] == bounds["min"]:
        return NEUTRAL
    val = 100.0 * (ctx["team_pass_rate"] - bounds["min"]) / (bounds["max"] - bounds["min"])
    return float(max(0.0, min(100.0, val)))


_LOOKUPS: dict[str, Callable] = {
    "WR": _lookup_wr_team_value,
    "RB": _lookup_rb_team_value,
    "TE": _lookup_te_team_value,
    "QB": _lookup_qb_team_value,
}


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Team score per player in [0, 100]."""
    out = players.copy()
    if position in _LOOKUPS:
        lookup = _LOOKUPS[position]
        out["team_value"] = [lookup(eid, CURRENT_SEASON) for eid in out["espn_id"]]
    else:
        out["team_value"] = NEUTRAL  # QB / TE in Phases 3-4
    return out


if __name__ == "__main__":
    ext = extended_training_frame()
    # WR snapshot
    print("\n--- 2025 WR team_value (top-target-share rep per team) ---")
    wr = ext[(ext["position_group"] == "WR") & (ext["season"] == 2025)].dropna(subset=["target_share", "espn_id"])
    reps_wr = wr.sort_values("target_share", ascending=False).drop_duplicates("team")
    reps_wr["team_value"] = [_lookup_wr_team_value(int(e), 2025) for e in reps_wr["espn_id"].astype(int)]
    print(reps_wr[["team", "name", "team_value"]].sort_values("team_value", ascending=False).to_string(index=False))
    # RB snapshot
    print("\n--- 2025 RB team_value (top-carries rep per team) ---")
    rb = ext[(ext["position_group"] == "RB") & (ext["season"] == 2025)].dropna(subset=["carries", "espn_id"])
    reps_rb = rb.sort_values("carries", ascending=False).drop_duplicates("team")
    reps_rb["team_value"] = [_lookup_rb_team_value(int(e), 2025) for e in reps_rb["espn_id"].astype(int)]
    print(reps_rb[["team", "name", "team_value"]].sort_values("team_value", ascending=False).to_string(index=False))
