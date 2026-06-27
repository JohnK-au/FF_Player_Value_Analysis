"""Team component — offensive environment scoring in [0, 100].

Captures the quality of the offense each player operates within. Different
position groups use different feature sets and weights; WR is implemented
(Phase 1C). RB / QB / TE land in Phases 2-4.

WR team_value = weighted z-scored combo of three signals, normalised within
season to [0, 100]:
    +0.38 * team_pass_rate_z         (passing volume -- more chances per game)
    +0.37 * team_pass_epa_z          (passing efficiency -- good QB lifts WRs)
    -0.25 * top_2_target_share_excl_self_z   (competition from OTHER team WRs)

Weights are EMPIRICALLY DERIVED via residual regression: fit Production model
on WR player features only, compute residuals, regress residuals on team
features (1,691 WR-seasons with complete team features across 2016-2025).
See docs/methodology/team.md for full methodology + diagnostics.

team_cpoe was dropped (wrong sign in the regression, near-zero contribution).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.population import extended_training_frame

NEUTRAL = 50.0

# Empirical weights from the residual regression. Re-derive when extending seasons.
WR_W_PASS_RATE = 0.38
WR_W_PASS_EPA = 0.37
WR_W_TOP_2_EXCL_SELF = -0.25

# Latest fully-completed NFL season anchor (mirrors production.CURRENT_SEASON).
CURRENT_SEASON = 2025

_FEATURES = ("team_pass_rate", "team_pass_epa", "top_2_excl_self")
_WEIGHTS = {
    "team_pass_rate": WR_W_PASS_RATE,
    "team_pass_epa": WR_W_PASS_EPA,
    "top_2_excl_self": WR_W_TOP_2_EXCL_SELF,
}


def _top2_excl_self_lookup(ext: pd.DataFrame) -> dict[tuple, float]:
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


@lru_cache(maxsize=1)
def _wr_team_artifacts() -> dict:
    """Cached: precomputed lookup tables + z-stats + per-season composite bounds.

    Built once per Python process. All subsequent score() calls are O(1)
    dict lookups, no DataFrame scans.
    """
    ext = extended_training_frame()
    wr = ext[ext["position_group"] == "WR"].copy()

    # per-(espn_id, season) player team-context (O(1) lookup in score())
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

    # per-(team, season, espn_id) top-2-excl-self lookup
    top2 = _top2_excl_self_lookup(ext)

    # Build per-row feature frame for z-stat fitting + per-season composite ranges
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

    # pooled z-stats (matches the residual regression's feature scaling)
    z_stats = {c: {"mean": float(feat_df[c].mean()), "std": float(feat_df[c].std())}
               for c in _FEATURES}

    # raw composite per row, then per-season min/max for [0, 100] mapping
    def _composite(rate: float, epa: float, top2: float) -> float:
        pr_z = (rate - z_stats["team_pass_rate"]["mean"]) / z_stats["team_pass_rate"]["std"]
        pe_z = (epa - z_stats["team_pass_epa"]["mean"]) / z_stats["team_pass_epa"]["std"]
        t2_z = (top2 - z_stats["top_2_excl_self"]["mean"]) / z_stats["top_2_excl_self"]["std"]
        return (WR_W_PASS_RATE * pr_z + WR_W_PASS_EPA * pe_z
                + WR_W_TOP_2_EXCL_SELF * t2_z)

    feat_df["composite"] = feat_df.apply(
        lambda r: _composite(r["team_pass_rate"], r["team_pass_epa"], r["top_2_excl_self"]),
        axis=1,
    )
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


def _lookup_wr_team_value(espn_id, season: int = CURRENT_SEASON) -> float:
    """team_value [0, 100] for a WR in ``season``. NEUTRAL when data missing."""
    art = _wr_team_artifacts()
    if pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)

    ctx = art["player_team_ctx"].get((espn_id, season))
    if ctx is None:
        return NEUTRAL

    t2 = art["top2"].get((ctx["team"], season, espn_id))
    if t2 is None or pd.isna(t2):
        # Player on team but no targets at all -- treat as average competition
        t2 = art["z_stats"]["top_2_excl_self"]["mean"]

    composite = art["composite_fn"](ctx["team_pass_rate"], ctx["team_pass_epa"], float(t2))
    bounds = art["season_bounds"].get(season)
    if not bounds or bounds["max"] == bounds["min"]:
        return NEUTRAL
    val = 100.0 * (composite - bounds["min"]) / (bounds["max"] - bounds["min"])
    return float(max(0.0, min(100.0, val)))


def _score_wr(players: pd.DataFrame) -> pd.DataFrame:
    """WR Team scoring for a slice of contract players."""
    out = players.copy()
    out["team_value"] = [_lookup_wr_team_value(eid, CURRENT_SEASON) for eid in out["espn_id"]]
    return out


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Team score per player in [0, 100]."""
    if position == "WR":
        return _score_wr(players)
    out = players.copy()
    out["team_value"] = NEUTRAL  # RB/QB/TE in Phases 2-4
    return out


if __name__ == "__main__":
    art = _wr_team_artifacts()
    ext = extended_training_frame()
    wr_2025 = ext[(ext["position_group"] == "WR") & (ext["season"] == 2025)].copy()
    wr_2025 = wr_2025.dropna(subset=["target_share", "espn_id"])
    reps = wr_2025.sort_values("target_share", ascending=False).drop_duplicates("team")
    reps["team_value"] = [
        _lookup_wr_team_value(int(eid), 2025) for eid in reps["espn_id"].astype(int)
    ]
    cols = ["team", "name", "team_pass_epa", "team_pass_rate", "team_value"]
    print("\n2025 team_value (rep = top-target-share WR per team):")
    print(
        reps[cols]
        .sort_values("team_value", ascending=False)
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
