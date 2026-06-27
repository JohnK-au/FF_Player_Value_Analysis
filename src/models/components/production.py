"""Production component — past on-field production scored to [0, 100].

Per-position models fit on the v2 extended training frame (2016-2025; see
docs/methodology/production.md). Each position uses ONLY player-driven
features — no age/years_exp (Age component owns those), no team stats (Team
component owns those), no QB-throwing-context for receivers.

Per-player Production = recency-weighted blend of per-season scores; each
per-season score is the model's predicted PPG mapped linearly within that
season's positional distribution between the position's replacement floor
(score 0) and the season's best (score 100).

WR is implemented (Phase 1B). RB / QB / TE land in Phases 2-4.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data.nflverse import combine_athleticism, draft_capital
from src.data.population import extended_training_frame

NEUTRAL = 50.0

# Per-position feature lists. ONLY player-driven signals (usage + skill + draft
# + combine). Excluded: age, years_exp (Age), team_* (Team), QB-context stats
# for receivers (Team).
WR_FEATURES: tuple[str, ...] = (
    # usage
    "target_share", "wopr", "racr", "snap_pct",
    # NGS receiving (player skill)
    "avg_separation", "yac_above_expected", "adot", "catch_pct",
    # EPA on the player's own catches
    "receiving_epa",
    # draft + combine (effective weight decays naturally as production accumulates)
    "draft_round", "draft_pick", "draft_value", "undrafted",
    "forty", "vertical", "broad_jump",
)

# Geometric recency decay applied to per-season scores. TODO: tune empirically.
RECENCY_WEIGHTS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

# Mid-season rollover anchor: latest fully-completed NFL season. Flips to the
# active year once ~Week 6 of new-season data is in. As of June 2026: 2025.
CURRENT_SEASON = 2025
MIN_GAMES = 4  # ignore tiny-sample seasons when fitting


def _pipeline() -> Pipeline:
    """HistGBR; NaN-native so missing NGS/PFR pre-2018 are handled."""
    return Pipeline([
        ("gbm", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            random_state=0,
        )),
    ])


def _wr_training() -> pd.DataFrame:
    """All WR-seasons in the v2 extended window with games >= MIN_GAMES."""
    df = extended_training_frame()
    df = df[df["position_group"] == "WR"].copy()
    df = df[df["games"].fillna(0) >= MIN_GAMES]
    df = df.dropna(subset=["ppg"])
    return df.reset_index(drop=True)


def _season_anchors(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Per-season (floor, top) for the position pool. Full-range mapping: floor = 0.

    No replacement-based cutoff -- every player gets a unique score within the
    season's distribution. 0 PPG -> 0, season's best WR predicted PPG -> 100.
    Preserves signal across all tiers (no 0 cluster); the elite cliff is
    preserved naturally by the underlying PPG distribution.
    """
    anchors = {}
    for season, sub in df.groupby("season"):
        ppg = sub["ppg"].dropna()
        if len(ppg) == 0:
            continue
        anchors[int(season)] = (0.0, float(ppg.max()))
    return anchors


def _ppg_to_score(predicted_ppg: float, floor: float, top: float) -> float:
    """Linear: floor -> 0, top -> 100, clipped to [0, 100]."""
    if top <= floor or pd.isna(predicted_ppg):
        return NEUTRAL
    return float(max(0.0, min(100.0, 100.0 * (predicted_ppg - floor) / (top - floor))))


def _recency_blend(per_season_scores: list[float]) -> float:
    """Geometric weighted average over per-season scores (most-recent first)."""
    if not per_season_scores:
        return NEUTRAL
    w = RECENCY_WEIGHTS[: len(per_season_scores)]
    wsum = sum(w)
    return sum(s * wi for s, wi in zip(per_season_scores, w)) / wsum


@lru_cache(maxsize=1)
def _wr_artifacts():
    """Cached: (model, feats, anchors, train_df). Built once per Python process."""
    train = _wr_training()
    feats = [c for c in WR_FEATURES if c in train.columns]
    model = _pipeline().fit(train[feats], train["ppg"])
    anchors = _season_anchors(train)
    return model, feats, anchors, train


def wr_oof_r2() -> tuple[float, float]:
    """5-fold OOF R^2 + MAE (diagnostic)."""
    from sklearn.metrics import mean_absolute_error, r2_score

    train = _wr_training()
    feats = [c for c in WR_FEATURES if c in train.columns]
    cv = KFold(5, shuffle=True, random_state=0)
    oof = cross_val_predict(_pipeline(), train[feats], train["ppg"], cv=cv)
    return float(r2_score(train["ppg"], oof)), float(mean_absolute_error(train["ppg"], oof))


def _score_one_wr(espn_id, model, feats: Iterable[str],
                  anchors: dict[int, tuple[float, float]], train_by_id) -> float:
    """Score a single WR by recency-blending their per-season predictions.

    Falls back to a draft/combine-only synthetic feature vector when the player
    has no NFL history (rookies / new arrivals); HistGBR is NaN-native, so the
    model still produces a defensible prediction from sparse features.
    """
    if espn_id is None or pd.isna(espn_id):
        return NEUTRAL
    espn_id = int(espn_id)
    history = train_by_id.get(espn_id)
    if history is not None and not history.empty:
        recent = history.head(len(RECENCY_WEIGHTS))
        preds = model.predict(recent[list(feats)])
        per_season = []
        for pred, season in zip(preds, recent["season"].values):
            season = int(season)
            if season in anchors:
                repl, top = anchors[season]
                per_season.append(_ppg_to_score(float(pred), repl, top))
        if per_season:
            return _recency_blend(per_season)

    # No NFL history — use the player's draft + combine only (everything else NaN)
    feature_row = {f: np.nan for f in feats}
    draft = draft_capital().set_index("espn_id")
    combine = combine_athleticism().set_index("espn_id")
    if espn_id in draft.index:
        for col in ("draft_round", "draft_pick", "draft_value", "undrafted"):
            if col in draft.columns and col in feature_row:
                feature_row[col] = draft.loc[espn_id, col]
    if espn_id in combine.index:
        for col in ("forty", "vertical", "broad_jump"):
            if col in combine.columns and col in feature_row:
                feature_row[col] = combine.loc[espn_id, col]
    pred = float(model.predict(pd.DataFrame([feature_row]))[0])
    if CURRENT_SEASON in anchors:
        repl, top = anchors[CURRENT_SEASON]
        return _ppg_to_score(pred, repl, top)
    return NEUTRAL


def _score_wr(players: pd.DataFrame) -> pd.DataFrame:
    """WR Production scoring for a slice of contract players."""
    out = players.copy()
    model, feats, anchors, train = _wr_artifacts()
    train_by_id = {
        int(espn_id): grp.sort_values("season", ascending=False)
        for espn_id, grp in train.dropna(subset=["espn_id"]).groupby("espn_id")
    }
    out["production_value"] = [
        _score_one_wr(eid, model, feats, anchors, train_by_id)
        for eid in out["espn_id"]
    ]
    return out


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Production score per player in [0, 100]."""
    if position == "WR":
        return _score_wr(players)
    out = players.copy()
    out["production_value"] = NEUTRAL  # RB/QB/TE land in Phases 2-4
    return out


if __name__ == "__main__":
    r2, mae = wr_oof_r2()
    print(f"WR Production model -- OOF R^2 = {r2:.3f}, MAE = {mae:.2f} PPG")
    _, _, anchors, _ = _wr_artifacts()
    print("\nPer-season WR PPG anchors (replacement -> top):")
    for s in sorted(anchors):
        repl, top = anchors[s]
        print(f"  {s}: {repl:.2f} -> {top:.2f}")
