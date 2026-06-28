"""Production component — past on-field production scored to [0, 100].

Per-position models fit on the v2 extended training frame (2016-2025; see
docs/methodology/production.md). Each position uses ONLY player-driven
features — no age/years_exp (Age component owns those), no team stats (Team
component owns those), no cross-position contamination (e.g. WR-specific NGS
stats excluded from the RB model).

Per-player Production = recency-weighted blend of per-season scores; each
per-season score is the model's predicted PPG mapped linearly within that
season's positional distribution between 0 PPG (floor) and the season's
best (top → 100). Same methodology across positions; only the feature set
changes.

WR and RB implemented (Phases 1B, 2). QB / TE land in Phases 3-4.
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
# + combine). Excluded: age, years_exp (Age), team_* (Team), cross-position
# stats (e.g. WR NGS stats out of RB model; QB stats out of receiver models).
WR_FEATURES: tuple[str, ...] = (
    # usage
    "target_share", "wopr", "racr", "snap_pct",
    # NGS receiving (player skill)
    "avg_separation", "yac_above_expected", "adot", "catch_pct",
    # EPA on the player's own catches
    "receiving_epa",
    # draft + combine
    "draft_round", "draft_pick", "draft_value", "undrafted",
    "forty", "vertical", "broad_jump",
)

# Per user (Phase 2): ybc_att dropped from RB Production -- it's an O-line
# proxy and goes to the RB Team component instead. yac_att kept (after-contact
# yards = player power/breaks tackles). receiving stats kept since dynasty
# value depends on 3-down-back role.
RB_FEATURES: tuple[str, ...] = (
    # volume
    "carries", "snap_pct", "target_share", "receptions",
    # rushing skill
    "rushing_epa", "ryoe_per_att", "time_to_los", "yac_att",
    # receiving (3-down backs)
    "receiving_epa", "catch_pct",
    # draft + combine
    "draft_round", "draft_pick", "draft_value", "undrafted",
    "forty", "vertical", "broad_jump",
)

POSITION_FEATURES: dict[str, tuple[str, ...]] = {
    "WR": WR_FEATURES,
    "RB": RB_FEATURES,
    # TE Phase 3: TEs are receivers; reuse the WR feature set unchanged.
    "TE": WR_FEATURES,
    # QB in Phase 4
}

# Geometric recency decay applied to per-season scores. TODO: tune empirically.
RECENCY_WEIGHTS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

# Mid-season rollover anchor: latest fully-completed NFL season.
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


def _position_training(position: str) -> pd.DataFrame:
    """All position-seasons in the v2 extended window with games >= MIN_GAMES."""
    df = extended_training_frame()
    df = df[df["position_group"] == position].copy()
    df = df[df["games"].fillna(0) >= MIN_GAMES]
    df = df.dropna(subset=["ppg"])
    return df.reset_index(drop=True)


def _season_anchors(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Per-season (floor, top) for the position pool. Full-range mapping: floor = 0."""
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


@lru_cache(maxsize=4)
def _position_artifacts(position: str):
    """Cached per-position: (model, feats, anchors, train_df, train_by_id)."""
    train = _position_training(position)
    feats = [c for c in POSITION_FEATURES[position] if c in train.columns]
    model = _pipeline().fit(train[feats], train["ppg"])
    anchors = _season_anchors(train)
    train_by_id = {
        int(espn_id): grp.sort_values("season", ascending=False)
        for espn_id, grp in train.dropna(subset=["espn_id"]).groupby("espn_id")
    }
    return model, feats, anchors, train, train_by_id


def position_oof_r2(position: str) -> tuple[float, float]:
    """5-fold OOF R^2 + MAE for the position model (diagnostic)."""
    from sklearn.metrics import mean_absolute_error, r2_score

    train = _position_training(position)
    feats = [c for c in POSITION_FEATURES[position] if c in train.columns]
    cv = KFold(5, shuffle=True, random_state=0)
    oof = cross_val_predict(_pipeline(), train[feats], train["ppg"], cv=cv)
    return float(r2_score(train["ppg"], oof)), float(mean_absolute_error(train["ppg"], oof))


def _score_one_player(
    espn_id, model, feats: Iterable[str],
    anchors: dict[int, tuple[float, float]], train_by_id,
) -> float:
    """Score a single player by recency-blending their per-season predictions.

    Falls back to a draft/combine-only synthetic feature vector when the
    player has no NFL history (rookies / new arrivals); HistGBR is NaN-native,
    so the model still produces a defensible prediction from sparse features.
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
                floor, top = anchors[season]
                per_season.append(_ppg_to_score(float(pred), floor, top))
        if per_season:
            return _recency_blend(per_season)

    # No NFL history -- use draft + combine only (everything else NaN)
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
        floor, top = anchors[CURRENT_SEASON]
        return _ppg_to_score(pred, floor, top)
    return NEUTRAL


def _score_position(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Production scoring for a slice of contract players at one position."""
    out = players.copy()
    model, feats, anchors, _, train_by_id = _position_artifacts(position)
    out["production_value"] = [
        _score_one_player(eid, model, feats, anchors, train_by_id)
        for eid in out["espn_id"]
    ]
    return out


def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Production score per player in [0, 100]."""
    if position in POSITION_FEATURES:
        return _score_position(players, position)
    out = players.copy()
    out["production_value"] = NEUTRAL  # QB / TE land in Phases 3-4
    return out


if __name__ == "__main__":
    for pos in ("WR", "RB"):
        r2, mae = position_oof_r2(pos)
        print(f"{pos} Production model -- OOF R^2 = {r2:.3f}, MAE = {mae:.2f} PPG")
    print()
    for pos in ("WR", "RB"):
        _, _, anchors, _, _ = _position_artifacts(pos)
        print(f"{pos} per-season PPG anchors (0 -> top):")
        for s in sorted(anchors):
            _, top = anchors[s]
            print(f"  {s}: top = {top:.2f}")
        print()
