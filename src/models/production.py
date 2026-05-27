"""Production model + value-over-replacement (the production-anchored basis).

Two pieces, both trained/derived **league-wide** on all NFL skill players ×
2022–2025 (thousands of rows), not just our ~156 rostered players:

1. **Production model** — expected fantasy PPG (our scoring) from usage,
   efficiency, draft capital, age and athletic profile. Its out-of-fold R² is a
   *diagnostic* of how well production is explained from a player's profile
   (high is good here — the opposite stance to the salary model, which we do not
   want to fit too well).
2. **Replacement levels + VOR** — using the league's actual starting lineup
   (rules §4: per team QB, 2×RB, 2×WR, WR/TE, TE, RB/WR/TE flex), the worst
   startable player at each position sets the replacement baseline, and
   ``vor = ppg − replacement``. VOR is the objective, market-independent measure
   of on-field value the fair-value engine prices against.

Feature-name lists live here (the lower-level module); ``models.value`` imports
them so there's a single source of truth and no circular import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..data.advanced import ADV_SEASON
from ..data.population import SKILL, season_frame, season_production, training_frame

# --- Feature names (canonical; models.value imports these) --------------------
ADV_NUM = [
    "target_share", "wopr", "racr", "receiving_epa", "carries", "rushing_epa",
    "passing_epa", "avg_separation", "yac_above_expected", "adot", "catch_pct",
    "ryoe_per_att", "time_to_los", "cpoe", "ybc_att", "yac_att", "pressure_pct",
    "on_tgt_pct", "snap_pct",
]
DRAFT_NUM = ["draft_value", "draft_round", "undrafted"]
COMBINE_NUM = ["forty", "vertical", "broad_jump"]

# --- Production model ----------------------------------------------------------
PROD_CAT = ["position_group"]
PROD_NUM = ["age", "years_exp"] + ADV_NUM + DRAFT_NUM + COMBINE_NUM
PROD_TARGET = "ppg"
MIN_GAMES_TRAIN = 4  # a stable season PPG needs a few games


def _pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), PROD_CAT)],
        remainder="passthrough",
    )
    gbm = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["position_group"].isin(SKILL)].copy()
    d = d[d[PROD_TARGET].notna() & (d["games"].fillna(0) >= MIN_GAMES_TRAIN)]
    return d.reset_index(drop=True)


def oof_predictions(d: pd.DataFrame) -> np.ndarray:
    X = d[PROD_CAT + PROD_NUM]
    cv = KFold(5, shuffle=True, random_state=0)
    return cross_val_predict(_pipeline(), X, d[PROD_TARGET], cv=cv)


def fit_production_model(df: pd.DataFrame | None = None) -> tuple[Pipeline, pd.DataFrame]:
    """Fit the PPG model on the full league-wide training frame."""
    d = _prep(training_frame() if df is None else df)
    return _pipeline().fit(d[PROD_CAT + PROD_NUM], d[PROD_TARGET]), d


def expected_ppg(season: int = ADV_SEASON, model: Pipeline | None = None) -> pd.DataFrame:
    """Model-expected PPG for a season's skill players (de-noised production)."""
    if model is None:
        model, _ = fit_production_model()
    fr = season_frame(season)
    fr = fr[fr["position_group"].isin(SKILL)].copy()
    fr["exp_ppg"] = model.predict(fr[PROD_CAT + PROD_NUM]).round(2)
    return fr[["espn_id", "name", "position_group", "ppg", "exp_ppg"]]


# --- Replacement levels + VOR --------------------------------------------------
N_TEAMS = 8
# Per-team starting skill slots (rules §4). Flex slots are filled after the fixed
# ones, more-restrictive flex first, by the best remaining eligible player.
FIXED = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX = [(("WR", "TE"), 1), (("RB", "WR", "TE"), 1)]
MIN_GAMES_POOL = 4  # ignore tiny-sample fluke PPGs when setting replacement level


def replacement_levels(pool: pd.DataFrame, value_col: str = "ppg") -> dict[str, float]:
    """Replacement PPG per position = best startable player who makes no lineup.

    Fills all 8 teams' starting slots from the top of the value-sorted pool, then
    reads off the highest-value *un-started* player at each position.
    """
    df = (
        pool.dropna(subset=[value_col, "position_group"])
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )
    started: set[int] = set()

    def fill(positions: list[str], count: int) -> None:
        avail = df.index[df["position_group"].isin(positions) & ~df.index.to_series().isin(started)]
        started.update(avail[:count])

    for pos, per in FIXED.items():
        fill([pos], per * N_TEAMS)
    for positions, per in FLEX:
        fill(list(positions), per * N_TEAMS)

    repl: dict[str, float] = {}
    for pos in SKILL:
        bench = df[(df["position_group"] == pos) & (~df.index.to_series().isin(started))]
        repl[pos] = float(bench[value_col].iloc[0]) if len(bench) else \
            float(df.loc[df["position_group"] == pos, value_col].min())
    return repl


def vor_table(
    season: int = ADV_SEASON, value_col: str = "ppg", min_games: int = MIN_GAMES_POOL
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-player value-over-replacement for a season's NFL skill pool."""
    pool = season_production(season)
    startable = pool[pool["games"].fillna(0) >= min_games]
    repl = replacement_levels(startable, value_col)
    out = pool.copy()
    out["replacement"] = out["position_group"].map(repl).round(2)
    out["vor"] = (out[value_col] - out["replacement"]).round(2)
    return out, repl


def weekly_consistency(season: int) -> pd.DataFrame:
    """Per-player **downside deviation** of weekly fantasy points (floor risk).

    The RMS of shortfalls *below a player's own average* across the weeks they
    played — it penalizes bust weeks (which lose head-to-head matchups) without
    punishing big ceiling weeks (a 90-pt week isn't "risk"). Keyed by ``espn_id``,
    computed from the 13-week box scores (rostered players only). ``stdev`` is the
    plain symmetric volatility, kept for reference.
    """
    from ..data.dataset import load_weekly

    w = load_weekly()
    w = w[w["season"] == season].copy()
    w["points"] = pd.to_numeric(w["points"], errors="coerce")
    w = w[w["points"].notna() & (w["points"] != 0)]  # drop DNP/0 weeks (matches PPG basis)

    def _downside(pts: pd.Series) -> float:
        short = (pts - pts.mean()).clip(upper=0)  # below-mean shortfalls only
        return float(np.sqrt((short ** 2).mean()))

    g = (
        w.groupby("espn_id")["points"]
        .agg(downside=_downside, stdev="std")
        .round(2)
        .reset_index()
    )
    g["espn_id"] = pd.to_numeric(g["espn_id"], errors="coerce").astype("Int64")
    return g.dropna(subset=["espn_id"])


if __name__ == "__main__":
    model, d = fit_production_model()
    oof = oof_predictions(d)
    print(f"Production model — trained on {len(d)} skill player-seasons "
          f"({d['season'].nunique()} seasons)")
    print(f"  out-of-fold R²: {r2_score(d[PROD_TARGET], oof):.3f} "
          f"(MAE {mean_absolute_error(d[PROD_TARGET], oof):.2f} PPG)")
    print("  (high R² is GOOD here — it means production is well-explained, "
          "unlike the salary model)")

    X = d[PROD_CAT + PROD_NUM]
    imp = permutation_importance(model, X, d[PROD_TARGET], n_repeats=8, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1][:10]
    names = PROD_CAT + PROD_NUM
    print("\nTop drivers of production (permutation importance):")
    for i in order:
        print(f"  {names[i]:<20} {imp.importances_mean[i]:.2f}")

    _, repl = vor_table(ADV_SEASON)
    print(f"\nReplacement PPG by position ({ADV_SEASON}, 8-team starting lineup):")
    for pos, v in repl.items():
        print(f"  {pos}: {v:.2f}")
