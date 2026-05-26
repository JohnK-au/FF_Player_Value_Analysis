"""Fair-value model.

Predicts a player's *fair* 2026 salary from production + age + position +
advanced usage/efficiency metrics + draft capital + combine, then
**surplus = actual − fair** flags over/under-valued players.

Trained on this league's 2026 contracts (skill positions). Fair values are
**out-of-fold** (each player scored by a model not fit on them), so surplus
isn't circular. Uses HistGradientBoosting, which handles missing values natively
(many advanced/combine metrics are sparse for low-volume players and rookies).
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

from ..config import PROCESSED_DIR
from ..data.dataset import build_model_features

SKILL = ["QB", "RB", "WR", "TE"]
TARGET = "salary_2026"
CAT_FEATURES = ["position_group"]

BASE_NUM = ["age", "ppg_2025", "ppg_2024", "games_2025", "stdev_2025", "years_exp"]
ADV_NUM = [
    "target_share", "wopr", "racr", "receiving_epa", "carries", "rushing_epa",
    "passing_epa", "avg_separation", "yac_above_expected", "adot", "catch_pct",
    "ryoe_per_att", "time_to_los", "cpoe", "ybc_att", "yac_att", "pressure_pct",
    "on_tgt_pct", "snap_pct",
]
DRAFT_NUM = ["draft_value", "draft_round", "undrafted"]
COMBINE_NUM = ["forty", "vertical", "broad_jump"]
NUM_FEATURES = BASE_NUM + ADV_NUM + DRAFT_NUM + COMBINE_NUM  # full set (viz imports this)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["position_group"].isin(SKILL)].copy()
    d = d[d[TARGET].notna() & d["ppg_2025"].notna()]
    d["ppg_2024"] = d["ppg_2024"].fillna(d["ppg_2025"])  # no prior year → assume similar
    # advanced/combine NaNs are left for the model to handle natively.
    return d.reset_index(drop=True)


def _pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
        remainder="passthrough",
    )
    gbm = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def _oof(d: pd.DataFrame, num_features: list[str]) -> np.ndarray:
    X = d[CAT_FEATURES + num_features]
    cv = KFold(5, shuffle=True, random_state=0)
    return cross_val_predict(_pipeline(), X, d[TARGET], cv=cv)


def fair_value_table(num_features: list[str] | None = None) -> tuple[pd.DataFrame, Pipeline]:
    """Per-player fair salary + surplus (out-of-fold) and the fitted model."""
    num_features = num_features or NUM_FEATURES
    d = _prepare(build_model_features())
    d["fair_salary"] = np.clip(_oof(d, num_features), 1, None).round(1)
    d["surplus"] = (d[TARGET] - d["fair_salary"]).round(1)  # + overpaid / − bargain
    model = _pipeline().fit(d[CAT_FEATURES + num_features], d[TARGET])
    return d, model


if __name__ == "__main__":
    d = _prepare(build_model_features())
    base_oof = _oof(d, BASE_NUM)
    full_oof = _oof(d, NUM_FEATURES)
    print(f"n={len(d)} skill players | out-of-fold R²:")
    print(f"  baseline (production+age+position): {r2_score(d[TARGET], base_oof):.3f} "
          f"(MAE {mean_absolute_error(d[TARGET], base_oof):.1f})")
    print(f"  + advanced + draft + combine:       {r2_score(d[TARGET], np.clip(full_oof,1,None)):.3f} "
          f"(MAE {mean_absolute_error(d[TARGET], np.clip(full_oof,1,None)):.1f})")

    # permutation importance on the full model
    X = d[CAT_FEATURES + NUM_FEATURES]
    model = _pipeline().fit(X, d[TARGET])
    imp = permutation_importance(model, X, d[TARGET], n_repeats=10, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    names = (CAT_FEATURES + NUM_FEATURES)
    print("\nTop features (permutation importance):")
    for i in order:
        print(f"  {names[i]:<20} {imp.importances_mean[i]:.1f}")

    tbl, _ = fair_value_table()
    cols = ["player", "team", "position_group", "salary_2026", "fair_salary", "surplus"]
    print("\nMost UNDER-valued (bargains):")
    print(tbl.nsmallest(8, "surplus")[cols].to_string(index=False))
    print("\nMost OVER-valued:")
    print(tbl.nlargest(8, "surplus")[cols].to_string(index=False))
    tbl.to_csv(PROCESSED_DIR / "fair_value_2026.csv", index=False)
    print(f"\nSaved -> {PROCESSED_DIR / 'fair_value_2026.csv'}")
