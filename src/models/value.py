"""Baseline fair-value model.

Predicts a player's *fair* 2026 salary from production + age + position +
consistency, then **surplus = actual − fair** flags over/under-valued players
(positive = paid above model, negative = a bargain).

Trained on this league's 2026 contracts (the players who have salaries). Fair
values are **out-of-fold** (each player scored by a model not fit on them), so
the surplus isn't circular. Current-season horizon; the dynasty (age-curve,
multi-year) version is a later refinement, as are advanced-usage features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import PROCESSED_DIR
from ..data.dataset import build_player_dataset

SKILL = ["QB", "RB", "WR", "TE"]
NUM_FEATURES = ["age", "ppg_2025", "ppg_2024", "games_2025", "stdev_2025", "years_exp"]
CAT_FEATURES = ["position_group"]
TARGET = "salary_2026"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Skill-position players with a salary and 2025 production; impute gaps."""
    d = df[df["position_group"].isin(SKILL)].copy()
    d = d[d[TARGET].notna() & d["ppg_2025"].notna()]
    d["ppg_2024"] = d["ppg_2024"].fillna(d["ppg_2025"])  # no prior year → assume similar
    d["stdev_2025"] = d["stdev_2025"].fillna(d["stdev_2025"].median())
    d["years_exp"] = d["years_exp"].fillna(0)
    d["age"] = d["age"].fillna(d["age"].median())
    return d.reset_index(drop=True)


def _pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
        remainder="passthrough",
    )
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=0
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def fair_value_table() -> tuple[pd.DataFrame, Pipeline]:
    """Return per-player fair salary + surplus (out-of-fold) and the fitted model."""
    d = _prepare(build_player_dataset())
    X = d[CAT_FEATURES + NUM_FEATURES]
    y = d[TARGET]

    model = _pipeline()
    oof = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    d["fair_salary"] = np.clip(oof, 1, None).round(1)
    d["surplus"] = (d[TARGET] - d["fair_salary"]).round(1)  # + overpaid / − bargain

    model.fit(X, y)  # final fit on all data (for scoring others later)
    return d, model


if __name__ == "__main__":
    d, _ = fair_value_table()
    print(
        f"n={len(d)} skill players | out-of-fold "
        f"R²={r2_score(d[TARGET], d['fair_salary']):.2f}  "
        f"MAE={mean_absolute_error(d[TARGET], d['fair_salary']):.1f} cap units"
    )
    cols = ["player", "team", "position_group", "age", "ppg_2025", "salary_2026", "fair_salary", "surplus"]
    print("\n=== Most UNDER-valued (bargains: actual ≪ fair) ===")
    print(d.nsmallest(12, "surplus")[cols].to_string(index=False))
    print("\n=== Most OVER-valued (actual ≫ fair) ===")
    print(d.nlargest(12, "surplus")[cols].to_string(index=False))

    out = PROCESSED_DIR / "fair_value_2026.csv"
    d.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
