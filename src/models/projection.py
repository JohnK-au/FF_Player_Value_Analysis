"""Next-season production projection (de-fluking the single-season value) + age curves.

Predicts next-season PPG (in our scoring) from current-season features + per-player
longitudinal context (`models/context.py`) + age + draft + combine. Trained on transition
pairs ``(features_t -> ppg_{t+1})`` across 2022→2025 (three transitions). HistGradient-
Boosting (NaN-native, so rookies' missing baselines are handled gracefully).

For dynasty (multi-year) extension, the 1-year projection is rolled forward across
remaining contract years via positional **age curves** (median YoY PPG ratio by position
and integer age, from the same historical transitions).

Public API:
- ``fit_projection_model(df=None)`` — train; returns ``(pipeline, train_df, feats)``.
- ``projected_production(target_season=2026, model, feats)`` — NFL-wide projected PPG.
- ``positional_age_curves(df=None)`` — multiplicative age multipliers by position × age.
- ``project_multiyear(start_ppg, position, start_age, n_years, curves)`` — recursive
  multi-year projection for dynasty value.

Honest expectation: projection R² is much lower than the same-season production R² (≈0.80);
0.3–0.5 is realistic and is the right diagnostic — projecting the future is the hard part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from ..config import PROCESSED_DIR
from ..data.context import (
    HEADLINE,
    RESULTS_METRICS,
    TEAM_METRICS,
    USAGE_METRICS,
    training_frame_with_context,
)
from ..data.population import SKILL
from .production import (
    ADV_NUM,
    COMBINE_NUM,
    DRAFT_NUM,
    MIN_GAMES_TRAIN,
    PROD_CAT,
    _pipeline,
)

PROJ_TARGET = "ppg_next"
DEFAULT_AGE_MULTIPLIER_FALLBACK = 0.95  # when no historical data for a (pos, age) bucket

# Current-season features (same as the same-season production model).
CURRENT_NUM = ["age", "years_exp"] + ADV_NUM + DRAFT_NUM + COMBINE_NUM

# Context features: per-metric baseline + z-score (the "is this sustainable vs the
# player's own history" signal). Plus the diagnostic rollups.
_BASELINE_METRICS = [HEADLINE] + USAGE_METRICS + RESULTS_METRICS + TEAM_METRICS
CONTEXT_NUM = [f"{m}_baseline" for m in _BASELINE_METRICS] + \
              [f"{m}_z" for m in _BASELINE_METRICS]
ROLLUP_NUM = ["usage_trend", "results_trend", "qb_context"]

PROJ_NUM = CURRENT_NUM + CONTEXT_NUM + ROLLUP_NUM


# --- Training pairs -----------------------------------------------------------
def _build_pairs(seasons: list[int] | None = None) -> pd.DataFrame:
    """Build (features at season t, ppg at season t+1) training pairs.

    Keeps only rows where next-season `games >= MIN_GAMES_TRAIN` (we don't train against
    injury-shortened "true" targets).
    """
    full = training_frame_with_context(seasons)
    full = full[full["position_group"].isin(SKILL)].copy()
    full = full.sort_values(["espn_id", "season"]).reset_index(drop=True)

    by = full.groupby("espn_id", sort=False)
    full[PROJ_TARGET] = by["ppg"].shift(-1)
    full["games_next"] = by["games"].shift(-1)

    pairs = full[full[PROJ_TARGET].notna() & (full["games_next"].fillna(0) >= MIN_GAMES_TRAIN)]
    return pairs.reset_index(drop=True)


# --- Model fitting ------------------------------------------------------------
def _present_features(df: pd.DataFrame) -> list[str]:
    return [c for c in PROJ_NUM if c in df.columns]


def fit_projection_model(df: pd.DataFrame | None = None):
    """Fit the projection model. Returns (pipeline, train_df, feats)."""
    d = df if df is not None else _build_pairs()
    feats = _present_features(d)
    pipe = _pipeline().fit(d[PROD_CAT + feats], d[PROJ_TARGET])
    return pipe, d, feats


def oof_predictions(d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = d[PROD_CAT + feats]
    cv = KFold(5, shuffle=True, random_state=0)
    return cross_val_predict(_pipeline(), X, d[PROJ_TARGET], cv=cv)


# --- 1-year projection scoring ------------------------------------------------
def projected_production(
    target_season: int = 2026,
    model=None,
    feats: list[str] | None = None,
    full: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """NFL-wide projected PPG for ``target_season`` scored from latest-available features."""
    full = full if full is not None else training_frame_with_context()
    src = full[(full["season"] == target_season - 1) & full["position_group"].isin(SKILL)].copy()
    if model is None:
        model, _, feats = fit_projection_model()
    feats = feats or _present_features(full)
    src["projected_ppg"] = model.predict(src[PROD_CAT + feats]).round(2)
    return src[["espn_id", "name", "position_group", "age", "ppg", "projected_ppg"]].copy()


# --- Replacement levels + VOR on projected production -------------------------
def projected_vor_table(
    target_season: int = 2026, value_col: str = "projected_ppg"
) -> tuple[pd.DataFrame, dict]:
    """Replacement-level VOR computed on projected next-season PPG.

    Mirrors ``production.vor_table`` but for projected production. No games filter is
    applied (the projection answers "what would they score over a full season"). The
    pool is the NFL-wide projected_production output.
    """
    from .production import replacement_levels

    pool = projected_production(target_season).copy()
    pool = pool[pool[value_col].notna()]
    repl = replacement_levels(pool, value_col)
    pool["replacement"] = pool["position_group"].map(repl).round(2)
    pool["vor"] = (pool[value_col] - pool["replacement"]).round(2)
    return pool, repl


# --- Age curves for multi-year (dynasty) projection ---------------------------
def positional_age_curves(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Median YoY PPG ratio by (position, integer age). Multiplicative age multiplier.

    From the same historical transition pairs as the projection model. Thin (3
    transitions, ~3,000 pairs total split by position × age), so individual cells can
    be noisy — treat magnitudes as directional.
    """
    d = df if df is not None else _build_pairs()
    d = d[d["ppg"] > 0].copy()
    d["yoy_ratio"] = d[PROJ_TARGET] / d["ppg"]
    d["age_int"] = d["age"].round().astype("Int64")
    curves = (
        d.groupby(["position_group", "age_int"])["yoy_ratio"]
        .median()
        .round(3)
        .reset_index()
    )
    return curves


def project_multiyear(
    start_ppg: float,
    position: str,
    start_age: float,
    n_years: int,
    curves: pd.DataFrame | None = None,
) -> list[float]:
    """Project ``start_ppg`` forward ``n_years`` via positional age multipliers.

    ``start_ppg`` is treated as year-1 of the projection (the next-season model output).
    Years 2..n apply the (position, age) YoY multiplier each step. Returns a list of
    projected PPG per year [year_1, year_2, ...] of length ``n_years``.
    """
    if curves is None:
        curves = positional_age_curves()
    pos_curve = curves[curves["position_group"] == position]
    out = [round(start_ppg, 2)]
    cur = start_ppg
    age = start_age
    for _ in range(n_years - 1):
        age += 1
        m = pos_curve.loc[pos_curve["age_int"] == int(round(age)), "yoy_ratio"]
        mult = float(m.iloc[0]) if len(m) else DEFAULT_AGE_MULTIPLIER_FALLBACK
        cur = cur * mult
        out.append(round(cur, 2))
    return out


# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    full = training_frame_with_context()
    pairs = _build_pairs()
    pipe, _, feats = fit_projection_model(pairs)
    oof = oof_predictions(pairs, feats)
    print(f"Projection model — trained on {len(pairs)} (t -> t+1) pairs "
          f"({pairs['season'].nunique()} source seasons, {len(feats)} numeric features)")
    print(f"  out-of-fold R^2: {r2_score(pairs[PROJ_TARGET], oof):.3f}  "
          f"(MAE {mean_absolute_error(pairs[PROJ_TARGET], oof):.2f} PPG)")
    print("  (projecting the future is harder than fitting the same season - 0.3-0.5 expected)")

    # 2026 projections
    proj = projected_production(2026, model=pipe, feats=feats, full=full)
    proj["delta"] = (proj["projected_ppg"] - proj["ppg"]).round(2)

    print(f"\n{len(proj)} skill players scored for 2026.")
    print("\nTop 10 projected 2026 PPG:")
    print(proj.nlargest(10, "projected_ppg")[
        ["name", "position_group", "age", "ppg", "projected_ppg", "delta"]
    ].to_string(index=False))
    print("\nBiggest projected rebounds (largest positive delta vs 2025 actual):")
    print(proj.nlargest(10, "delta")[
        ["name", "position_group", "age", "ppg", "projected_ppg", "delta"]
    ].to_string(index=False))
    print("\nBiggest projected declines:")
    print(proj.nsmallest(10, "delta")[
        ["name", "position_group", "age", "ppg", "projected_ppg", "delta"]
    ].to_string(index=False))

    # Age curves
    curves = positional_age_curves(pairs)
    print("\nPositional age curves (median YoY PPG ratio):")
    print(curves.pivot(index="age_int", columns="position_group", values="yoy_ratio")
          .to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "projected_production_2026.csv"
    proj.to_csv(out, index=False)
    print(f"\nSaved projected production -> {out}")
