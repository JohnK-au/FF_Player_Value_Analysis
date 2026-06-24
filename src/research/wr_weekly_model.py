"""WR weekly archetype model + analysis.

Loads the per-week WR feature table built by ``wr_weekly.py``, computes per-player
**rolling-history** features (shift then 4-week rolling mean per `espn_id`, no leakage),
**drops same-week outcome features** (which would leak the target), then:

1. Fits a HistGradientBoostingRegressor for predictive performance — OOF R² + MAE,
   compared to trivial baselines (mean / trailing-4-week PPG).
2. Fits a shallow DecisionTreeRegressor (depth 4, min-leaf 80) for interpretable
   **archetype rules** — "WR with prior-4-week target_share above X AND opponent
   def pass-epa-allowed above Y → expected PPG Z."
3. Permutation importance to rank features by predictive contribution.
4. Conditional 2-D heat tables on the top features (PPG by feature bin × feature bin).

Honest framing: we're predicting per-week fantasy points pre-game from *prior* signals,
which is hard. A R² of 0.10–0.25 here would already be a meaningful lift over the
naive "his trailing PPG is X, predict X" baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text  # noqa: E402

from src.config import PROCESSED_DIR  # noqa: E402

RESEARCH_DIR = PROCESSED_DIR / "research"
TARGET = "fantasy_points"

# ---- Feature partitioning ---------------------------------------------------
# Same-week stats that are downstream of the target (DROP for predictive modeling).
LEAKY_COLS = [
    "targets", "receptions", "air_yards", "rec_yards", "yac", "rec_tds", "rec_epa",
    "target_share", "air_yards_share",
    "team_pass_plays", "team_pass_epa_play", "team_cpoe", "team_completion_pct",
    "team_air_yards", "team_pass_tds", "team_targets", "team_rush_plays", "team_pass_rate",
    "avg_cushion", "avg_separation", "avg_intended_air_yards", "catch_percentage",
    "avg_yac_above_expectation",
]

STATIC_FEATS = ["age", "years_exp", "draft_round", "draft_pick", "draft_value",
                "undrafted", "forty", "vertical", "broad_jump"]
ENV_FEATS = ["is_home", "team_spread", "team_implied_total", "opp_implied_total", "total_line"]
DEF_ROLL_FEATS = [
    "def_pass_epa_play_roll", "def_cpoe_allowed_roll",
    "def_completion_pct_allowed_roll", "def_pass_tds_allowed_roll", "def_yards_allowed_roll",
]

# Per-player rolling histories — computed here from the same-week columns (which are
# leaky if used at t, but legitimate as a *prior* signal when shifted by 1 week).
PLAYER_ROLL_BASE = [
    "fantasy_points", "targets", "target_share",
    "air_yards", "air_yards_share", "rec_epa",
    "team_pass_epa_play", "team_cpoe",
    "avg_separation", "catch_percentage",
]
ROLL_WINDOW = 4
PLAYER_ROLL_FEATS = [f"{c}_roll{ROLL_WINDOW}" for c in PLAYER_ROLL_BASE]

ALL_FEATS = STATIC_FEATS + ENV_FEATS + DEF_ROLL_FEATS + PLAYER_ROLL_FEATS


# ---- Data loading + rolling helpers ----------------------------------------
def add_player_rolling(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Add trailing ``window``-week per-player rolling means (no leakage via shift)."""
    df = df.sort_values(["espn_id", "season", "week"]).copy()
    gb = df.groupby("espn_id", sort=False)
    for c in PLAYER_ROLL_BASE:
        if c not in df.columns:
            df[f"{c}_roll{window}"] = np.nan
            continue
        df[f"{c}_roll{window}"] = (
            gb[c].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    return df


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RESEARCH_DIR / "wr_weekly_features.csv")
    return add_player_rolling(df)


# ---- Modeling --------------------------------------------------------------
def fit_predictive_model(df: pd.DataFrame, feats: list[str] | None = None):
    """Fit HistGBR on (rolling + env + defense + static) → fantasy_points."""
    feats = [c for c in (feats or ALL_FEATS) if c in df.columns]
    d = df.dropna(subset=[TARGET])
    # Require at least one rolling-history value (drop a player's first week with no prior).
    d = d.dropna(subset=PLAYER_ROLL_FEATS, how="all").copy()
    X, y = d[feats], d[TARGET]
    pipe = HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
        random_state=0,
    )
    cv = KFold(5, shuffle=True, random_state=0)
    oof = cross_val_predict(pipe, X, y, cv=cv)
    pipe.fit(X, y)
    return pipe, d, feats, oof


def baselines(d: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compare to trivial baselines: constant mean, naive trailing-N PPG."""
    y = d[TARGET]
    pred_mean = np.full_like(y, y.mean())
    out: dict[str, tuple[float, float]] = {
        "constant-mean": (
            r2_score(y, pred_mean), mean_absolute_error(y, pred_mean)
        )
    }
    if f"fantasy_points_roll{ROLL_WINDOW}" in d.columns:
        pred_trail = d[f"fantasy_points_roll{ROLL_WINDOW}"].fillna(y.mean())
        out[f"trailing-{ROLL_WINDOW}-wk PPG"] = (
            r2_score(y, pred_trail), mean_absolute_error(y, pred_trail)
        )
    return out


def shallow_archetype_tree(d: pd.DataFrame, feats: list[str], max_depth: int = 4,
                            min_samples_leaf: int = 80):
    """Decision tree with leaves big enough to be archetypes, not noise."""
    feats = [c for c in feats if c in d.columns]
    X = d[feats].copy()
    # DecisionTreeRegressor doesn't handle NaN — fill with column median.
    X = X.fillna(X.median(numeric_only=True))
    y = d[TARGET]
    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0,
    )
    tree.fit(X, y)
    return tree, feats


def conditional_heat(d: pd.DataFrame, x: str, y: str, target: str = TARGET,
                     bins: int = 4) -> pd.DataFrame:
    """Mean target by quantile bins of two features."""
    sub = d.dropna(subset=[x, y, target]).copy()
    sub["_x"] = pd.qcut(sub[x], q=bins, labels=[f"Q{i+1}" for i in range(bins)], duplicates="drop")
    sub["_y"] = pd.qcut(sub[y], q=bins, labels=[f"Q{i+1}" for i in range(bins)], duplicates="drop")
    pivot = sub.groupby(["_x", "_y"], observed=True)[target].agg(["mean", "count"]).round(2)
    return pivot


# ---- CLI ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df)} WR-weeks ({df['espn_id'].nunique()} unique WRs, "
          f"{df['season'].nunique()} seasons).")

    pipe, d, feats, oof = fit_predictive_model(df)
    y = d[TARGET]
    print(f"\nPredictive model — n={len(d)} WR-weeks, {len(feats)} features")
    print(f"  OOF R² = {r2_score(y, oof):.3f}   MAE = {mean_absolute_error(y, oof):.2f} pts")
    print("\nBaselines:")
    for name, (r2, mae) in baselines(d).items():
        print(f"  {name:30}  R² {r2:+.3f}   MAE {mae:.2f}")

    print("\nTop 15 features by permutation importance:")
    imp = permutation_importance(pipe, d[feats], y, n_repeats=4, random_state=0, n_jobs=-1)
    ranks = sorted(zip(feats, imp.importances_mean), key=lambda x: -x[1])
    for n, v in ranks[:15]:
        print(f"  {n:40}  {v:.3f}")

    # --- Archetype tree -----------------------------------------------------
    print("\n=== Shallow decision tree (archetype rules, depth 4, min-leaf 80) ===")
    tree, tfeats = shallow_archetype_tree(d, feats, max_depth=4, min_samples_leaf=80)
    print(export_text(tree, feature_names=tfeats, decimals=2))

    # --- Conditional heats on top-pair features -----------------------------
    top_two = [n for n, _ in ranks[:6] if n in d.columns][:2]
    if len(top_two) == 2:
        print(f"\n=== Mean WR fantasy_points by quartile bins of {top_two[0]} × {top_two[1]} ===")
        print(conditional_heat(d, top_two[0], top_two[1]).to_string())
