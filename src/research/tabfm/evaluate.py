"""Phase 2 — the evaluation harness (metrics, backtest splitter, bootstrap).

Built ONCE, reused by every model in Phases 2-3 (baselines AND TabFM). Runs in
the TabFM venv (py3.11), reading the parquet Phase 1 wrote.

    from src.research.tabfm import evaluate as ev
    train, test = ev.backtest_split(pairs, 2023)   # predict 2024 from 2023
    X_tr, y_tr = ev.feature_matrix(train)
    ...
    ev.score(y_true, y_pred)   # -> {"mae":..., "r2":..., "spearman":..., "slope":...}

READ FIRST
    02_design_rationale.md sections 4-5; cheatsheet "Metrics", "The backtest
    splitter", "Paired bootstrap".

WHY A SHARED HARNESS
    Every model -- persistence, Ridge, HistGBR, TabFM -- speaks the same
    (train_df, test_df) -> predictions contract, so the metrics and backtests
    are written once and never special-cased per model. That uniformity is the
    whole reason baselines come before TabFM: debug the ruler on cheap models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Tier constants imported from the Phase-1 builder = single source of truth.
from src.research.tabfm.build_dataset import (  # noqa: E402
    ADVANCED_STATS,
    BOX_STATS,
    CONSISTENCY_MISSING_FLAG,
    CONSISTENCY_STATS,
)

PARQUET = _ROOT / "data" / "processed" / "research" / "tabfm_transitions.parquet"
TARGET = "target_ppg"

# Core features = profile + raw box score + season-t league output. `team` and
# `name` are identity, not features; `position_group` is one-hot encoded below;
# targets and keys are never features.
CORE_NUMERIC = ["age", "years_exp", "games", "points", "ppg"]
_KEYS_AND_TARGETS = ["espn_id", "season", "name", "team", "position_group",
                     "target_ppg", "target_games"]


def load_pairs() -> pd.DataFrame:
    """The transitions parquet Phase 1 built."""
    if not PARQUET.exists():
        raise SystemExit(
            f"{PARQUET} missing -- build it first with the PROJECT venv:\n"
            "  .venv/bin/python src/research/tabfm/build_dataset.py")
    return pd.read_parquet(PARQUET)


# ------------------------------------------------------------ feature matrix
# Pre-filled plumbing. The ablation tiers (Phase 3) are toggled here: flip
# include_consistency / include_advanced to add or remove a whole bundle.

def feature_matrix(df: pd.DataFrame, *, include_consistency: bool = True,
                   include_advanced: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for modelling. y = target_ppg; X = the selected tiers.

    position_group -> one-hot columns (models need numbers). team is dropped
    (32 sparse levels, weak next-season signal). NaNs are LEFT as NaN -- each
    model handles them its own way (HistGBR native; Ridge via its imputer;
    TabFM internally), which is part of what we're measuring.
    """
    cols = list(CORE_NUMERIC) + [c for c in BOX_STATS if c in df.columns]
    if include_consistency:
        cols += [c for c in CONSISTENCY_STATS if c in df.columns]
        cols += [CONSISTENCY_MISSING_FLAG]
    if include_advanced:
        cols += [c for c in ADVANCED_STATS if c in df.columns]

    X = df[cols].copy()
    # one-hot the position (QB/RB/WR/TE) so linear/tree models can use it
    pos = pd.get_dummies(df["position_group"], prefix="pos")
    X = pd.concat([X.reset_index(drop=True), pos.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)
    return X, y


# =========================================================================
# TODO(you) 2.1 -- the four metric functions.
#
# Each takes (y_true, y_pred) as array-likes and returns one float.
#   mae(y_true, y_pred)                mean absolute error, in PPG
#   r2(y_true, y_pred)                 R^2 (sklearn's r2_score)
#   spearman(y_true, y_pred)           Spearman rank correlation (ranking quality)
#   calibration_slope(y_true, y_pred)  slope of ACTUAL regressed on PREDICTED;
#                                      1.0 = calibrated, <1 = compressed to mean
#
# Cheatsheet section "Metrics" has all four one-liners. Imports you'll want:
#   from sklearn.metrics import mean_absolute_error, r2_score
#   from scipy.stats import spearmanr
#   (np is already imported)
#
# WHY four and not just R^2: R^2 rewards spreading predictions out; your league
# decisions are RANKINGS (Spearman) and you care whether elites are mispriced
# (slope). See 02 section 5.
#
# Stuck after a real attempt? -> 04_solutions.md (2.1)
# =========================================================================
def mae(y_true, y_pred) -> float:
    raise NotImplementedError("TODO(you) 2.1")


def r2(y_true, y_pred) -> float:
    raise NotImplementedError("TODO(you) 2.1")


def spearman(y_true, y_pred) -> float:
    raise NotImplementedError("TODO(you) 2.1")


def calibration_slope(y_true, y_pred) -> float:
    raise NotImplementedError("TODO(you) 2.1")


def score(y_true, y_pred) -> dict:
    """Bundle the four metrics into one dict (uses your TODO 2.1 functions)."""
    return {"mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred),
            "spearman": spearman(y_true, y_pred),
            "slope": calibration_slope(y_true, y_pred)}


# =========================================================================
# TODO(you) 2.2 -- the rolling-origin backtest splitter.
#
# Given the transitions table and a `test_season`, return (train, test):
#   train = every pair whose season t is STRICTLY BEFORE test_season
#   test  = every pair whose season t EQUALS test_season
#
# Remember `season` is season t of the pair, so the pair predicting 2024 has
# season == 2023. To "predict 2024", call backtest_split(pairs, 2023).
#
# WHY strictly-before: the model must never see a transition from the future it
# is predicting. This is the whole reason we don't shuffle (random K-fold would
# leak future rows into training). See 02 section 5.
#
# Stuck after a real attempt? -> 04_solutions.md (2.2)
# =========================================================================
def backtest_split(pairs: pd.DataFrame,
                   test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError("TODO(you) 2.2")


# --------------------------------------------------------- paired bootstrap
# Pre-filled plumbing (read it -- the concept matters, the mechanics are fiddly).
# "Is model A's edge over model B real, or did we get lucky on which players
# were in the test set?" Resample PLAYERS (not rows) with replacement many
# times; if the MAE difference stays the same sign across resamples, it's real.

def paired_bootstrap_mae(test: pd.DataFrame, y_true, pred_a, pred_b,
                         n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI on (MAE_a - MAE_b), resampling players. CI excluding 0 => real.

    Resamples ESPN player ids, not rows, because a player's transitions are
    correlated -- treating rows as independent would make the CI falsely narrow.
    """
    from sklearn.metrics import mean_absolute_error
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    players = test["espn_id"].to_numpy()
    unique = np.unique(players)
    deltas = []
    for _ in range(n):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        # multiplicity-correct: a player drawn twice contributes twice
        idx = np.concatenate([np.where(players == p)[0] for p in drawn])
        deltas.append(mean_absolute_error(y_true[idx], pred_a[idx])
                      - mean_absolute_error(y_true[idx], pred_b[idx]))
    return tuple(np.percentile(deltas, [2.5, 97.5]))
