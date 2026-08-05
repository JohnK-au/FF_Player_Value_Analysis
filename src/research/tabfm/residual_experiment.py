"""Phase 4b — residualized consistency: deconfound weekly_std from ppg.

    .venv-tabfm/bin/python src/research/tabfm/residual_experiment.py

Consistency looked useless in Phases 3/4a. Was it genuinely useless, or just
tangled with PPG (corr 0.84)? Here we DECONFOUND: replace raw weekly_std with
its residual after regressing on ppg -- "volatility beyond what your scoring
level predicts" -- and re-test. Fast (Ridge + HistGBR, no TabFM).

READ FIRST
    02_design_rationale.md section 3; _now.md panel (the leak-free rule).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.research.tabfm import baselines as bl  # noqa: E402
from src.research.tabfm import evaluate as ev  # noqa: E402


# =========================================================================
# TODO(you) 4.2 -- residualize weekly_std on ppg, LEAK-FREE.
#
# Goal: add a column 'std_residual' = how much more/less volatile a player is
# than expected for their scoring level.
#
# Steps:
#   1. Fit a straight line  weekly_std ~ ppg  on the TRAINING rows only, and
#      only where weekly_std is present (not NaN). np.polyfit(x, y, 1) returns
#      [slope, intercept].  Mask NaNs first: m = train["weekly_std"].notna()
#   2. For BOTH train and test, compute expected_std = intercept + slope*ppg,
#      then std_residual = weekly_std - expected_std.
#      (Rows with NaN weekly_std -> NaN residual; that's fine, the model's
#       imputer handles it -- same as raw std.)
#   3. Return (train, test) each with the new 'std_residual' column.
#
# Work on COPIES (train = train.copy()) so you don't mutate the caller's frames.
#
# WHY fit on train only: the residual line is part of the model. Fitting it on
# the test season would leak the future into your features -- the same sin the
# whole project is built to avoid.
#
# Stuck after a real attempt? -> 04_solutions.md (4.2)
# =========================================================================
def add_std_residual(train: pd.DataFrame,
                     test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()
    # fit weekly_std ~ ppg on TRAIN rows where std is present (polyfit hates NaN)
    m = train["weekly_std"].notna()
    slope, intercept = np.polyfit(train.loc[m, "ppg"], train.loc[m, "weekly_std"], 1)
    # apply the SAME line to both frames -> residual = actual - expected
    for df in (train, test):
        expected_std = slope * df["ppg"] + intercept
        df["std_residual"] = df["weekly_std"] - expected_std
    return train, test


# ------------------------------------------------------------------ plumbing
# Fit a model on CORE features (no consistency/advanced tiers) plus whatever
# extra columns we hand it, and return test-set MAE. Lets us compare "core +
# raw std" vs "core + residual" with everything else held fixed.

def _mae_with(train, test, model_fn, extra_cols) -> float:
    Xtr, ytr = ev.feature_matrix(train, include_consistency=False,
                                 include_advanced=False)
    Xte, _ = ev.feature_matrix(test, include_consistency=False,
                               include_advanced=False)
    for c in extra_cols:                       # bolt on the consistency variant
        Xtr = Xtr.assign(**{c: train[c].to_numpy()})
        Xte = Xte.assign(**{c: test[c].to_numpy()})
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    model = model_fn()
    model.fit(Xtr, ytr)
    return ev.mae(test["target_ppg"].to_numpy(), model.predict(Xte))


def _ridge():
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))


def _histgbr():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=0)


MODELS = {"ridge": _ridge, "histgbr": _histgbr}


def main() -> None:
    pairs = ev.load_pairs()
    for bt_name, test_season in bl.BACKTESTS.items():
        train, test = ev.backtest_split(pairs, test_season)
        test = test[test["target_games"] >= bl.MIN_TARGET_GAMES]
        train, test = add_std_residual(train, test)          # TODO 4.2

        print(f"\n=== {bt_name}  (MAE; lower = better) ===")
        print(f"  {'model':8s} {'core':>7} {'+raw_std':>9} {'+residual':>10}")
        for m_name, m_fn in MODELS.items():
            base = _mae_with(train, test, m_fn, [])
            raw = _mae_with(train, test, m_fn, ["weekly_std"])
            res = _mae_with(train, test, m_fn, ["std_residual"])
            print(f"  {m_name:8s} {base:7.3f} {raw:9.3f} {res:10.3f}")
        print("  (does +residual beat +raw_std? and does either beat core?)")


if __name__ == "__main__":
    main()
