"""Phase 2 — the four baselines, run through the rolling-origin backtests.

    .venv-tabfm/bin/python src/research/tabfm/baselines.py

Every model here (and TabFM later) is a function (train_df, test_df) -> preds,
so they all drop into the same harness with no special-casing.

READ FIRST
    02_design_rationale.md section 4 (why baselines first, which ones).

WHY THIS RUNS BEFORE TABFM
    These four are cheap, fast, deterministic. Debug the metrics + backtests on
    them, and publish the bar (persistence) to yourself, BEFORE the headline
    model runs -- so you can't move a goalpost you've already written down.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.research.tabfm import evaluate as ev  # noqa: E402

# Predict season t+1 from season t. season == t, so predicting 2024 uses the
# season==2023 rows as the test set.
BACKTESTS = {"predict_2024": 2023, "predict_2025": 2024}
MIN_TARGET_GAMES = 4  # survivorship filter on the OUTCOME (decided; disclosed)


# ------------------------------------------------------------------- models
# Each: (train_df, test_df) -> np.ndarray of predictions aligned to test_df.

# =========================================================================
# TODO(you) 2.3 -- the persistence baseline: "next year = this year".
#
# Return each test player's OWN season-t PPG as the prediction for t+1.
# It uses no training at all -- that's the point. It is the bar every real
# model must clear, and in fantasy it's a surprisingly high bar (last year's
# PPG already encodes talent, role, and offense).
#
# One line. `test["ppg"]` is season-t PPG; return it as a numpy array.
#
# Stuck after a real attempt? -> 04_solutions.md (2.3)
# =========================================================================
def persistence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test["ppg"].to_numpy()


def position_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Regression-to-the-mean: shrink each player's PPG halfway to their
    position's average next-season PPG (learned from train). Extreme seasons
    regress; this baseline bakes that in."""
    anchor = train.groupby("position_group")["target_ppg"].mean()
    pos_anchor = test["position_group"].map(anchor).to_numpy()
    own = test["ppg"].to_numpy()
    return 0.5 * own + 0.5 * pos_anchor


def _sk_predict(model, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit an sklearn model on the feature matrix and predict the test rows."""
    X_tr, y_tr = ev.feature_matrix(train)
    X_te, _ = ev.feature_matrix(test)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)  # align one-hot cols
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def ridge(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))
    return _sk_predict(m, train, test)


def histgbr(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    return _sk_predict(HistGradientBoostingRegressor(random_state=0), train, test)


MODELS = {"persistence": persistence, "position_mean": position_mean,
          "ridge": ridge, "histgbr": histgbr}


# -------------------------------------------------------------------- runner

def main() -> None:
    pairs = ev.load_pairs()
    print(f"transitions: {len(pairs)} rows, seasons {pairs.season.min()}-"
          f"{pairs.season.max()}\n")

    rows = []
    for bt_name, test_season in BACKTESTS.items():
        train, test = ev.backtest_split(pairs, test_season)   # TODO 2.2
        # Survivorship filter: only score players who actually played t+1.
        test = test[test["target_games"] >= MIN_TARGET_GAMES]
        y_true = test["target_ppg"].to_numpy()
        print(f"=== {bt_name}: train {len(train)} rows (<= {test_season-1}), "
              f"test {len(test)} players (>= {MIN_TARGET_GAMES} games) ===")
        preds = {}
        for m_name, model in MODELS.items():
            pred = np.asarray(model(train, test))
            preds[m_name] = pred
            s = ev.score(y_true, pred)                          # TODO 2.1
            rows.append({"backtest": bt_name, "model": m_name, **s})
            print(f"  {m_name:14s} MAE {s['mae']:5.2f}  R2 {s['r2']:5.2f}  "
                  f"Spearman {s['spearman']:5.2f}  slope {s['slope']:5.2f}")

        # Is the best model's edge over persistence real?
        best = min(MODELS, key=lambda m: ev.score(y_true, preds[m])["mae"])
        if best != "persistence":
            lo, hi = ev.paired_bootstrap_mae(test, y_true, preds[best],
                                             preds["persistence"])
            verdict = "REAL (CI excludes 0)" if hi < 0 else "not clearly real"
            print(f"  -> {best} vs persistence, MAE-delta 95% CI "
                  f"[{lo:+.2f}, {hi:+.2f}]  {verdict}")
        print()

    out = _ROOT / "data" / "processed" / "research" / "tabfm_baselines.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
