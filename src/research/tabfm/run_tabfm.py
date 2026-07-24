"""Phase 3 — TabFM through the same harness, plus the tier ablation.

    .venv-tabfm/bin/python src/research/tabfm/run_tabfm.py

The question the project has built toward: does the foundation model beat a plain
Ridge at forecasting next-season PPG? Same (train_df, test_df) -> preds contract
as the baselines, so TabFM just drops into the harness.

SLOW + CACHED
    TabFM predicts in one big forward pass over the whole training context, so a
    single prediction is minutes, not seconds. Every prediction is cached to
    data/processed/research/tabfm_preds/ by a key; re-runs (to tweak reporting)
    are then instant. Delete that folder to force recompute. Run in background.

READ FIRST
    01_how_tabfm_works.md sections 3 and 6; your own smoke_test.py.
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
from src.research.tabfm._weights import load_core  # noqa: E402

PRED_DIR = _ROOT / "data" / "processed" / "research" / "tabfm_preds"
SEEDS = (0, 1)                       # ICL ensembling has mild run-to-run variance
ABLATIONS = {                        # feature bundles toggled via feature_matrix
    "full":         dict(include_consistency=True,  include_advanced=True),
    "no_advanced":  dict(include_consistency=True,  include_advanced=False),
    "no_consistency": dict(include_consistency=False, include_advanced=True),
}


# =========================================================================
# TODO(you) 3.1 -- the TabFM prediction core. This is your smoke_test.py, now
# inside the harness. Given the pretrained `core`, the training feature matrix
# (X_tr, y_tr) and the test features X_te, return TabFM's predictions.
#
#   1. Wrap `core` in TabFMRegressor(model=core, random_state=seed).
#   2. fit(X_tr, y_tr)   -- instant, just stores context.
#   3. return .predict(X_te)   -- the slow forward pass.
#
# (feature_matrix building + column alignment are done for you in tabfm_model
# below -- you only write the 3 model lines here.)
#
# Stuck after a real attempt? -> 04_solutions.md (3.1)
# =========================================================================
def tabfm_predict(core, X_tr, y_tr, X_te, seed: int = 0) -> np.ndarray:
    raise NotImplementedError("TODO(you) 3.1 -- see smoke_test.py")


# ------------------------------------------------------------ model wrapper
# Pre-filled plumbing: builds the feature matrix (with the ablation toggles),
# aligns one-hot columns, and delegates to your tabfm_predict.

def tabfm_model(train: pd.DataFrame, test: pd.DataFrame, *, seed: int = 0,
                include_consistency: bool = True,
                include_advanced: bool = True) -> np.ndarray:
    X_tr, y_tr = ev.feature_matrix(train, include_consistency=include_consistency,
                                   include_advanced=include_advanced)
    X_te, _ = ev.feature_matrix(test, include_consistency=include_consistency,
                                include_advanced=include_advanced)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    return tabfm_predict(load_core("regression"), X_tr, y_tr, X_te, seed=seed)


# ------------------------------------------------------------------ caching
def _cached(key: str, compute) -> np.ndarray:
    """Return cached predictions for `key`, or compute + save them."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{key}.npy"
    if path.exists():
        print(f"    [cache] {key}")
        return np.load(path)
    print(f"    [compute] {key} ...", flush=True)
    pred = np.asarray(compute())
    np.save(path, pred)
    return pred


# -------------------------------------------------------------------- runner
def main() -> None:
    pairs = ev.load_pairs()

    for bt_name, test_season in bl.BACKTESTS.items():
        train, test = ev.backtest_split(pairs, test_season)
        test = test[test["target_games"] >= bl.MIN_TARGET_GAMES]
        y = test["target_ppg"].to_numpy()
        print(f"\n=== {bt_name}: train {len(train)}, test {len(test)} players ===")

        # --- headline: TabFM (avg of seeds) vs Ridge vs persistence ---
        ridge = _cached(f"{bt_name}__ridge",
                        lambda: bl.ridge(train, test))
        persist = bl.persistence(train, test)
        seed_preds = [
            _cached(f"{bt_name}__tabfm_full_s{s}",
                    lambda s=s: tabfm_model(train, test, seed=s))
            for s in SEEDS
        ]
        tabfm = np.mean(seed_preds, axis=0)

        print("  --- headline (full features) ---")
        for name, pred in [("persistence", persist), ("ridge", ridge),
                           ("tabfm", tabfm)]:
            sc = ev.score(y, pred)
            print(f"    {name:12s} MAE {sc['mae']:5.2f}  R2 {sc['r2']:5.2f}  "
                  f"Spearman {sc['spearman']:5.2f}  slope {sc['slope']:5.2f}")
        # seed spread: how much does TabFM wobble run to run?
        seed_maes = [ev.mae(y, p) for p in seed_preds]
        print(f"    tabfm seed MAEs: {[round(m,2) for m in seed_maes]} "
              f"(spread {max(seed_maes)-min(seed_maes):.2f})")
        # is TabFM's edge over Ridge real?
        lo, hi = ev.paired_bootstrap_mae(test, y, tabfm, ridge)
        verdict = ("TabFM better, real" if hi < 0 else
                   "Ridge better, real" if lo > 0 else "no clear winner")
        print(f"    tabfm vs ridge, MAE-delta 95% CI [{lo:+.2f}, {hi:+.2f}]  {verdict}")

        # --- ablation: what is each feature tier worth to TabFM? ---
        print("  --- ablation (TabFM, seed 0) ---")
        for ab_name, cfg in ABLATIONS.items():
            pred = _cached(f"{bt_name}__tabfm_{ab_name}_s0",
                           lambda cfg=cfg: tabfm_model(train, test, seed=0, **cfg))
            print(f"    {ab_name:16s} MAE {ev.mae(y, pred):5.2f}")

    print(f"\npredictions cached in {PRED_DIR.relative_to(_ROOT)}/ "
          "(delete to recompute)")


if __name__ == "__main__":
    main()
