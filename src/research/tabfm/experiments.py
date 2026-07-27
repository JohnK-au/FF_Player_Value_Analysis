"""Phase 4a/4b — the deeper consistency experiments.

    .venv-tabfm/bin/python src/research/tabfm/experiments.py

4a STRATIFIED ABLATION (fast -- reuses cached predictions)
    The tier ablation showed consistency adds nothing ON AVERAGE. But does it
    help for ELITES specifically? Split test players into high- and low-ppg
    groups and compare full vs no-consistency MAE within each. Directly answers
    the Phase-1 question: "maybe consistency matters above a ppg threshold."

4b RESIDUALIZED CONSISTENCY (Phase 4b -- appended later)
    Deconfound weekly_std from ppg and test the residual as a feature.

READ FIRST
    02_design_rationale.md section 3 (the consistency decisions).
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

PRED_DIR = _ROOT / "data" / "processed" / "research" / "tabfm_preds"


# ------------------------------------------------------------------ plumbing
def _test_set(pairs: pd.DataFrame, test_season: int) -> pd.DataFrame:
    """The same test set the runner scored: split, then >=4 games, index reset
    so it lines up positionally with the cached prediction arrays."""
    _, test = ev.backtest_split(pairs, test_season)
    return test[test["target_games"] >= bl.MIN_TARGET_GAMES].reset_index(drop=True)


def _cached(bt_name: str, key: str) -> np.ndarray:
    path = PRED_DIR / f"{bt_name}__{key}.npy"
    if not path.exists():
        raise SystemExit(f"missing {path.name} -- run run_tabfm.py first")
    return np.load(path)


# =========================================================================
# TODO(you) 4.1 -- the stratified ablation.
#
# Inputs (all aligned row-for-row, same length = number of test players):
#   test        the test DataFrame (has 'ppg' = season-t PPG, 'target_ppg')
#   pred_full   TabFM predictions WITH consistency features
#   pred_nocon  TabFM predictions WITHOUT the consistency tier
#   threshold   the season-t PPG that splits "high" from "low"
#
# Return a dict of MAEs, one per (group, config):
#   {"high_full":.., "high_nocon":.., "low_full":.., "low_nocon":..}
#
# How: build a boolean mask on test["ppg"] (>= threshold = high, < = low).
# Use it to select the matching rows of the TRUTH (test["target_ppg"]) and of
# each prediction array, then ev.mae() each subset. Numpy arrays accept the same
# boolean mask as the DataFrame column -- that's why the index was reset above.
#
# The question the numbers answer: is (high_full < high_nocon)? i.e. does
# dropping consistency HURT more for elites than for everyone? If the gap only
# shows up in the high group, consistency matters above a ppg threshold.
#
# Stuck after a real attempt? -> 04_solutions.md (4.1)
# =========================================================================

CUTOFFS = {
        "QB" : (12, 18),
        "RB" : (7, 13),
        "WR" : (7, 13),
        "TE" : (5, 11),
    }

def stratified_ablation(test, preds: dict) -> pd.DataFrame:
    """MAE per (position, ppg-band, feature-config). `preds` maps a config name
    to its prediction array; add a config = add a dict entry, nothing else."""
    y = test["target_ppg"]
    rows = []
    for pos, (low, high) in CUTOFFS.items():
        in_pos = test["position_group"] == pos
        bands = {
            "below":  in_pos & (test["ppg"] < low),
            "middle": in_pos & (test["ppg"] >= low) & (test["ppg"] < high),
            "above":  in_pos & (test["ppg"] >= high),
        }
        for band, mask in bands.items():
            n = int(mask.sum())
            if n == 0:                      # empty cell -> MAE undefined, skip
                continue
            for cfg, arr in preds.items():
                rows.append({"pos": pos, "band": band, "config": cfg,
                             "n": n, "mae": ev.mae(y[mask], arr[mask])})
    return pd.DataFrame(rows)


_BAND_ORDER = {"below": 0, "middle": 1, "above": 2}


def main() -> None:
    pairs = ev.load_pairs()
    for bt_name, test_season in bl.BACKTESTS.items():
        test = _test_set(pairs, test_season)
        preds = {
            "full":  _cached(bt_name, "tabfm_full_s0"),
            "nocon": _cached(bt_name, "tabfm_no_consistency_s0"),
            "noadv": _cached(bt_name, "tabfm_no_advanced_s0"),
        }

        long = stratified_ablation(test, preds)   # TODO 4.1
        # pivot so the configs sit side by side, add a delta per dropped tier
        wide = (long.pivot_table(index=["pos", "band", "n"], columns="config",
                                 values="mae").reset_index())
        wide["d_consist"]  = wide["nocon"] - wide["full"]   # + = consistency helped
        wide["d_advanced"] = wide["noadv"] - wide["full"]   # + = advanced helped
        wide["_order"] = wide["band"].map(_BAND_ORDER)
        wide = wide.sort_values(["pos", "_order"]).drop(columns="_order")
        print(f"\n=== {bt_name}  (delta > 0 = that tier HELPS this cell) ===")
        print(wide.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
