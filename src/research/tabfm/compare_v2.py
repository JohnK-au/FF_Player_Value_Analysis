"""Phase 4c — agreement analysis: TabFM vs the V2 engine ("two experts").

    .venv-tabfm/bin/python src/research/tabfm/compare_v2.py

Do the foundation model's forecasts and your hand-built V2 engine agree on who's
good? Join TabFM's 2025 PPG forecast to V2's dynasty_value on espn_id, rank-
correlate, and surface the biggest disagreements. Fast (cached preds).

CAVEAT (kept honest in the output): the two measure different things -- TabFM
predicts next-season PPG, V2 is a 0-100 dynasty value on a different horizon. So
this is a RANK comparison ("do they order players the same?"), not an accuracy
head-to-head. As a tiebreaker we also rank both against ACTUAL 2025 PPG.

READ FIRST
    02_design_rationale.md section 1 (the two-experts framing).
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
V2_MASTER = _ROOT / "data" / "processed" / "player_value_v2_2026.csv"
V2_VALUE_COL = "dynasty_value"   # which V2 lens to compare TabFM against


# ------------------------------------------------------------------ plumbing
def build_merged() -> pd.DataFrame:
    """One row per predict-2025 player: identity + TabFM forecast + actual 2025
    PPG + V2's dynasty_value. TabFM pred = mean over the two cached seeds."""
    _, test = ev.backtest_split(ev.load_pairs(), 2024)
    test = test[test["target_games"] >= bl.MIN_TARGET_GAMES].reset_index(drop=True)
    seed_preds = [np.load(PRED_DIR / f"predict_2025__tabfm_full_s{s}.npy")
                  for s in (0, 1)]
    df = test[["espn_id", "name", "position_group", "target_ppg"]].copy()
    df = df.rename(columns={"target_ppg": "actual_2025"})
    df["tabfm_pred"] = np.mean(seed_preds, axis=0)
    v2 = pd.read_csv(V2_MASTER)[["espn_id", V2_VALUE_COL]]
    return df.merge(v2, on="espn_id", how="inner")


# =========================================================================
# TODO(you) 4.3 -- quantify the agreement between the two experts.
#
# Given `df` with columns `pred_col` (TabFM) and `value_col` (V2), return:
#   (rho, ranked)
#   rho     Spearman rank correlation between the two columns (one float).
#           from scipy.stats import spearmanr; spearmanr(a, b).statistic
#   ranked  a COPY of df with a new "rank_gap" column:
#             rank each column (df[col].rank()), then
#             rank_gap = rank(pred_col) - rank(value_col)
#           Big positive gap = TabFM ranks the player much higher than V2 does;
#           big negative = V2 ranks them higher. Sorting by rank_gap surfaces the
#           disagreements at both ends.
#
# Both .rank() calls must use the SAME direction (default ascending is fine) so
# the subtraction is meaningful.
#
# Stuck after a real attempt? -> 04_solutions.md (4.3)
# =========================================================================
def rank_agreement(df: pd.DataFrame, pred_col: str = "tabfm_pred",
                   value_col: str = V2_VALUE_COL) -> tuple[float, pd.DataFrame]:
    from scipy.stats import spearmanr
    rho = spearmanr(df[pred_col], df[value_col]).statistic
    ranked = df.copy()
    ranked["rank_gap"] = ranked[pred_col].rank() - ranked[value_col].rank()
    return rho, ranked


def main() -> None:
    df = build_merged()
    rho, ranked = rank_agreement(df)                     # TODO 4.3

    print(f"=== TabFM forecast vs V2 {V2_VALUE_COL} — {len(df)} players ===")
    print(f"Spearman rank correlation: {rho:.3f}  "
          f"(1 = identical ordering, 0 = unrelated)")
    print("NOTE: different targets/horizons -- a RANK agreement check, not an "
          "accuracy test.\n")

    cols = ["name", "position_group", "tabfm_pred", V2_VALUE_COL,
            "actual_2025", "rank_gap"]
    ranked = ranked.sort_values("rank_gap")
    print(f"--- V2 rates far above TabFM (V2 high, TabFM low) ---")
    print(ranked.head(8)[cols].round(1).to_string(index=False))
    print(f"\n--- TabFM rates far above V2 (TabFM high, V2 low) ---")
    print(ranked.tail(8)[cols].round(1).to_string(index=False))

    # tiebreaker: which lens ordered ACTUAL 2025 better?
    a_tab = ev.spearman(df["actual_2025"], df["tabfm_pred"])
    a_v2 = ev.spearman(df["actual_2025"], df[V2_VALUE_COL])
    print(f"\n--- vs reality: Spearman with ACTUAL 2025 PPG ---")
    print(f"  TabFM forecast : {a_tab:.3f}")
    print(f"  V2 {V2_VALUE_COL}: {a_v2:.3f}")
    print("  (higher = that lens ranked who-actually-scored better)")


if __name__ == "__main__":
    main()
