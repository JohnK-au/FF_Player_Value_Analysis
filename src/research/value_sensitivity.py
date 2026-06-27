"""Sensitivity sweep on the value-engine tunables.

Three tunables drive the dollar magnitudes in the production-anchored value layer:
    DEEP_FACTOR     -- how deep below starter-replacement we set the pricing floor
    RISK_LAMBDA     -- how much weekly downside discounts a player's adjusted production
    DISCOUNT_RATE   -- per-year discount applied to future contract years (dynasty)

These were originally set by intuition (0.5 / 0.5 / 0.10). This module sweeps each
in isolation (others held at default) and reports:

  * how much the league rate ($/deep-VOR point) shifts,
  * how many players' surplus changes SIGN (overpaid <-> bargain) across the sweep,
  * Spearman rank correlation of surplus_2026 across adjacent settings (stability),
  * tracked-player movement: top bargains/overpays in 2026 + a curated watch list.

A *robust* tunable would show: small rate change, very few sign flips, near-perfect
rank correlation across reasonable settings. A *fragile* tunable would show the
opposite -- and that's a signal to tighten the recipe.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.models.value import (  # noqa: E402
    DEEP_FACTOR, DISCOUNT_RATE, RISK_LAMBDA,
    dynasty_value_table, projected_value_table,
)

WATCH = [
    "Puka Nacua", "Bijan Robinson", "Jahmyr Gibbs", "Devon Achane",
    "Justin Jefferson", "AJ Brown", "Patrick Mahomes", "Lamar Jackson",
    "Drake Maye", "Mark Andrews", "TJ Hockenson",
]


def _summarize_run(label: str, base, prev_surplus: pd.Series | None,
                   base_surplus: pd.Series) -> dict:
    """Compute the headline metrics for one parameter setting."""
    priced = base[base["value_2026"].notna()].copy()
    s = priced.set_index("player")["surplus_2026"]
    out = {
        "label": label,
        "n_priced": len(priced),
        "n_zero_value": int((priced["value_2026"] == 0).sum()),
        "median_fair": round(float(priced["value_2026"].median()), 1),
        "top_fair": round(float(priced["value_2026"].max()), 1),
    }
    if prev_surplus is not None:
        common = s.index.intersection(prev_surplus.index)
        if len(common) > 5:
            rho, _ = spearmanr(s[common], prev_surplus[common])
            out["rho_vs_prev"] = round(float(rho), 3)
            sign_flip = int(((s[common] > 0) != (prev_surplus[common] > 0)).sum())
            out["sign_flips_vs_prev"] = sign_flip
    rho_base, _ = spearmanr(s, base_surplus.reindex(s.index))
    out["rho_vs_default"] = round(float(rho_base), 3)
    sign_flip_base = int(((s > 0) != (base_surplus.reindex(s.index) > 0)).sum())
    out["sign_flips_vs_default"] = sign_flip_base
    return out


def sweep_deep_factor(values=(0.3, 0.4, 0.5, 0.6, 0.7)) -> tuple[pd.DataFrame, dict]:
    """Vary DEEP_FACTOR holding RISK_LAMBDA, DISCOUNT_RATE at defaults."""
    print(f"\n=== Sensitivity sweep: DEEP_FACTOR  (RISK_LAMBDA={RISK_LAMBDA}, "
          f"DISCOUNT_RATE={DISCOUNT_RATE}) ===")
    rows: list[dict] = []
    watch_tracks: dict[str, list[float]] = {p: [] for p in WATCH}
    base_df, _, _ = projected_value_table(deep_factor=DEEP_FACTOR)
    base_surplus = base_df.set_index("player")["surplus_2026"]
    prev_surplus = None
    for v in values:
        df, _, _ = projected_value_table(deep_factor=v)
        summary = _summarize_run(f"DF={v}", df, prev_surplus, base_surplus)
        rows.append(summary)
        s = df.set_index("player")["surplus_2026"]
        for p in WATCH:
            watch_tracks[p].append(float(s.get(p, np.nan)))
        prev_surplus = s
    return pd.DataFrame(rows), watch_tracks


def sweep_risk_lambda(values=(0.0, 0.25, 0.5, 0.75, 1.0)) -> tuple[pd.DataFrame, dict]:
    """Vary RISK_LAMBDA holding DEEP_FACTOR, DISCOUNT_RATE at defaults."""
    print(f"\n=== Sensitivity sweep: RISK_LAMBDA  (DEEP_FACTOR={DEEP_FACTOR}, "
          f"DISCOUNT_RATE={DISCOUNT_RATE}) ===")
    rows: list[dict] = []
    watch_tracks: dict[str, list[float]] = {p: [] for p in WATCH}
    base_df, _, _ = projected_value_table(risk_lambda=RISK_LAMBDA)
    base_surplus = base_df.set_index("player")["surplus_2026"]
    prev_surplus = None
    for v in values:
        df, _, _ = projected_value_table(risk_lambda=v)
        summary = _summarize_run(f"RL={v}", df, prev_surplus, base_surplus)
        rows.append(summary)
        s = df.set_index("player")["surplus_2026"]
        for p in WATCH:
            watch_tracks[p].append(float(s.get(p, np.nan)))
        prev_surplus = s
    return pd.DataFrame(rows), watch_tracks


def sweep_discount_rate(values=(0.05, 0.10, 0.15, 0.20, 0.25)) -> tuple[pd.DataFrame, dict]:
    """Vary DISCOUNT_RATE for dynasty value (other tunables at defaults)."""
    print(f"\n=== Sensitivity sweep: DISCOUNT_RATE  (DEEP_FACTOR={DEEP_FACTOR}, "
          f"RISK_LAMBDA={RISK_LAMBDA}) ===")
    rows: list[dict] = []
    watch_tracks: dict[str, list[float]] = {p: [] for p in WATCH}
    base_df, _, _ = dynasty_value_table(discount_rate=DISCOUNT_RATE)
    base_surplus = base_df.set_index("player")["dynasty_surplus"]
    prev_surplus = None
    for v in values:
        df, _, _ = dynasty_value_table(discount_rate=v)
        priced = df[df["dynasty_value"].notna()].copy()
        s = priced.set_index("player")["dynasty_surplus"]
        # Build a custom summary for dynasty (uses dynasty_surplus)
        summary = {
            "label": f"DR={v}",
            "n_priced": len(priced),
            "median_dynasty_value": round(float(priced["dynasty_value"].median()), 1),
            "top_dynasty_value": round(float(priced["dynasty_value"].max()), 1),
        }
        if prev_surplus is not None:
            common = s.index.intersection(prev_surplus.index)
            if len(common) > 5:
                rho, _ = spearmanr(s[common], prev_surplus[common])
                summary["rho_vs_prev"] = round(float(rho), 3)
                summary["sign_flips_vs_prev"] = int(
                    ((s[common] > 0) != (prev_surplus[common] > 0)).sum()
                )
        rho_base, _ = spearmanr(s, base_surplus.reindex(s.index))
        summary["rho_vs_default"] = round(float(rho_base), 3)
        summary["sign_flips_vs_default"] = int(
            ((s > 0) != (base_surplus.reindex(s.index) > 0)).sum()
        )
        rows.append(summary)
        for p in WATCH:
            watch_tracks[p].append(float(s.get(p, np.nan)))
        prev_surplus = s
    return pd.DataFrame(rows), watch_tracks


def _print_watch(label: str, settings: tuple, tracks: dict, value_label: str):
    """Print the watch-list table: rows = players, cols = setting values."""
    rows = []
    for p in WATCH:
        vals = tracks[p]
        row = {"player": p}
        for setting, val in zip(settings, vals):
            row[f"{setting}"] = round(val, 1) if not np.isnan(val) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n--- {label}: watch-list {value_label} across the sweep ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    pd.options.display.width = 200
    pd.options.display.max_rows = 200

    # DEEP_FACTOR sweep
    df_summary, df_watch = sweep_deep_factor()
    print(df_summary.to_string(index=False))
    _print_watch("DEEP_FACTOR", (0.3, 0.4, 0.5, 0.6, 0.7), df_watch, "surplus_2026")

    # RISK_LAMBDA sweep
    rl_summary, rl_watch = sweep_risk_lambda()
    print(rl_summary.to_string(index=False))
    _print_watch("RISK_LAMBDA", (0.0, 0.25, 0.5, 0.75, 1.0), rl_watch, "surplus_2026")

    # DISCOUNT_RATE sweep
    dr_summary, dr_watch = sweep_discount_rate()
    print(dr_summary.to_string(index=False))
    _print_watch("DISCOUNT_RATE", (0.05, 0.10, 0.15, 0.20, 0.25), dr_watch, "dynasty_surplus")
