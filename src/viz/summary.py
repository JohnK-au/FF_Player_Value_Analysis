"""One-page summary of preliminary value results (2026).

A 2x2 figure that tells the story so far:
  A. Most over/under-valued players (production-anchored surplus).
  B. Salary vs production-fair value (bargains above the line, overpays below).
  C. Replacement level by position — why elite QBs have low value-over-replacement.
  D. The two lenses (market-fit vs production-anchored) agree on who's mispriced.

Reads the already-built ``fair_value_2026.csv``; pulls replacement levels live.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import FIG_VALUE, PROCESSED_DIR
from ..models.production import vor_table

POS = ["QB", "RB", "WR", "TE"]
POS_COLOR = {"QB": "#1f77b4", "RB": "#ff7f0e", "WR": "#2ca02c", "TE": "#9467bd"}
GREEN, RED = "#2ca02c", "#d62728"
MY_TEAM = "Kerr"


def _panel_movers(ax, priced: pd.DataFrame, n: int = 10) -> None:
    bargains = priced.nsmallest(n, "surplus_prod")
    overpays = priced.nlargest(n, "surplus_prod")
    sel = pd.concat([overpays, bargains]).drop_duplicates("player").sort_values("surplus_prod")
    colors = [GREEN if s < 0 else RED for s in sel["surplus_prod"]]
    y = np.arange(len(sel))
    ax.barh(y, sel["surplus_prod"], color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.player} ({r.team})" for r in sel.itertuples()], fontsize=8)
    for yi, r in zip(y, sel.itertuples()):
        off = 8 if r.surplus_prod < 0 else -8
        ha = "left" if r.surplus_prod < 0 else "right"
        ax.annotate(f"{r.salary_2026:.0f}->{r.prod_fair:.0f}", (r.surplus_prod, yi),
                    xytext=(off, 0), textcoords="offset points", va="center", ha=ha, fontsize=6.5,
                    color="#333")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("surplus = salary - production-fair value  (cap units)")
    ax.set_title("A. Most over- / under-valued (production lens)", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], color=GREEN, label="under-valued (bargain)")
    ax.scatter([], [], color=RED, label="over-valued")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def _panel_scatter(ax, priced: pd.DataFrame) -> None:
    under = priced["surplus_prod"] < 0
    ax.scatter(priced.loc[under, "salary_2026"], priced.loc[under, "prod_fair"],
               c=GREEN, alpha=0.6, s=28, edgecolor="grey", linewidth=0.3, label="under-valued")
    ax.scatter(priced.loc[~under, "salary_2026"], priced.loc[~under, "prod_fair"],
               c=RED, alpha=0.6, s=28, edgecolor="grey", linewidth=0.3, label="over-valued")
    mine = priced[priced["team"] == MY_TEAM]
    ax.scatter(mine["salary_2026"], mine["prod_fair"], facecolor="none", edgecolor="black",
               linewidth=1.8, s=70, zorder=5, label=f"your team ({MY_TEAM})")
    lim = [1, max(priced["salary_2026"].max(), priced["prod_fair"].max()) * 1.1]
    ax.plot(lim, lim, "--", color="dimgrey", lw=1, label="fair (salary = value)")
    for r in pd.concat([priced.nsmallest(4, "surplus_prod"), priced.nlargest(4, "surplus_prod")]).itertuples():
        ax.annotate(r.player, (r.salary_2026, r.prod_fair), fontsize=6.5, color="#333",
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("2026 salary (cap units, log)")
    ax.set_ylabel("production-fair value (log)")
    ax.set_title("B. Salary vs production value (above line = bargain)", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)


def _panel_replacement(ax, pool: pd.DataFrame, repl: dict) -> None:
    pool = pool[pool["games"].fillna(0) >= 4]
    rng = np.random.default_rng(0)
    for i, pos in enumerate(POS):
        v = pool.loc[pool["position_group"] == pos, "ppg"].dropna()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, alpha=0.35,
                   color=POS_COLOR[pos])
        ax.hlines(repl[pos], i - 0.32, i + 0.32, color="black", lw=2, zorder=5)
        ax.annotate(f"repl {repl[pos]:.0f}", (i, repl[pos]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(POS)))
    ax.set_xticklabels(POS)
    ax.set_ylabel("2025 PPG (full season)")
    ax.set_title("C. Replacement level by position (high QB repl = low QB value)",
                 fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_lenses(ax, priced: pd.DataFrame) -> None:
    d = priced.dropna(subset=["surplus_market", "surplus_prod"])
    for pos in POS:
        s = d[d["position_group"] == pos]
        ax.scatter(s["surplus_market"], s["surplus_prod"], s=26, alpha=0.7,
                   color=POS_COLOR[pos], label=pos)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    for r in d.nlargest(4, "surplus_prod").itertuples():
        ax.annotate(r.player, (r.surplus_market, r.surplus_prod), fontsize=6.5, color="#333",
                    xytext=(3, 2), textcoords="offset points")
    ax.text(0.98, 0.98, "both overpaid", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#999")
    ax.text(0.02, 0.02, "both bargains", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="#999")
    ax.set_xlabel("market-fit surplus (salary - market-predicted)")
    ax.set_ylabel("production surplus (salary - production value)")
    ax.set_title("D. Both lenses agree on mispricing (top-right = both overpaid)",
                 fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False, title="position")


def plot_summary(out=None):
    d = pd.read_csv(PROCESSED_DIR / "fair_value_2026.csv")
    priced = d[d["prod_fair"].notna()].copy()
    pool, repl = vor_table(2025)

    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    _panel_movers(axes[0, 0], priced)
    _panel_scatter(axes[0, 1], priced)
    _panel_replacement(axes[1, 0], pool, repl)
    _panel_lenses(axes[1, 1], priced)
    fig.suptitle("Player Value — Preliminary Results (2026): production-anchored value vs salary",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = out or (FIG_VALUE / "value_summary_2026.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Saved", plot_summary())
