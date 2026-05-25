"""Player value scatter: contract salary (x) vs points per game (y).

Colored by position, with median quadrant lines (bargains / stars / depth /
overpaid) and the biggest bargains & overpays (by the fair-value model's surplus)
labelled. Skill positions only, regulars (>=5 games) to keep PPG meaningful.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..config import FIGURES_DIR
from ..models.value import fair_value_table

POS_COLORS = {"QB": "#d62728", "RB": "#2ca02c", "WR": "#1f77b4", "TE": "#ff7f0e"}
MIN_GAMES = 5


def plot_value_scatter(out: Path | None = None, annotate: int = 7) -> Path:
    d, _ = fair_value_table()
    d = d[d["games_2025"] >= MIN_GAMES].copy()

    fig, ax = plt.subplots(figsize=(13, 8.5))
    for pos, color in POS_COLORS.items():
        s = d[d["position_group"] == pos]
        ax.scatter(s["salary_2026"], s["ppg_2025"], c=color, s=46, alpha=0.82,
                   edgecolor="white", linewidth=0.5, label=pos)

    # median quadrant lines + corner labels
    smed, pmed = d["salary_2026"].median(), d["ppg_2025"].median()
    xmax, ymax = d["salary_2026"].max() * 1.06, d["ppg_2025"].max() * 1.12
    ax.axvline(smed, color="grey", ls="--", lw=0.8)
    ax.axhline(pmed, color="grey", ls="--", lw=0.8)
    for x, y, ha, va, txt in [
        (smed * 0.5, ymax * 0.97, "center", "top", "BARGAINS\n(cheap, productive)"),
        (smed + (xmax - smed) * 0.5, ymax * 0.97, "center", "top", "STARS\n(pricey, productive)"),
        (smed * 0.5, pmed * 0.35, "center", "center", "Low-cost depth"),
        (smed + (xmax - smed) * 0.5, pmed * 0.35, "center", "center", "OVERPAID\n(pricey, low output)"),
    ]:
        ax.text(x, y, txt, color="grey", fontsize=9, ha=ha, va=va, alpha=0.8)

    # label the biggest bargains (most negative surplus) and overpays (most positive)
    for _, r in d.nsmallest(annotate, "surplus").iterrows():
        ax.annotate(r["player"], (r["salary_2026"], r["ppg_2025"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points", color="#114")
    for _, r in d.nlargest(annotate, "surplus").iterrows():
        ax.annotate(r["player"], (r["salary_2026"], r["ppg_2025"]),
                    fontsize=7, xytext=(4, -8), textcoords="offset points", color="#611")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("2026 contract salary (cap units)")
    ax.set_ylabel("2025 points per game (fantasy weeks 1–13)")
    ax.set_title("Player Value — Salary vs Production (2026)", fontsize=15, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c, label=p) for p, c in POS_COLORS.items()],
              title="Position", loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    out = out or (FIGURES_DIR / "value_scatter_2026.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Saved", plot_value_scatter())
