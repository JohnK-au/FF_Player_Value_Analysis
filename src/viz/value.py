"""Player value scatter — salary vs production, faceted by position.

Per-position panels: x = 2026 salary (log scale), y = points/game, colour =
fair-value surplus (green bargain → red overpaid), marker size ∝ age, and the
fitted model's fair-value curve overlaid. The user's players (ESPN team) are
ringed in black and always labelled; high earners and high scorers are labelled
too. ``plot_value_facets`` also supports a 2-year-average y-axis.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import FIGURES_DIR
from ..data.espn import get_league
from ..models.value import CAT_FEATURES, NUM_FEATURES, fair_value_table

POS_ORDER = ["QB", "RB", "WR", "TE"]
MIN_GAMES = 5


def my_player_ids() -> tuple[set, str]:
    """ESPN player ids on the user's team (best-effort; empty if ESPN is down)."""
    team_id = int(os.environ.get("ESPN_TEAM_ID", 8))
    try:
        lg = get_league()
        team = next(t for t in lg.teams if t.team_id == team_id)
        return {p.playerId for p in team.roster}, team.team_name
    except Exception as e:
        print(f"(could not load ESPN roster: {type(e).__name__}: {str(e)[:60]})")
        return set(), ""


def _age_to_size(age: pd.Series) -> pd.Series:
    return ((age.fillna(26) - 20).clip(lower=3, upper=18)) * 9


def _fair_curve(model, position: str, d_pos: pd.DataFrame):
    """Model's fair salary across the PPG range (other features at pos medians)."""
    ppgs = np.linspace(d_pos["ppg_2025"].min(), d_pos["ppg_2025"].max(), 40)
    base = {f: d_pos[f].median() for f in NUM_FEATURES}
    rows = []
    for v in ppgs:
        r = dict(base)
        r["ppg_2025"] = v
        r["ppg_2024"] = v
        r["position_group"] = position
        rows.append(r)
    X = pd.DataFrame(rows)[CAT_FEATURES + NUM_FEATURES]
    return model.predict(X), ppgs


def plot_value_facets(
    y_metric: str = "ppg_2025", logx: bool = True, label_top: int = 5,
    out: Path | None = None,
) -> Path:
    d, model = fair_value_table()
    d = d[d["games_2025"] >= MIN_GAMES].copy()
    if y_metric == "ppg_2yr":
        d["ppg_2yr"] = d[["ppg_2025", "ppg_2024"]].mean(axis=1)
    ylabel = {"ppg_2025": "2025 PPG (wk 1–13)", "ppg_2yr": "2024–25 avg PPG"}[y_metric]

    my_ids, my_team = my_player_ids()
    d["mine"] = d["espn_id"].isin(my_ids)
    d["size"] = _age_to_size(d["age"])

    vmax = float(d["surplus"].abs().quantile(0.9)) or 1.0
    norm = plt.Normalize(-vmax, vmax)
    cmap = plt.get_cmap("RdYlGn_r")  # − surplus (bargain) green, + (overpaid) red

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sc = None
    for ax, pos in zip(axes.ravel(), POS_ORDER):
        s = d[d["position_group"] == pos]
        sc = ax.scatter(s["salary_2026"], s[y_metric], c=s["surplus"], cmap=cmap,
                        norm=norm, s=s["size"], alpha=0.85, edgecolor="grey", linewidth=0.4)
        mine = s[s["mine"]]
        ax.scatter(mine["salary_2026"], mine[y_metric], s=mine["size"], facecolor="none",
                   edgecolor="black", linewidth=2.2, zorder=5)

        if y_metric == "ppg_2025" and len(s) > 3:
            xs, ys = _fair_curve(model, pos, s)
            ax.plot(xs, ys, color="dimgrey", ls="--", lw=1, alpha=0.7, zorder=1,
                    label="model fair value")

        to_label = pd.concat([
            mine, s.nlargest(label_top, "salary_2026"), s.nlargest(label_top, y_metric)
        ]).drop_duplicates("espn_id")
        for _, r in to_label.iterrows():
            ax.annotate(r["player"], (r["salary_2026"], r[y_metric]), fontsize=7,
                        fontweight="bold" if r["mine"] else "normal",
                        color="black" if r["mine"] else "#444",
                        xytext=(4, 3), textcoords="offset points", zorder=6)

        if logx:
            ax.set_xscale("log")
        ax.set_title(pos, fontweight="bold")
        ax.set_xlabel("2026 salary (cap units, log)" if logx else "2026 salary (cap units)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("surplus = actual − fair  (green = bargain, red = overpaid)")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=2, markersize=10,
               label=f"your players{f' ({my_team})' if my_team else ''}"),
        Line2D([0], [0], ls="--", color="dimgrey", label="model fair-value curve"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=6,
               label="marker size ∝ age"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(f"Player Value by Position — Salary vs {ylabel}", fontsize=16,
                 fontweight="bold", y=0.99)

    out = out or (FIGURES_DIR / f"value_facets_{y_metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Saved", plot_value_facets("ppg_2025", logx=True))
    print("Saved", plot_value_facets("ppg_2yr", logx=True))
