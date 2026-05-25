"""Cap-distribution figures.

- ``plot_cap_distribution`` (primary): per-team salary-cap composition for one
  season (default 2026) — total salary vs the 1500 cap, broken down into where
  the salary goes (contracts, extensions, rookies, tags, practice squad, cuts,
  penalties), with cap space shown.
- ``plot_cap_projection``: the same composition across 2025-2029, so you can see
  how commitments roll off and cap space opens up over time.

Output PNGs contain league data, so they're written to the git-ignored ``figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..data.cap import CAP_TOTAL, cap_breakdown
from ..data.contracts import TEAMS, UPCOMING_SEASON

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "figures"

# Stacking order: (column, label, colour). "Adjustments" is computed (signed).
COMPONENTS = [
    ("contracts", "Contracts (active)", "#1f77b4"),
    ("extensions", "Extensions", "#17becf"),
    ("rookies", "Rookies", "#2ca02c"),
    ("tags", "Franchise tags", "#9467bd"),
    ("practice_squad", "Practice squad", "#8c564b"),
    ("cuts", "Cuts (dead cap)", "#d62728"),
    ("penalties", "Penalties", "#ff9896"),
]
ADJ_LABEL, ADJ_COLOR = "IR / trade / other", "#c7c7c7"
SPACE_COLOR = "#edf2f7"
_COMP_COLS = [c for c, _, _ in COMPONENTS]


def _legend_handles():
    handles = [Patch(facecolor=col, label=lab) for _, lab, col in COMPONENTS]
    handles.append(Patch(facecolor=ADJ_COLOR, label=ADJ_LABEL))
    handles.append(Patch(facecolor=SPACE_COLOR, edgecolor="grey", label="Cap space"))
    return handles


def _stack_row(ax, y, row, *, horizontal=True, thickness=0.62):
    """Draw one team's stacked bar (components + signed adjustment), return used."""
    cum = 0.0
    for col, _, color in COMPONENTS:
        val = float(row[col])
        if val:
            (ax.barh if horizontal else ax.bar)(
                y, val, **({"left": cum} if horizontal else {"bottom": cum}),
                color=color, edgecolor="white", linewidth=0.4,
                **({"height": thickness} if horizontal else {"width": thickness}),
            )
        cum += val
    used = float(row["sheet_used"])
    adj = used - cum  # trade credits + IR returns + residual (signed)
    if abs(adj) > 0.05:
        (ax.barh if horizontal else ax.bar)(
            y, adj, **({"left": cum} if horizontal else {"bottom": cum}),
            color=ADJ_COLOR, edgecolor="white", linewidth=0.4,
            **({"height": thickness} if horizontal else {"width": thickness}),
        )
    # cap space to the 1500 cap
    (ax.barh if horizontal else ax.bar)(
        y, CAP_TOTAL - used, **({"left": used} if horizontal else {"bottom": used}),
        color=SPACE_COLOR, edgecolor="lightgrey", linewidth=0.4,
        **({"height": thickness} if horizontal else {"width": thickness}),
    )
    return used


def plot_cap_distribution(season: int = UPCOMING_SEASON, out: Path | None = None) -> Path:
    """Primary: one season's per-team cap composition vs the 1500 cap."""
    df = cap_breakdown(season).set_index("team").reindex(TEAMS)
    df = df.sort_values("sheet_used")  # smallest at bottom; biggest on top after barh

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for i, (team, row) in enumerate(df.iterrows()):
        used = _stack_row(ax, i, row)
        ax.text(used - 8, i, f"{used:.0f}", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")
        ax.text(CAP_TOTAL - 8, i, f"space {row['cap_space']:.0f}", va="center",
                ha="right", fontsize=8, color="#555")

    ax.axvline(CAP_TOTAL, color="black", linewidth=1.3)
    ax.text(CAP_TOTAL, len(df) - 0.3, f" cap {CAP_TOTAL}", fontsize=9, va="bottom")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_xlim(0, CAP_TOTAL * 1.02)
    ax.set_xlabel("Cap units")
    ax.set_title(f"{season} Salary-Cap Distribution by Team", fontsize=15, fontweight="bold")
    ax.legend(handles=_legend_handles(), ncol=3, fontsize=8,
              loc="lower right", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)

    out = out or (FIGURES_DIR / f"cap_distribution_{season}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cap_projection(
    seasons: range = range(2025, 2030), out: Path | None = None
) -> Path:
    """Per-team cap composition across seasons (commitments rolling off over time)."""
    data = {s: cap_breakdown(s).set_index("team") for s in seasons}
    fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharey=True)
    fig.suptitle(
        "Cap Distribution Projection by Team (2025–2029)\n"
        "stacked salary vs the 1500 cap; cap space is the light remainder",
        fontsize=15, fontweight="bold",
    )
    xs = list(seasons)
    for ax, team in zip(axes.ravel(), TEAMS):
        for j, s in enumerate(xs):
            _stack_row(ax, j, data[s].loc[team], horizontal=False, thickness=0.6)
        ax.axhline(CAP_TOTAL, color="black", linewidth=1.0)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(s) for s in xs], fontsize=9)
        ax.set_ylim(0, CAP_TOTAL * 1.05)
        ax.set_title(team, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[0].legend(handles=_legend_handles(), ncol=2, fontsize=7, loc="upper right")

    out = out or (FIGURES_DIR / "cap_projection_2025_2029.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = plot_cap_distribution(UPCOMING_SEASON)
    p2 = plot_cap_projection()
    print(f"Saved {p1}\nSaved {p2}")
