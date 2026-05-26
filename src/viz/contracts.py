"""Visualize each team's 2026 contract commitments as a timeline (Gantt) grid.

One panel per team: every player is a horizontal bar spanning the seasons they're
under contract (2026 onward), coloured by salary. Extensions are outlined and
marked with a star. Scope = active rosters rolled forward + extensions; rookies,
tags, IR and dead cap are not yet included.

The output PNG contains league data (names + salaries), so it is written to the
git-ignored ``figures/`` directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..config import FIG_CONTRACTS
from ..data.contracts import UPCOMING_SEASON, TEAMS, build_2026_contracts

_CMAP = plt.get_cmap("YlOrRd")


def plot_team_contracts(
    df=None, *, out: Path | None = None
) -> tuple[Path, list[str]]:
    """Render the per-team 2026 contract timeline grid; return (path, notes)."""
    notes: list[str] = []
    if df is None:
        df, notes = build_2026_contracts()

    last_season = int(df["last_season"].max())
    seasons = list(range(UPCOMING_SEASON, last_season + 1))
    norm = Normalize(vmin=float(df["salary_2026"].min()), vmax=float(df["salary_2026"].max()))

    fig, axes = plt.subplots(4, 2, figsize=(17, 26))
    fig.suptitle(
        "2026 Contracts by Team  —  active rosters rolled forward + extensions\n"
        "(bar length = years under contract, colour = salary; "
        "★ = extension. Excludes rookies, tags, IR & dead cap.)",
        fontsize=15, fontweight="bold",
    )

    for ax, team in zip(axes.ravel(), TEAMS):
        sub = (
            df[df["team"] == team]
            .sort_values(["last_season", "salary_2026"], ascending=[True, True])
            .reset_index(drop=True)
        )
        committed = int(sub["salary_2026"].sum())
        labels = []
        for y, row in sub.iterrows():
            is_ext = row["source"] == "extension"
            ax.barh(
                y, width=row["years_2026"], left=UPCOMING_SEASON, height=0.7,
                color=_CMAP(norm(row["salary_2026"])),
                edgecolor="navy" if is_ext else "grey",
                linewidth=1.8 if is_ext else 0.5,
            )
            ax.text(
                UPCOMING_SEASON + row["years_2026"] + 0.05, y,
                f"{int(row['salary_2026'])}",
                va="center", ha="left", fontsize=7,
            )
            labels.append(("★ " if is_ext else "") + str(row["player"]))

        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_xlim(UPCOMING_SEASON, last_season + 1)
        ax.set_xticks([s + 0.5 for s in seasons])
        ax.set_xticklabels([str(s) for s in seasons], fontsize=8)
        for s in seasons:
            ax.axvline(s, color="lightgrey", linewidth=0.6, zorder=0)
        ax.set_title(f"{team}  —  {len(sub)} players, {committed} committed", fontsize=11)
        ax.invert_yaxis()  # biggest/longest deals at the top

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=_CMAP), ax=axes, fraction=0.025, pad=0.02
    )
    cbar.set_label("Salary (cap units)")

    out = out or (FIG_CONTRACTS / "team_contracts_2026.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out, notes


if __name__ == "__main__":
    path, notes = plot_team_contracts()
    if notes:
        print(f"Reconciliation notes ({len(notes)}):")
        for n in notes:
            print("  -", n)
    print(f"\nSaved contract timeline to {path}")
