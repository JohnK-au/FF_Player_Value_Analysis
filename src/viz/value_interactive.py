"""Interactive (Plotly) player-value scatter — open the HTML in a browser.

Same idea as the static facets but hover-driven: every dot shows player / team /
salary / PPG / age / fair salary / surplus on hover, so nothing needs labelling.
Faceted by position, log-x salary, colour = surplus, size = age; the user's
players are drawn as stars. Self-contained HTML (works offline).
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import FIG_VALUE
from ..data.espn import get_league
from ..models.value import fair_value_table

MIN_GAMES = 5


def _my_ids() -> set:
    try:
        tid = int(os.environ.get("ESPN_TEAM_ID", 8))
        lg = get_league()
        return {p.playerId for t in lg.teams if t.team_id == tid for p in t.roster}
    except Exception as e:
        print(f"(could not load ESPN roster: {type(e).__name__}: {str(e)[:60]})")
        return set()


def plot_interactive(y_metric: str = "ppg_2025", out: Path | None = None) -> Path:
    import plotly.express as px

    d, _ = fair_value_table()
    d = d[d["games_2025"] >= MIN_GAMES].copy()
    d["roster"] = d["espn_id"].isin(_my_ids()).map({True: "Mine", False: "Other"})

    fig = px.scatter(
        d, x="salary_2026", y=y_metric, color="surplus", size="age", symbol="roster",
        facet_col="position_group", facet_col_wrap=2,
        category_orders={"position_group": ["QB", "RB", "WR", "TE"]},
        symbol_map={"Mine": "star", "Other": "circle"},
        color_continuous_scale="RdYlGn_r", color_continuous_midpoint=0,
        log_x=True, size_max=18, hover_name="player",
        hover_data={
            "team": True, "salary_2026": ":.0f", y_metric: ":.1f", "age": ":.1f",
            "fair_salary": ":.0f", "surplus": ":.0f", "roster": False,
            "position_group": False,
        },
        labels={"salary_2026": "2026 salary (cap units)", "surplus": "surplus"},
        title="Player Value — Salary vs Production (hover for detail; ★ = your players)",
        height=850,
    )
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    out = out or (FIG_VALUE / "value_interactive.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="inline")
    return out


if __name__ == "__main__":
    print("Saved", plot_interactive())
