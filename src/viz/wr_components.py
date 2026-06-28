"""Render a color-coded HTML table of all WRs with their v2 components.

Usage:
    python -m src.viz.wr_components

Writes ``figures/wr_components_2026.html`` (gitignored). Open in any browser.
Green = good, red = bad on each value column. Identity / contract columns
are uncolored; age is colored with low (young) = green for dynasty intuition.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT

MASTER_CSV = Path("data/processed/player_value_v2_2026.csv")
OUT_PATH = Path(REPO_ROOT) / "figures" / "wr_components_2026.html"

DISPLAY_COLS = [
    "player", "team", "nfl_team_2025", "salary_2026", "age",
    "production_value", "team_value", "age_value",
    "injury_value", "position_value", "intangibles_value",
    "on_field_value", "dynasty_value",
]

VALUE_COLS = [
    "production_value", "team_value", "age_value",
    "injury_value", "position_value", "intangibles_value",
    "on_field_value", "dynasty_value",
]

PRETTY_NAMES = {
    "player": "Player",
    "team": "League",
    "nfl_team_2025": "NFL Team",
    "salary_2026": "Salary",
    "age": "Age",
    "production_value": "Production",
    "team_value": "Team (val)",
    "age_value": "Age (val)",
    "injury_value": "Injury",
    "position_value": "Position*",
    "intangibles_value": "Intangibles*",
    "on_field_value": "On-Field Value",
    "dynasty_value": "Dynasty Value",
}


def render():
    df = pd.read_csv(MASTER_CSV)
    wr = df[df["position_group"] == "WR"].copy()
    wr = wr.sort_values("dynasty_value", ascending=False).reset_index(drop=True)
    wr.insert(0, "rank", range(1, len(wr) + 1))

    show = wr[["rank"] + DISPLAY_COLS].copy()
    show["salary_2026"] = show["salary_2026"].round(0).astype(int)
    show["age"] = show["age"].round(1)
    for c in VALUE_COLS:
        show[c] = show[c].round(1)
    # Rename columns up-front with unique pretty names (no collisions)
    show = show.rename(columns={**PRETTY_NAMES, "rank": "#"})

    pretty_value_cols = [PRETTY_NAMES[c] for c in VALUE_COLS]
    pretty_age = PRETTY_NAMES["age"]

    styled = (
        show.style
        .background_gradient(subset=pretty_value_cols, cmap="RdYlGn", vmin=0, vmax=100)
        .background_gradient(subset=[pretty_age], cmap="RdYlGn_r", vmin=22, vmax=33)
        .format({
            PRETTY_NAMES["salary_2026"]: "{:.0f}",
            pretty_age: "{:.1f}",
            **{c: "{:.1f}" for c in pretty_value_cols},
        })
        .set_caption(
            "<h2 style='text-align:left;margin-bottom:8px'>WR v2 components (2026 contract roster)</h2>"
            "<div style='text-align:left;font-size:12px;color:#666;margin-bottom:14px'>"
            "Green = good, red = bad on value columns (0-100). Age column: green = young (dynasty-favourable). "
            "Position + Intangibles are neutral stubs (Phase 4.5 / user input). "
            "Dynasty Value is uniform-1/6 combine of the 6 components.</div>"
        )
        .set_table_styles([
            {"selector": "th", "props": "background-color: #2c3e50; color: white; padding: 8px; text-align: center; font-weight: 600;"},
            {"selector": "td", "props": "padding: 5px 10px; text-align: right; font-family: monospace;"},
            {"selector": "td.col0, td.col1, td.col2", "props": "text-align: left;"},
            {"selector": "table", "props": "border-collapse: collapse; border: 1px solid #ccc; font-size: 13px;"},
            {"selector": "caption", "props": "caption-side: top;"},
            {"selector": f"th.col_heading.level0", "props": "border-bottom: 2px solid #2c3e50;"},
            {"selector": f"th.col{DISPLAY_COLS.index('dynasty_value') + 1}", "props": "background-color: #1a252f; border-left: 3px solid #f39c12;"},
            {"selector": f"th.col{DISPLAY_COLS.index('on_field_value') + 1}", "props": "background-color: #34495e; border-left: 2px solid #95a5a6;"},
        ])
        .set_properties(**{"border": "1px solid #e0e0e0"})
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = styled.to_html()
    # Wrap in a minimal HTML doc with viewport meta + base styling
    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WR v2 components (2026)</title>
<style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 20px; background: #f9f9f9; }}</style>
</head><body>
{html}
</body></html>
"""
    OUT_PATH.write_text(full_html, encoding="utf-8")
    print(f"Wrote {len(show)} WR rows -> {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    render()
