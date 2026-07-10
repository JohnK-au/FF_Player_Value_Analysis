"""Render cross-position HTML tables for multiple Position-weight variants.

Compares how different weightings of the 4 Position sub-metrics (Marginal
Gap, Total Impact, Supply-Demand, Slot Count) shift cross-position dynasty
value rankings. Each variant produces a single HTML at
``figures/cross_position_{variant}.html`` with all 4 positions in one
ranked table.

Usage:
    python -m src.viz.cross_position_variants
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.models.components.combine import DEFAULT_OFV_WEIGHTS

MASTER_CSV = Path("data/processed/player_value_v2_2026.csv")
OUT_DIR = Path(REPO_ROOT) / "figures"

# PPG-based sub-metrics (M and T in absolute PPG units, cross-position comparable).
# Derived 2026-06-28 from 2025 PPG (games >= 4) for the 490 priced players in
# master CSV. Multiple replacement-tier definitions:
#   STRICT: replacement = single player at rank (8 * S_p + 1)
#   BACKUP: replacement = avg PPG of ranks N+1 to 2N (first-bench tier)
#   FA:     replacement = avg PPG of ranks 2N+1 to 3N (realistic FA tier; user's QB 16-24 framing)
#   DEEP:   replacement = avg PPG of ranks 3N+1 to 4N (deep FA tier)
SUBMETRICS_STRICT = {
    "QB": {"S": 1.0, "M": 3.04, "T": 3.04,  "D": 0.434},
    "RB": {"S": 2.5, "M": 4.58, "T": 11.45, "D": 0.432},
    "WR": {"S": 3.0, "M": 3.81, "T": 11.43, "D": 0.338},
    "TE": {"S": 1.5, "M": 2.81, "T": 4.22,  "D": 0.345},
}
SUBMETRICS_BACKUP = {
    "QB": {"S": 1.0, "M": 5.05,  "T": 5.05,  "D": 0.434},
    "RB": {"S": 2.5, "M": 7.79,  "T": 19.49, "D": 0.432},
    "WR": {"S": 3.0, "M": 5.22,  "T": 15.66, "D": 0.338},
    "TE": {"S": 1.5, "M": 4.44,  "T": 6.67,  "D": 0.345},
}
SUBMETRICS_FA = {
    "QB": {"S": 1.0, "M": 9.64,  "T": 9.64,  "D": 0.434},
    "RB": {"S": 2.5, "M": 11.39, "T": 28.48, "D": 0.432},
    "WR": {"S": 3.0, "M": 7.83,  "T": 23.50, "D": 0.338},
    "TE": {"S": 1.5, "M": 7.64,  "T": 11.46, "D": 0.345},
}
SUBMETRICS_DEEP = {
    "QB": {"S": 1.0, "M": 12.82, "T": 12.82, "D": 0.434},
    "RB": {"S": 2.5, "M": 13.49, "T": 33.74, "D": 0.432},
    "WR": {"S": 3.0, "M": 10.77, "T": 32.30, "D": 0.338},
    "TE": {"S": 1.5, "M": 9.74,  "T": 14.61, "D": 0.345},
}
# Default for back-compat
SUBMETRICS = SUBMETRICS_STRICT
METRICS = ("M", "T", "D", "S")

# Weighting variants to compare. Each is a dict of metric -> weight; weights
# need not sum to 1 (we just z-score and dot-product).
# Each variant is (weights, submetrics_dict). Weights need not sum to 1.
VARIANTS: dict[str, dict] = {
    "v1_equal":              {"weights": {"M": 0.25, "T": 0.25, "D": 0.25, "S": 0.25}, "submetrics": SUBMETRICS_STRICT},
    "v2_marginal_gap_only":  {"weights": {"M": 1.0,  "T": 0.0,  "D": 0.0,  "S": 0.0},  "submetrics": SUBMETRICS_STRICT},
    "v3_total_impact_only":  {"weights": {"M": 0.0,  "T": 1.0,  "D": 0.0,  "S": 0.0},  "submetrics": SUBMETRICS_STRICT},
    "v4_supply_demand_only": {"weights": {"M": 0.0,  "T": 0.0,  "D": 1.0,  "S": 0.0},  "submetrics": SUBMETRICS_STRICT},
    "v5_hybrid_m_favored":   {"weights": {"M": 0.4,  "T": 0.3,  "D": 0.2,  "S": 0.1},  "submetrics": SUBMETRICS_STRICT},
    "v6_vorp_backup_T":      {"weights": {"M": 0.0,  "T": 1.0,  "D": 0.0,  "S": 0.0},  "submetrics": SUBMETRICS_BACKUP},
    "v7_vorp_fa_T":          {"weights": {"M": 0.0,  "T": 1.0,  "D": 0.0,  "S": 0.0},  "submetrics": SUBMETRICS_FA},
    "v8_vorp_deep_T":        {"weights": {"M": 0.0,  "T": 1.0,  "D": 0.0,  "S": 0.0},  "submetrics": SUBMETRICS_DEEP},
}

VARIANT_DESCRIPTIONS = {
    "v1_equal":              "Equal weights (M=T=D=S=0.25). Replacement = strict (single player at rank N+1).",
    "v2_marginal_gap_only":  "Marginal Gap only (M=1.0). Strict replacement. Pure per-slot elite value; ignores slot count and scarcity.",
    "v3_total_impact_only":  "Total Impact only (T=1.0). Strict replacement. Pure 'winning the position' framing; T = M x S includes slot count.",
    "v4_supply_demand_only": "Supply-Demand only (D=1.0). Pure economic scarcity (league demand / NFL pool size).",
    "v5_hybrid_m_favored":   "Hybrid (M=0.4, T=0.3, D=0.2, S=0.1). Strict replacement. Value-gap heavy with light demand/scarcity.",
    "v6_vorp_backup_T":      "VORP-BACKUP Total Impact only. Replacement = avg PPG of ranks N+1 to 2N (first-bench tier). For QB: ranks 9-16.",
    "v7_vorp_fa_T":          "VORP-FA Total Impact only. Replacement = avg PPG of ranks 2N+1 to 3N (realistic FA tier). For QB: ranks 17-24 (user's specific framing).",
    "v8_vorp_deep_T":        "VORP-DEEP Total Impact only. Replacement = avg PPG of ranks 3N+1 to 4N (deep FA tier). For QB: ranks 25-32. Biggest gaps; WR climbs to near-tied with RB.",
}

POSITION_ORDER = ("QB", "RB", "WR", "TE")
POS_WEIGHT_IN_COMBINE = DEFAULT_OFV_WEIGHTS["position_value"]  # 0.05

# Locked v1 scores that were used to compute current dynasty_value in master CSV
CURRENT_POSITION_SCORES = {"QB": 0.0, "RB": 100.0, "WR": 35.2, "TE": 21.2}


def compute_position_scores(weights: dict[str, float], submetrics: dict = None) -> dict[str, float]:
    """Compute per-position [0, 100] scores given sub-metric weights + submetrics.

    Methodology: z-score each sub-metric across the 4 positions, weighted-sum
    into composite, min-max normalize to [0, 100].
    """
    if submetrics is None:
        submetrics = SUBMETRICS
    z_scores = {}
    for m in METRICS:
        vals = [submetrics[p][m] for p in POSITION_ORDER]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        z_scores[m] = {p: (submetrics[p][m] - mean) / std if std > 0 else 0.0
                       for p in POSITION_ORDER}

    composite = {
        p: sum(weights[m] * z_scores[m][p] for m in METRICS)
        for p in POSITION_ORDER
    }
    lo, hi = min(composite.values()), max(composite.values())
    if hi == lo:
        return {p: 50.0 for p in POSITION_ORDER}
    return {p: round(100 * (composite[p] - lo) / (hi - lo), 1) for p in POSITION_ORDER}


DISPLAY_COLS = [
    "rank", "player", "position_group", "team", "nfl_team_2025", "roster_status",
    "salary_2026", "age",
    "production_value", "team_value", "age_value", "injury_value",
    "_pos_variant", "on_field_value", "_dv_variant",
]

PRETTY_NAMES = {
    "rank": "#",
    "player": "Player",
    "position_group": "Pos",
    "team": "League",
    "nfl_team_2025": "NFL",
    "roster_status": "Status",
    "salary_2026": "Salary",
    "age": "Age",
    "production_value": "Production",
    "team_value": "Team",
    "age_value": "Age (val)",
    "injury_value": "Injury",
    "_pos_variant": "Position",
    "on_field_value": "On-Field",
    "_dv_variant": "Dynasty",
}

STATUS_LABELS = {
    "active": "Active", "extension": "Ext.", "rookie": "Rookie",
    "practice_squad": "PSquad", "fa": "FA",
}

VALUE_COLS = ["production_value", "team_value", "age_value", "injury_value",
              "_pos_variant", "on_field_value", "_dv_variant"]


def render_variant(variant_name: str, variant_cfg: dict):
    weights = variant_cfg["weights"]
    submetrics = variant_cfg.get("submetrics", SUBMETRICS)
    scores = compute_position_scores(weights, submetrics)
    desc = VARIANT_DESCRIPTIONS[variant_name]

    df = pd.read_csv(MASTER_CSV)
    df["_pos_variant"] = df["position_group"].map(scores).fillna(50.0)
    # Adjust dynasty_value: subtract current position contribution, add variant's
    df["_dv_variant"] = (
        df["dynasty_value"]
        - POS_WEIGHT_IN_COMBINE * df["position_group"].map(CURRENT_POSITION_SCORES).fillna(50.0)
        + POS_WEIGHT_IN_COMBINE * df["_pos_variant"]
    )
    df = df.sort_values("_dv_variant", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    show = df[DISPLAY_COLS].copy()
    show["salary_2026"] = pd.to_numeric(show["salary_2026"], errors="coerce")
    show["age"] = pd.to_numeric(show["age"], errors="coerce").round(1)
    for c in VALUE_COLS:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(1)
    show["roster_status"] = show["roster_status"].map(STATUS_LABELS).fillna(show["roster_status"])
    show["team"] = show["team"].fillna("-")
    show["nfl_team_2025"] = show["nfl_team_2025"].fillna("-")
    show = show.rename(columns=PRETTY_NAMES)

    pretty_value_cols = [PRETTY_NAMES[c] for c in VALUE_COLS]
    pretty_age = PRETTY_NAMES["age"]

    def _fmt_salary(v):
        return "-" if pd.isna(v) else f"{int(v):d}"

    score_table_html = "<br>".join(
        f"&nbsp;&nbsp;<b>{p}</b>: {scores[p]:.1f}" for p in POSITION_ORDER
    )

    styled = (
        show.style
        .background_gradient(subset=pretty_value_cols, cmap="RdYlGn", vmin=0, vmax=100)
        .background_gradient(subset=[pretty_age], cmap="RdYlGn_r", vmin=22, vmax=33)
        .format({
            PRETTY_NAMES["salary_2026"]: _fmt_salary,
            pretty_age: "{:.1f}",
            **{c: "{:.1f}" for c in pretty_value_cols},
        })
        .set_caption(
            f"<h2 style='text-align:left;margin-bottom:6px'>Cross-position Dynasty Value ({variant_name})</h2>"
            f"<div style='text-align:left;font-size:13px;color:#444;margin-bottom:6px'>{desc}</div>"
            f"<div style='text-align:left;font-size:13px;color:#333;margin-bottom:14px'>"
            f"<b>Position scores under this variant:</b><br>{score_table_html}</div>"
        )
        .set_table_styles([
            {"selector": "th", "props": "background-color: #2c3e50; color: white; padding: 7px 10px; text-align: center; font-weight: 600;"},
            {"selector": "td", "props": "padding: 4px 8px; text-align: right; font-family: monospace; font-size: 12px;"},
            {"selector": "td.col0, td.col1, td.col2, td.col3, td.col4, td.col5", "props": "text-align: left;"},
            {"selector": "table", "props": "border-collapse: collapse; border: 1px solid #ccc;"},
            {"selector": "caption", "props": "caption-side: top;"},
            {"selector": f"th.col{DISPLAY_COLS.index('_dv_variant')}", "props": "background-color: #1a252f; border-left: 3px solid #f39c12;"},
            {"selector": f"th.col{DISPLAY_COLS.index('on_field_value')}", "props": "background-color: #34495e; border-left: 2px solid #95a5a6;"},
            {"selector": f"th.col{DISPLAY_COLS.index('_pos_variant')}", "props": "background-color: #34495e; border-left: 2px solid #95a5a6;"},
        ])
        .set_properties(**{"border": "1px solid #e0e0e0"})
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"cross_position_{variant_name}.html"
    html = styled.to_html()
    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-position ({variant_name})</title>
<style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 20px; background: #f9f9f9; }}</style>
</head><body>
{html}
</body></html>
"""
    out_path.write_text(full_html, encoding="utf-8")
    print(f"  {variant_name}: scores QB={scores['QB']:.1f} RB={scores['RB']:.1f} WR={scores['WR']:.1f} TE={scores['TE']:.1f} -> {out_path.name}")


def render_index(variant_names: list[str]):
    """Render a small index page linking to all variant outputs."""
    items = "\n".join(
        f'        <li><a href="cross_position_{v}.html"><b>{v}</b></a>: {VARIANT_DESCRIPTIONS[v]}</li>'
        for v in variant_names
    )
    index_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Position-weight variants</title>
<style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 24px; max-width: 900px; }} li {{ margin: 10px 0; }}</style>
</head><body>
<h2>Cross-position Dynasty Value -- Position-weight variants</h2>
<p>Each variant uses a different weighting of the 4 Position sub-metrics
(<b>M</b> Marginal Gap, <b>T</b> Total Impact, <b>D</b> Supply-Demand,
<b>S</b> Slot Count). The Position component contributes 5% to Dynasty Value
in the OFV-weighted combine, so the max swing per player is +/- 2.5 DV from
the variant choice.</p>
<ul>
{items}
</ul>
</body></html>
"""
    (OUT_DIR / "cross_position_index.html").write_text(index_html, encoding="utf-8")
    print(f"  index -> cross_position_index.html")


if __name__ == "__main__":
    print(f"Rendering {len(VARIANTS)} cross-position weight variants...")
    for name, cfg in VARIANTS.items():
        render_variant(name, cfg)
    render_index(list(VARIANTS.keys()))
    print(f"\nOpen figures/cross_position_index.html to compare variants.")
