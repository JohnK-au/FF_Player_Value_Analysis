"""Render cross-position HTML tables for the 4-preset cap-pricing workshop (Phase 5.5).

Each preset combines (alpha, baseline_tier, age_band) parameters for the multi-stage
pipeline (see src/models/pricing.py). Presets:

  A_conservative     -- alpha=1.0, baseline=deep, age_band=(0.90, 1.10)
  B_moderate         -- alpha=1.3, baseline=fa,   age_band=(0.85, 1.15)   [Recommended]
  C_aggressive_elite -- alpha=1.5, baseline=fa,   age_band=(0.85, 1.15)
  D_extreme_elite    -- alpha=1.7, baseline=fa,   age_band=(0.80, 1.20)

Each preset produces an HTML at figures/pricing_{preset}.html with all 490 priced
players ranked by surplus_2026 (most overpaid at top -> biggest bargain at bottom).

Usage:
    python -m src.viz.pricing_variants
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.models.pricing import PRESETS, SKILL_POSITIONS, build_pricing

OUT_DIR = Path(REPO_ROOT) / "figures"

DISPLAY_COLS = [
    "rank", "player", "position_group", "team", "nfl_team_2025", "roster_status",
    "age", "salary_2026", "years_2026",
    "dynasty_value", "replacement_dv", "above_baseline_dv", "age_mult",
    "fair_value_2026", "surplus_2026",
    "fair_value_dynasty", "surplus_dynasty",
]

PRETTY_NAMES = {
    "rank": "#", "player": "Player", "position_group": "Pos",
    "team": "League", "nfl_team_2025": "NFL", "roster_status": "Status",
    "age": "Age", "salary_2026": "Salary", "years_2026": "Yrs",
    "dynasty_value": "DV", "replacement_dv": "Repl DV",
    "above_baseline_dv": "Above BL", "age_mult": "Age Mult",
    "fair_value_2026": "Fair 2026", "surplus_2026": "Surplus 2026",
    "fair_value_dynasty": "Fair Dyn.", "surplus_dynasty": "Surplus Dyn.",
}

STATUS_LABELS = {
    "active": "Active", "extension": "Ext.", "rookie": "Rookie",
    "practice_squad": "PSquad", "fa": "FA",
}

QUALITY_COLS = ["dynasty_value", "replacement_dv", "above_baseline_dv"]
FAIR_COLS = ["fair_value_2026", "fair_value_dynasty"]
SURPLUS_COLS = ["surplus_2026", "surplus_dynasty"]


def render_preset(preset_key: str):
    p = PRESETS[preset_key]
    df = build_pricing(
        basis="dynasty_value",
        pool_method="empirical",
        pool_scale=p.get("pool_scale", 1.0),
        baseline_tier=p["baseline_tier"],
        baseline_offset=p.get("baseline_offset", 0.0),
        baseline_override=p.get("baseline_override"),
        alpha=p["alpha"],
        age_band=p["age_band"],
    )
    pool = float(df["pricing_pool"].iloc[0])
    rate = float(df["pricing_rate"].iloc[0])

    baselines = {
        pos: float(df[df["position_group"] == pos]["replacement_dv"].iloc[0])
        for pos in SKILL_POSITIONS if (df["position_group"] == pos).any()
    }

    # Sort by surplus_2026 desc (most overpaid at top -> biggest bargain at bottom)
    df = df.sort_values("surplus_2026", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    show = df[DISPLAY_COLS].copy()
    show["salary_2026"] = pd.to_numeric(show["salary_2026"], errors="coerce")
    show["years_2026"] = pd.to_numeric(show["years_2026"], errors="coerce")
    show["age"] = pd.to_numeric(show["age"], errors="coerce").round(1)
    for c in QUALITY_COLS + FAIR_COLS + SURPLUS_COLS + ["age_mult"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    show["roster_status"] = show["roster_status"].map(STATUS_LABELS).fillna(show["roster_status"])
    show["team"] = show["team"].fillna("-")
    show["nfl_team_2025"] = show["nfl_team_2025"].fillna("-")
    show = show.rename(columns=PRETTY_NAMES)

    pretty_quality = [PRETTY_NAMES[c] for c in QUALITY_COLS]
    pretty_fair    = [PRETTY_NAMES[c] for c in FAIR_COLS]
    pretty_surplus = [PRETTY_NAMES[c] for c in SURPLUS_COLS]

    # Symmetric surplus color scale
    surplus_max = max(abs(float(df["surplus_2026"].min())), abs(float(df["surplus_2026"].max())), 50)
    surplus_dyn_max = max(abs(float(df["surplus_dynasty"].min())), abs(float(df["surplus_dynasty"].max())), 100)
    fair_max = max(100, float(df["fair_value_2026"].max()))
    age_mult_lo, age_mult_hi = p["age_band"]

    def _fmt_int(v):
        return "-" if pd.isna(v) else f"{int(v):d}"

    styled = (
        show.style
        .background_gradient(subset=pretty_quality, cmap="RdYlGn", vmin=0, vmax=100)
        .background_gradient(subset=[PRETTY_NAMES["age_mult"]], cmap="RdYlGn", vmin=age_mult_lo, vmax=age_mult_hi)
        .background_gradient(subset=pretty_fair, cmap="Blues", vmin=0, vmax=fair_max)
        .background_gradient(subset=[PRETTY_NAMES["surplus_2026"]], cmap="RdYlGn_r", vmin=-surplus_max, vmax=surplus_max)
        .background_gradient(subset=[PRETTY_NAMES["surplus_dynasty"]], cmap="RdYlGn_r", vmin=-surplus_dyn_max, vmax=surplus_dyn_max)
        .background_gradient(subset=[PRETTY_NAMES["age"]], cmap="RdYlGn_r", vmin=22, vmax=33)
        .format({
            PRETTY_NAMES["salary_2026"]: _fmt_int,
            PRETTY_NAMES["years_2026"]: _fmt_int,
            PRETTY_NAMES["age"]: "{:.1f}",
            PRETTY_NAMES["age_mult"]: "{:.3f}",
            **{c: "{:.1f}" for c in pretty_quality + pretty_fair + pretty_surplus},
        })
        .set_caption(
            f"<h2 style='text-align:left;margin-bottom:6px'>Cap pricing preset: {p['label']}</h2>"
            f"<div style='text-align:left;font-size:13px;color:#444;margin-bottom:10px'>{p['desc']}</div>"
            f"<div style='font-size:12px;color:#666;margin-bottom:6px'>"
            f"Params: <b>alpha={p['alpha']}</b>, "
            f"pool_scale=<b>{p.get('pool_scale', 1.0):.2f}</b>, "
            f"baselines <b>{'(user override)' if p.get('baseline_override') else f'tier={p.get(chr(34)+chr(98)+chr(97)+chr(115)+chr(101)+chr(108)+chr(105)+chr(110)+chr(101)+chr(95)+chr(116)+chr(105)+chr(101)+chr(114)+chr(34), chr(102)+chr(97))}, offset=-{p.get(chr(34)+chr(98)+chr(97)+chr(115)+chr(101)+chr(108)+chr(105)+chr(110)+chr(101)+chr(95)+chr(111)+chr(102)+chr(102)+chr(115)+chr(101)+chr(116)+chr(34), 0.0):.0f}'}</b>, "
            f"age_band=<b>[{age_mult_lo:.2f}, {age_mult_hi:.2f}]</b><br>"
            f"Pool: <b>{pool:.0f}</b> cap units (empirical rostered skill spend x pool_scale); "
            f"Rate: <b>{rate:.4f}</b> cap units per scarcity-value point"
            f"</div>"
            f"<div style='font-size:12px;color:#666;margin-bottom:14px'>"
            f"Per-position replacement DV baselines: " +
            "&nbsp;&nbsp;".join(f"<b>{pos}</b>: {v:.1f}" for pos, v in baselines.items()) +
            f"<br>Sorted by surplus_2026 desc (most overpaid at top; biggest bargains at bottom)"
            f"</div>"
        )
        .set_table_styles([
            {"selector": "th", "props": "background-color: #2c3e50; color: white; padding: 7px 10px; text-align: center; font-weight: 600;"},
            {"selector": "td", "props": "padding: 4px 8px; text-align: right; font-family: monospace; font-size: 12px;"},
            {"selector": "td.col0, td.col1, td.col2, td.col3, td.col4, td.col5", "props": "text-align: left;"},
            {"selector": "table", "props": "border-collapse: collapse; border: 1px solid #ccc;"},
            {"selector": "caption", "props": "caption-side: top;"},
            {"selector": f"th.col{DISPLAY_COLS.index('surplus_2026')}", "props": "background-color: #1a252f; border-left: 3px solid #f39c12;"},
            {"selector": f"th.col{DISPLAY_COLS.index('fair_value_2026')}", "props": "background-color: #34495e; border-left: 2px solid #95a5a6;"},
            {"selector": f"th.col{DISPLAY_COLS.index('surplus_dynasty')}", "props": "border-left: 2px solid #95a5a6;"},
        ])
        .set_properties(**{"border": "1px solid #e0e0e0"})
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pricing_{preset_key}.html"
    full_html = (
        '<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Pricing preset ({p["label"]})</title>'
        '<style>body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 20px; background: #f9f9f9; }</style>'
        '</head><body>\n'
        f"{styled.to_html()}\n"
        '</body></html>\n'
    )
    out_path.write_text(full_html, encoding="utf-8")

    ros = df[df["roster_status"] != "fa"]
    top5_mean = ros.nlargest(5, "fair_value_2026")["fair_value_2026"].mean()
    top1 = ros["fair_value_2026"].max()
    median = ros["fair_value_2026"].median()
    print(f"  {preset_key:20s} alpha={p['alpha']:.2f} tier={p['baseline_tier']:6s}  "
          f"top1={top1:6.1f}  top5_mean={top5_mean:6.1f}  median={median:5.1f}  "
          f"mean_surplus={ros['surplus_2026'].mean():+5.1f}  -> {out_path.name}")
    return df


def render_index(preset_keys: list[str]):
    items = "\n".join(
        f'        <li><a href="pricing_{k}.html"><b>{PRESETS[k]["label"]}</b></a><br>'
        f'           <span style="color:#555;font-size:13px">{PRESETS[k]["desc"]}</span></li>'
        for k in preset_keys
    )
    index_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Cap pricing preset workshop -- Phase 5.5</title>
<style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 24px; max-width: 1100px; }} li {{ margin: 14px 0; }}</style>
</head><body>
<h2>Cap pricing preset workshop -- Phase 5.5</h2>
<p>4 preset parameter combinations for the multi-stage pricing pipeline:
<ol>
<li>Per-position <b>replacement baseline</b> (mid-tier collapse to fair=0 when replaceable from FA)</li>
<li><b>Non-linear elite premium</b> (alpha &gt; 1 concentrates pool on top producers)</li>
<li><b>Age multiplier</b> ON TOP of DV's 20% Age weight</li>
<li>Multi-year contract <b>age decay</b> (year-by-year age projection)</li>
</ol>
Basis: <b>dynasty_value</b>; pool: <b>empirical</b> (sum of rostered skill spend, self-balancing).
<br><br>
Sign convention: <b style="color:#c0392b">positive surplus = overpaid</b>;
<b style="color:#27ae60">negative surplus = bargain</b>. Sorted by surplus_2026 desc.</p>
<ul>
{items}
</ul>
</body></html>
"""
    (OUT_DIR / "pricing_index.html").write_text(index_html, encoding="utf-8")
    print(f"  index -> pricing_index.html")


if __name__ == "__main__":
    print(f"Rendering {len(PRESETS)} pricing presets...")
    for key in PRESETS.keys():
        render_preset(key)
    render_index(list(PRESETS.keys()))
    print(f"\nOpen figures/pricing_index.html to compare presets.")
