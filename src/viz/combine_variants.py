"""Render cross-position HTML tables for multiple Dynasty-Value combine variants.

Phase 5 workshop: compares 5 candidate combine methods for folding the 5
post-OFV components (OFV / Age / Injury / Position / Intangibles) into the
final Dynasty Value headline.

Variants:
  v1_current             -- weighted sum, current defaults (the locked baseline)
  v2_ofv_age_boost       -- weighted sum, user-specified rebalance (OFV 0.65, Age 0.30, Injury 0.10)
  v3_hybrid              -- weighted base (OFV+Age) x multiplicative overlays (Injury, Position, Intangibles)
  v4_multiplicative      -- OFV base x all 4 off-field as bounded multipliers
  v5_position_boosted    -- weighted sum, Position raised to 0.10 (give v2 Position more voice)

Each variant produces a single HTML at figures/combine_{variant}.html with all
490 priced players ranked by the variant's dynasty_value.

Usage:
    python -m src.viz.combine_variants
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT

MASTER_CSV = Path("data/processed/player_value_v2_2026.csv")
OUT_DIR = Path(REPO_ROOT) / "figures"

# Multiplier band constants for hybrid + multiplicative variants.
# Format: (low, high) -- score 0 -> low, score 100 -> high.
BAND_AGE         = (0.85, 1.15)  # biggest off-field band; age curve has real teeth
BAND_INJURY      = (0.90, 1.10)  # moderate; durability matters but not dominant
BAND_POSITION    = (0.95, 1.05)  # light; v2 Position is one constant per position
BAND_INTANGIBLES = (0.95, 1.05)  # light; stub today, room to grow when real

# Hybrid base weights (OFV + Age weighted sum, sum=1)
HYBRID_BASE = {"on_field_value": 0.70, "age_value": 0.30}


def _band_multiplier(score: pd.Series, band: tuple[float, float]) -> pd.Series:
    """Linear interpolation: score=0 -> band[0], score=100 -> band[1]."""
    lo, hi = band
    return lo + (score.clip(0, 100) / 100.0) * (hi - lo)


# ---- Variant combine functions ----

def v1_current(df: pd.DataFrame) -> pd.Series:
    """Current locked default: OFV 0.55 / Age 0.20 / Injury 0.15 / Position 0.05 / Intang 0.05."""
    return (
        0.55 * df["on_field_value"]
        + 0.20 * df["age_value"]
        + 0.15 * df["injury_value"]
        + 0.05 * df["position_value"]
        + 0.05 * df["intangibles_value"]
    )


def v2_ofv_age_boost(df: pd.DataFrame) -> pd.Series:
    """User-specified: OFV 0.65, Age 0.30, Injury 0.10, Position 0, Intang 0.
    Raw sum = 1.05; auto-normalized so weights sum to 1."""
    weights = {
        "on_field_value": 0.65, "age_value": 0.30, "injury_value": 0.10,
        "position_value": 0.0, "intangibles_value": 0.0,
    }
    total = sum(weights.values())
    return sum(df[col] * (w / total) for col, w in weights.items() if w > 0)


def v3_hybrid(df: pd.DataFrame) -> pd.Series:
    """Weighted base (OFV+Age) x multiplicative overlays (Injury, Position, Intangibles)."""
    base = sum(df[col] * w for col, w in HYBRID_BASE.items())
    return (
        base
        * _band_multiplier(df["injury_value"], BAND_INJURY)
        * _band_multiplier(df["position_value"], BAND_POSITION)
        * _band_multiplier(df["intangibles_value"], BAND_INTANGIBLES)
    )


def v4_multiplicative(df: pd.DataFrame) -> pd.Series:
    """OFV base x all 4 off-field as bounded multipliers. Range ~[0, 140]."""
    return (
        df["on_field_value"]
        * _band_multiplier(df["age_value"], BAND_AGE)
        * _band_multiplier(df["injury_value"], BAND_INJURY)
        * _band_multiplier(df["position_value"], BAND_POSITION)
        * _band_multiplier(df["intangibles_value"], BAND_INTANGIBLES)
    )


def v5_position_boosted(df: pd.DataFrame) -> pd.Series:
    """Weighted sum, Position raised to 0.10 (twice current). My rec: give v2 Position more voice."""
    return (
        0.50 * df["on_field_value"]
        + 0.20 * df["age_value"]
        + 0.15 * df["injury_value"]
        + 0.10 * df["position_value"]
        + 0.05 * df["intangibles_value"]
    )


VARIANTS = {
    "v1_current":          v1_current,
    "v2_ofv_age_boost":    v2_ofv_age_boost,
    "v3_hybrid":           v3_hybrid,
    "v4_multiplicative":   v4_multiplicative,
    "v5_position_boosted": v5_position_boosted,
}

VARIANT_DESCRIPTIONS = {
    "v1_current": (
        "<b>Weighted sum, current locked defaults.</b><br>"
        "0.55 OFV + 0.20 Age + 0.15 Injury + 0.05 Position + 0.05 Intangibles. "
        "Range [0, 100]."
    ),
    "v2_ofv_age_boost": (
        "<b>Weighted sum, user-specified rebalance.</b><br>"
        "0.65 OFV + 0.30 Age + 0.10 Injury + 0.0 Position + 0.0 Intangibles (raw sum 1.05). "
        "Auto-normalized to 0.619 / 0.286 / 0.095 / 0 / 0 so weights sum to 1. "
        "Drops Position + Intangibles entirely; doubles down on on-field + age."
    ),
    "v3_hybrid": (
        "<b>Hybrid combine: weighted base x multiplicative overlays.</b><br>"
        "base = 0.70*OFV + 0.30*Age (range [0, 100])<br>"
        "dynasty = base x injury_mult [0.90, 1.10] x position_mult [0.95, 1.05] x intangibles_mult [0.95, 1.05].<br>"
        "Big rocks (OFV+Age) get explicit weights; off-field bumps multiply on top. Range ~[0, 121]."
    ),
    "v4_multiplicative": (
        "<b>Full multiplicative: OFV base x all 4 off-field multipliers.</b><br>"
        "base = OFV<br>"
        "dynasty = OFV x age_mult [0.85, 1.15] x injury_mult [0.90, 1.10] x position_mult [0.95, 1.05] x intangibles_mult [0.95, 1.05].<br>"
        "Elite OFV players get amplified by good off-field; weak OFV gets penalized hard. Range ~[0, 140]."
    ),
    "v5_position_boosted": (
        "<b>Weighted sum, Position raised to 0.10.</b><br>"
        "0.50 OFV + 0.20 Age + 0.15 Injury + 0.10 Position + 0.05 Intangibles. "
        "Recommendation: now that v2 Position is empirically derived (RB=100, WR=93, TE=9, QB=0), "
        "double its weight from 0.05 -> 0.10 to give it more voice in the final ranking. "
        "Range [0, 100]."
    ),
}

# ---- Rendering ----

DISPLAY_COLS = [
    "rank", "player", "position_group", "team", "nfl_team_2025", "roster_status",
    "salary_2026", "age",
    "on_field_value", "age_value", "injury_value", "position_value", "intangibles_value",
    "_dv_variant",
]

PRETTY_NAMES = {
    "rank": "#", "player": "Player", "position_group": "Pos",
    "team": "League", "nfl_team_2025": "NFL", "roster_status": "Status",
    "salary_2026": "Salary", "age": "Age",
    "on_field_value": "OFV", "age_value": "Age (val)", "injury_value": "Injury",
    "position_value": "Position", "intangibles_value": "Intang",
    "_dv_variant": "Dynasty",
}

STATUS_LABELS = {
    "active": "Active", "extension": "Ext.", "rookie": "Rookie",
    "practice_squad": "PSquad", "fa": "FA",
}

VALUE_COLS = ["on_field_value", "age_value", "injury_value", "position_value",
              "intangibles_value", "_dv_variant"]


def render_variant(variant_name: str, fn):
    df = pd.read_csv(MASTER_CSV)
    df["_dv_variant"] = fn(df)
    df = df.sort_values("_dv_variant", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Compute variant range for the caption + color-gradient anchors
    dv_min, dv_max = float(df["_dv_variant"].min()), float(df["_dv_variant"].max())

    show = df[DISPLAY_COLS].copy()
    show["salary_2026"] = pd.to_numeric(show["salary_2026"], errors="coerce")
    show["age"] = pd.to_numeric(show["age"], errors="coerce").round(1)
    for c in ("on_field_value", "age_value", "injury_value", "position_value",
              "intangibles_value", "_dv_variant"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(1)
    show["roster_status"] = show["roster_status"].map(STATUS_LABELS).fillna(show["roster_status"])
    show["team"] = show["team"].fillna("-")
    show["nfl_team_2025"] = show["nfl_team_2025"].fillna("-")
    show = show.rename(columns=PRETTY_NAMES)

    pretty_value_cols_100 = [PRETTY_NAMES[c] for c in
                              ("on_field_value", "age_value", "injury_value",
                               "position_value", "intangibles_value")]
    pretty_dv = PRETTY_NAMES["_dv_variant"]
    pretty_age = PRETTY_NAMES["age"]

    def _fmt_salary(v):
        return "-" if pd.isna(v) else f"{int(v):d}"

    desc = VARIANT_DESCRIPTIONS[variant_name]
    range_html = (f"<div style='font-size:12px;color:#666;margin-top:6px'>"
                  f"Dynasty value range in this variant: <b>{dv_min:.1f}</b> to <b>{dv_max:.1f}</b></div>")

    styled = (
        show.style
        .background_gradient(subset=pretty_value_cols_100, cmap="RdYlGn", vmin=0, vmax=100)
        .background_gradient(subset=[pretty_dv], cmap="RdYlGn", vmin=dv_min, vmax=dv_max)
        .background_gradient(subset=[pretty_age], cmap="RdYlGn_r", vmin=22, vmax=33)
        .format({
            PRETTY_NAMES["salary_2026"]: _fmt_salary,
            pretty_age: "{:.1f}",
            **{c: "{:.1f}" for c in pretty_value_cols_100 + [pretty_dv]},
        })
        .set_caption(
            f"<h2 style='text-align:left;margin-bottom:6px'>Dynasty Value combine variant ({variant_name})</h2>"
            f"<div style='text-align:left;font-size:13px;color:#444;margin-bottom:6px'>{desc}</div>"
            f"{range_html}"
        )
        .set_table_styles([
            {"selector": "th", "props": "background-color: #2c3e50; color: white; padding: 7px 10px; text-align: center; font-weight: 600;"},
            {"selector": "td", "props": "padding: 4px 8px; text-align: right; font-family: monospace; font-size: 12px;"},
            {"selector": "td.col0, td.col1, td.col2, td.col3, td.col4, td.col5", "props": "text-align: left;"},
            {"selector": "table", "props": "border-collapse: collapse; border: 1px solid #ccc;"},
            {"selector": "caption", "props": "caption-side: top;"},
            {"selector": f"th.col{DISPLAY_COLS.index('_dv_variant')}", "props": "background-color: #1a252f; border-left: 3px solid #f39c12;"},
            {"selector": f"th.col{DISPLAY_COLS.index('on_field_value')}", "props": "background-color: #34495e; border-left: 2px solid #95a5a6;"},
        ])
        .set_properties(**{"border": "1px solid #e0e0e0"})
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"combine_{variant_name}.html"
    full_html = (
        '<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Combine variant ({variant_name})</title>'
        '<style>body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 20px; background: #f9f9f9; }</style>'
        '</head><body>\n'
        f"{styled.to_html()}\n"
        '</body></html>\n'
    )
    out_path.write_text(full_html, encoding="utf-8")
    print(f"  {variant_name}: dv range [{dv_min:.1f}, {dv_max:.1f}] -> {out_path.name}")
    return df


def render_index(variant_names: list[str]):
    items = "\n".join(
        f'        <li><a href="combine_{v}.html"><b>{v}</b></a><br>'
        f'           <span style="color:#555;font-size:13px">{VARIANT_DESCRIPTIONS[v]}</span></li>'
        for v in variant_names
    )
    index_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Dynasty Value combine variants</title>
<style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 24px; max-width: 1100px; }} li {{ margin: 14px 0; }}</style>
</head><body>
<h2>Dynasty Value combine variants -- Phase 5 workshop</h2>
<p>Each variant uses a different method for folding the 5 post-OFV components
(<b>OFV</b>, <b>Age</b>, <b>Injury</b>, <b>Position</b>, <b>Intangibles</b>)
into the headline Dynasty Value. Range varies by variant -- compare RANKS,
not absolute values.</p>
<ul>
{items}
</ul>
<p style="color:#777;font-size:12px;margin-top:24px">
Multiplier bands used by hybrid + multiplicative variants:
Age [0.85, 1.15], Injury [0.90, 1.10], Position [0.95, 1.05], Intangibles [0.95, 1.05].
</p>
</body></html>
"""
    (OUT_DIR / "combine_index.html").write_text(index_html, encoding="utf-8")
    print(f"  index -> combine_index.html")


if __name__ == "__main__":
    print(f"Rendering {len(VARIANTS)} Dynasty-Value combine variants...")
    top5_per_variant = {}
    for name, fn in VARIANTS.items():
        df_sorted = render_variant(name, fn)
        top5_per_variant[name] = df_sorted.head(5)[["player", "position_group", "_dv_variant"]].values.tolist()
    render_index(list(VARIANTS.keys()))
    print(f"\nOpen figures/combine_index.html to compare variants.\n")
    print("Top 5 per variant (player, pos, dv):")
    for name, top5 in top5_per_variant.items():
        print(f"  {name}:")
        for player, pos, dv in top5:
            print(f"    {pos:3s}  {dv:6.2f}  {player}")
