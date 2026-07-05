"""V2 Player Card -- per-player deep dive.

Sections:
  Header:                      player, position, teams, roster status, age, salary, years
  Headline metrics (4 cards):  OFV, Dynasty Value, Fair 2026 (+surplus), Fair Dynasty
  6-component quality bars:    each [0, 100] with 3 reference lines per position:
                                 league_avg  |  top-3 avg  |  replacement DV
  Pricing decomposition:       DV -> above_baseline -> scarcity (^alpha) x rate
                                 x age_mult = fair_2026
  Multi-year projection:       year-by-year fair over remaining contract with age decay
  Peer comparison:              top 5 at same position + this player's rank

Prototyping stage -- keep iterating on layout/content based on user feedback.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for p in (_THIS.parents[1], _THIS.parents[3]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from _lib import (  # noqa: E402
    COMPONENT_COLS, POS_ORDER,
    fmt_float_or_dash, fmt_int_or_dash, load_master,
)
from src.models.pricing import build_pricing  # noqa: E402

st.set_page_config(page_title="Player Card (V2)", layout="wide")
st.title("Player Card")

master = load_master()

# Player selector -- default to Bijan if present
players = sorted(master["player"].dropna().unique().tolist())
default_idx = players.index("Bijan Robinson") if "Bijan Robinson" in players else 0
choice = st.selectbox("Search player", players, index=default_idx)

p = master[master["player"] == choice].iloc[0]
pos = p["position_group"]
age = p.get("age")
salary = p.get("salary_2026")
years = p.get("years_2026")


def _fmt_salary(x):
    return fmt_int_or_dash(x)


# --- Header ------------------------------------------------------------------
h1, h2, h3, h4, h5 = st.columns([2.4, 1, 1, 1, 1])
h1.subheader(f"{p['player']}  ·  {pos}  ·  {p.get('nfl_team_2025', '-')}  ·  league team **{p.get('team', '-') or 'FA'}**")
h2.metric("Age", fmt_float_or_dash(age, n=1))
h3.metric("Salary 2026", _fmt_salary(salary))
h4.metric("Years remaining", _fmt_salary(years))
h5.metric("Roster status", str(p.get("roster_status", "-")))

# --- Headline metrics (4 cards) ---------------------------------------------
v1, v2, v3, v4 = st.columns(4)

# Delta = -surplus so bargain (negative surplus) shows as a positive (green) delta.
surplus_2026 = p.get("surplus_2026")
surplus_dyn = p.get("surplus_dynasty")

v1.metric(
    "On-Field Value",
    fmt_float_or_dash(p.get("on_field_value"), n=1),
    help="Production x Team multiplier. What the player is producing given their environment. [0, 100].",
)
v2.metric(
    "Dynasty Value",
    fmt_float_or_dash(p.get("dynasty_value"), n=1),
    help="Full 6-component combine (Production + Team + Age + Injury + Position + Intangibles). [0, 100].",
)
v3.metric(
    "Fair Value 2026",
    _fmt_salary(p.get("fair_value_2026")),
    delta=f"{-surplus_2026:+.0f}" if pd.notna(surplus_2026) else None,
    help="Single-season cap-unit fair value. Delta = -surplus (positive delta = bargain, green).",
)
v4.metric(
    "Fair Value Dynasty",
    _fmt_salary(p.get("fair_value_dynasty")),
    delta=f"{-surplus_dyn:+.0f}" if pd.notna(surplus_dyn) else None,
    help="Sum of fair over remaining contract years with per-year age decay.",
)

# --- 6-component quality breakdown ------------------------------------------
st.subheader("Quality components (V2 framework)")

pos_peers = master[master["position_group"] == pos].copy()
league_peers = master[master["position_group"].isin(POS_ORDER)].copy()

# References per component: league avg (all skill), position avg, top-3 avg
top3_at_pos = pos_peers.nlargest(3, "dynasty_value")
replacement_dv_for_pos = float(p.get("replacement_dv") or 0)

comp_data = []
for c in COMPONENT_COLS:
    label = c.replace("_value", "").replace("_", " ").title()
    val = p.get(c)
    league_avg = float(league_peers[c].mean()) if c in league_peers.columns else None
    pos_avg = float(pos_peers[c].mean()) if c in pos_peers.columns else None
    top3_avg = float(top3_at_pos[c].mean()) if c in top3_at_pos.columns and len(top3_at_pos) else None
    comp_data.append({
        "component": label,
        "player": float(val) if pd.notna(val) else None,
        "position_avg": pos_avg,
        "top3_avg": top3_avg,
        "league_avg": league_avg,
    })
comp_df = pd.DataFrame(comp_data)

# Horizontal bar chart: player values with reference lines/dots
fig = go.Figure()
fig.add_trace(go.Bar(
    y=comp_df["component"],
    x=comp_df["player"],
    orientation="h",
    marker=dict(color="#2E86AB", line=dict(color="#1a5378", width=1)),
    text=[f"{v:.1f}" if pd.notna(v) else "-" for v in comp_df["player"]],
    textposition="outside",
    name=p["player"],
))
# Overlay reference markers as scatter points
for ref_col, symbol, color, ref_name in [
    ("position_avg", "line-ns", "#95a5a6", f"{pos} avg"),
    ("top3_avg",     "diamond", "#f39c12", f"top-3 {pos} avg"),
    ("league_avg",   "circle",  "#7f8c8d", "league avg (all skill)"),
]:
    fig.add_trace(go.Scatter(
        y=comp_df["component"],
        x=comp_df[ref_col],
        mode="markers",
        marker=dict(symbol=symbol, size=10, color=color, line=dict(color="black", width=1)),
        name=ref_name,
    ))

# Replacement-DV vertical line (worth-rostering threshold)
if replacement_dv_for_pos > 0:
    fig.add_vline(
        x=replacement_dv_for_pos, line=dict(color="#c0392b", dash="dash", width=2),
        annotation_text=f"replacement DV = {replacement_dv_for_pos:.1f}",
        annotation_position="top right",
    )

fig.update_layout(
    height=420,
    xaxis=dict(range=[0, 105], title="Score [0, 100]"),
    yaxis=dict(autorange="reversed"),
    margin=dict(t=30, b=40, l=40, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Diamonds = **top-3 {pos} avg** (elite tier). Vertical dashes = **{pos} avg**. "
    f"Circles = **league avg** across all skill positions. "
    f"Red dashed line at DV = {replacement_dv_for_pos:.1f} = **replacement threshold** "
    f"(players at/below get fair_value = 0)."
)

# --- Pricing decomposition (waterfall-ish) ----------------------------------
st.subheader("Pricing decomposition -- how DV becomes fair_value_2026")

pd_cols = st.columns(6)
pd_cols[0].metric("DV", fmt_float_or_dash(p.get("dynasty_value"), n=1))
pd_cols[1].metric("- Replacement", fmt_float_or_dash(p.get("replacement_dv"), n=1))
pd_cols[2].metric("= Above baseline", fmt_float_or_dash(p.get("above_baseline_dv"), n=1))
pd_cols[3].metric(
    f"^ alpha={float(master['pricing_alpha'].iloc[0]):.2f}",
    fmt_float_or_dash(p.get("scarcity_value"), n=2),
    help="Non-linear elite premium: scarcity_value = above_baseline_dv ^ alpha",
)
pd_cols[4].metric(
    f"× age_mult ({p.get('age_mult', 1.0):.3f})",
    fmt_float_or_dash(p.get("base_fair"), n=1),
    help="base_fair = scarcity_value × rate; year-1 fair applies the age multiplier.",
)
pd_cols[5].metric("= Fair 2026", _fmt_salary(p.get("fair_value_2026")))

rate = float(master["pricing_rate"].iloc[0])
alpha = float(master["pricing_alpha"].iloc[0])
lo, hi = float(master["pricing_age_band_lo"].iloc[0]), float(master["pricing_age_band_hi"].iloc[0])
pool_scale = float(master["pricing_pool_scale"].iloc[0])
st.caption(
    f"Pipeline: DV - replacement_dv = above_baseline_dv; scarcity = above^{alpha:.2f}; "
    f"base_fair = scarcity × rate ({rate:.4f}); fair_2026 = base_fair × age_mult (band [{lo:.2f}, {hi:.2f}], pool_scale {pool_scale:.2f})."
)

# --- Multi-year contract projection -----------------------------------------
if pd.notna(years) and int(years) > 1:
    st.subheader(f"Multi-year projection ({int(years)} years, per-year age decay)")

    # Rerun pricing with current preset to get per-year fair breakdown for THIS player
    # by projecting age forward. Simpler: use age_curve directly.
    from src.models.components.age import AGE_PARAMS, _logistic_age_value

    base_fair = float(p.get("base_fair") or 0)
    age_start = float(age) if pd.notna(age) else None
    per_year = []
    if age_start is not None and base_fair > 0 and pos in AGE_PARAMS:
        params = AGE_PARAMS[pos]
        for y in range(1, int(years) + 1):
            proj_age = age_start + (y - 1)
            av = _logistic_age_value(proj_age, params["center"], params["steepness"])
            mult = lo + (av / 100.0) * (hi - lo)
            per_year.append({"season": 2025 + y, "projected_age": proj_age,
                             "age_mult": mult, "fair": base_fair * mult})
    if per_year:
        py_df = pd.DataFrame(per_year)
        fig_y = go.Figure()
        fig_y.add_trace(go.Bar(
            x=py_df["season"].astype(str),
            y=py_df["fair"],
            marker=dict(color="#27ae60"),
            text=[f"{v:.0f}" for v in py_df["fair"]],
            textposition="outside",
        ))
        fig_y.update_layout(
            height=280, yaxis_title="Fair value (cap units)",
            xaxis_title="Season", margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_y, use_container_width=True)
        total = float(py_df["fair"].sum())
        st.caption(
            f"Age at start = {age_start:.1f}; {pos} curve center = {AGE_PARAMS[pos]['center']}, "
            f"steepness = {AGE_PARAMS[pos]['steepness']}. Total fair across "
            f"{int(years)} years = **{total:.0f}** cap units; "
            f"dynasty_total_salary = {p.get('dynasty_total_salary', 0):.0f}."
        )

# --- Position market context ------------------------------------------------
st.subheader(f"{pos} market context")

# Actual salary references (rostered only)
rostered_pos = pos_peers[pos_peers["roster_status"] != "fa"].copy()
top3_paid = rostered_pos.nlargest(3, "salary_2026") if len(rostered_pos) else pd.DataFrame()

mc = st.columns(4)
if len(top3_paid):
    top3_avg_paid = float(top3_paid["salary_2026"].mean())
    mc[0].metric(f"Top-3 paid {pos} avg salary", f"{top3_avg_paid:.0f}",
                 help=", ".join([f"{r['player']} ({int(r['salary_2026'])})" for _, r in top3_paid.iterrows()]))
if len(rostered_pos):
    mc[1].metric(f"{pos} rostered avg salary", f"{rostered_pos['salary_2026'].mean():.0f}",
                 help=f"n={len(rostered_pos)} rostered")
if len(top3_at_pos):
    top3_fair_avg = float(top3_at_pos["fair_value_2026"].mean())
    mc[2].metric(f"Top-3 {pos} by DV: fair avg", f"{top3_fair_avg:.0f}",
                 help=", ".join([f"{r['player']} (fair {r['fair_value_2026']:.0f})" for _, r in top3_at_pos.iterrows()]))
mc[3].metric(f"Replacement fair", "0",
             help=f"Players at/below replacement_dv={replacement_dv_for_pos:.1f} get fair_value=0. "
                  "This is the 'worth rostering' break line.")

# --- Peer comparison table --------------------------------------------------
st.subheader(f"Top 10 {pos} by Dynasty Value")

peer_cols = ["player", "team", "roster_status", "age", "salary_2026", "years_2026",
             "dynasty_value", "fair_value_2026", "surplus_2026", "fair_value_dynasty", "surplus_dynasty"]
peer_cols = [c for c in peer_cols if c in pos_peers.columns]
top10 = pos_peers.nlargest(10, "dynasty_value")[peer_cols].reset_index(drop=True)
top10.index = top10.index + 1

# Highlight the current player's row
def _highlight_current(row):
    is_current = row["player"] == choice
    return ["background-color: rgba(46, 160, 67, 0.20); font-weight: 600" if is_current else "" for _ in row]

styled = top10.style.apply(_highlight_current, axis=1).format({
    "age": "{:.1f}", "salary_2026": "{:.0f}", "years_2026": "{:.0f}",
    "dynasty_value": "{:.1f}", "fair_value_2026": "{:.0f}",
    "surplus_2026": "{:+.0f}", "fair_value_dynasty": "{:.0f}", "surplus_dynasty": "{:+.0f}",
})
st.dataframe(styled, use_container_width=True, height=min(420, 60 + 40 * len(top10)))

# Rank line
rank_pos = int((pos_peers["dynasty_value"] > float(p["dynasty_value"])).sum()) + 1 if pd.notna(p.get("dynasty_value")) else None
st.caption(
    f"{choice} ranks #{rank_pos} of {len(pos_peers)} {pos}s in the priced universe by Dynasty Value. "
    f"Prices come from the current pricing preset (alpha={alpha:.2f}, pool_scale={pool_scale:.2f}, "
    f"replacement_dv[{pos}]={replacement_dv_for_pos:.1f})."
)
