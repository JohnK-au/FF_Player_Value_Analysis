"""Player Value Card — search a player, see their full value breakdown.

Header: salary / age / years remaining + 4 headline metrics (current value, dynasty,
year_type, projected PPG). A year-by-year projected-PPG chart over the remaining contract.
Context block: usage_trend / results_trend / qb_context / PPG baseline. Both lenses panel
compares market-fit vs production-anchored.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Path setup mirrors Home.py for direct page-link launches.
_THIS = Path(__file__).resolve()
for p in (_THIS.parents[1], _THIS.parents[3]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from _lib import POS_ORDER, load_master  # noqa: E402

st.set_page_config(page_title="Player Card", layout="wide")
st.title("Player Value Card")

master = load_master()
players = sorted(master["player"].dropna().unique().tolist())
default_idx = players.index("Bijan Robinson") if "Bijan Robinson" in players else 0
choice = st.selectbox("Search player", players, index=default_idx)

p = master[master["player"] == choice].iloc[0]


def _fmt(x, kind="f", n=1):
    if pd.isna(x):
        return "—"
    if kind == "d":
        return f"{int(x)}"
    return f"{float(x):.{n}f}"


# --- Header -------------------------------------------------------------------
h1, h2, h3, h4 = st.columns([2.4, 1, 1, 1])
h1.subheader(f"{p['player']}  ·  {p['position_group']}  ·  team {p['team']}")
h2.metric("Age", _fmt(p.get("age"), n=1))
h3.metric("Salary 2026", _fmt(p.get("salary_2026"), kind="d"))
h4.metric("Years remaining", _fmt(p.get("years_2026"), kind="d"))

# --- Headline value -----------------------------------------------------------
v1, v2, v3, v4 = st.columns(4)
# Delta = -surplus, so bargain (neg surplus) shows as a positive (green) delta.
surplus_2026 = p.get("surplus_2026")
dynasty_surplus = p.get("dynasty_surplus")
v1.metric("Value 2026",
          _fmt(p.get("value_2026"), kind="d"),
          delta=f"{-surplus_2026:+.0f}" if pd.notna(surplus_2026) else None,
          help="Production-anchored fair value off the projection model; delta = -surplus.")
v2.metric("Dynasty value (discounted)",
          _fmt(p.get("dynasty_value"), kind="d"),
          delta=f"{-dynasty_surplus:+.0f}" if pd.notna(dynasty_surplus) else None,
          help="Multi-year discounted sum across remaining contract years.")
v3.metric("Projected 2026 PPG", _fmt(p.get("projected_ppg"), n=1))
v4.metric("Year type", str(p.get("year_type", "—")),
          help="Glance flag off PPG vs the player's own trailing baseline.")

# --- Multi-year projection chart ---------------------------------------------
st.subheader("Projected PPG over the remaining contract")
if (
    pd.notna(p.get("projected_ppg"))
    and pd.notna(p.get("age"))
    and pd.notna(p.get("years_2026"))
):
    from src.models.projection import positional_age_curves, project_multiyear

    curves = positional_age_curves()
    n = int(p["years_2026"])
    years = [2026 + i for i in range(n)]
    per_year = project_multiyear(
        float(p["projected_ppg"]), p["position_group"], float(p["age"]), n, curves
    )
    fig = go.Figure(
        data=[go.Bar(x=[str(y) for y in years], y=per_year, name="Projected PPG")]
    )
    fig.update_layout(height=320, xaxis_title="Season", yaxis_title="Projected PPG",
                      margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{p['position_group']} age curve applied year-over-year. "
        f"Total raw projected production: {sum(per_year):.1f} PPG across {n} year(s)."
    )
else:
    st.info("No projection available (rookie or missing features).")

# --- Context block ------------------------------------------------------------
st.subheader("Context — how this season compares to the player's own history")
ctx_cols = st.columns(4)
ctx_cols[0].metric("Usage trend (sticky)", _fmt(p.get("usage_trend"), n=2),
                   help="Mean z-score across target_share, wopr, snap_pct, carries, adot. "
                        "Positive = role held/grew.")
ctx_cols[1].metric("Results trend (volatile)", _fmt(p.get("results_trend"), n=2),
                   help="Mean z-score across EPA, catch%, RACR, etc. Negative + positive usage = rebound signal.")
ctx_cols[2].metric("QB / offense context", _fmt(p.get("qb_context"), n=2),
                   help="Team passing-EPA z (WR/TE/QB) or rushing-EPA z (RB). Bad offense ≠ bad player.")
ctx_cols[3].metric("PPG baseline (3-yr trailing)", _fmt(p.get("ppg_baseline"), n=1),
                   help="The player's own rolling-3-year PPG baseline (no leakage).")

# --- Two lenses ---------------------------------------------------------------
st.subheader("Both lenses")
l1, l2 = st.columns(2)
l1.metric("Market-fit fair (salary ~ features)",
          _fmt(p.get("market_fair"), kind="d"),
          delta=f"{-p['surplus_market']:+.0f}" if pd.notna(p.get("surplus_market")) else None,
          help="Predicts the salary the league market would assign. Diagnostic, not optimized.")
l2.metric("Production-anchored (projected, current year)",
          _fmt(p.get("value_2026"), kind="d"),
          delta=f"{-surplus_2026:+.0f}" if pd.notna(surplus_2026) else None,
          help="The PRIMARY value — VOR redistribution off projected production.")
