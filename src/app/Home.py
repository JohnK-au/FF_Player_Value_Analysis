"""Player Value Engine — Streamlit app entry point.

Landing page = the over/under-valued board. Filter by position/team/year_type/horizon,
sort by surplus, and use the sidebar to navigate to the Player Card or Market/Driver
Explorer pages.

Run: ``streamlit run src/app/Home.py``  (from the project root, .venv activated).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root + this app dir are importable (works for direct + page launches).
_THIS = Path(__file__).resolve()
for p in (_THIS.parent, _THIS.parents[2]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from _lib import POS_ORDER, TEAMS, YEAR_TYPE_OPTIONS, load_master, style_surplus  # noqa: E402

st.set_page_config(page_title="Player Value Engine", page_icon="🏈", layout="wide")
st.title("🏈 Player Value Engine — 2026")
st.markdown(
    "Production-anchored fair value with a next-season **projection** model + dynasty "
    "(multi-year discounted) view. Use the sidebar to switch to the **Player Card** "
    "(per-player deep-dive) or the **Market / Driver Explorer** (which metrics predict "
    "success + what-if simulator + over-pay map)."
)

master = load_master()
priced = master[master["value_2026"].notna()].copy()

# --- Filters --------------------------------------------------------------
fc1, fc2, fc3, fc4 = st.columns([1, 1.4, 1.4, 1.4])
horizon = fc1.radio("Horizon", ["Current 2026", "Dynasty"], horizontal=False)
positions = fc2.multiselect("Position", POS_ORDER, default=POS_ORDER)
teams = fc3.multiselect("Team", TEAMS, default=TEAMS)
year_types = fc4.multiselect("Year type", YEAR_TYPE_OPTIONS, default=YEAR_TYPE_OPTIONS)

f = priced[
    priced["position_group"].isin(positions)
    & priced["team"].isin(teams)
    & priced["year_type"].fillna("rookie").isin(year_types)
].copy()

if horizon == "Current 2026":
    surplus_col, value_col = "surplus_2026", "value_2026"
    show_cols = [
        "player", "team", "position_group", "salary_2026", "ppg_2025",
        "projected_ppg", "value_2026", "surplus_2026",
        "year_type", "usage_trend", "results_trend", "qb_context",
        "market_fair",
    ]
else:
    surplus_col, value_col = "dynasty_surplus", "dynasty_value"
    show_cols = [
        "player", "team", "position_group", "salary_2026", "years_2026",
        "dynasty_total_salary", "dynasty_value", "dynasty_surplus",
        "year_type", "usage_trend", "qb_context",
    ]
show_cols = [c for c in show_cols if c in f.columns]
f = f.sort_values(surplus_col)

# --- Summary metrics ------------------------------------------------------
bargains = int((f[surplus_col] < -10).sum())
fair = int(f[surplus_col].between(-10, 10).sum())
overpays = int((f[surplus_col] > 10).sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Players (filtered)", len(f))
m2.metric("Bargains (surplus < −10)", bargains)
m3.metric("Fair (|surplus| ≤ 10)", fair)
m4.metric("Overpays (surplus > +10)", overpays)

st.dataframe(
    style_surplus(f[show_cols], surplus_col),
    use_container_width=True,
    height=720,
    hide_index=True,
)

st.caption(
    f"Sorted by `{surplus_col}` (most under-valued first). "
    "Green = bargain, red = overpaid (against the production-anchored fair value)."
)
