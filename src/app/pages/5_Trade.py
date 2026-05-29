"""Trade Evaluator — input players on each side, see value + cap delta.

Two-sided multi-select. The page totals each side's salary, locked dynasty cap
commitment, projected current-year value, and discounted dynasty value, then shows
the net delta for each side (positive = the side received more value than it gave).
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for p in (_THIS.parents[1], _THIS.parents[3]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from _lib import MY_TEAM, TEAMS, load_master, style_surplus  # noqa: E402

st.set_page_config(page_title="Trade Evaluator", layout="wide")
st.title("Trade Evaluator")
st.markdown(
    "Select players each side gives up. The evaluator totals salary, dynasty cap "
    "commitment, current-year projected value, and discounted dynasty value for each "
    "side, then shows the **net** for each (positive = received more value than gave)."
)

master = load_master()

c1, c2 = st.columns(2)
with c1:
    teamA = st.selectbox("Team A", TEAMS, index=TEAMS.index(MY_TEAM), key="teamA")
    a_pool = sorted(master[master["team"] == teamA]["player"].dropna().unique())
    a_picks = st.multiselect(f"{teamA} sends", a_pool, key="a_picks")
with c2:
    other = [t for t in TEAMS if t != teamA]
    teamB = st.selectbox("Team B", other, index=0, key="teamB")
    b_pool = sorted(master[master["team"] == teamB]["player"].dropna().unique())
    b_picks = st.multiselect(f"{teamB} sends", b_pool, key="b_picks")

A = master[master["player"].isin(a_picks) & (master["team"] == teamA)].copy()
B = master[master["player"].isin(b_picks) & (master["team"] == teamB)].copy()


def _totals(df: pd.DataFrame) -> dict:
    if not len(df):
        return {"salary": 0.0, "dyn_salary": 0.0, "value": 0.0, "dyn_value": 0.0}
    return {
        "salary": float(df["salary_2026"].sum()),
        "dyn_salary": float(df.get("dynasty_total_salary", pd.Series(0)).sum()),
        "value": float(df["value_2026"].sum()),
        "dyn_value": float(df.get("dynasty_value", pd.Series(0)).sum()),
    }


TA, TB = _totals(A), _totals(B)
net_A = {  # what team A receives - what team A gives
    "salary": TB["salary"] - TA["salary"],
    "dyn_salary": TB["dyn_salary"] - TA["dyn_salary"],
    "value": TB["value"] - TA["value"],
    "dyn_value": TB["dyn_value"] - TA["dyn_value"],
}
net_B = {k: -v for k, v in net_A.items()}

st.subheader("Net impact")
nc1, nc2 = st.columns(2)
with nc1:
    st.markdown(f"**{teamA} (Team A)**")
    a1, a2 = st.columns(2)
    a1.metric("Net current value", f"{net_A['value']:+.0f}",
              help="What A receives minus what A gives, in projected 2026 fair value.")
    a2.metric("Net dynasty value", f"{net_A['dyn_value']:+.0f}",
              help="Multi-year discounted value swing for A.")
    a3, a4 = st.columns(2)
    a3.metric("Net 2026 salary", f"{net_A['salary']:+.0f}",
              help="Positive = A's 2026 cap goes UP (took on bigger contracts).")
    a4.metric("Net dynasty cap commitment", f"{net_A['dyn_salary']:+.0f}",
              help="Multi-year cap commitment swing.")
with nc2:
    st.markdown(f"**{teamB} (Team B)**")
    b1, b2 = st.columns(2)
    b1.metric("Net current value", f"{net_B['value']:+.0f}")
    b2.metric("Net dynasty value", f"{net_B['dyn_value']:+.0f}")
    b3, b4 = st.columns(2)
    b3.metric("Net 2026 salary", f"{net_B['salary']:+.0f}")
    b4.metric("Net dynasty cap commitment", f"{net_B['dyn_salary']:+.0f}")

# Side breakdowns
st.subheader("Side-by-side breakdown")
show_cols = ["player", "position_group", "salary_2026", "years_2026", "projected_ppg",
             "value_2026", "surplus_2026", "dynasty_value", "dynasty_surplus", "year_type"]


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in show_cols if c in df.columns]
    return df[cols]


bc1, bc2 = st.columns(2)
with bc1:
    st.markdown(f"**{teamA} sends ({len(A)})**")
    if len(A):
        st.dataframe(style_surplus(_frame(A), "surplus_2026"),
                     use_container_width=True, hide_index=True,
                     height=min(420, 80 + 40 * len(A)))
    else:
        st.info("Select players for Team A to include.")
with bc2:
    st.markdown(f"**{teamB} sends ({len(B)})**")
    if len(B):
        st.dataframe(style_surplus(_frame(B), "surplus_2026"),
                     use_container_width=True, hide_index=True,
                     height=min(420, 80 + 40 * len(B)))
    else:
        st.info("Select players for Team B to include.")
