"""Roster view — team selector + cap summary + drop/extend/tag/keep recommendations.

Heuristic recommendations encode the league's actionable decisions (rules §6–§9):
TAG candidates are last-year-of-contract elite + cheap players; EXTEND candidates are
in the year before the final year with significant under-payment; DROP candidates are
overpaid with real salary; everything else KEEP.
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

from _lib import MY_TEAM, TEAMS, load_master, recommend_action, style_surplus  # noqa: E402

st.set_page_config(page_title="Roster View", layout="wide")
st.title("Roster — value + cap + recommendations")

master = load_master()
team = st.selectbox("Team", TEAMS, index=TEAMS.index(MY_TEAM))
roster = master[master["team"] == team].copy()
roster = roster.sort_values(["position_group", "surplus_2026"])

CAP_TOTAL = 1500
sal_used = float(roster["salary_2026"].sum())
dyn_locked = float(roster["dynasty_total_salary"].sum()) if "dynasty_total_salary" in roster.columns else 0.0
cur_value = float(roster["value_2026"].sum())
dyn_value = float(roster["dynasty_value"].sum()) if "dynasty_value" in roster.columns else 0.0
cur_surplus_total = float(roster["surplus_2026"].sum())
dyn_surplus_total = float(roster["dynasty_surplus"].sum()) if "dynasty_surplus" in roster.columns else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Players (skill)", len(roster))
m2.metric("Salary used (2026, skill)", f"{sal_used:.0f}",
          delta=f"of {CAP_TOTAL} cap (skill-only)")
m3.metric("Net surplus — current", f"{cur_surplus_total:+.0f}",
          delta="lower = more bargain", delta_color="inverse")
m4.metric("Net surplus — dynasty", f"{dyn_surplus_total:+.0f}",
          delta="lower = more bargain", delta_color="inverse")

st.caption(
    f"Roster value (current) = {cur_value:.0f}  ·  dynasty value (discounted) = {dyn_value:.0f}  "
    f"·  dynasty commitment = {dyn_locked:.0f} cap units across remaining years."
)

# --- Recommendations -----------------------------------------------------------
roster["recommendation"] = roster.apply(recommend_action, axis=1)
rec_counts = roster["recommendation"].value_counts().reindex(["DROP", "TAG", "EXTEND", "KEEP"]).fillna(0).astype(int)
r1, r2, r3, r4 = st.columns(4)
r1.metric("DROP candidates", rec_counts.get("DROP", 0))
r2.metric("TAG candidates", rec_counts.get("TAG", 0), help="Last year of contract + elite & under-paid.")
r3.metric("EXTEND candidates", rec_counts.get("EXTEND", 0), help="Year before final year + significantly under-paid.")
r4.metric("KEEP", rec_counts.get("KEEP", 0))

# Show DROP / TAG / EXTEND lists first if any
priority = roster[roster["recommendation"].isin(["DROP", "TAG", "EXTEND"])]
if len(priority):
    st.subheader("Action candidates")
    cols = ["player", "position_group", "salary_2026", "years_2026", "ppg_2025",
            "projected_ppg", "value_2026", "surplus_2026", "dynasty_value",
            "dynasty_surplus", "year_type", "recommendation"]
    cols = [c for c in cols if c in priority.columns]
    st.dataframe(
        style_surplus(priority[cols].sort_values("surplus_2026"), "surplus_2026"),
        use_container_width=True, hide_index=True, height=min(420, 60 + 40 * len(priority)),
    )
    st.caption(
        "Heuristic rules (tunable in `_lib.recommend_action`): DROP = salary ≥ 50 and "
        "surplus_2026 > +50  ·  TAG = years_2026 == 1 and surplus_2026 < −30 and value_2026 > 50  ·  "
        "EXTEND = years_2026 == 2 and surplus_2026 < −30."
    )

# Full roster table
st.subheader(f"Full roster — {team}")
full_cols = ["player", "position_group", "salary_2026", "years_2026",
             "ppg_2025", "projected_ppg", "value_2026", "surplus_2026",
             "dynasty_value", "dynasty_surplus", "year_type", "recommendation"]
full_cols = [c for c in full_cols if c in roster.columns]
st.dataframe(
    style_surplus(roster[full_cols], "surplus_2026"),
    use_container_width=True, hide_index=True, height=560,
)
