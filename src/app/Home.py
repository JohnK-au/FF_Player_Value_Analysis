"""V2 Player Value Engine -- Streamlit app entry point.

Landing page = over/under-valued board across the whole league (155 rostered +
335 dynasty-league FAs). Filter by position/team/status/horizon; sort by surplus.

Fair values come from `src/models/pricing.py`'s 4-stage pipeline
(V2 dynasty_value -> replacement baseline -> above-baseline -> non-linear
scarcity -> rate x age multiplier x multi-year age decay).

Run: ``streamlit run src/app/Home.py``  (from project root, .venv activated).
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

from _lib import (  # noqa: E402
    POS_ORDER, ROSTER_STATUSES, TEAMS,
    fmt_int_or_dash, load_master, style_surplus,
)

st.set_page_config(page_title="Player Value Engine (V2)", page_icon="football", layout="wide")
st.title("Player Value Engine -- 2026 (V2)")
st.markdown(
    "6-component quality framework (Production / Team / Age / Injury / Position / Intangibles)  "
    "combined into **Dynasty Value**, then translated into **cap-unit fair values** by the "
    "4-stage pricing engine. Filter below; sort by surplus to find bargains/overpays."
)

master = load_master()

# --- Filters --------------------------------------------------------------
fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1.2, 1.2, 1.2, 1.4])
horizon = fc1.radio("Horizon", ["Current 2026", "Dynasty (multi-year)"])
positions = fc2.multiselect("Position", POS_ORDER, default=POS_ORDER)
_show_teams = TEAMS + ["(FA)"]
teams = fc3.multiselect("League team", _show_teams, default=_show_teams)
statuses = fc4.multiselect("Roster status", ROSTER_STATUSES, default=ROSTER_STATUSES)
search = fc5.text_input("Player name contains", "")

f = master.copy()
# Team filter: FA rows have NaN team; treat "(FA)" as roster_status=='fa'
team_mask = f["team"].isin(teams)
if "(FA)" in teams:
    team_mask = team_mask | (f["roster_status"] == "fa")
f = f[
    f["position_group"].isin(positions)
    & team_mask
    & f["roster_status"].isin(statuses)
]
if search.strip():
    f = f[f["player"].str.contains(search.strip(), case=False, na=False)]

if horizon == "Current 2026":
    surplus_col = "surplus_2026"
    fair_col = "fair_value_2026"
    show_cols = [
        "player", "team", "position_group", "roster_status", "age",
        "salary_2026", "years_2026",
        "dynasty_value", "on_field_value",
        "replacement_dv", "above_baseline_dv",
        fair_col, surplus_col,
    ]
else:
    surplus_col = "surplus_dynasty"
    fair_col = "fair_value_dynasty"
    show_cols = [
        "player", "team", "position_group", "roster_status", "age",
        "salary_2026", "years_2026", "dynasty_total_salary",
        "dynasty_value", "on_field_value",
        fair_col, surplus_col,
    ]
show_cols = [c for c in show_cols if c in f.columns]
f = f.sort_values(surplus_col)

# --- Summary metrics ------------------------------------------------------
bargains = int((f[surplus_col] <= -30).sum())
fair_ct = int(f[surplus_col].between(-30, 20).sum())
overpays = int((f[surplus_col] > 20).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Players (filtered)", len(f))
m2.metric("Bargains (surplus < -30)", bargains)
m3.metric("Fair (-30 to +20)", fair_ct)
m4.metric("Overpays (surplus > +20)", overpays)

# --- Board -----------------------------------------------------------------
display = f[show_cols].copy()
# Format numeric columns
for col in ["salary_2026", "years_2026", "dynasty_total_salary"]:
    if col in display.columns:
        display[col] = display[col].apply(fmt_int_or_dash)
for col in ["age", "dynasty_value", "on_field_value", "replacement_dv",
            "above_baseline_dv", fair_col, surplus_col]:
    if col in display.columns:
        display[col] = pd.to_numeric(display[col], errors="coerce").round(1)

st.dataframe(
    style_surplus(display, surplus_col),
    use_container_width=True,
    height=720,
    hide_index=True,
)

# --- Caption --------------------------------------------------------------
pool_scale = float(master["pricing_pool_scale"].iloc[0]) if "pricing_pool_scale" in master.columns else 1.0
alpha = float(master["pricing_alpha"].iloc[0]) if "pricing_alpha" in master.columns else None
st.caption(
    f"Sorted by `{surplus_col}` (bargains first). "
    f"**Pricing preset**: alpha={alpha:.2f}, pool_scale={pool_scale:.2f}, "
    f"per-position replacement baselines shown per row. "
    "**Positive surplus = overpaid (red); negative = bargain (green).** "
    "In the dynasty view, surplus_dynasty = total contract cost minus discounted fair over remaining years."
)
