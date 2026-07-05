"""V2 Trade Evaluator -- two-sided player swap analysis.

Pick two teams; select what each side sends. Model computes:
  - Per-side totals: salary, dynasty_total_salary, DV, on-field, fair, surplus
  - Net delta (Team A - Team B): who "wins" on each dimension
  - Verdict: single-season fair delta vs dynasty fair delta

Doesn't handle draft picks or extension rights yet (rules sec 13 mentions
these as tradeable). Noted for future iteration.
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

from _lib import (  # noqa: E402
    MY_TEAM, TEAMS,
    load_master, style_surplus,
)

st.set_page_config(page_title="Trade Evaluator (V2)", layout="wide")
st.title("Trade Evaluator")

master = load_master()


def _roster_names(team: str) -> list[str]:
    r = master[(master["team"] == team) & master["position_group"].isin(["QB","RB","WR","TE"])]
    return sorted(r["player"].dropna().unique().tolist())


# --- Team selectors + player picks ------------------------------------------
top = st.columns(2)
with top[0]:
    team_a = st.selectbox("Team A", TEAMS, index=TEAMS.index(MY_TEAM), key="team_a")
    picks_a = st.multiselect(f"{team_a} sends", _roster_names(team_a), key="picks_a")
with top[1]:
    # Default Team B to the first team that isn't Team A
    default_b_idx = 0 if team_a != TEAMS[0] else 1
    team_b = st.selectbox("Team B", TEAMS, index=default_b_idx, key="team_b")
    picks_b = st.multiselect(f"{team_b} sends", _roster_names(team_b), key="picks_b")

if team_a == team_b:
    st.warning("Team A and Team B are the same. Pick different teams.")
    st.stop()

if not picks_a and not picks_b:
    st.info("Pick at least one player from each side to see the trade breakdown.")
    st.stop()

# --- Slice per side ----------------------------------------------------------
side_a = master[(master["team"] == team_a) & master["player"].isin(picks_a)].copy()
side_b = master[(master["team"] == team_b) & master["player"].isin(picks_b)].copy()

VAL_COLS = [
    "salary_2026", "dynasty_total_salary",
    "dynasty_value", "on_field_value",
    "fair_value_2026", "surplus_2026",
    "fair_value_dynasty", "surplus_dynasty",
]


def _totals(side: pd.DataFrame) -> dict[str, float]:
    if not len(side):
        return {c: 0.0 for c in VAL_COLS}
    return {c: float(pd.to_numeric(side[c], errors="coerce").sum()) for c in VAL_COLS}


tot_a = _totals(side_a)
tot_b = _totals(side_b)

# --- Side breakdown tables --------------------------------------------------
mid = st.columns(2)
show_cols = [
    "player", "position_group", "age", "roster_status",
    "salary_2026", "years_2026",
    "dynasty_value", "on_field_value",
    "fair_value_2026", "surplus_2026",
    "fair_value_dynasty", "surplus_dynasty",
]

for col, side, tot, label in [
    (mid[0], side_a, tot_a, f"{team_a} sends"),
    (mid[1], side_b, tot_b, f"{team_b} sends"),
]:
    with col:
        st.subheader(label)
        if not len(side):
            st.caption("_(nothing selected)_")
            continue

        show = side[[c for c in show_cols if c in side.columns]].copy()
        for c in ("age", "dynasty_value", "on_field_value"):
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(1)
        for c in ("salary_2026", "years_2026", "fair_value_2026", "surplus_2026",
                  "fair_value_dynasty", "surplus_dynasty"):
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(0)
        st.dataframe(
            style_surplus(show, "surplus_2026"),
            use_container_width=True, hide_index=True,
            height=min(320, 60 + 40 * len(show)),
        )

        # Per-side totals row
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Salary 2026", f"{tot['salary_2026']:.0f}")
        s2.metric("Dynasty salary total", f"{tot['dynasty_total_salary']:.0f}")
        s3.metric("Fair 2026", f"{tot['fair_value_2026']:.0f}",
                  delta=f"{-tot['surplus_2026']:+.0f} vs salary",
                  help="Negative surplus (green delta) = paying less than fair.")
        s4.metric("Fair dynasty", f"{tot['fair_value_dynasty']:.0f}",
                  delta=f"{-tot['surplus_dynasty']:+.0f} vs total salary")

# --- Net delta (Team A's perspective) ---------------------------------------
st.markdown("---")
st.subheader(f"Net delta for {team_a} (positive = {team_a} gains)")

# Net = what Team A RECEIVES - what Team A SENDS
# Team A receives Side B, sends Side A
recv = tot_b
sends = tot_a


def _delta(key: str) -> float:
    return recv[key] - sends[key]


nc = st.columns(6)

nc[0].metric(
    "Delta salary 2026",
    f"{_delta('salary_2026'):+.0f}",
    help="Net change in 2026 skill salary. Positive = Team A takes on more salary.",
    delta_color="off",
)
nc[1].metric(
    "Delta dynasty salary",
    f"{_delta('dynasty_total_salary'):+.0f}",
    help="Net change in locked multi-year cap commitment.",
    delta_color="off",
)
nc[2].metric(
    "Delta Dynasty Value",
    f"{_delta('dynasty_value'):+.1f}",
    help="Net change in V2 quality score total (higher = better roster).",
)
nc[3].metric(
    "Delta On-Field Value",
    f"{_delta('on_field_value'):+.1f}",
    help="Net change in current-year production x team quality (higher = better).",
)
nc[4].metric(
    "Delta fair 2026",
    f"{_delta('fair_value_2026'):+.0f}",
    help="Net change in single-season intrinsic cap-unit value. Positive = Team A gains fair value.",
)
nc[5].metric(
    "Delta fair dynasty",
    f"{_delta('fair_value_dynasty'):+.0f}",
    help="Net change in multi-year intrinsic value.",
)

# Surplus delta
sd1, sd2, sd3 = st.columns([1.2, 1.2, 2.6])
delta_surplus_2026 = _delta("surplus_2026")  # positive = A takes on more overpayment
delta_surplus_dyn = _delta("surplus_dynasty")
sd1.metric(
    "Delta surplus 2026",
    f"{delta_surplus_2026:+.0f}",
    delta=f"{-delta_surplus_2026:+.0f} vs paying-fair",
    delta_color="normal",
    help="Positive = Team A absorbs more OVERPAYMENT. Negative = Team A gains bargain-value.",
)
sd2.metric(
    "Delta surplus dynasty",
    f"{delta_surplus_dyn:+.0f}",
    delta=f"{-delta_surplus_dyn:+.0f} vs paying-fair",
    delta_color="normal",
)

# --- Verdict ---------------------------------------------------------------
fair_2026_win = _delta("fair_value_2026")
fair_dyn_win = _delta("fair_value_dynasty")

verdict_lines = []
if abs(fair_2026_win) < 10:
    verdict_lines.append(f"Roughly even on 2026 fair value ({fair_2026_win:+.0f}).")
elif fair_2026_win > 0:
    verdict_lines.append(f"**{team_a} WINS** 2026 fair value by **{fair_2026_win:+.0f}** cap units.")
else:
    verdict_lines.append(f"**{team_b} WINS** 2026 fair value by **{-fair_2026_win:+.0f}** cap units.")

if abs(fair_dyn_win) < 20:
    verdict_lines.append(f"Roughly even on dynasty fair value ({fair_dyn_win:+.0f}).")
elif fair_dyn_win > 0:
    verdict_lines.append(f"**{team_a} WINS** dynasty fair value by **{fair_dyn_win:+.0f}** cap units.")
else:
    verdict_lines.append(f"**{team_b} WINS** dynasty fair value by **{-fair_dyn_win:+.0f}** cap units.")

with sd3:
    st.markdown("### Verdict")
    for line in verdict_lines:
        st.markdown(f"- {line}")

st.caption(
    "**Fair-value delta** compares intrinsic cap-unit values (what each side is really worth). "
    "**Surplus delta** compares the market inefficiency each side carries "
    "(positive surplus = overpaid vs fair). "
    "Draft picks + extension rights (rules sec 13) not yet supported."
)
