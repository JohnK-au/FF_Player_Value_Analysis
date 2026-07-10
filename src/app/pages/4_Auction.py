"""V2 Auction Bidder -- FA pool with recommended max bids.

The V2 master already includes dynasty-league FAs (roster_status='fa') --
NFL skill players with 2025 games >= 4 not on any of the 8 rosters. Each has
a computed fair_value_2026 via the same pricing pipeline as rostered players,
which is the model's max-fair-bid signal.

Sections:
  Filters (position, age band, min DV, name search)
  Position tabs (All / QB / RB / WR / TE) with counts
  Bid target table sorted by fair_value_2026 desc
  Roster fit check: budget input, cross-reference vs your team's cap space
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
    MY_TEAM, POS_ORDER, TEAMS,
    fa_pool, load_master, style_surplus,
)

st.set_page_config(page_title="Auction Bidder (V2)", layout="wide")
st.title("Auction Bidder -- FA pool max fair bids")
st.markdown(
    "Dynasty-league free agents (NFL skill players **not** on any of the 8 rosters). "
    "`fair_value_2026` is the model's **max fair single-season bid** -- pay less "
    "and it's a bargain."
)

master = load_master()
pool = fa_pool(master)

# --- Filters ---------------------------------------------------------------
f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.2, 1.6])
positions = f1.multiselect("Position", POS_ORDER, default=POS_ORDER)
age_min, age_max = f2.slider(
    "Age range",
    min_value=20, max_value=45,
    value=(20, 40),
)
min_fair = f3.number_input("Min fair 2026", value=0, min_value=0, max_value=200, step=5,
                            help="Hide FAs below this fair value (usually below-baseline).")
search = f4.text_input("Player name contains", "")

f = pool[
    pool["position_group"].isin(positions)
    & pool["age"].between(age_min, age_max)
    & (pool["fair_value_2026"] >= min_fair)
]
if search.strip():
    f = f[f["player"].str.contains(search.strip(), case=False, na=False)]

# --- Roster-fit sidebar ---------------------------------------------------
with st.sidebar:
    st.subheader("Roster fit")
    my_team = st.selectbox("My team", TEAMS, index=TEAMS.index(MY_TEAM))
    my_roster = master[master["team"] == my_team]
    my_salary = float(my_roster["salary_2026"].sum()) if len(my_roster) else 0.0
    CAP_TOTAL = 1500
    cap_space = CAP_TOTAL - my_salary
    st.metric("Skill salary used", f"{my_salary:.0f}")
    st.metric("Est. skill cap space", f"{cap_space:.0f}",
              help="Roughly what you have left before non-skill (K/P/HC/etc) commitments.")

    st.markdown("---")
    st.markdown("**Cap-space guide** (rough thresholds):")
    st.markdown(
        f"- Elite bid ceiling: {cap_space:.0f}\n"
        f"- 5-player mid-tier avg: {cap_space/5:.0f}/each\n"
        f"- 10 depth flyers: {cap_space/10:.0f}/each"
    )

# --- Position tabs --------------------------------------------------------
tabs = st.tabs(["All"] + [f"{p} ({int((f['position_group'] == p).sum())})" for p in POS_ORDER])

show_cols = [
    "player", "nfl_team_2025", "position_group", "age", "years_exp",
    "dynasty_value", "on_field_value",
    "replacement_dv", "above_baseline_dv",
    "fair_value_2026",
]


def _render_table(df: pd.DataFrame, subtitle: str = ""):
    if not len(df):
        st.info("No FAs match the current filters.")
        return
    df = df.sort_values("fair_value_2026", ascending=False).head(200)
    show = df[[c for c in show_cols if c in df.columns]].copy()
    for c in ("age", "dynasty_value", "on_field_value", "replacement_dv", "above_baseline_dv"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1)
    for c in ("years_exp", "fair_value_2026"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(0)
    st.dataframe(show, use_container_width=True, hide_index=True, height=min(720, 60 + 32 * len(show)))
    if subtitle:
        st.caption(subtitle)


with tabs[0]:
    st.subheader("All FAs (top 200 by fair 2026)")
    _render_table(f, f"{len(f)} FAs match filters. Sorted by max fair bid desc.")

for i, pos in enumerate(POS_ORDER, start=1):
    with tabs[i]:
        sub = f[f["position_group"] == pos]
        st.subheader(f"{pos} FAs")
        _render_table(sub, f"{len(sub)} {pos} FAs match filters.")

# --- Bottom caption -------------------------------------------------------
st.markdown("---")
st.caption(
    "**fair_value_2026** = max fair *single-season* bid from the V2 pricing pipeline "
    "(dynasty_value scored via the 6-component framework, then baseline-collapsed, "
    "non-linear-scaled, and age-adjusted). If you can sign the player for less than "
    "the fair, that's negative surplus = a bargain. "
    "Multi-year commitments should compare `fair_value_dynasty` (not shown -- FAs "
    "have no years yet)."
)
