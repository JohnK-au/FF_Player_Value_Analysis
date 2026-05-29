"""Auction Bid Targets — FA pool projected fair values (max-fair-bid per player).

Free agents = NFL skill players not currently under league contract. For each FA we
project 2026 PPG (S3 model), apply the same deep-baseline pricing recipe (with no
volatility penalty since we lack weekly data for non-rostered players), and report
the **max fair bid** — i.e., the cap-unit amount at which paying for them would equal
their projected production-anchored value.

For the user's own cap context, show projected total skill-cap usage if currently
on the user's team (useful for sizing bids against remaining headroom).
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

from _lib import MY_TEAM, POS_ORDER, TEAMS, load_fa_pool, load_master  # noqa: E402

st.set_page_config(page_title="Auction Bid Targets", layout="wide")
st.title("Auction Bid Targets")
st.markdown(
    "Free agents (NFL skill players not under league contract) priced at their "
    "projected production-anchored value. **`max_fair_bid`** is the cap-unit amount at "
    "which paying for them equals their projected value — bid up to it for a neutral "
    "deal; bid below for a bargain."
)

fa, rate = load_fa_pool()
master = load_master()

# --- Filters ---
fc1, fc2, fc3 = st.columns([1, 1, 1.4])
team = fc1.selectbox("Your team", TEAMS, index=TEAMS.index(MY_TEAM))
positions = fc2.multiselect("Position", POS_ORDER, default=POS_ORDER)
min_bid = fc3.slider("Minimum max_fair_bid", 0, int(fa["max_fair_bid"].max() or 1),
                     value=10, help="Hide near-zero FA values to focus on real targets.")

f = fa[fa["position_group"].isin(positions) & (fa["max_fair_bid"] >= min_bid)].copy()

# --- Cap context for the selected team ---
my_roster = master[master["team"] == team]
my_skill_used = float(my_roster["salary_2026"].sum())
CAP_TOTAL = 1500
m1, m2, m3, m4 = st.columns(4)
m1.metric("Your skill cap used", f"{my_skill_used:.0f}")
m2.metric("FA pool size (filtered)", len(f))
m3.metric("Top max_fair_bid", f"{f['max_fair_bid'].max():.0f}" if len(f) else "—")
m4.metric("Cap rate", f"{rate:.1f}", help="Cap units per deep-VOR point (from value engine).")

st.caption(
    "Note: FA volatility (downside deviation) is unavailable for non-rostered players, "
    "so the bid assumes consistency_factor = 1.0. For volatile profiles, discount the bid "
    "in your head, or wait for in-season weekly data to refine."
)

# --- Top targets by position ---
st.subheader("Top targets by position")
tabs = st.tabs(POS_ORDER)
for tab, pos in zip(tabs, POS_ORDER):
    with tab:
        s = f[f["position_group"] == pos].head(25)
        if len(s) == 0:
            st.info(f"No {pos} free agents meeting the filter.")
            continue
        cols = ["name", "age", "ppg", "projected_ppg", "deep_vor", "max_fair_bid"]
        cols = [c for c in cols if c in s.columns]
        st.dataframe(
            s[cols].rename(columns={
                "name": "player", "ppg": "2025 PPG",
                "projected_ppg": "projected 2026 PPG",
                "deep_vor": "deep VOR", "max_fair_bid": "max fair bid",
            }),
            use_container_width=True, hide_index=True, height=min(600, 80 + 35 * len(s)),
        )
