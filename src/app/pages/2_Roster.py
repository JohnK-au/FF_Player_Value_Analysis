"""V2 Roster page -- per-team view with V2-aware recommendations.

Sections:
  Team selector (defaults to MY_TEAM)
  Cap summary + net surplus metrics (current + dynasty)
  Per-position breakdown (count, avg DV, salary, fair, above-replacement count)
  Recommendation counts (DROP / TAG / EXTEND / KEEP) using V2 recommend_action
  Action priority table -- players marked DROP/TAG/EXTEND at top
  Full roster table sorted by position + surplus

Prototyping stage -- polish iteratively.
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
    fmt_int_or_dash, load_master, recommend_action, style_surplus,
    team_roster,
)

st.set_page_config(page_title="Roster (V2)", layout="wide")
st.title("Roster -- value + cap + recommendations")

CAP_TOTAL = 1500  # per team, from rules sec 3

master = load_master()

team = st.selectbox("Team", TEAMS, index=TEAMS.index(MY_TEAM))
roster = team_roster(master, team)

if not len(roster):
    st.warning(f"No skill players found for team {team!r}.")
    st.stop()

# --- Cap summary ---------------------------------------------------------
sal_used_skill = float(roster["salary_2026"].sum())
dyn_locked = float(roster["dynasty_total_salary"].sum()) if "dynasty_total_salary" in roster.columns else 0.0

total_ofv = float(roster["on_field_value"].sum()) if "on_field_value" in roster.columns else 0.0
total_dv = float(roster["dynasty_value"].sum()) if "dynasty_value" in roster.columns else 0.0
total_fair_2026 = float(roster["fair_value_2026"].sum())
total_fair_dyn = float(roster["fair_value_dynasty"].sum())
net_surplus_2026 = float(roster["surplus_2026"].sum())
net_surplus_dyn = float(roster["surplus_dynasty"].sum())

st.subheader(f"{team} -- summary")
m = st.columns(4)
m[0].metric("Skill players", len(roster))
m[1].metric(
    "Skill salary 2026",
    f"{sal_used_skill:.0f}",
    delta=f"of {CAP_TOTAL} team cap ({sal_used_skill/CAP_TOTAL:.0%})",
    delta_color="off",
)
m[2].metric(
    "Net surplus 2026",
    f"{net_surplus_2026:+.0f}",
    help="Sum of salary - fair over rostered skill. Negative = getting more value than paying.",
    delta_color="off",
)
m[3].metric(
    "Net surplus dynasty",
    f"{net_surplus_dyn:+.0f}",
    help="Multi-year: sum of dynasty_total_salary - fair_value_dynasty.",
    delta_color="off",
)

st.caption(
    f"Skill Dynasty Value total = **{total_dv:.0f}** (avg {total_dv/len(roster):.1f})  |  "
    f"Total fair 2026 = **{total_fair_2026:.0f}**  |  "
    f"Total fair dynasty = **{total_fair_dyn:.0f}**  |  "
    f"Dynasty cap commitment (locked salary through remaining years) = **{dyn_locked:.0f}**."
)

# --- Per-position breakdown ---------------------------------------------
st.subheader("Per-position breakdown")

# Build a per-position summary + above-replacement count
def _pos_summary(sub: pd.DataFrame) -> pd.Series:
    n = len(sub)
    if n == 0:
        return pd.Series({
            "n": 0, "avg_age": None, "avg_dv": None,
            "salary": 0, "fair_2026": 0, "surplus_2026": 0,
            "above_repl": 0,
        })
    return pd.Series({
        "n": n,
        "avg_age": float(sub["age"].mean()),
        "avg_dv": float(sub["dynasty_value"].mean()),
        "salary": float(sub["salary_2026"].sum()),
        "fair_2026": float(sub["fair_2026" if "fair_2026" in sub.columns else "fair_value_2026"].sum()),
        "surplus_2026": float(sub["surplus_2026"].sum()),
        "above_repl": int((sub["above_baseline_dv"] > 0).sum()),
    })


pos_rows = []
for pos in POS_ORDER:
    sub = roster[roster["position_group"] == pos]
    row = _pos_summary(sub).to_dict()
    row["position"] = pos
    # Include position's replacement_dv from any player at that pos (constant per pos)
    row["replacement_dv"] = float(sub["replacement_dv"].iloc[0]) if len(sub) else None
    pos_rows.append(row)

pos_df = pd.DataFrame(pos_rows)[
    ["position", "n", "avg_age", "avg_dv", "replacement_dv",
     "above_repl", "salary", "fair_2026", "surplus_2026"]
]
pos_df = pos_df.rename(columns={
    "n": "count",
    "avg_age": "avg age",
    "avg_dv": "avg DV",
    "replacement_dv": "replacement DV",
    "above_repl": "startable (DV > replacement)",
    "salary": "salary total",
    "fair_2026": "fair 2026 total",
    "surplus_2026": "net surplus 2026",
})

st.dataframe(
    pos_df.style.format({
        "avg age": "{:.1f}", "avg DV": "{:.1f}", "replacement DV": "{:.1f}",
        "salary total": "{:.0f}", "fair 2026 total": "{:.0f}",
        "net surplus 2026": "{:+.0f}",
    }),
    use_container_width=True, hide_index=True, height=180,
)
st.caption(
    "**startable** = count of your roster at this position with DV > replacement threshold. "
    "**net surplus** > 0 means the position group is collectively overpaid."
)

# --- Recommendations -----------------------------------------------------
st.subheader("Recommendations (V2 heuristic)")

roster["recommendation"] = roster.apply(recommend_action, axis=1)
rec_counts = roster["recommendation"].value_counts().reindex(["DROP", "TAG", "EXTEND", "KEEP"]).fillna(0).astype(int)

r = st.columns(4)
r[0].metric("DROP candidates", int(rec_counts.get("DROP", 0)), delta_color="off")
r[1].metric("TAG candidates", int(rec_counts.get("TAG", 0)),
            help="Last year of contract + elite quality + sizeable bargain.")
r[2].metric("EXTEND candidates", int(rec_counts.get("EXTEND", 0)),
            help="Year before final + significant bargain.")
r[3].metric("KEEP", int(rec_counts.get("KEEP", 0)), delta_color="off")

st.caption(
    "Heuristic thresholds (in `_lib.recommend_action`): "
    "**DROP** = salary >= 40 and surplus_2026 > 60  ·  "
    "**TAG** = years_2026 == 1 and DV >= 70 and surplus <= -60 and fair >= 100  ·  "
    "**EXTEND** = years_2026 == 2 and surplus <= -50 and DV >= 55."
)

# Priority list: DROP > TAG > EXTEND
priority = roster[roster["recommendation"].isin(["DROP", "TAG", "EXTEND"])].copy()
if len(priority):
    st.subheader("Action candidates")
    cols_priority = [
        "player", "position_group", "age", "salary_2026", "years_2026",
        "dynasty_value", "replacement_dv",
        "fair_value_2026", "surplus_2026", "fair_value_dynasty", "surplus_dynasty",
        "recommendation",
    ]
    cols_priority = [c for c in cols_priority if c in priority.columns]
    # Order: DROP first, then TAG, then EXTEND
    rec_order = {"DROP": 0, "TAG": 1, "EXTEND": 2}
    priority = priority.assign(_ord=priority["recommendation"].map(rec_order)).sort_values(
        ["_ord", "surplus_2026"], ascending=[True, False]
    ).drop(columns=["_ord"])
    for c in ("age", "dynasty_value", "replacement_dv"):
        if c in priority.columns:
            priority[c] = pd.to_numeric(priority[c], errors="coerce").round(1)
    for c in ("salary_2026", "years_2026", "fair_value_2026", "surplus_2026",
              "fair_value_dynasty", "surplus_dynasty"):
        if c in priority.columns:
            priority[c] = pd.to_numeric(priority[c], errors="coerce").round(0)
    st.dataframe(
        style_surplus(priority[cols_priority], "surplus_2026"),
        use_container_width=True, hide_index=True,
        height=min(420, 60 + 40 * len(priority)),
    )

# --- Full roster ---------------------------------------------------------
st.subheader(f"Full skill roster -- {team}")
full_cols = [
    "player", "position_group", "roster_status", "age",
    "salary_2026", "years_2026", "dynasty_value", "on_field_value",
    "replacement_dv", "above_baseline_dv",
    "fair_value_2026", "surplus_2026",
    "fair_value_dynasty", "surplus_dynasty",
    "recommendation",
]
full_cols = [c for c in full_cols if c in roster.columns]

full = roster[full_cols].copy()
for c in ("age", "dynasty_value", "on_field_value", "replacement_dv", "above_baseline_dv"):
    if c in full.columns:
        full[c] = pd.to_numeric(full[c], errors="coerce").round(1)
for c in ("salary_2026", "years_2026", "fair_value_2026", "surplus_2026",
          "fair_value_dynasty", "surplus_dynasty"):
    if c in full.columns:
        full[c] = pd.to_numeric(full[c], errors="coerce").round(0)

st.dataframe(
    style_surplus(full, "surplus_2026"),
    use_container_width=True, hide_index=True, height=560,
)
