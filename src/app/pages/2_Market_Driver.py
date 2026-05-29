"""Market / Driver Explorer — the four core interactions.

- **Driver Ranking**: which metrics predict the chosen success target, by raw
  correlation (exploratory) or model permutation importance (explanation).
- **Relationship Explorer**: pick a metric → scatter + linear trend across player-seasons.
- **What-if Simulator**: pick a player → adjust their key metrics with sliders → re-predict
  projected PPG live.
- **Over-pay Map**: total surplus by team × position (which teams systematically over/underpay).
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for p in (_THIS.parents[1], _THIS.parents[3]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from _lib import (  # noqa: E402
    POS_ORDER,
    load_master,
    load_population,
    load_projection_model,
    model_permutation_importance,
)

st.set_page_config(page_title="Market / Driver Explorer", layout="wide")
st.title("Market / Driver Explorer")
st.markdown(
    "Discover which advanced metrics predict fantasy success; see what the model learned; "
    "simulate counterfactuals (e.g. \"what if Jefferson had average QB play?\"); and spot "
    "which teams systematically over/under-pay each position."
)

# --- Top selectors ------------------------------------------------------------
sc1, sc2 = st.columns([1, 2])
position = sc1.selectbox("Position", POS_ORDER, index=POS_ORDER.index("WR"))
target_choice = sc1.radio(
    "Success target",
    ["Next-season PPG (projection)", "Current PPG (same season)"],
    index=0,
)

tab_rank, tab_rel, tab_whatif, tab_overpay = st.tabs(
    ["Driver Ranking", "Relationship Explorer", "What-if Simulator", "Over-pay Map"]
)

# --- Common data slice --------------------------------------------------------
pop = load_population()
sub = pop[pop["position_group"] == position].copy()
TARGET_COL = "ppg"
if target_choice.startswith("Next"):
    pipe, train_df, feats = load_projection_model()
    from src.models.production import PROD_CAT

    score_rows = sub.dropna(subset=feats, how="all").copy()
    if len(score_rows):
        score_rows["proj_ppg"] = pipe.predict(score_rows[PROD_CAT + feats]).round(2)
        sub = score_rows
        TARGET_COL = "proj_ppg"

# Candidate metrics for ranking / relationship views
from src.data.context import (  # noqa: E402
    RESULTS_METRICS,
    TEAM_METRICS,
    USAGE_METRICS,
)
from src.models.production import ADV_NUM, COMBINE_NUM, DRAFT_NUM  # noqa: E402

CANDIDATE_METRICS = (
    ["age", "years_exp"]
    + ADV_NUM
    + DRAFT_NUM
    + COMBINE_NUM
    + [f"{m}_baseline" for m in USAGE_METRICS + RESULTS_METRICS + TEAM_METRICS]
    + ["usage_trend", "results_trend", "qb_context"]
)
CANDIDATE_METRICS = [c for c in CANDIDATE_METRICS if c in sub.columns]

# --- Tab: Driver Ranking ------------------------------------------------------
with tab_rank:
    st.subheader(f"What predicts {position} success — {target_choice}?")
    flavor = st.radio(
        "Ranking flavor",
        ["Raw correlation (exploratory)", "Model permutation importance"],
        horizontal=True,
    )
    top_n = st.slider("How many to show", 5, 30, 18)

    if flavor.startswith("Raw"):
        rows = []
        for m in CANDIDATE_METRICS:
            ss = sub[[m, TARGET_COL]].dropna()
            if len(ss) > 5:
                rows.append({"metric": m, "correlation": ss[m].corr(ss[TARGET_COL])})
        rank = (
            pd.DataFrame(rows).dropna()
            .assign(abs_corr=lambda d: d["correlation"].abs())
            .sort_values("abs_corr", ascending=False)
            .head(top_n)
        )
        fig = px.bar(
            rank, x="correlation", y="metric", orientation="h",
            color="correlation", color_continuous_scale="RdBu", range_color=[-1, 1],
            title=f"Top {top_n} metrics by raw correlation with {TARGET_COL} ({position})",
        )
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Univariate Pearson correlation. Positive = higher metric → higher PPG. "
            "Use this to explore which features are individually associated with success."
        )
    else:
        imp = model_permutation_importance().head(top_n)
        fig = px.bar(
            imp.sort_values("importance"),
            x="importance", y="feature", orientation="h",
            title=f"Top {top_n} features by model permutation importance (projection model)",
        )
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Permutation importance from the next-season projection model. Reflects what "
            "the model *uses* (controlling for other features); the projection model "
            "ranks features across all positions, so the same chart applies regardless of "
            "the position selector."
        )

# --- Tab: Relationship Explorer ----------------------------------------------
with tab_rel:
    st.subheader(f"Metric vs. {TARGET_COL} — {position}")
    metric = st.selectbox("Metric (x-axis)", CANDIDATE_METRICS, index=0)
    s = sub[[metric, TARGET_COL, "name", "season"]].dropna(subset=[metric, TARGET_COL]).copy()
    s["label"] = s["name"] + " (" + s["season"].astype(str) + ")"
    if len(s) > 5:
        fig = px.scatter(
            s, x=metric, y=TARGET_COL, hover_name="label",
            opacity=0.7,
            title=f"{metric} vs. {TARGET_COL} ({position}, n={len(s)})",
        )
        # Linear-fit trendline (no statsmodels dependency)
        slope, intercept = np.polyfit(s[metric].astype(float), s[TARGET_COL].astype(float), 1)
        xs = np.linspace(s[metric].min(), s[metric].max(), 60)
        fig.add_trace(
            go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                       name="linear fit", line=dict(color="#444", dash="dash"))
        )
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
        corr = s[metric].corr(s[TARGET_COL])
        st.caption(f"Pearson r = {corr:+.3f}  ·  slope = {slope:+.3f}  ·  intercept = {intercept:.2f}")
    else:
        st.warning("Not enough data for this metric/position pair.")

# --- Tab: What-if Simulator ---------------------------------------------------
with tab_whatif:
    st.subheader("What-if: adjust a player's metrics → re-predict projected PPG")
    master = load_master()
    pos_players = sorted(
        master[master["position_group"] == position]["player"].dropna().unique()
    )
    default_player = (
        "Justin Jefferson" if position == "WR" and "Justin Jefferson" in pos_players
        else (pos_players[0] if pos_players else None)
    )
    player_choice = st.selectbox(
        "Player", pos_players, index=pos_players.index(default_player) if default_player else 0,
        key="whatif_player",
    )
    row = master[master["player"] == player_choice].iloc[0]
    pop_row = pop[(pop["espn_id"] == row["espn_id"]) & (pop["season"] == 2025)]
    if len(pop_row) == 0:
        st.warning("No 2025 feature row found for this player.")
    else:
        pop_row = pop_row.iloc[0]
        pipe, _, feats = load_projection_model()
        from src.models.production import PROD_CAT

        candidates = [
            ("team_pass_epa", "QB / offense efficiency (team pass EPA)"),
            ("team_cpoe", "Team CPOE (QB accuracy over expected)"),
            ("team_rush_epa", "Team rushing EPA"),
            ("target_share", "Target share"),
            ("snap_pct", "Snap share"),
            ("receiving_epa", "Receiving EPA (sum)"),
            ("rushing_epa", "Rushing EPA (sum)"),
            ("carries", "Carries"),
            ("age", "Age"),
        ]
        candidates = [(m, l) for m, l in candidates if m in feats]

        st.markdown(f"Adjust **{player_choice}**'s 2025 metric values — projected 2026 PPG updates live.")
        sl_cols = st.columns(2)
        sliders = {}
        for i, (m, label) in enumerate(candidates):
            col = sl_cols[i % 2]
            current = pop_row.get(m)
            if pd.isna(current):
                continue
            current = float(current)
            series = pop[m].dropna()
            if len(series) < 10:
                continue
            lo, hi = float(series.quantile(0.02)), float(series.quantile(0.98))
            if lo >= hi:
                lo, hi = current - max(abs(current), 1.0), current + max(abs(current), 1.0)
            value = col.slider(
                label, min_value=round(lo, 3), max_value=round(hi, 3),
                value=round(min(max(current, lo), hi), 3),
                step=round((hi - lo) / 100, 3) or 0.01,
                key=f"sl_{m}",
            )
            sliders[m] = value

        # Baseline vs modified predictions
        modified = pop_row.copy()
        for m, v in sliders.items():
            modified[m] = v
        X_base = pd.DataFrame([pop_row[PROD_CAT + feats]])
        X_mod = pd.DataFrame([modified[PROD_CAT + feats]])
        proj_base = float(pipe.predict(X_base)[0])
        proj_mod = float(pipe.predict(X_mod)[0])

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Current actual (2025 PPG)", f"{row['ppg_2025']:.2f}" if pd.notna(row.get("ppg_2025")) else "—")
        rc2.metric("Baseline projected 2026 PPG", f"{proj_base:.2f}")
        rc3.metric(
            "With your adjustments",
            f"{proj_mod:.2f}",
            delta=f"{proj_mod - proj_base:+.2f}",
            help="Live re-prediction from the projection model with your slider values.",
        )
        st.caption(
            "Example: pick Justin Jefferson and drag *team_pass_epa* up to the 80th percentile — "
            "you'll see his projected 2026 PPG rise. That isolates how much of his projection "
            "is offense-context vs. his own profile."
        )

# --- Tab: Over-pay Map --------------------------------------------------------
with tab_overpay:
    st.subheader("Over-pay map — which teams systematically over/underpay each position")
    master = load_master()
    horizon_map = st.radio(
        "Horizon", ["Current 2026", "Dynasty"], horizontal=True, key="overpay_horizon"
    )
    surplus_col = "surplus_2026" if horizon_map == "Current 2026" else "dynasty_surplus"
    grid = master.groupby(["team", "position_group"])[surplus_col].sum().reset_index()
    pivot = (
        grid.pivot(index="team", columns="position_group", values=surplus_col)
        .fillna(0)
        .round(0)
        .reindex(columns=POS_ORDER)
    )
    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels={"x": "Position", "y": "Team", "color": f"sum {surplus_col}"},
        title=f"Total {surplus_col} by team × position  (red = team overpaid; green = bargain)",
    )
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Sum of surplus across the team's players at each position. Red cells = the team is "
        "systematically over-paying that position relative to production; green = under-paying."
    )
