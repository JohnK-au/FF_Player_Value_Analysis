"""Shared loaders + helpers for the Streamlit app.

Heavy artifacts (consolidated value table, multi-season context frame, fitted projection
model, permutation-importance bar) are cached so each is built once per app session.
Pages import from this module. The first import also adds the project root to sys.path so
`from src.<...>` imports work whether the file was launched as the Streamlit entry or as a
page (pages share the same Python interpreter, so the setup persists).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable for `from src.<...>` style imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

POS_ORDER = ["QB", "RB", "WR", "TE"]
TEAMS = ["Nate", "Seeb", "Silv", "Kerr", "Will", "Drew", "Couc", "Haft"]
MY_TEAM = "Kerr"

YEAR_TYPE_OPTIONS = ["up", "par", "down", "rookie", "partial"]


@st.cache_data(show_spinner="Loading consolidated value table...")
def load_master() -> pd.DataFrame:
    """The master player_value_2026.csv — current + dynasty + market + context."""
    from src.config import PROCESSED_DIR
    return pd.read_csv(PROCESSED_DIR / "player_value_2026.csv")


@st.cache_data(show_spinner="Building multi-season context frame (one-time, ~30 s)...")
def load_population() -> pd.DataFrame:
    """Stacked training_frame_with_context — the data behind the market/driver views."""
    from src.data.context import training_frame_with_context
    return training_frame_with_context()


@st.cache_resource(show_spinner="Fitting projection model...")
def load_projection_model():
    """Returns (fitted pipeline, training_df, feature_list)."""
    from src.models.projection import fit_projection_model
    return fit_projection_model()


@st.cache_data(show_spinner="Computing permutation importance...")
def model_permutation_importance() -> pd.DataFrame:
    """Permutation importance of the projection model's features (sorted desc)."""
    from sklearn.inspection import permutation_importance

    from src.models.production import PROD_CAT
    from src.models.projection import PROJ_TARGET

    pipe, d, feats = load_projection_model()
    X = d[PROD_CAT + feats]
    imp = permutation_importance(
        pipe, X, d[PROJ_TARGET], n_repeats=5, random_state=0, n_jobs=-1
    )
    return (
        pd.DataFrame({"feature": PROD_CAT + feats, "importance": imp.importances_mean.round(4)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="Computing FA pool projected values...")
def load_fa_pool() -> tuple[pd.DataFrame, float]:
    """The free-agent pool (NFL skill players NOT under league contract) with
    projected fair values (max fair bid)."""
    from src.config import PROCESSED_DIR
    from src.models.value import DEEP_FACTOR, projected_value_table

    proj = pd.read_csv(PROCESSED_DIR / "projected_production_2026.csv")
    master = load_master()
    rostered_ids = set(master["espn_id"].dropna().astype(int).tolist())
    fa = proj[~proj["espn_id"].astype(int).isin(rostered_ids)].copy()

    # Use the same rate + deep baselines the value engine derived for our roster.
    _, rate, repl = projected_value_table(2026)
    deep_bl = {pos: DEEP_FACTOR * v for pos, v in repl.items()}
    # FAs have no downside data → no volatility penalty (consistency_factor = 1).
    fa["consistency_factor"] = 1.0
    fa["prod_adj"] = fa["projected_ppg"]
    fa["deep_baseline"] = fa["position_group"].map(deep_bl)
    fa["deep_vor"] = (fa["prod_adj"] - fa["deep_baseline"]).clip(lower=0)
    fa["max_fair_bid"] = (fa["deep_vor"] * rate).round(1)
    fa = fa[fa["max_fair_bid"] > 0].sort_values("max_fair_bid", ascending=False).reset_index(drop=True)
    return fa, float(rate)


def recommend_action(row: pd.Series) -> str:
    """Heuristic drop / extend / tag / keep recommendation per player.

    Encodes the league's actionable decisions (rules §6–§9): a player in his final
    year (years_2026 == 1) with elite production is a TAG candidate; a player in the
    year before his final year (years_2026 == 2) with significant under-payment is
    an EXTEND candidate; an overpaid player with real salary is a DROP candidate;
    everyone else KEEP.
    """
    if pd.isna(row.get("surplus_2026")):
        return "—"
    surplus = float(row["surplus_2026"])
    years = int(row.get("years_2026") or 0)
    salary = float(row.get("salary_2026") or 0)
    value = float(row.get("value_2026") or 0)
    if salary >= 50 and surplus > 50:
        return "DROP"
    if years == 1 and surplus < -30 and value > 50:
        return "TAG"
    if years == 2 and surplus < -30:
        return "EXTEND"
    return "KEEP"


def style_surplus(df: pd.DataFrame, col: str):
    """Style a surplus column: red for positive (overpaid), green for negative (bargain)."""
    def _color(v):
        if pd.isna(v):
            return ""
        if v < -10:
            return "background-color: rgba(46, 160, 67, 0.30); color: #0a3a17"
        if v > 10:
            return "background-color: rgba(214, 67, 67, 0.30); color: #4a0a0a"
        return ""
    return df.style.map(_color, subset=[col])
