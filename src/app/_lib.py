"""V2 shared loaders + helpers for the Streamlit app.

Joins the V2 quality-framework master CSV with the cap-pricing output on
`espn_id` into a single unified DataFrame that the pages consume. V2 master
is authoritative for identity/quality; pricing extends with cap-unit fair
values, per-player replacement baselines, and provenance columns.

Cached so each load happens once per app session.
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
ROSTER_STATUSES = ["active", "extension", "rookie", "practice_squad", "fa"]

# All 6 V2 component columns for card breakdowns / roster views.
COMPONENT_COLS = [
    "production_value", "team_value", "age_value",
    "injury_value", "position_value", "intangibles_value",
]

# V2 quality-derived intermediate/output columns.
V2_HEADLINE_COLS = ["on_field_value", "dynasty_value", "contract_value"]

# Pricing engine output columns (from src/models/pricing.py).
PRICING_COLS = [
    "replacement_dv", "above_baseline_dv", "scarcity_value",
    "base_fair", "age_mult",
    "fair_value_2026", "surplus_2026",
    "fair_value_dynasty", "surplus_dynasty",
]

# Pricing engine provenance columns (params used for the current build).
PRICING_META_COLS = [
    "pricing_basis", "pricing_pool_method", "pricing_pool_scale",
    "pricing_pool", "pricing_baseline_tier", "pricing_baseline_offset",
    "pricing_alpha", "pricing_age_band_lo", "pricing_age_band_hi",
    "pricing_rate",
]


@st.cache_data(show_spinner="Loading V2 master + pricing...")
def load_master() -> pd.DataFrame:
    """Return the V2 master joined with the cap-pricing table on espn_id.

    The V2 master contains player identity, all 6 component scores, OFV, and DV.
    The pricing table adds cap-unit fair values, surplus, and per-position
    replacement baselines. Sign convention: positive surplus = overpaid.
    """
    from src.config import PROCESSED_DIR

    v2 = pd.read_csv(PROCESSED_DIR / "player_value_v2_2026.csv")
    pricing = pd.read_csv(PROCESSED_DIR / "player_pricing_2026.csv")

    # Pricing CSV was built from V2 master and inherits all V2 columns.
    # Trust pricing as the fuller superset -- just return it verbatim.
    if all(col in pricing.columns for col in ("espn_id", "player", "dynasty_value")):
        return pricing.copy()

    # Fallback: manual merge if pricing was built with a slimmer output shape.
    keep_from_v2 = [c for c in v2.columns if c not in pricing.columns or c == "espn_id"]
    return pricing.merge(v2[keep_from_v2], on="espn_id", how="left")


def recommend_action(row: pd.Series) -> str:
    """V2 heuristic drop/extend/tag/keep recommendation per player.

    Encodes the league's actionable decisions (rules sec 6-9) using V2 pricing:
      - TAG    -- last year of contract, elite quality, sizeable bargain
      - EXTEND -- year before final year, significant bargain
      - DROP   -- overpaid with real salary
      - KEEP   -- everything else
    """
    surplus = row.get("surplus_2026")
    if pd.isna(surplus):
        return "-"
    surplus = float(surplus)
    salary = float(row.get("salary_2026") or 0)
    years = int(row.get("years_2026") or 0)
    dv = float(row.get("dynasty_value") or 0)
    fair = float(row.get("fair_value_2026") or 0)

    # DROP: overpaid on non-trivial contract
    if salary >= 40 and surplus > 60:
        return "DROP"
    # TAG: last year of contract, elite player, sizeable bargain
    if years == 1 and dv >= 70 and surplus <= -60 and fair >= 100:
        return "TAG"
    # EXTEND: penultimate year, significant bargain
    if years == 2 and surplus <= -50 and dv >= 55:
        return "EXTEND"
    return "KEEP"


def style_surplus(df: pd.DataFrame, col: str):
    """Style a surplus column: red for positive (overpaid), green for negative (bargain).

    Matches V1 convention so the visual language is unchanged across engines.
    """
    def _color(v):
        if pd.isna(v):
            return ""
        v = float(v)
        if v < -30:
            return "background-color: rgba(46, 160, 67, 0.30); color: #0a3a17"
        if v > 30:
            return "background-color: rgba(214, 67, 67, 0.30); color: #4a0a0a"
        return ""
    return df.style.map(_color, subset=[col])


def fmt_int_or_dash(v) -> str:
    """Format a possibly-nan numeric as int or '-'."""
    if pd.isna(v):
        return "-"
    return f"{int(v):d}"


def fmt_float_or_dash(v, n: int = 1) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{n}f}"


def team_roster(master: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return the current roster for `team` (skill positions only), sorted."""
    r = master[
        (master["team"] == team)
        & (master["position_group"].isin(POS_ORDER))
    ].copy()
    return r.sort_values(["position_group", "surplus_2026"])


def fa_pool(master: pd.DataFrame) -> pd.DataFrame:
    """Return dynasty-league free agents (roster_status == 'fa')."""
    return master[master["roster_status"] == "fa"].copy()
