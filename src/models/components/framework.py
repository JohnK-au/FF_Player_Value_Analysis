"""Framework orchestrator — builds the v2 master player value table.

Pulls the 2026 contract roster, runs each of the 6 component scoring
modules per position, combines them into Dynasty Value, derives a generic
Contract Value, and writes the master CSV at
``data/processed/player_value_v2_2026.csv``.

Coexists with the legacy ``data/processed/player_value_2026.csv`` produced
by ``src/models/value.py``. The legacy file is unchanged; v2 lands alongside.

Generic Contract Value (v1 derivation):
    contract_value_per_year = dynasty_value / max(years_2026, 1)
i.e. annualised intrinsic worth. This is the simplest interpretable
derivation; roster-aware Contract Value (need-weighting) is a Phase 5+ task.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.contracts import build_2026_contracts
from src.data.players import attributes_table
from src.models.components import age, combine, injury, intangibles, position, production, team

OUT_PATH = Path(PROCESSED_DIR) / "player_value_v2_2026.csv"
POSITIONS = ("QB", "RB", "WR", "TE")

IDENTITY_COLS = ("espn_id", "player", "team", "position_group", "age", "years_exp")
CONTRACT_COLS = ("salary_2026", "years_2026", "dynasty_total_salary")
COMPONENT_COLS = (
    "production_value",
    "age_value",
    "team_value",
    "injury_value",
    "position_value",
    "intangibles_value",
)
OUTPUT_COLS = (
    *IDENTITY_COLS,
    *CONTRACT_COLS,
    *COMPONENT_COLS,
    "dynasty_value",
    "contract_value",
    "dynasty_surplus",
    "contract_surplus",
)


def _contract_roster() -> pd.DataFrame:
    """Build the 2026 contract roster joined to player attributes."""
    contracts, _notes = build_2026_contracts()
    attrs = attributes_table().drop_duplicates("player")
    out = contracts.merge(attrs, on="player", how="left")
    out["position_group"] = out["position_group"].fillna("Other")
    out["dynasty_total_salary"] = out["salary_2026"] * out["years_2026"].clip(lower=1)
    return out


def _score_position(players: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Run each component scoring module for one position slice."""
    out = players.copy()
    for module in (production, age, team, injury, position, intangibles):
        out = module.score(out, pos)
    return out


def build_player_values_v2(combination_method: str = "uniform_weighted_sum") -> pd.DataFrame:
    """Top-level entry point: produces and returns the v2 master player value frame."""
    roster = _contract_roster()

    pieces = []
    for pos in POSITIONS:
        slice_df = roster[roster["position_group"] == pos]
        if slice_df.empty:
            continue
        pieces.append(_score_position(slice_df, pos))
    scored = pd.concat(pieces, ignore_index=True) if pieces else roster.iloc[0:0]

    if not scored.empty:
        scored["dynasty_value"] = combine.combine(scored, method=combination_method)
        years = scored["years_2026"].clip(lower=1)
        scored["contract_value"] = scored["dynasty_value"] / years
        scored["dynasty_surplus"] = scored["dynasty_value"] - scored.get("dynasty_total_salary", 0)
        scored["contract_surplus"] = scored["contract_value"] - scored["salary_2026"]

    out_cols = [c for c in OUTPUT_COLS if c in scored.columns]
    return scored[out_cols].sort_values("dynasty_value", ascending=False, na_position="last")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_player_values_v2()
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows -> {OUT_PATH}")
    if not df.empty:
        print("first 5 rows (dynasty_value desc):")
        cols = ["player", "team", "position_group", "salary_2026", "years_2026", "dynasty_value", "contract_value"]
        print(df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
