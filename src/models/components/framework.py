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
from src.data.cap import player_salaries_2026
from src.data.contracts import build_2026_contracts
from src.data.population import extended_training_frame
from src.models.components import age, combine, injury, intangibles, position, production, team

OUT_PATH = Path(PROCESSED_DIR) / "player_value_v2_2026.csv"
POSITIONS = ("QB", "RB", "WR", "TE")
NFL_TEAM_SEASON = 2025  # season we look up modal NFL team from; mirrors CURRENT_SEASON

IDENTITY_COLS = ("espn_id", "player", "team", "nfl_team_2025", "roster_status", "position_group", "age", "years_exp")
CONTRACT_COLS = ("salary_2026", "years_2026", "dynasty_total_salary")
FA_MIN_GAMES_2025 = 4  # minimum 2025 games for "currently relevant" FA inclusion
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
    "on_field_value",   # Production x Team multiplier (Phase 1D); see combination.md
    "dynasty_value",
    "contract_value",
    # NOTE: surplus intentionally NOT computed here. dynasty_value is a 0-100
    # quality score; subtracting cap-unit salaries from it is dimensionally
    # meaningless. Surplus lives in pricing.py (surplus = salary - fair,
    # positive = overpaid), in cap units, per docs/methodology/pricing.md.
)


_SOURCE_PRIORITY: dict[str, int] = {
    "active": 0, "extension": 1, "rookie": 2, "practice_squad": 3,
}


def _contract_roster() -> pd.DataFrame:
    """Build the 2026 contract roster (active + extensions + rookies + practice squad)
    joined to player attributes.

    ``player_salaries_2026()`` already merges in espn_id, position_group, age,
    years_exp via the attributes crosswalk (see ``src/data/cap.py``). For
    ``years_2026`` we join from ``build_2026_contracts()`` (only active +
    extensions carry an explicit term); rookie + practice-squad players default
    to 1 (mirrors the legacy engine, see ``models/value.py::dynasty_value_table``).
    NFL team for the 2025 season (their modal posteam) is joined from the
    extended training frame so the master CSV carries an NFL-team column
    alongside the league team.

    Deduplicates on ``espn_id`` (keeping the highest-priority source) to handle
    contract-sheet typo variants of the same player. E.g., the Browns'
    Shedeur Sanders is listed in the sheet both as "sheduer" (rookies section,
    $10) and "Shadeur" (practice-squad section, $10). The crosswalk resolves
    both name spellings to the same ``espn_id``, but `player_salaries_2026`
    dedups only on ``(team, player)`` so both variants survive. We catch them
    here by deduping on espn_id with source priority active > extension >
    rookie > practice_squad.
    """
    sal = player_salaries_2026()
    contracts, _notes = build_2026_contracts()
    years = contracts[["team", "player", "years_2026"]].drop_duplicates(["team", "player"])
    out = sal.merge(years, on=["team", "player"], how="left")
    out["years_2026"] = out["years_2026"].fillna(1).clip(lower=1, upper=5).astype(int)
    out["dynasty_total_salary"] = out["salary_2026"] * out["years_2026"]

    # Drop rows without an espn_id (un-crosswalked players: kickers, HCs,
    # any typo'd names that didn't resolve to a known player) -- can't score them
    no_id = out["espn_id"].isna().sum()
    out = out.dropna(subset=["espn_id"]).copy()

    # Dedupe duplicates by espn_id (typo'd name variants, multi-section listings
    # like Shedeur Sanders appearing as both "sheduer"/rookie and "Shadeur"/
    # practice_squad). Keep the highest-priority source.
    out["_priority"] = out["source"].map(_SOURCE_PRIORITY).fillna(99)
    pre_dedup = len(out)
    out = (
        out.sort_values(["espn_id", "_priority"])
        .drop_duplicates(subset=["espn_id"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    dups_dropped = pre_dedup - len(out)
    if no_id or dups_dropped:
        print(f"  _contract_roster: dropped {no_id} no-espn_id rows + {dups_dropped} espn_id duplicates")

    # NFL team for the latest fully-completed NFL season (modal posteam)
    ext = extended_training_frame()
    nfl_team_lookup = (
        ext[ext["season"] == NFL_TEAM_SEASON]
        .dropna(subset=["espn_id", "team"])
        .drop_duplicates("espn_id")
        .set_index("espn_id")["team"]
        .to_dict()
    )
    out["nfl_team_2025"] = out["espn_id"].map(nfl_team_lookup)

    # roster_status mirrors source for rostered players
    out["roster_status"] = out["source"]
    return out


def _fa_roster(rostered_ids: set[int]) -> pd.DataFrame:
    """Dynasty-league FAs: NFL skill players in 2025 with games >= FA_MIN_GAMES_2025
    who are NOT on any of our 8 contract rosters.

    Same schema as the rostered roster but contract-derived fields are NaN
    (no salary, no years, no dynasty_total_salary, downstream no surplus).
    """
    ext = extended_training_frame()
    fa = ext[
        (ext["season"] == NFL_TEAM_SEASON)
        & ext["position_group"].isin(POSITIONS)
        & (ext["games"].fillna(0) >= FA_MIN_GAMES_2025)
        & ext["espn_id"].notna()
        & ~ext["espn_id"].astype("Int64").isin(rostered_ids)
    ].copy()
    fa = fa.drop_duplicates("espn_id")

    out = pd.DataFrame({
        "espn_id": fa["espn_id"].astype("Int64"),
        "player": fa["name"],
        "team": pd.NA,                       # no league team
        "nfl_team_2025": fa["team"],          # their 2025 NFL team
        "roster_status": "fa",
        "position_group": fa["position_group"],
        "age": fa["age"],
        "years_exp": fa["years_exp"],
        "salary_2026": pd.NA,
        "years_2026": pd.NA,
        "dynasty_total_salary": pd.NA,
        "source": "fa",
    })
    return out.reset_index(drop=True)


def _score_position(players: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Run each component scoring module for one position slice."""
    out = players.copy()
    for module in (production, age, team, injury, position, intangibles):
        out = module.score(out, pos)
    return out


def build_player_values_v2(
    combination_method: str = combine.DEFAULT_METHOD,
    include_fas: bool = True,
) -> pd.DataFrame:
    """Top-level entry point: produces and returns the v2 master player value frame.

    When ``include_fas=True`` (default), also scores dynasty-league free agents
    (NFL skill players who played >= ``FA_MIN_GAMES_2025`` games in 2025 and
    are NOT on any of our 8 contract rosters). FAs are appended to the same
    master CSV with ``roster_status='fa'``; their salary / years / contract
    surplus fields are NaN.
    """
    rostered = _contract_roster()
    if include_fas:
        rostered_ids = set(rostered["espn_id"].dropna().astype(int).tolist())
        fas = _fa_roster(rostered_ids)
        print(f"  build_player_values_v2: {len(rostered)} rostered + {len(fas)} FAs = {len(rostered) + len(fas)} total")
        roster = pd.concat([rostered, fas], ignore_index=True)
    else:
        roster = rostered

    pieces = []
    for pos in POSITIONS:
        slice_df = roster[roster["position_group"] == pos]
        if slice_df.empty:
            continue
        pieces.append(_score_position(slice_df, pos))
    scored = pd.concat(pieces, ignore_index=True) if pieces else roster.iloc[0:0]

    if not scored.empty:
        scored["on_field_value"] = combine.on_field_value(
            scored["production_value"], scored["team_value"], scored["position_group"]
        )
        scored["dynasty_value"] = combine.combine(scored, method=combination_method)
        # NaN-safe years calc for FAs (no contract)
        years = pd.to_numeric(scored["years_2026"], errors="coerce").clip(lower=1)
        scored["contract_value"] = scored["dynasty_value"] / years

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
