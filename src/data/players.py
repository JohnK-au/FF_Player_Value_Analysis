"""Join contract-sheet players to ESPN players (positions, ids).

The contract sheet uses shorthand names ("Amon Ra", "Devon Achane") while ESPN
has full names ("Amon-Ra St. Brown", "De'Von Achane"). This module normalizes
names, matches each contract player to an ESPN player (exact-on-normalized, then
fuzzy), and produces a crosswalk carrying position / proTeam / espn_id — the
keystone for the salary-by-position view and the downstream value model
(ESPN ↔ nflverse joins on espn_id).
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

from .contracts import PROCESSED_DIR, parse_active_contracts, parse_extensions
from .espn import get_league

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Known hard aliases (sheet name -> ESPN name); extend as the report surfaces them.
ALIASES: dict[str, str] = {
    "Amon Ra": "Amon-Ra St. Brown",
    "Tet McMillain": "Tetairoa McMillan",
    "Foye Oluokon": "Foyesade Oluokun",
    "J Jennings": "Jauan Jennings",
    "J. Jennings": "Jauan Jennings",
    "Shadeur": "Shedeur Sanders",
    "sheduer": "Shedeur Sanders",
}


def _normalize(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"[.,'’`]", "", s)        # drop periods / commas / apostrophes
    s = re.sub(r"[-/]", " ", s)          # hyphens / slashes -> space
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(toks)


def espn_player_table(year: int = 2025, fa_size: int = 1500) -> pd.DataFrame:
    """ESPN players for a season: rostered + the free-agent pool (broad coverage).

    Columns: espn_id, espn_name, position, pro_team, norm.
    """
    lg = get_league(year)
    rows = []
    for team in lg.teams:
        for p in team.roster:
            rows.append(
                {
                    "espn_id": p.playerId,
                    "espn_name": p.name,
                    "position": getattr(p, "position", None),
                    "pro_team": getattr(p, "proTeam", None),
                }
            )
    try:
        for p in lg.free_agents(size=fa_size):
            rows.append(
                {
                    "espn_id": p.playerId,
                    "espn_name": p.name,
                    "position": getattr(p, "position", None),
                    "pro_team": getattr(p, "proTeam", None),
                }
            )
    except Exception as e:  # free-agent pull is best-effort
        print(f"(free-agent pull failed: {type(e).__name__}: {str(e)[:80]})")
    df = pd.DataFrame(rows).drop_duplicates("espn_id").reset_index(drop=True)
    df["norm"] = df["espn_name"].map(_normalize)
    return df


# Sheet names whose only fuzzy candidate is a DIFFERENT player -- block the
# spurious match so they stay unmatched (deep rookies with no ESPN entry yet).
# Verified 2026-08: "Chris Bell"->Chris Boswell(K), "Omar Cooper Jr"->Amari Cooper.
_BLOCK_FUZZY = {"Chris Bell", "Omar Cooper Jr"}


def _match_one(name: str, espn_df: pd.DataFrame, norm_index: list[str]):
    target = ALIASES.get(str(name).strip(), name)
    n = _normalize(target)
    if not n:
        return None, 0.0, "none"
    exact = espn_df[espn_df["norm"] == n]
    if len(exact):
        return exact.iloc[0], 1.0, "exact"
    if str(name).strip() in _BLOCK_FUZZY:
        return None, 0.0, "none"
    close = difflib.get_close_matches(n, norm_index, n=1, cutoff=0.84)
    if close:
        hit = espn_df[espn_df["norm"] == close[0]].iloc[0]
        score = difflib.SequenceMatcher(None, n, close[0]).ratio()
        return hit, round(score, 3), "fuzzy"
    return None, 0.0, "none"


def current_players() -> pd.DataFrame:
    """Distinct players currently under contract (active + extension + rookie + IR/PS)."""
    from .cap import parse_ir, parse_rookies

    frames = [
        parse_active_contracts()[["team", "player"]].assign(role="active"),
        parse_extensions()[["team", "player"]].assign(role="extension"),
        parse_rookies()[["team", "player"]].assign(role="rookie"),
        parse_ir().rename(columns={"designation": "role"})[["team", "player", "role"]],
    ]
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[allp["player"].notna() & (allp["player"].astype(str).str.strip() != "")]
    return allp.reset_index(drop=True)


def build_crosswalk(year: int = 2025) -> pd.DataFrame:
    """Match each distinct contract player to an ESPN player; carry position/id."""
    espn_df = espn_player_table(year)
    norm_index = espn_df["norm"].tolist()
    names = sorted(current_players()["player"].astype(str).str.strip().unique())

    recs = []
    for name in names:
        if re.search(r"\bhc\b$", name, re.I):  # "Pats HC", "49ers HC" — a roster slot
            recs.append({"player": name, "espn_name": None, "espn_id": None,
                         "position": "HC", "pro_team": None, "match_score": 1.0,
                         "method": "hc"})
            continue
        hit, score, method = _match_one(name, espn_df, norm_index)
        recs.append(
            {
                "player": name,
                "espn_name": None if hit is None else hit["espn_name"],
                "espn_id": None if hit is None else hit["espn_id"],
                "position": None if hit is None else hit["position"],
                "pro_team": None if hit is None else hit["pro_team"],
                "match_score": score,
                "method": method,
            }
        )
    return pd.DataFrame(recs)


# Collapse ESPN positions into the groups we report on.
POSITION_GROUP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "K": "K/P", "P": "K/P", "HC": "HC",
    "CB": "IDP", "S": "IDP", "DB": "IDP", "DE": "IDP",
    "DT": "IDP", "DL": "IDP", "LB": "IDP", "EDGE": "IDP",
}


def load_crosswalk(rebuild: bool = False) -> pd.DataFrame:
    """Load the cached player crosswalk, building it if missing or ``rebuild``."""
    path = PROCESSED_DIR / "player_crosswalk.csv"
    if path.exists() and not rebuild:
        return pd.read_csv(path)
    return save_crosswalk_and_return()


def save_crosswalk_and_return() -> pd.DataFrame:
    cw = build_crosswalk()
    save_crosswalk(cw)
    return cw


def attributes_table(rebuild: bool = False) -> pd.DataFrame:
    """Crosswalk + nflverse age/experience: player, espn_id, position, group, age."""
    from .nflverse import player_attributes as nfl_attrs

    cw = load_crosswalk(rebuild=rebuild)[["player", "espn_name", "espn_id", "position"]].copy()
    cw["espn_id"] = pd.to_numeric(cw["espn_id"], errors="coerce").astype("Int64")
    cw["position_group"] = cw["position"].map(POSITION_GROUP).fillna("Other")
    ages = nfl_attrs()[["espn_id", "age", "years_exp"]]
    # espn_id is nullable Int64; pandas<2 merge can't factorize NA keys (rookies
    # that didn't crosswalk + head-coach slots). Join only the id'd rows and
    # re-attach the rest with NaN age -- they still flow downstream via the
    # name-based join in cap.player_salaries_2026.
    have_id = cw[cw["espn_id"].notna()].merge(ages, on="espn_id", how="left")
    no_id = cw[cw["espn_id"].isna()].copy()
    return pd.concat([have_id, no_id], ignore_index=True)


def save_crosswalk(df: pd.DataFrame | None = None) -> "pd.Path":
    df = build_crosswalk() if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "player_crosswalk.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    cw = build_crosswalk()
    n = len(cw)
    matched = cw[cw["method"] != "none"]
    print(f"Matched {len(matched)}/{n} contract players to ESPN "
          f"({len(matched) / n:.0%}).")
    print(f"  exact: {(cw.method=='exact').sum()}  fuzzy: {(cw.method=='fuzzy').sum()}"
          f"  none: {(cw.method=='none').sum()}")

    fuzzy = cw[cw.method == "fuzzy"].sort_values("match_score")
    if len(fuzzy):
        print("\n--- fuzzy matches to eyeball (lowest score first) ---")
        print(fuzzy[["player", "espn_name", "position", "match_score"]].head(20).to_string(index=False))

    unmatched = cw[cw.method == "none"]
    if len(unmatched):
        print(f"\n--- unmatched ({len(unmatched)}) ---")
        print(", ".join(unmatched["player"].tolist()))

    path = save_crosswalk(cw)
    print(f"\nSaved crosswalk to {path}")
