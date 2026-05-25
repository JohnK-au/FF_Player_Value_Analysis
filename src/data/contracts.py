"""Parse the contract Google Sheet into tidy per-player tables.

The sheet is a visual, multi-team layout (see CLAUDE.md): eight team blocks sit
side by side, separated by blank columns, with several stacked sections below the
main rosters. This module reshapes the **active roster** section — each team's
current multi-year contracts — into a tidy DataFrame of one row per
``(team, player)`` with salary and years remaining.

Columns are located by their header *labels* rather than fixed offsets, because
the leftmost team block (Nate) carries an extra column (the sheet's global
``Season:`` / years-remaining markers), so absolute positions differ per block.

This module also parses the separate **Contract Extensions** tab
(``parse_extensions``). Other Master Cap Sheet sections (rookies, tags,
IR/practice-squad, cuts/dead-cap, draft picks) are documented in the sheet but
not yet parsed here — see the section map in CLAUDE.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sheets import RAW_DATA_DIR, load_tab

# Team nicknames as they appear in the sheet header row (left to right).
TEAMS = ["Nate", "Seeb", "Silv", "Kerr", "Will", "Drew", "Couc", "Haft"]

# Some tabs use an alternate spelling for a team (the Contract Extensions tab
# heads Nate's block "N8").
_TEAM_ALIASES = {"N8": "Nate"}

# Header label (lower-cased, stripped) -> tidy column name.
_FIELD_LABELS = {
    "contract slot": "contract_slot",
    "player": "player",
    "salary": "salary",
    "yrs left @ acquisition": "years_left_at_acq",
    "season @ acquisition": "season_at_acq",
    "yrs remain": "years_remaining",
}

_NUMERIC_COLS = (
    "salary",
    "years_remaining",
    "years_left_at_acq",
    "season_at_acq",
    "contract_slot",
)

# Contract Extensions tab: header label -> tidy column name.
_EXT_FIELD_LABELS = {
    "player": "player",
    "salary": "salary",
    "years": "years",
    "goes into effect:": "effective_season",
    "yrs until end": "years_until_end",
    "contract starts": "contract_starts",
}
_EXT_NUMERIC_COLS = (
    "salary",
    "years",
    "effective_season",
    "years_until_end",
    "contract_starts",
)

PROCESSED_DIR = RAW_DATA_DIR.parent / "processed"


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Load the raw **Master Cap Sheet** tab as strings (no header interpreted)."""
    return load_tab("master_cap", use_cache=use_cache)


def load_extensions(use_cache: bool = True) -> pd.DataFrame:
    """Load the raw **Contract Extensions** tab as strings."""
    return load_tab("contract_extensions", use_cache=use_cache)


def _canonical_team(name: str) -> str | None:
    """Return the canonical team nickname for a header cell, or None."""
    name = str(name).strip()
    if name in TEAMS:
        return name
    return _TEAM_ALIASES.get(name)


def _team_start_columns(header: pd.Series) -> dict[str, int]:
    """Map each team nickname to the column index where its block starts."""
    starts: dict[str, int] = {}
    for col, val in header.items():
        team = _canonical_team(val)
        if team and team not in starts:
            starts[team] = int(col)
    return starts


def _field_columns(
    header: pd.Series, start: int, end: int, labels: dict[str, str] = _FIELD_LABELS
) -> dict[str, int]:
    """Within a block's column span, map tidy field name -> absolute column."""
    found: dict[str, int] = {}
    for col in range(start, end):
        label = str(header.iloc[col]).strip().lower()
        if label in labels:
            found[labels[label]] = col
    return found


def _active_row_range(df: pd.DataFrame) -> tuple[int, int]:
    """Rows holding active-roster data: from just after the header to the TAG row."""
    for i in range(1, len(df)):
        if (df.iloc[i].astype(str).str.strip() == "TAG").any():
            return 1, i
    return 1, len(df)


def _clean_player(raw: str) -> tuple[str, str | None]:
    """Strip whitespace; split an embedded ``*`` note off the player name."""
    text = str(raw).strip()
    if "*" in text:
        name, _, note = text.partition("*")
        return name.strip(), note.strip() or None
    return text, None


def parse_active_contracts(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reshape the active-roster section into a tidy per-player DataFrame."""
    if df is None:
        df = load_raw()

    header = df.iloc[0]
    starts = _team_start_columns(header)
    if len(starts) != len(TEAMS):
        raise ValueError(
            f"Expected {len(TEAMS)} team blocks, found {len(starts)}: {sorted(starts)}"
        )

    ordered = sorted(starts.items(), key=lambda kv: kv[1])  # (team, start) by column
    row_lo, row_hi = _active_row_range(df)

    records: list[dict] = []
    for n, (team, start) in enumerate(ordered):
        end = ordered[n + 1][1] if n + 1 < len(ordered) else df.shape[1]
        fcols = _field_columns(header, start, end)
        if "player" not in fcols:
            raise ValueError(f"No 'player' column found for team block {team!r}")
        for i in range(row_lo, row_hi):
            row = df.iloc[i]
            raw_player = row.iloc[fcols["player"]]
            if pd.isna(raw_player) or not str(raw_player).strip():
                continue
            player, note = _clean_player(raw_player)
            if not player or player.lower() == "nan":
                continue
            rec: dict = {"team": team, "player": player}
            for field in _NUMERIC_COLS:
                if field in fcols:
                    rec[field] = row.iloc[fcols[field]]
            if note:
                rec["note"] = note
            records.append(rec)

    out = pd.DataFrame.from_records(records)
    for col in _NUMERIC_COLS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def sanity_report(df: pd.DataFrame) -> None:
    """Print quick checks so obvious parsing/data issues surface immediately."""
    print(f"Parsed {len(df)} active contracts across {df['team'].nunique()} teams\n")

    summary = (
        df.groupby("team")
        .agg(players=("player", "size"), total_salary=("salary", "sum"))
        .reindex(TEAMS)
    )
    print("Per-team active roster size and salary total:")
    print(summary.to_string())

    bad_salary = int(df["salary"].isna().sum())
    print(f"\nRows with non-numeric/missing salary: {bad_salary}")

    yr = df["years_remaining"]
    out_of_range = df[(yr < 1) | (yr > 5) | yr.isna()]
    print(
        f"years_remaining outside 1-5 (likely sheet/formula quirks): {len(out_of_range)}"
    )
    if len(out_of_range):
        print(
            out_of_range[["team", "player", "years_remaining"]].to_string(index=False)
        )


def save_active_contracts(df: pd.DataFrame | None = None) -> Path:
    """Parse and write the tidy active-roster table to data/processed/."""
    df = parse_active_contracts(df) if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "contracts_active.csv"
    df.to_csv(out, index=False)
    return out


def parse_extensions(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reshape the Contract Extensions tab into a tidy per-player DataFrame.

    One row per negotiated extension; ``effective_season`` is the season it takes
    effect (extensions are signed the year before a player's final contract year).
    """
    if df is None:
        df = load_extensions()

    header = df.iloc[0]
    starts = _team_start_columns(header)
    if len(starts) != len(TEAMS):
        raise ValueError(
            f"Expected {len(TEAMS)} team blocks, found {len(starts)}: {sorted(starts)}"
        )

    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    records: list[dict] = []
    for n, (team, start) in enumerate(ordered):
        end = ordered[n + 1][1] if n + 1 < len(ordered) else df.shape[1]
        fcols = _field_columns(header, start, end, _EXT_FIELD_LABELS)
        if "player" not in fcols:
            raise ValueError(f"No 'player' column found for team block {team!r}")
        for i in range(1, len(df)):
            raw_player = df.iloc[i].iloc[fcols["player"]]
            if pd.isna(raw_player) or not str(raw_player).strip():
                continue
            player, note = _clean_player(raw_player)
            if not player or player.lower() == "nan":
                continue
            rec: dict = {"team": team, "player": player}
            for field in _EXT_NUMERIC_COLS:
                if field in fcols:
                    rec[field] = df.iloc[i].iloc[fcols[field]]
            if note:
                rec["note"] = note
            records.append(rec)

    out = pd.DataFrame.from_records(records)
    for col in _EXT_NUMERIC_COLS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_extensions(df: pd.DataFrame | None = None) -> Path:
    """Parse and write the tidy contract-extensions table to data/processed/."""
    df = parse_extensions(df) if df is None else df
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "contract_extensions.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    contracts = parse_active_contracts()
    sanity_report(contracts)
    path = save_active_contracts(contracts)
    print(f"\nSaved tidy active contracts to {path}")

    print("\n" + "=" * 60)
    extensions = parse_extensions()
    print(
        f"Parsed {len(extensions)} contract extensions across "
        f"{extensions['team'].nunique()} teams:\n"
    )
    print(extensions.to_string(index=False))
    ext_path = save_extensions(extensions)
    print(f"\nSaved tidy extensions to {ext_path}")
