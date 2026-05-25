"""Parse the remaining Master Cap Sheet sections and reconcile a cap ledger.

Adds the sections beyond active rosters/extensions — rookies, franchise tags,
IR, cuts (dead cap), and the cap summary — then reconstructs each team's CAP USED
per season and compares it to the sheet's own figures, so we can validate the
parsing and the cap accounting. Dead cap is intentionally left out of the
reconstruction for now (its sheet layout isn't fully understood), so it shows up
as the residual.
"""
from __future__ import annotations

import pandas as pd

from .contracts import (
    CURRENT_SEASON,
    TEAMS,
    UPCOMING_SEASON,
    _clean_player,
    _field_columns,
    _team_start_columns,
    load_raw,
    parse_active_contracts,
    parse_extensions,
)


def _section_header_row(df: pd.DataFrame, marker: str) -> int:
    for i in range(len(df)):
        if (df.iloc[i].astype(str).str.strip() == marker).any():
            return i
    raise ValueError(f"section marker {marker!r} not found")


def _team_spans(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    starts = sorted(_team_start_columns(df.iloc[0]).items(), key=lambda kv: kv[1])
    spans = []
    for n, (team, start) in enumerate(starts):
        end = starts[n + 1][1] if n + 1 < len(starts) else df.shape[1]
        spans.append((team, start, end))
    return spans


def _num(series_like) -> pd.Series:
    return pd.to_numeric(series_like, errors="coerce")


# --- Section parsers -----------------------------------------------------------

_CAP_LABELS = {"cap used": "cap_used", "dead cap": "dead_cap", "cap space": "cap_space"}


def parse_cap_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The sheet's stated CAP USED / DEAD CAP / CAP SPACE per team per season."""
    df = load_raw() if df is None else df
    hdr = _section_header_row(df, "CAP USED")
    header = df.iloc[hdr]
    recs = []
    for team, start, end in _team_spans(df):
        fcols = _field_columns(header, start, end, _CAP_LABELS)
        for i in range(hdr + 1, hdr + 6):  # 5 seasons (2025-2029)
            season = df.iloc[i, start + 1]
            if pd.isna(season):
                continue
            rec = {"team": team, "season": season}
            for f, c in fcols.items():
                rec[f] = df.iloc[i, c]
            recs.append(rec)
    out = pd.DataFrame(recs)
    for c in ["season", "cap_used", "dead_cap", "cap_space"]:
        out[c] = _num(out[c])
    return out


_ROOKIE_LABELS = {
    "draft year": "draft_year",
    "player": "player",
    "drafted salary": "drafted_salary",
    "true salary": "true_salary",
    "true years remain": "true_years_remain",
}


def parse_rookies(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rookie deals: drafted vs. true salary, option flag, years remaining."""
    df = load_raw() if df is None else df
    hdr = _section_header_row(df, "ROOKIES")
    stop = _section_header_row(df, "CAP USED")
    header = df.iloc[hdr]
    recs = []
    for team, start, end in _team_spans(df):
        fcols = _field_columns(header, start, end, _ROOKIE_LABELS)
        if "player" not in fcols:
            continue
        for i in range(hdr + 1, stop):
            raw = df.iloc[i, fcols["player"]]
            if pd.isna(raw) or not str(raw).strip():
                continue
            player, note = _clean_player(raw)
            if not player or player.lower() == "nan":
                continue
            rec = {"team": team, "player": player, "option": str(df.iloc[i, start]).strip()}
            for f in ("draft_year", "drafted_salary", "true_salary", "true_years_remain"):
                if f in fcols:
                    rec[f] = df.iloc[i, fcols[f]]
            recs.append(rec)
    out = pd.DataFrame(recs)
    for c in ("draft_year", "drafted_salary", "true_salary", "true_years_remain"):
        out[c] = _num(out[c])
    return out


def rookie_season_salary(row: pd.Series, season: int) -> float:
    """Cap hit for a rookie in a given season (0 if not under contract).

    3-year deal from draft_year; if the option was exercised (true > drafted, or
    option flag 'y'), a 4th year is added and years 3-4 are paid the true salary.
    """
    start = row.get("draft_year")
    if pd.isna(start):
        return 0.0
    start = int(start)
    opt = str(row.get("option", "")).strip().lower().startswith("y")
    length = 4 if opt else 3
    if not (start <= season <= start + length - 1):
        return 0.0
    year_index = season - start + 1
    if opt and year_index >= 3 and not pd.isna(row.get("true_salary")):
        return float(row["true_salary"])
    drafted = row.get("drafted_salary")
    return float(drafted) if not pd.isna(drafted) else 0.0


_IR_LABELS = {
    "contract slot": "contract_slot",
    "player": "player",
    "replacement player": "replacement_player",
    "original player's salary": "original_salary",
    "yrs left (not incl this year)": "yrs_left_excl",
}


def parse_ir(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """IR players: the injured player, their replacement, and original salary."""
    df = load_raw() if df is None else df
    hdr = _section_header_row(df, "IR")  # header row is the one above the IR rows
    header_row = hdr - 1  # the 'contract slot / replacement player' labels
    stop = _section_header_row(df, "Amnesty")
    header = df.iloc[header_row]
    recs = []
    for team, start, end in _team_spans(df):
        fcols = _field_columns(header, start, end, _IR_LABELS)
        if "player" not in fcols:
            continue
        for i in range(hdr, stop):
            tag = str(df.iloc[i, start]).strip()
            if tag not in ("IR", "PSquad"):
                continue
            raw = df.iloc[i, fcols["player"]]
            if pd.isna(raw) or not str(raw).strip():
                continue
            player, _ = _clean_player(raw)
            rec = {"team": team, "designation": tag, "player": player}
            for f in ("replacement_player", "original_salary", "yrs_left_excl", "contract_slot"):
                if f in fcols:
                    rec[f] = df.iloc[i, fcols[f]]
            recs.append(rec)
    out = pd.DataFrame(recs)
    for c in ("original_salary", "yrs_left_excl"):
        if c in out:
            out[c] = _num(out[c])
    return out


_CUTS_LABELS = {
    "season cut in": "season_cut",
    "yrs left (1-5)": "yrs_left",
    "player": "player",
    "salary owed": "salary_owed",
}


def parse_cuts(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cut players and their dead-cap 'salary owed' (raw; semantics TBD).

    Captures the unlabeled leading number (block's first column) as ``lead_num``.
    Rows whose player is a 'CAP HITS' subtotal are kept but flagged.
    """
    df = load_raw() if df is None else df
    hdr = _section_header_row(df, "Cuts")
    try:
        stop = _section_header_row(df, "DRAFT PICKS:")
    except ValueError:
        stop = len(df)
    header = df.iloc[hdr]
    recs = []
    for team, start, end in _team_spans(df):
        fcols = _field_columns(header, start, end, _CUTS_LABELS)
        if "player" not in fcols:
            continue
        for i in range(hdr + 1, stop):
            raw = df.iloc[i, fcols["player"]]
            if pd.isna(raw) or not str(raw).strip():
                continue
            player = str(raw).strip()
            rec = {
                "team": team,
                "lead_num": df.iloc[i, start],
                "player": player,
                "is_cap_hits_row": player.upper().startswith("CAP HITS"),
            }
            for f in ("season_cut", "yrs_left", "salary_owed"):
                if f in fcols:
                    rec[f] = df.iloc[i, fcols[f]]
            recs.append(rec)
    out = pd.DataFrame(recs)
    for c in ("lead_num", "season_cut", "yrs_left", "salary_owed"):
        if c in out:
            out[c] = _num(out[c])
    return out


# --- Reconciliation ------------------------------------------------------------

def reconcile(season: int) -> pd.DataFrame:
    """Reconstruct CAP USED for a season from the parts we understand vs the sheet.

    Components: active contracts (flat salary, season-appropriate) + rookie
    salaries (+ extensions/IR for 2026). Dead cap (cuts) is NOT added, so the
    residual ≈ each team's dead cap for that season.
    """
    active = parse_active_contracts()
    active["years_remaining"] = _num(active["years_remaining"])
    active["salary"] = _num(active["salary"])
    rookies = parse_rookies()
    caps = parse_cap_summary().query("season == @season").set_index("team")["cap_used"]

    rows = []
    for team in TEAMS:
        a = active[active["team"] == team]
        # Active in `season`: years_remaining (as of CURRENT_SEASON) must reach it.
        active_sal = a.loc[
            a["years_remaining"] >= (season - CURRENT_SEASON + 1), "salary"
        ].sum()

        rk = rookies[rookies["team"] == team]
        rookie_sal = sum(rookie_season_salary(r, season) for _, r in rk.iterrows())

        extra = 0.0
        note = ""
        if season >= UPCOMING_SEASON:
            ext = parse_extensions()
            e = ext[ext["team"] == team]
            ext_players = set(parse_extensions()["player"])
            # active sum above double-counts extended players at original salary;
            # remove them and add the extension salary instead.
            active_sal = a.loc[
                (a["years_remaining"] >= (season - CURRENT_SEASON + 1))
                & (~a["player"].isin(ext_players)),
                "salary",
            ].sum()
            extra = _num(e["salary"]).sum()
            note = "active(excl ext)+ext"

        recon = active_sal + rookie_sal + extra
        sheet = caps.get(team, float("nan"))
        rows.append(
            {
                "team": team,
                "active": round(active_sal, 1),
                "rookies": round(rookie_sal, 1),
                "ext": round(extra, 1),
                "reconstructed": round(recon, 1),
                "sheet_cap_used": sheet,
                "residual (≈dead cap)": round(sheet - recon, 1),
            }
        )
    out = pd.DataFrame(rows)
    return out


if __name__ == "__main__":
    for yr in (CURRENT_SEASON, UPCOMING_SEASON):
        print(f"\n===== {yr} CAP USED reconciliation (residual ≈ dead cap) =====")
        r = reconcile(yr)
        print(r.to_string(index=False))
        print(f"  league totals: reconstructed={r['reconstructed'].sum():.1f}, "
              f"sheet={r['sheet_cap_used'].sum():.1f}, "
              f"residual={r['residual (≈dead cap)'].sum():.1f}")
