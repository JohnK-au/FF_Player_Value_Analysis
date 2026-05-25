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


def parse_tags(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Franchise tags (the TAG row is both header and data; placeholders skipped)."""
    df = load_raw() if df is None else df
    hdr = _section_header_row(df, "TAG")
    recs = []
    for team, start, end in _team_spans(df):
        row = [df.iloc[hdr, c] for c in range(start, end)]
        ly_idx = None
        for k, v in enumerate(row):
            vn = pd.to_numeric(v, errors="coerce")
            if pd.notna(vn) and 2024 <= vn <= 2031:
                ly_idx = k
        if ly_idx is None or ly_idx < 2:
            continue
        player = str(row[ly_idx - 2]).strip()
        if not player or "put here" in player.lower() or player.lower() == "nan":
            continue
        recs.append(
            {
                "team": team,
                "player": player,
                "salary": pd.to_numeric(row[ly_idx - 1], errors="coerce"),
                "league_year": int(pd.to_numeric(row[ly_idx])),
                "tag_year": pd.to_numeric(row[1], errors="coerce"),
            }
        )
    return pd.DataFrame(recs)


# --- Dead cap & reconciliation -------------------------------------------------

# Existing cuts are charged at 20%/yr; the league's 50% rule isn't applied to them
# yet (per the manager). Use 0.50 for *new* cuts when projecting forward.
DEAD_CAP_RATE = 0.20

# Salary cap per team per season (confirmed: CAP USED + CAP SPACE = 1500).
CAP_TOTAL = 1500


def dead_cap(season: int, cuts: pd.DataFrame | None = None) -> pd.Series:
    """Per-team dead cap hitting ``season``.

    Each cut/penalty charges ``DEAD_CAP_RATE`` × salary-owed per year across its
    window ``[season_cut, season_cut + yrs_left - 1]`` (CAP HITS admin penalties
    included, same rate).
    """
    cuts = parse_cuts() if cuts is None else cuts
    c = cuts[cuts["season_cut"].notna() & cuts["yrs_left"].notna()].copy()
    active = (c["season_cut"] <= season) & (season <= c["season_cut"] + c["yrs_left"] - 1)
    hit = DEAD_CAP_RATE * c["salary_owed"].where(active, 0.0)
    return hit.groupby(c["team"]).sum().reindex(TEAMS).fillna(0.0)


def _dead_cap_split(season: int) -> tuple[pd.Series, pd.Series]:
    """Dead cap hitting ``season``, split into player cuts vs admin penalties."""
    c = parse_cuts()
    c = c[c["season_cut"].notna() & c["yrs_left"].notna()].copy()
    c = c[(c["season_cut"] <= season) & (season <= c["season_cut"] + c["yrs_left"] - 1)]
    cut = (
        DEAD_CAP_RATE
        * c[~c["is_cap_hits_row"]].groupby("team")["salary_owed"].sum()
    ).reindex(TEAMS).fillna(0.0)
    pen = (
        DEAD_CAP_RATE
        * c[c["is_cap_hits_row"]].groupby("team")["salary_owed"].sum()
    ).reindex(TEAMS).fillna(0.0)
    return cut, pen


def cap_breakdown(season: int) -> pd.DataFrame:
    """Per-team salary-cap composition for ``season`` (for figures).

    Columns: the salary components (contracts/extensions/rookies/tags/
    practice_squad/cuts/penalties), the sheet's CAP USED, and CAP SPACE
    (= 1500 − CAP USED). Components reconcile to CAP USED within a small
    adjustment (trade credits + IR-returns), which figures show as a residual.
    """
    active = parse_active_contracts()
    active["years_remaining"] = _num(active["years_remaining"])
    active["salary"] = _num(active["salary"])
    rookies = parse_rookies()
    ext = parse_extensions()
    tags = parse_tags()
    ir = parse_ir()
    caps = parse_cap_summary().query("season == @season").set_index("team")
    cut_dc, pen_dc = _dead_cap_split(season)
    ext_players = set(ext["player"])
    need = season - CURRENT_SEASON + 1
    psq = ir[ir["designation"] == "PSquad"].copy()
    psq["end"] = CURRENT_SEASON + _num(psq["yrs_left_excl"]).fillna(0)

    rows = []
    for team in TEAMS:
        a = active[active["team"] == team]
        if season >= UPCOMING_SEASON:
            contracts = a.loc[
                (a["years_remaining"] >= need) & (~a["player"].isin(ext_players)),
                "salary",
            ].sum()
            extensions = _num(ext.loc[ext["team"] == team, "salary"]).sum()
        else:
            contracts = a.loc[a["years_remaining"] >= need, "salary"].sum()
            extensions = 0.0
        rook = sum(
            rookie_season_salary(r, season)
            for _, r in rookies[rookies["team"] == team].iterrows()
        )
        tag = _num(
            tags.loc[(tags["team"] == team) & (tags["league_year"] == season), "salary"]
        ).sum()
        ps = _num(
            psq.loc[(psq["team"] == team) & (psq["end"] >= season), "original_salary"]
        ).sum()
        sheet_used = caps.loc[team, "cap_used"] if team in caps.index else float("nan")
        rows.append(
            {
                "team": team,
                "contracts": round(contracts, 1),
                "extensions": round(extensions, 1),
                "rookies": round(rook, 1),
                "tags": round(tag, 1),
                "practice_squad": round(ps, 1),
                "cuts": round(cut_dc[team], 1),
                "penalties": round(pen_dc[team], 1),
                "sheet_used": round(sheet_used, 1),
                "cap_space": round(CAP_TOTAL - sheet_used, 1),
            }
        )
    return pd.DataFrame(rows)


def player_salaries_2026() -> pd.DataFrame:
    """Per-player 2026 cap salary (active + extensions + rookies + practice squad),
    joined to position group via the player crosswalk."""
    from .contracts import build_2026_contracts
    from .players import position_group_map

    combined, _ = build_2026_contracts()  # active + extensions, salary_2026
    parts = [combined[["team", "player", "salary_2026", "source"]]]

    rk = parse_rookies()
    rk["salary_2026"] = [rookie_season_salary(r, UPCOMING_SEASON) for _, r in rk.iterrows()]
    parts.append(
        rk.loc[rk["salary_2026"] > 0, ["team", "player", "salary_2026"]].assign(source="rookie")
    )

    ir = parse_ir()
    ps = ir[ir["designation"] == "PSquad"].copy()
    ps["salary_2026"] = _num(ps["original_salary"])
    parts.append(
        ps.loc[ps["salary_2026"] > 0, ["team", "player", "salary_2026"]].assign(source="practice_squad")
    )

    allp = pd.concat(parts, ignore_index=True).drop_duplicates(["team", "player"])
    pg = position_group_map()
    allp["position_group"] = allp["player"].map(pg).fillna("Other")
    return allp


def salary_by_position() -> pd.DataFrame:
    """Team × position-group matrix of 2026 player salary."""
    allp = player_salaries_2026()
    return (
        allp.groupby(["team", "position_group"])["salary_2026"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(TEAMS)
    )


def reconcile(season: int) -> pd.DataFrame:
    """Reconstruct each team's CAP USED for a season vs the sheet's own figure.

    Components: active contracts + rookies + franchise tags + dead cap + a
    trade cap-adjustment (the sheet's DEAD CAP column); for 2026+ extended
    players are swapped to their extension salary.
    """
    active = parse_active_contracts()
    active["years_remaining"] = _num(active["years_remaining"])
    active["salary"] = _num(active["salary"])
    rookies = parse_rookies()
    ext = parse_extensions()
    tags = parse_tags()
    ir = parse_ir()
    caps = parse_cap_summary().query("season == @season").set_index("team")
    dc = dead_cap(season)
    ext_players = set(ext["player"])
    need = season - CURRENT_SEASON + 1  # years_remaining (as of 2025) to reach season

    # Practice-squad salaries DO count against the cap (the IR player's does not —
    # its replacement, already on the active roster, counts instead).
    psq = ir[ir["designation"] == "PSquad"].copy()
    psq["end"] = CURRENT_SEASON + _num(psq["yrs_left_excl"]).fillna(0)

    rows = []
    for team in TEAMS:
        a = active[active["team"] == team]
        if season >= UPCOMING_SEASON:
            active_sal = a.loc[
                (a["years_remaining"] >= need) & (~a["player"].isin(ext_players)),
                "salary",
            ].sum()
            ext_sal = _num(ext.loc[ext["team"] == team, "salary"]).sum()
        else:
            active_sal = a.loc[a["years_remaining"] >= need, "salary"].sum()
            ext_sal = 0.0
        rook = sum(
            rookie_season_salary(r, season)
            for _, r in rookies[rookies["team"] == team].iterrows()
        )
        tag_sal = _num(
            tags.loc[(tags["team"] == team) & (tags["league_year"] == season), "salary"]
        ).sum()
        ps_sal = _num(
            psq.loc[(psq["team"] == team) & (psq["end"] >= season), "original_salary"]
        ).sum()
        trade_adj = (
            caps.loc[team, "dead_cap"]
            if team in caps.index and pd.notna(caps.loc[team, "dead_cap"])
            else 0.0
        )
        recon = active_sal + rook + ext_sal + tag_sal + ps_sal + dc[team] + trade_adj
        sheet = caps.loc[team, "cap_used"] if team in caps.index else float("nan")
        rows.append(
            {
                "team": team,
                "active": round(active_sal, 1),
                "rookie": round(rook, 1),
                "ext": round(ext_sal, 1),
                "tag": round(tag_sal, 1),
                "ps": round(ps_sal, 1),
                "dead": round(dc[team], 1),
                "trade": round(trade_adj, 1),
                "recon": round(recon, 1),
                "sheet": sheet,
                "resid": round(sheet - recon, 1),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for yr in (CURRENT_SEASON, UPCOMING_SEASON):
        print(f"\n===== {yr} CAP USED reconciliation =====")
        r = reconcile(yr)
        print(r.to_string(index=False))
        print(
            f"  league totals: recon={r['recon'].sum():.1f}, "
            f"sheet={r['sheet'].sum():.1f}, resid={r['resid'].sum():.1f}"
        )
