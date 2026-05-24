# CLAUDE.md

Working notes for Claude Code (and humans) on this repo. Keep this file current
as the project evolves; it's the fastest way to resume work in a new session.

## What this project is

ML analysis of **player performance** (from ESPN) combined with **contract
economics** (from a Google Sheet) for an 8-team **dynasty** fantasy football
league that runs an offline, NFL-style salary cap and contract system. Goal:
find over/under-valued players, project performance over contract length, and
support roster decisions under the cap. See [README.md](README.md) for the
fuller pitch.

- Public repo: `JohnK-au/FF_Player_Value_Analysis`
- Language: Python 3 (a local `.venv` is used for dependencies)

## Current state — as of 2026-05-24

Both data sources are wired up and verified end-to-end:

| Piece | Status | Where |
| --- | --- | --- |
| Project scaffolding (README, gitignore, requirements) | ✅ done | repo root |
| Contract-sheet ingestion ("Option A": unauthenticated CSV export) | ✅ working | [src/data/sheets.py](src/data/sheets.py) |
| ESPN league ingestion (authenticated via `espn-api`) | ✅ working | [src/data/espn.py](src/data/espn.py) |
| Parsing contract sheet into tidy data | ⬜ not started | — |
| Joining performance + contracts | ⬜ not started | — |
| ML models | ⬜ not started | — |
| League rules documentation | ⬜ not started | `docs/rules.md` (planned) |

`src/data/espn.py` was verified pulling the **2025** season (8 teams, records).
`src/data/sheets.py` caches the contract tab to `data/raw/contracts_gid0.csv`.

## Important constraints (read before changing anything)

1. **The repo is PUBLIC — never commit league identifiers or secrets.** All
   sensitive values live in the git-ignored `.env` (documented by
   [.env.example](.env.example)): the contract sheet ID, ESPN league/team IDs,
   and ESPN auth cookies. The sheet link was previously scrubbed from git
   history for this reason.
2. **The ESPN league is PRIVATE**, so the API requires the user's auth cookies
   (`ESPN_S2`, `ESPN_SWID`). These **expire periodically** — when calls return
   HTTP 401, refresh them from a logged-in browser (DevTools → Application →
   Cookies → `.espn.com`) and paste into `.env`.
3. **The contract Google Sheet must stay shared as "anyone with the link can
   view"** for Option A (no-auth CSV export) to keep working.
4. **Use the venv**: run commands as `.venv/bin/python ...` (or activate it).

## Setup / how to run

```bash
python3 -m venv .venv && source .venv/bin/activate   # first time
pip install -r requirements.txt                       # first time
cp .env.example .env                                  # then fill in real values

python -m src.data.espn      # prints the league's teams + records
python -m src.data.sheets    # caches the contract sheet to data/raw/
```

## Data-shape gotcha (contract sheet)

The contract sheet is a **visual layout, not a tidy table**: eight team blocks
sit side by side, each with columns
`[team, contract slot, player, salary, yrs left @ acquisition, season @ acquisition, yrs remaining]`,
separated by blank columns. The left edge carries `Season:` / year markers and a
years-remaining grouping (5 down to 1). Below the main rosters there are further
sections: tagged (franchise) players, rookies (drafted vs. true salary), a cap
summary (salary used / dead cap / cap space per team per season), and special
designations (IR, practice squad, amnesty, cut-with-retained-salary). Turning
this into tidy `(season, team, player, salary, years_remaining, status)` rows is
the main parsing task ahead.

## Next steps (suggested order)

1. **Parse the contract sheet into a tidy per-player table.** Recommended next —
   it's the join key everything else hangs off. Add e.g. `src/data/contracts.py`
   that reshapes `data/raw/contracts_gid0.csv` into a clean DataFrame, and write
   a few sanity checks (8 teams, salaries numeric, years in range).
2. **Pull richer ESPN data** beyond team records: weekly player scores, season
   stats, and ESPN's own projections (extra `espn-api` calls / API views).
3. **Build a unified player-value dataset** joining performance + contract cost
   (match players across ESPN names and the sheet's names — expect name-mismatch
   cleanup).
4. **Document the league rules** in `docs/rules.md` (needs clarification from the
   commissioner — see open questions). Encode the cap logic the analysis relies on.
5. **Baseline models**: value-vs-salary (which players beat their cap hit) and
   performance projection over remaining contract years.
6. **Roster optimization** under the salary cap (which keep/cut/tag/extend moves
   maximize value).

## Open questions to clarify (league rules)

- Salary cap amount and how it changes year to year.
- Franchise-tag rules (cost, eligibility, limits).
- Rookie contract scale ("drafted" vs "true" salary distinction).
- Dead cap, amnesty, and retained-salary mechanics.
- Contract length limits and extension/restructure rules.
- Scoring settings (PPR etc.) — likely readable from ESPN's settings view.
- Which season to center the analysis on (2025 completed vs 2026 upcoming).
