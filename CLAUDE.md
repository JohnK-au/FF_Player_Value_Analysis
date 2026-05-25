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

## Current state — as of 2026-05-25

Both data sources are wired up and verified end-to-end:

| Piece | Status | Where |
| --- | --- | --- |
| Project scaffolding (README, gitignore, requirements) | ✅ done | repo root |
| Contract-sheet ingestion (Option A; workbook read **by tab name** via xlsx) | ✅ working | [src/data/sheets.py](src/data/sheets.py) |
| ESPN league ingestion (authenticated via `espn-api`) | ✅ working | [src/data/espn.py](src/data/espn.py) |
| Parse contract sheet → tidy table (**active rosters**) | ✅ working | [src/data/contracts.py](src/data/contracts.py) |
| Parse **Contract Extensions** tab → tidy table | ✅ working | [src/data/contracts.py](src/data/contracts.py) |
| Parse MCS sections: cap summary, rookies, tags, IR, cuts | ✅ working (picks pending) | [src/data/cap.py](src/data/cap.py) |
| 2026 contract view (active rolled forward + extensions) | ✅ working | `contracts.build_2026_contracts` |
| Per-team contract timeline figure | ✅ working | [src/viz/contracts.py](src/viz/contracts.py) |
| Cap ledger reconciled to sheet CAP USED | ✅ 2025 exact (7/8), 2026 close | `cap.reconcile` |
| Refine residuals (IR returns in 2026; rookie option edges) | ⬜ minor | — |
| Joining performance + contracts | ⬜ not started | — |
| ML models | ⬜ not started | — |
| League rules documentation | 🟨 drafted (needs confirmation) | [docs/rules.md](docs/rules.md) |

`src/data/espn.py` was verified pulling the **2025** season (8 teams, records).
`src/data/sheets.py` caches the contract tab to `data/raw/contracts_gid0.csv`.
`src/data/contracts.py` parses the **active-roster** section into a tidy
`(team, player, salary, years_remaining, …)` table (168 contracts, 8 teams) and
writes `data/processed/contracts_active.csv`. It locates columns by header label
(not fixed offsets) because the leftmost block carries an extra column. Known
sheet data quirk it surfaces: Justin Fields & Christian Kirk (Haft) have a broken
`years_remaining` formula (blank "season @ acquisition" → shows `-2024`).

**Key league facts** (full rules in [docs/rules.md](docs/rules.md)):
- The workbook has 11 tabs but **only three matter**: `Master Cap Sheet` (tab 0,
  `gid=0` — the one parsed), `Trade Log`, and `Contract Extensions`. The rest
  (yearly rookie drafts / free agents / defunct) are ignored.
- **Active/upcoming season is 2026** — center the analysis there.
- Salary cap is **1500 units/team/season, fixed** (confirmed). Roster = 14
  starters + 14 bench = **28 spots = the 28 veteran contract slots** (pool:
  5×1yr, 5×2yr, 7×3yr, 6×4yr, 5×5yr), + 4 IR, + 1 offline practice-squad player.
  Cut dead cap = a % of full salary per remaining year (**rule 50%, but existing
  cuts still at legacy 20%** — not yet converted); **amnesty** wipes one cut
  penalty-free once every 3 seasons. `cap.reconcile()` rebuilds CAP USED from
  active+rookies+tags+dead-cap+trade-adj and matches the sheet (2025 near-exact).
- Ingestion reads the workbook as **xlsx and caches the 3 tabs by name**
  (`sheets.cache_tabs()` → `data/raw/contracts_<key>.csv`, keys `master_cap`,
  `trade_log`, `contract_extensions`); `sheets.load_tab(key)` loads them.
- **Source of truth (confirmed):** when the active roster and the Contract
  Extensions tab disagree on a player's team, the **Extensions tab is current** —
  the active-roster section lags trades. `build_2026_contracts` already treats
  extensions as authoritative and returns mismatches as notes.
  **Caveat:** traded-but-*not*-extended players may still sit under their old team
  on the active roster, and the Trade Log is too sparse (1 entry) to correct them.

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
python -m src.data.sheets    # caches the 3 relevant tabs to data/raw/
python -m src.data.contracts # parse active rosters + extensions -> data/processed/
python -m src.data.cap       # parse cap sections + reconcile CAP USED vs the sheet
python -m src.viz.contracts  # render figures/team_contracts_2026.png (git-ignored)
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

Section map (1-indexed rows of the Master Cap Sheet tab), discovered by
reading the full cached export:

| Section | Rows | Notes |
| --- | --- | --- |
| Active rosters | 2–29 | ✅ parsed by `contracts.py`; grouped by years-remaining tier 5→1 |
| TAG (franchise tags) | 30 | player, salary, league year |
| ROOKIES | 31–45 | option (y/n), draft year, drafted vs. true salary, true years remain |
| CAP summary | 46–51 | CAP USED / DEAD CAP / CAP SPACE per team, seasons 2025–2029 |
| IR / Practice Squad | 52–57 | includes the *replacement* player + original salary |
| Amnesty | 58–59 | season amnestied, player, next season allowed |
| Cuts (dead cap) | 60–117 | season cut, yrs left, player, salary owed; "CAP HITS … /5" subtotals |
| Draft picks | 119–159 | per team/year, with elaborate swap-condition notes |
| Trade log | 164–175 | a side table in the middle columns only |

Team nicknames in the sheet are `Nate, Seeb, Silv, Kerr, Will, Drew, Couc, Haft`
— these will need mapping to ESPN team ids for the eventual join.

## Next steps (suggested order)

1. ~~Parse the active-roster section into a tidy table.~~ ✅ done
   (`src/data/contracts.py`). **Remaining sheet parsing** (optional, do as needed):
   rookies, tags, IR/practice-squad, and cuts/dead-cap → fold into one tidy table
   with a `status` column, reusing the header-label-matching approach. Draft picks
   and the trade log are lower priority. Also add a sheet-nickname → ESPN-team-id
   map for the join.
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

## League rules

Rules are now substantially confirmed and documented in
[docs/rules.md](docs/rules.md) (living doc, tagged ✅/🟡/❓): cap, roster + slot
pool, the pre-season free-agency auction, rookie scale + 4th-year option,
franchise-tag formula, extensions, cuts/dead-cap, amnesty, and IR. Only a few
minor items remain open (how extension salary/length is set, draft order, trade
deadline). Scoring/lineup is read straight from ESPN, not transcribed.
