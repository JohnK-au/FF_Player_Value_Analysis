# Data Sources & Scraping

How the project pulls its two data domains and stitches them together:

1. **Contracts & cap** — from a Google Sheet (the league's offline cap system).
2. **Player stats & attributes** — from ESPN (fantasy points in our scoring) and
   nflverse (age, experience, advanced metrics).

Everything is joined on the ESPN player id (`espn_id`). All modules live in
[`src/data/`](../src/data); processed outputs are cached under `data/` (git-ignored).

> **Secrets & privacy.** This repo is public. No league identifiers or credentials
> are committed — they live only in the git-ignored `.env` (see
> [.env.example](../.env.example)). Generated CSVs and figures contain player
> names/salaries and are git-ignored too. See [Configuration](#configuration).

---

## Configuration

All sensitive values come from `.env` (copy `.env.example` and fill in):

| Variable | Used for |
| --- | --- |
| `CONTRACTS_SHEET_ID` | Google Sheet workbook id (contracts) |
| `CONTRACTS_DEFAULT_GID` | default tab gid for legacy CSV export |
| `ESPN_LEAGUE_ID` | ESPN fantasy league id |
| `ESPN_TEAM_ID` | the user's team id (8) |
| `ESPN_SEASON` | default season to read (e.g. 2025) |
| `ESPN_S2`, `ESPN_SWID` | ESPN auth cookies (the league is private) |

**Things that periodically break:**
- **ESPN cookies expire.** When ESPN calls return HTTP 401, refresh `ESPN_S2` /
  `ESPN_SWID` from a logged-in browser (DevTools → Application → Cookies →
  `.espn.com`) and update `.env`.
- **The contract sheet must stay shared "anyone with the link can view"** — the
  unauthenticated export ("Option A") relies on it.

---

## 1. Contracts & cap (Google Sheet)

### Ingestion — [`sheets.py`](../src/data/sheets.py)
The sheet is read with **no authentication** via Google's export endpoints
("Option A"). The whole workbook is pulled as **xlsx and indexed by tab name**
(no per-tab gids needed).

- `fetch_workbook()` → download the xlsx bytes.
- `read_tab(key)` → one tab as a raw header-less DataFrame.
- `cache_tabs()` → download once, cache the relevant tabs to
  `data/raw/contracts_<key>.csv`.
- `load_tab(key)` → load a tab (cached CSV if present, else fetch).
- `fetch_tab(gid)` → legacy CSV-by-gid export (ad-hoc use).

The workbook has 11 tabs but **only three matter** (`TABS`): `master_cap`
(Master Cap Sheet), `trade_log` (Trade Log), `contract_extensions`
(Contract Extensions).

### The Master Cap Sheet layout (important)
It's a **visual layout, not a tidy table**: eight team blocks side by side,
separated by blank columns, with stacked sections below. Columns are located by
their **header labels**, not fixed offsets, because the leftmost block (Nate)
carries an extra column and the Contract Extensions tab heads Nate's block `N8`.

Section map (1-indexed rows of the Master Cap Sheet):

| Section | Rows | Parsed by |
| --- | --- | --- |
| Active rosters | 2–29 | `contracts.parse_active_contracts` |
| Franchise tags | 30 | `cap.parse_tags` |
| Rookies | 31–45 | `cap.parse_rookies` |
| Cap summary (USED/DEAD/SPACE, 2025–29) | 46–51 | `cap.parse_cap_summary` |
| IR / practice squad | 52–57 | `cap.parse_ir` |
| Amnesty | 58–59 | — |
| Cuts (dead cap) | 60–117 | `cap.parse_cuts` |
| Draft picks | 119–159 | — |

### Parsing & cap model — [`contracts.py`](../src/data/contracts.py), [`cap.py`](../src/data/cap.py)
- `contracts.parse_active_contracts()` → tidy `(team, player, salary,
  years_remaining, …)` for active rosters.
- `contracts.parse_extensions()` → the Contract Extensions tab.
- `contracts.build_2026_contracts()` → roll active deals forward to 2026 and
  apply extensions (extensions are the source of truth on team disagreements).
- `cap.cap_breakdown(season)` / `cap.reconcile(season)` → per-team CAP USED from
  components (active + rookies + tags + practice squad + dead cap + trade adj),
  validated against the sheet's own figures (2025 reconstructs near-exactly).
- See [docs/rules.md](rules.md) for the cap rules these encode (cap = 1500/team;
  cuts = 20%/yr dead cap; etc.).

---

## 2. Player stats & attributes

### ESPN league — [`espn.py`](../src/data/espn.py)
`get_league(year)` returns an authenticated `espn-api` `League` (the league is
private → needs the cookies above). ESPN already computes fantasy points in the
league's **custom scoring**, so we read points directly rather than recomputing.

### Contract ↔ ESPN crosswalk — [`players.py`](../src/data/players.py)
Maps each contract-sheet player to an ESPN player to attach `position`,
`proTeam`, and **`espn_id`** (the key everything else joins on).
- `espn_player_table(year)` → rostered + free-agent pool (~1,700 players).
- `build_crosswalk()` → normalized + fuzzy name matching with a small `ALIASES`
  table and head-coach handling; **100% of contract players matched**. Cached to
  `data/processed/player_crosswalk.csv`.
- `attributes_table()` → crosswalk + nflverse age/experience.

### Performance — [`performance.py`](../src/data/performance.py)
Two grains, both in the league's scoring:
- **Season** (`season_player_points` / `pull_performance`) → full-NFL-season
  points / PPG / games for the whole pool, 2022–2025 → `data/processed/performance.csv`.
  Broad coverage, but spans the 17-game NFL season.
- **Weekly** (`weekly_player_points` / `pull_weekly_performance`) → per-player
  per-week points from box scores over weeks `1..reg_season_count` (**13** — the
  league's regular season) → `data/processed/performance_weekly.csv`. Restricts
  production to the weeks that count here, and supports consistency (stdev) via
  `fantasy_season_summary`. Covers players rostered in-league that week.

> ⚠️ Season totals include NFL weeks 14–18 that fall **after** this league's
> season; prefer the weekly (wk 1–13) numbers for value work.

### Attributes (nflverse) — [`nflverse.py`](../src/data/nflverse.py)
`player_attributes()` pulls age (as of the upcoming season) and experience from
nflverse seasonal rosters via `nfl_data_py`, joined on `espn_id` (~97% coverage;
head coaches absent, as expected).

### Unified dataset — [`dataset.py`](../src/data/dataset.py)
`build_player_dataset()` → one row per 2026 contract player: salary, position
group, age, experience, and recent production. Prefers **fantasy-season (wk 1–13)**
production and **falls back to full-season** where weekly is missing, with a
`prod_source` flag (`fantasy_wk1_13` / `full_season_proxy` / `none`). Cached to
`data/processed/player_dataset_2026.csv`.

---

## Cached outputs (all git-ignored)

| File | Produced by |
| --- | --- |
| `data/raw/contracts_master_cap.csv` (+ `trade_log`, `contract_extensions`) | `sheets.cache_tabs` |
| `data/processed/contracts_active.csv`, `contract_extensions.csv` | `contracts.py` |
| `data/processed/player_crosswalk.csv` | `players.build_crosswalk` |
| `data/processed/performance.csv`, `performance_weekly.csv` | `performance.py` |
| `data/processed/player_dataset_2026.csv`, `position_salary_2026.csv` | `dataset.py` / `cap.py` |

## Re-running the pipeline

```bash
python -m src.data.sheets       # cache the 3 contract tabs
python -m src.data.contracts    # parse active rosters + extensions
python -m src.data.cap          # parse cap sections + reconcile vs the sheet
python -m src.data.players      # build contract<->ESPN crosswalk (positions, espn_id)
python -m src.data.performance  # cache season + weekly (wk1-13) points 2022-2025
python -m src.data.dataset      # build the unified 2026 player dataset
```
