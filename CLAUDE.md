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

**Docs:** [docs/architecture.md](docs/architecture.md) — codebase layering & where
features land; [docs/data_sources.md](docs/data_sources.md) — data-scraping
reference; [docs/data_dictionary.md](docs/data_dictionary.md) — data assets, scope,
& stat/acronym glossary; [docs/figures.md](docs/figures.md) — figure catalog (organized under
`figures/{contracts,cap,value}/`); [docs/rules.md](docs/rules.md) — league cap
rules; [docs/analysis_plan.md](docs/analysis_plan.md) — the ML roadmap.

Shared paths & league constants live in [`src/config.py`](src/config.py).

- Public repo: `JohnK-au/FF_Player_Value_Analysis`
- Language: Python 3 (a local `.venv` is used for dependencies)

## Current state — as of 2026-05-27

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
| Cap-distribution figures (2026 + 2025–29 projection) | ✅ working | [src/viz/cap.py](src/viz/cap.py) |
| Contract ↔ ESPN player join (position, espn_id) | ✅ 100% matched | [src/data/players.py](src/data/players.py) |
| Player age/experience (nflverse) join | ✅ 97% (HCs excl.) | [src/data/nflverse.py](src/data/nflverse.py) |
| ESPN performance 2022–2025 (season + weekly wk1–13) | ✅ working | [src/data/performance.py](src/data/performance.py) |
| Unified 2026 player dataset (salary+age+production) | ✅ working | [src/data/dataset.py](src/data/dataset.py) |
| Advanced metrics (pbp/NGS/PFR/snaps + draft + combine) | ✅ working | [src/data/advanced.py](src/data/advanced.py), [nflverse.py](src/data/nflverse.py) |
| League-wide training frame (all NFL skill players × 2022–2025) | ✅ 2,659 rows | [src/data/population.py](src/data/population.py) |
| Production model (expected PPG) + replacement levels/VOR | ✅ OOF R² 0.80 | [src/models/production.py](src/models/production.py) |
| Fair-value model — **2 lenses**: production-anchored (VOR→$, downside-risk-adjusted) + market-fit | ✅ working | [src/models/value.py](src/models/value.py) |
| Cap ledger reconciled to sheet CAP USED | ✅ 2025 exact (7/8), 2026 close | `cap.reconcile` |
| Refine residuals (IR returns in 2026; rookie option edges) | ⬜ minor | — |
| Joining performance + contracts | ✅ done | [src/data/dataset.py](src/data/dataset.py) |
| ML models (production + fair value) | ✅ first cut done | [src/models/](src/models/) |
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
python -m src.viz.cap        # cap_distribution_2026 + cap_projection_2025_2029 + salary_by_position_2026
python -m src.data.players   # build contract<->ESPN crosswalk (positions, espn_id)
python -m src.data.performance # cache season + weekly (wk1-13) points 2022-2025
python -m src.data.dataset   # build unified 2026 player dataset + efficiency teaser
python -m src.data.population # build all-NFL-skill-players × 2022-2025 training frame
python -m src.data.context   # per-player baseline/delta/z + year_type (down/up/par diagnostic)
python -m src.models.production # production model (exp PPG, OOF R²) + replacement levels/VOR
python -m src.models.value   # both-lens fair value (production-anchored + market-fit) + over/under lists
python -m src.viz.value      # faceted value scatter (salary vs PPG; figures/value_facets_*.png)
python -m src.viz.value_interactive # interactive hover scatter -> figures/value_interactive.html
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

## Next steps

**Done so far:** contract/cap parsing + reconciliation; figures (contracts, cap,
salary-by-position, value scatters + a preliminary [value_summary_2026](src/viz/summary.py));
contract↔ESPN player join; ages; ESPN performance (season + weekly wk1–13); advanced
metrics (pbp/NGS/PFR/snaps) + **team/offense efficiency context** (`team_pass_epa`/`team_cpoe`/
`team_rush_epa`/`team_pass_rate`) + draft + combine (cached to Parquet); the **expanded
training set** (all NFL skill players × 2022–2025 = 2,659 rows, [population.py](src/data/population.py));
a **production model** (expected PPG, OOF **R² 0.81**, [production.py](src/models/production.py));
a **two-lens fair-value engine** ([value.py](src/models/value.py)); and a
[data dictionary](docs/data_dictionary.md). Full roadmap in [docs/analysis_plan.md](docs/analysis_plan.md).

**Current build (in progress):** an approved plan (saved at
`~/.claude/plans/ok-i-am-entering-curious-map.md`) to add per-player context (down/up/par),
a next-season **projection** model + age curves, **dynasty** value, and an interactive
**Streamlit** app (open-source/free; current AND dynasty value side by side; market views =
relationship explorer + driver ranking + what-if simulator + over-pay map, all positions).
Work is on branch **`value-engine-projection-app`** (pushed). Done on it: M1 data dictionary +
PPG policy ([[ppg-basis-policy]]); S1 team/offense context; **production-pricing fix**
(deep-baseline + multiplicative consistency factor — top fair 1,090→291, sub-baseline 84%→34%);
**S2 per-player longitudinal context** (`src/data/context.py`: prior/baseline/delta/z + a
`year_type` ∈ `{up, par, down, rookie, partial}` glance flag, no-leakage `shift(1).rolling`;
Jefferson 2025 = `down` + usage intact + results crashed + bad QB context = the rebound signature
we wanted, identified automatically). Preliminary summary figure.

**Key design decision (2026-05-27):** fair value is **anchored to objective
production (VOR), not fit to actual salaries.** A `salary ~ features` model learns
the market's *average* pricing, so it structurally cannot flag *systematic*
mispricing (e.g. the league overpaying elite QBs) — and the user believes the
market is inefficient. So the **primary lens is production-anchored**: value over
replacement (per the 8-team starting lineup, rules §4) priced into cap units by
redistributing the league's total skill-cap spend by VOR. The old `salary~features`
model is **kept as a secondary "market price" lens**; its R² is now a *diagnostic
we deliberately do NOT maximize* (a perfect fit would call everyone fair). The gap
between the two lenses is the signal. (Top output insight: in a 1-QB league
replacement QB ≈ 22 PPG, so Mahomes/Lamar at 100–117 read as heavily overpaid;
elite young RB/WR on cheap deals read as big bargains.) VOR is **risk-adjusted for
consistency**: `vor_adj = vor − 0.5·downside_deviation` (`value.RISK_LAMBDA`),
penalizing *bust* weeks (floor risk in a weekly H2H league) but **not** big ceiling
weeks — boom weeks must not hurt elite RBs like Gibbs/Bijan (user-confirmed).

**✅ Pricing fix (2026-05-27, see [[value-pricing-degenerate-fix]]):** the prior subtractive
risk penalty + $1 floor (84% of players floored to `prod_fair=1`, Puka ≈ 1,090) was replaced
with (a) a **multiplicative consistency factor** bounded in `[0.5, 1]` (volatile-but-startable
players are penalized but never zeroed) and (b) **deep-baseline pricing**: redistribute the cap
pool over `deep_vor = prod_adj − 0.5·replacement[pos]`. Sub-baseline players now get
`prod_fair = 0` (their salary registers as surplus). Result: rate 87→15, top fair 1,090→291,
sub-baseline 84%→34%, AJ Brown overpaid by 144 (fair 56), Jefferson by 132 (fair 18). The
remaining single-season distortions (Jefferson's down 2025) are what the projection (S2–S4) fixes.

**Immediate next steps when resuming (priority order):**
1. **S3 — next-season projection** ([projection.py] new) + age curves → consumes S2's context
   features as predictors; fixes single-season distortions (Jefferson/AJ Brown down years);
   enables dynasty value.
2. **S4 — value off projected production** (current + **dynasty** via `years_2026`) +
   consolidated `player_value_2026.csv`.
3. **Streamlit app** (M5–M6): market/driver explorer, over/under board, player value card,
   roster view + drop/extend/tag/keep, auction bid targets, trade evaluator.

Minor open items: trade reconciliation (active roster lags trades; Extensions tab
is authoritative); a couple of league-rule unknowns (extension salary-setting,
draft order). The user's team = sheet nickname **"Kerr"** (ESPN "Sydney Surf Sharks").

## League rules

Rules are now substantially confirmed and documented in
[docs/rules.md](docs/rules.md) (living doc, tagged ✅/🟡/❓): cap, roster + slot
pool, the pre-season free-agency auction, rookie scale + 4th-year option,
franchise-tag formula, extensions, cuts/dead-cap, amnesty, and IR. Only a few
minor items remain open (how extension salary/length is set, draft order, trade
deadline). Scoring/lineup is read straight from ESPN, not transcribed.
