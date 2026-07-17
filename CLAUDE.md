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

## Current state — as of 2026-07-17

**All shipped and merged to `main`** (through PR #17): the V2 six-component
framework, the cap-unit pricing engine, and the Streamlit rebuild. **V1 engine
deleted** in `c049e99` — code lives in git history if it ever needs reviving
(see "V1 history" under [Next steps](#next-steps)).

### V2 six-component framework — Phase 4.5 v2 complete (commit `213c9db`)

The six-component scoring framework is fully implemented for all four skill
positions (WR/RB/QB/TE) on the FA-inclusive player universe. Master table:
[`data/processed/player_value_v2_2026.csv`](data/processed/player_value_v2_2026.csv)
(490 priced players: 155 rostered + 335 dynasty-league FAs).

| Component | Status | Module / spec |
| --- | --- | --- |
| Production | ✅ live (per-position predicted PPG → [0, 100], recency-weighted) | [`src/models/components/production.py`](src/models/components/production.py) · [production.md](docs/methodology/production.md) |
| Team | ✅ live (per-position empirical residual regression → multiplier band; WR 0.875-1.125, RB 0.85-1.15, TE 0.90-1.10, QB 0.95-1.05) | [`team.py`](src/models/components/team.py) · [team.md](docs/methodology/team.md) |
| On-Field Value (OFV) | ✅ live (= Production × Team multiplier; written to master CSV) | [`combine.py`](src/models/components/combine.py) · [combination.md](docs/methodology/combination.md) |
| Age | ✅ live (logistic decay sigmoid per position) | [`age.py`](src/models/components/age.py) · [age.md](docs/methodology/age.md) |
| Injury | ✅ live (durability score from games played + IR designation) | [`injury.py`](src/models/components/injury.py) · [injury.md](docs/methodology/injury.md) |
| Position | ✅ **v2 locked** — VORP-Deep Total Impact (PPG-based; RB=100, WR=93.1, TE=8.6, QB=0) | [`position.py`](src/models/components/position.py) · [position.md](docs/methodology/position.md) |
| Intangibles | 🟡 stub at neutral 50 (reserved for trade-target / coaching-fit signal) | [`intangibles.py`](src/models/components/intangibles.py) · [intangibles.md](docs/methodology/intangibles.md) |
| Dynasty Value combine | ✅ live (OFV 0.55 + Age 0.20 + Injury 0.15 + Position 0.05 + Intangibles 0.05) | [`combine.py`](src/models/components/combine.py) · [combination.md](docs/methodology/combination.md) |

**Position component v2 design decision (2026-06-28):** v1 used OFV-based
4-sub-metric composite, but OFV is normalized within position so cross-position
M comparisons were meaningless (TE looked artificially scarce). v2 fix: re-derive
M in absolute 2025 PPG with **VORP-Deep replacement** (avg PPG of ranks 3N+1 to
4N — the "deep FA tier"), T-only weighting (T = M × S). Result: WR climbs from
35→93 (now near-tied with RB), TE drops from 21→8.6 (its absolute PPG gap is
genuinely small), QB stays at 0 (smallest gap × 1 slot). See
[position.md](docs/methodology/position.md) for the full variant exploration and
sub-metric snapshot. **Multi-tier (elite/startable/bench/reserve) concentration
is deferred as an overlay/filter on top of results, NOT recomputed into the
score** ([[multi-tier-position-overlay]]).

### Cap-unit pricing engine — Phase 5.5 complete (commit `444a7d6`)

Layered on top of Dynasty Value. `src/models/pricing.py` runs the 4-stage
pipeline (per-position replacement baseline → above-baseline DV → non-linear
scarcity via α exponent → rate × age multiplier × multi-year age decay). See
[pricing.md](docs/methodology/pricing.md) for the full spec.

**Locked v1 parameters:**
- basis: `dynasty_value`
- pool_method: `empirical`, pool_scale: **1.45** (model implies ~30% league underspend)
- baselines (user-picked): QB=41, RB=29, WR=34, TE=31
- α: **1.25** (modest non-linear elite premium)
- age_band: [0.85, 1.15]

Output: `data/processed/player_pricing_2026.csv` — joins to V2 master on `espn_id`.
Sign convention matches original app: positive surplus = overpaid (red).

### Streamlit rebuild — Phase 6 complete

`src/app/` fully rebuilt on the V2+pricing stack. Loader (`_lib.py`) joins the
V2 master and pricing CSVs into a single unified DataFrame; Home + 5 pages:

- **Home** ([`Home.py`](src/app/Home.py)) — over/under board with pos/team/status/name
  filters and horizon toggle (2026 vs dynasty).
- **Player Card** ([`pages/1_Player_Card.py`](src/app/pages/1_Player_Card.py)) —
  6-component quality bars with league / top-3 / replacement overlays; pricing
  decomposition; multi-year projection with per-year age decay; top-10 at position.
- **Roster** ([`pages/2_Roster.py`](src/app/pages/2_Roster.py)) — cap summary,
  per-position breakdown (startable count / net surplus), DROP/TAG/EXTEND/KEEP
  recommendations via V2 `recommend_action` thresholds.
- **Trade Evaluator** ([`pages/3_Trade.py`](src/app/pages/3_Trade.py)) — two-sided
  swap with per-side totals + net delta strip + verdict box.
- **Auction Bidder** ([`pages/4_Auction.py`](src/app/pages/4_Auction.py)) — FA pool
  filtered by fair max-bid; per-position tabs; roster-fit sidebar with cap-space
  budget guide.
- **Compare** ([`pages/5_Compare.py`](src/app/pages/5_Compare.py)) — "This or That"
  head-to-head: pick the more valuable player across three lenses (2026 /
  Dynasty / real-life NFL). Winner-stays king-of-the-hill; anonymous mode hides
  name + team + salary for stat-blind evaluation. Picks persist to
  `data/research/user_comparisons.csv`, comments to `player_comments.csv` (both
  allowlisted in `.gitignore`).

Run with `streamlit run src/app/Home.py`.

### Data foundation (unchanged by V1 archival)

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

# V2 framework + pricing
python -m src.models.components.framework  # build V2 master CSV (490 x 6 components)
python -m src.models.pricing               # build pricing CSV (fair values + surplus)

# Workshop tools (param/weight tuning)
python -m src.viz.position_components WR   # per-position component grid HTML
python -m src.viz.cross_position_variants  # Position-component weight variants
python -m src.viz.combine_variants         # Dynasty Value combine variants
python -m src.viz.pricing_variants         # pricing pipeline preset variants

# App (V2 Streamlit rebuild)
streamlit run src/app/Home.py
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

**Done so far (V2 line):** V2 six-component framework (Production / Team / OFV /
Age / Injury / Position / Intangibles → Dynasty Value); cap-unit pricing engine
(4-stage pipeline: replacement baseline → non-linear scarcity → age-adjusted
multi-year decay); Streamlit rebuild (Home / Player Card / Roster / Trade /
Auction / Compare) reading V2 master + pricing. V1 engine deleted 2026-07-05;
history preserved in git.

**V1 history (superseded — do not treat as current).** The pre-V2 engine was
production-anchored VOR pricing (`salary~features` as a secondary "market" lens),
a per-player context module, a next-season **projection** model
(`src/models/projection.py`, OOF R² 0.48 on 1,167 transition pairs), and a
`player_value_2026.csv` master read by a 5-page app including a Market Driver
page. **All of it was deleted in `c049e99` (Phase 7, 2026-07-05)** — the modules,
the CSV, and the pages no longer exist. The design rationale is preserved in the
git history and in [docs/methodology/](docs/methodology/); reach for
`git show c049e99^:<path>` if any of it needs reviving (the projection model is
the most likely candidate — nothing in V2 currently predicts *next* season).

**WR weekly research + PyTorch scaffold (landed via PR #14).** Exploratory
week-level WR model + archetype tree built on per-week pbp aggregation + NGS
weekly + Vegas lines + rolling defense + rolling player history; plus a PyTorch
sequence-model scaffold on top of it. Findings in
[docs/research/wr_weekly_archetypes.md](docs/research/wr_weekly_archetypes.md)
and [docs/research/wr_weekly_torch.md](docs/research/wr_weekly_torch.md); code in
[`src/research/`](src/research/). Headline: per-week **predictive** R² is tiny
(0.024) — single weeks are high-variance — but the **descriptive** ceiling is
R² 0.68 (knowing role + efficiency that week), and aggregating the weekly model
to a season hits R² 0.68 on PPG. Counterintuitive finding: aDOT alone is slightly
*negative* for fantasy production; the dominant archetype is "volume + accurate
QB + good YAC", not "deep threat + accurate QB". Next: extend to RB/TE/QB.

> ⚠️ **Re-baseline that 0.68 before trusting it.** The research doc frames it as
> "0.68 vs 0.48, wins handily" and attributes the 0.48 to `models/production.py`.
> That's a misattribution: **0.48 was the deleted `projection.py`** (next-season
> prediction — a strictly harder task, as the doc's own caveat notes), not the
> Production component. The live same-season Production model is **WR R² 0.816**
> (see below), which is *higher* than the weekly model's 0.68. The weekly work may
> still be valuable as an **in-season** tool — that's a genuinely different job
> from what Production does — but the "wins handily" comparison as written does
> not hold. Flagged 2026-07-17; not corrected in the research doc itself.

**Live model baselines (measured 2026-07-17, `position_oof_r2`):** the Production
component is a per-position HistGBR — **QB R² 0.688 · RB 0.829 · WR 0.816 ·
TE 0.776**. These are player-leaky (KFold over player-seasons; the same player
appears on both sides of a split), so they are comparable between models on
identical folds but optimistic as absolute accuracy. See
[production.md](docs/methodology/production.md).

**Immediate next steps when resuming (priority order):**
1. **Use it and iterate.** The full V2 stack (framework + pricing + Streamlit) is live.
   Open `streamlit run src/app/Home.py`, explore real decisions, feed back signals.
   When a fair value looks suspicious, trace the chain:
   `dynasty_value` → subtract `replacement_dv` → `above_baseline_dv` → `^ alpha` →
   `× rate` → `× age_mult` = `fair_value_2026`. See [pricing.md](docs/methodology/pricing.md).
2. **UI polish + additional Streamlit features.** The V2 rebuild is prototyping-stage.
   Known refinements: TAG salary formula (rules §8) shown on Roster page, draft picks
   + extension rights on Trade page, Player Card charting polish, per-position drilldowns.
3. **Pedigree signal** for Intangibles (currently stub at 50). User's earlier observation
   that Alec Pierce ≠ AJ Brown at similar DV points to a missing consistency/history
   signal. Belongs in the Intangibles component. Future work.
4. **Later refinements**: extend historical seasons beyond 2016-25 for a more stable
   training basis; reconstruct weekly skill scoring from nflverse so FA pool gets a
   real consistency factor; trade reconciliation (active roster still lags trades);
   multi-tier position overlay UI ([[multi-tier-position-overlay]]).
5. **Known modelling debt** (found 2026-07-16/17, none of it breaking):
   - `receptions` is wanted by the RB Production model but absent from the
     training frame — adding it shifts the DV distribution and therefore
     silently miscalibrates `pricing.USER_BASELINES` + `pool_scale`. Needs its
     own pass with a re-calibration ([production.md](docs/methodology/production.md)).
   - The Production CV is **player-leaky** (`KFold` over player-seasons);
     `GroupKFold` on `espn_id` would give honest numbers and re-baseline the docs.
   - `extended_training_frame()` is uncached and rebuilt ~10× per `framework.py`
     run (~200 nflverse round-trips) — pure perf, no behaviour change.

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
