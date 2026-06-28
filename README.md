# FF Player Value Analysis

Machine learning analysis of player performance and contract value for an
8-team dynasty fantasy football league that uses an offline, NFL-style
salary cap and contract system.

> **Status:** Active development. The original V1 fair-value engine (production-
> anchored VOR + market-fit lens) is shipped and powers an interactive Streamlit
> app. A **V2 six-component scoring framework** (Production, Team, On-Field
> Value, Age, Injury, Position) is now fully implemented across all four skill
> positions; integrating it into the app is the next milestone.

## Purpose

This project combines two data sources that are normally kept separate in
fantasy football:

1. **On-field performance** — weekly and seasonal stats (ESPN + nflverse
   pbp/NGS/PFR/snaps, all keyed on `espn_id`)
2. **Contract economics** — the league's custom salary-cap and multi-year
   contract system, parsed from a Google Sheet

…and uses ML to answer questions a real NFL front office would ask:

- Which players are **over- or under-valued** vs their salary and contract length?
- What is a player's **projected performance** over their remaining deal?
- How should **age curves, dead cap, tags, and IR** shape roster decisions?
- Which trades, extensions, cuts, or tags **maximize roster value** under the cap?

## The League

- **Format:** Dynasty (rosters carry over season to season). Active season: **2026**.
- **Source:** ESPN Fantasy Football (private league; requires user auth cookies).
- **Teams:** 8 (Nate, Seeb, Silv, Kerr, Will, Drew, Couc, Haft).
- **Cap:** 1,500 units/team/season, fixed.
- **Roster:** 14 starters + 14 bench (= 28 veteran contract slots) + 4 IR + 1 offline practice squad.

The cap rules (rookie scale + 4th-year option, franchise tag, extensions,
cuts/dead-cap, amnesty, IR) are documented and largely confirmed in
[`docs/rules.md`](docs/rules.md).

## What's working today

### Data foundation
- ✅ Google Sheet contract ingestion (active rosters + extensions + cap sections,
  reconciled against the sheet's CAP USED)
- ✅ ESPN league pull (8 teams; season + weekly wk 1–13 fantasy points)
- ✅ nflverse integration (ages, draft capital, combine, advanced metrics:
  EPA / NGS / PFR / snaps, windowed to wk 1–13)
- ✅ Contract ↔ ESPN crosswalk (100% matched, 242 players)
- ✅ Unified 2026 player dataset + NFL-wide training frame
  (2016–2025, ~5,600 player-seasons, fantasy scoring reconstructed from nflverse)

### Models — V1 fair-value engine (production)
- ✅ Production model (HistGBR per position, OOF R² 0.81)
- ✅ Next-season projection model (OOF R² 0.48) + positional age curves
- ✅ Two-lens fair value (production-anchored VOR with risk-adjusted consistency +
  market-fit) — `data/processed/player_value_2026.csv`
- ✅ Projection-based current + dynasty value (multi-year discount across `years_2026`)

### Models — V2 six-component framework ([`docs/methodology/`](docs/methodology/))
- ✅ **Production** (per-position predicted PPG → [0, 100], recency-weighted)
- ✅ **Team** (per-position empirical residual regression → multiplier in [0.875, 1.125])
- ✅ **On-Field Value** = Production × Team multiplier
- ✅ **Age** (logistic decay sigmoid per position)
- ✅ **Injury** (durability score from games played + IR designation)
- ✅ **Position** (cross-position importance via VOR-Deep Total Impact in absolute PPG —
  see [Phase 4.5 v2](docs/methodology/position.md))
- 🟡 **Intangibles** (stub at neutral 50; reserved for trade-target/coaching-fit signal)
- ✅ **Dynasty Value** combine = OFV 0.55 + Age 0.20 + Injury 0.15 + Position 0.05 + Intangibles 0.05
- Output: `data/processed/player_value_v2_2026.csv` (490 priced players: 155 rostered + 335 dynasty-league FAs)

### Interactive Streamlit app (`src/app/`)
- ✅ **Home** — over/under board with horizon toggle, filters, colored surplus styling
- ✅ **Player Card** — per-player deep-dive with multi-year projection chart + context
- ✅ **Market / Driver Explorer** — driver ranking, relationship explorer, what-if simulator, over-pay map
- ✅ **Roster** — team selector + cap summary + drop/extend/tag/keep recommendations
- ✅ **Auction Bid Targets** — FA pool ranked by `max_fair_bid`, per-position tabs
- ✅ **Trade Evaluator** — two-sided trade with net current + dynasty value/cap deltas

> Currently reads the V1 engine output (`player_value_2026.csv`). Migration to
> the V2 framework outputs is the next milestone.

## Tech Stack

- **Language:** Python 3 (uses a local `.venv`)
- **Core libs:** `pandas`, `numpy`, `scikit-learn` (HistGBR), `nfl_data_py`, `espn-api`
- **App:** `streamlit`, `plotly`
- **Viz:** `matplotlib`, `plotly`

## Documentation map

| Doc | Purpose |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | Living dev state — what's working, what's next, design decisions, gotchas |
| [`docs/architecture.md`](docs/architecture.md) | Code layering: where features land |
| [`docs/data_sources.md`](docs/data_sources.md) | How each data source is scraped/pulled |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Asset/grain/scope + stat glossary |
| [`docs/rules.md`](docs/rules.md) | League cap/contract rules (✅/🟡/❓ tagged) |
| [`docs/methodology/`](docs/methodology/) | V2 framework — per-component spec (production, team, age, injury, position, combination) |
| [`docs/figures.md`](docs/figures.md) | Figure catalog (under `figures/{contracts,cap,value}/`) |
| [`docs/analysis_plan.md`](docs/analysis_plan.md) | Historical roadmap (predates V2 framework; see CLAUDE.md for current state) |

## Project Structure

```
.
├── data/                          # raw + processed data (git-ignored)
├── docs/                          # rules, architecture, methodology, data dictionary
├── figures/                       # rendered figures (git-ignored)
├── src/
│   ├── config.py                  # paths + league constants
│   ├── data/                      # ingestion + parsing (sheets, ESPN, nflverse, cap, contracts, dataset, context)
│   ├── models/
│   │   ├── components/            # V2 six-component framework (one module per component + combine + framework orchestrator)
│   │   ├── production.py          # V1 production model
│   │   ├── projection.py          # V1 next-season projection model + age curves
│   │   └── value.py               # V1 fair-value engine (two-lens)
│   ├── viz/                       # figures + interactive HTMLs
│   └── app/                       # Streamlit app (Home + 5 pages)
├── .env.example                   # template for local config
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/JohnK-au/FF_Player_Value_Analysis.git
cd FF_Player_Value_Analysis

# create + activate the venv (use the appropriate path for your OS)
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env   # then fill in real values
```

### Configuration

League-specific identifiers — contract sheet ID, ESPN league/team IDs, ESPN
auth cookies (`ESPN_S2`, `ESPN_SWID`) — live in a local `.env` (git-ignored).
The contract sheet must be shared as **"Anyone with the link can view"** for
the no-auth CSV export to work. ESPN cookies expire periodically; refresh
from a logged-in browser when calls return HTTP 401.

### Common commands

> On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

```bash
# Data pipeline
.venv/bin/python -m src.data.sheets         # cache 3 relevant contract tabs
.venv/bin/python -m src.data.contracts      # parse active + extensions
.venv/bin/python -m src.data.cap            # parse + reconcile cap sections
.venv/bin/python -m src.data.players        # contract <-> ESPN crosswalk
.venv/bin/python -m src.data.performance    # season + weekly points 2022-2025
.venv/bin/python -m src.data.dataset        # 2026 player dataset
.venv/bin/python -m src.data.population     # all-NFL training frame
.venv/bin/python -m src.data.context        # per-player baseline/delta/z + year_type

# V1 engine
.venv/bin/python -m src.models.production   # production model + replacement levels
.venv/bin/python -m src.models.projection   # next-season projection + age curves
.venv/bin/python -m src.models.value        # fair value (production + market lenses)

# V2 framework
.venv/bin/python -m src.models.components.framework   # build master CSV
.venv/bin/python -m src.viz.position_components WR    # per-position HTML viz
.venv/bin/python -m src.viz.cross_position_variants   # cross-position weight variants

# Figures
.venv/bin/python -m src.viz.contracts   # per-team contract timelines
.venv/bin/python -m src.viz.cap         # cap distribution + projection + salary-by-position
.venv/bin/python -m src.viz.value       # value scatters

# Streamlit app
streamlit run src/app/Home.py
```

## Data Sources

- **ESPN league** — `espn-api`; private, auth-cookie-gated
- **Contract spreadsheet** — Google Sheet CSV-export (`CONTRACTS_SHEET_ID`);
  reads 3 tabs by name: `Master Cap Sheet`, `Trade Log`, `Contract Extensions`
- **nflverse** — `nfl_data_py` for ages, draft capital, combine, play-by-play,
  NGS, PFR, snap counts

## License

Personal project. Not licensed for redistribution.
