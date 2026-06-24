# FF Player Value Analysis

Machine-learning analysis of player performance and contract value for an 8-team
dynasty fantasy football league that uses an offline, NFL-style salary cap and
multi-year contract system.

> **Status:** Engine + interactive app are built and operational. A 2026 player
> value table (current-season + dynasty) is produced from the production-anchored
> fair-value model and a next-season PPG projection, surfaced through a local
> Streamlit dashboard. Active research direction: week-level archetype discovery
> for per-position projection refinement.

## What this gives you

For every player in the 8-team league (and every NFL skill free agent):

- **Production-anchored fair value** for 2026 (`value_2026`) and a discounted
  **multi-year dynasty value** (`dynasty_value`) — both with `surplus` columns
  flagging bargains / overpays.
- **Down/up/par diagnostic** for each player's most recent season via a
  longitudinal context layer that compares them to their own history (no leakage).
- **Next-season PPG projection** (`projected_ppg`) trained on 2,659 NFL skill
  player-seasons with positional age curves for the multi-year extension.
- An **interactive Streamlit app** with 5 pages: an over/under-valued board, a
  per-player value card (year-by-year projection chart, context, both lenses),
  a market/driver explorer (relationship, ranking, what-if simulator, over-pay
  map), a per-team roster view with drop/extend/tag/keep recommendations, an
  auction bid-target list, and a two-sided trade evaluator.

## The League

- **Format:** dynasty (rosters carry season to season), 8 teams.
- **Source:** ESPN Fantasy Football, custom scoring.
- **Cap:** 1500 cap units/team/season, fixed.
- **Roster:** 14 starters + 14 bench (28 veteran contract slots: 5×1yr, 5×2yr,
  7×3yr, 6×4yr, 5×5yr) + 4 IR + 1 offline practice squad.

Full rules are documented in [`docs/rules.md`](docs/rules.md).

## How the engine works (one screen)

```
contracts (Google Sheet)  +  ESPN performance  +  nflverse (pbp/NGS/PFR/snaps)
        ↓
src/data/population.py   →  training_frame.csv (2,659 skill player-seasons, 2022-25)
        ↓
src/data/context.py      →  per-player baseline / delta / z + year_type (down/up/par)
        ↓
src/models/projection.py →  next-season PPG (OOF R² 0.48) + positional age curves
        ↓
src/models/value.py
   • multiplicative consistency factor (volatility penalty, bounded [0.5, 1])
   • deep-baseline pricing (0.5 × starter replacement, NOT a $1 floor)
   • redistribute cap pool over deep_vor → value_2026
   • multi-year + age curve + 10%/yr discount → dynasty_value
        ↓
data/processed/player_value_2026.csv (master table the app reads)
        ↓
streamlit run src/app/Home.py
```

**Key design decision:** fair value is anchored to objective production (VOR /
replacement-level) rather than fit to actual salaries. Fitting actual salaries
makes the model unable to flag *systematic* mispricing — and the league is
believed to be inefficient. The market-fit model is retained as a secondary
"market price" lens, and the gap between the two is the signal.

## Tech Stack

- **Language:** Python 3 (local `.venv`).
- **Data:** `pandas`, `pyarrow`, `nfl_data_py`, `espn-api`, `python-dotenv`, `openpyxl`.
- **Modeling:** `scikit-learn` (`HistGradientBoostingRegressor`, `DecisionTreeRegressor`).
- **Viz:** `matplotlib`, `plotly`.
- **App:** `streamlit` (open-source, runs locally; league data stays on your machine).

## Project Structure

```
.
├── data/
│   ├── raw/                            # Cached sheet/API pulls (git-ignored)
│   └── processed/                      # Tidy + derived tables (git-ignored)
│       ├── training_frame_context.parquet      # Modeling base
│       ├── projected_production_2026.csv       # NFL-wide projected PPG
│       ├── player_value_2026.csv               # Master value table the app reads
│       └── research/                           # Exploration outputs
├── src/
│   ├── config.py
│   ├── data/                           # Ingestion + feature assembly
│   │   ├── sheets.py / espn.py
│   │   ├── contracts.py / cap.py
│   │   ├── players.py / nflverse.py
│   │   ├── performance.py / advanced.py
│   │   ├── dataset.py / population.py
│   │   └── context.py                  # Per-player longitudinal context (S2)
│   ├── models/
│   │   ├── production.py               # PPG model + VOR + replacement levels
│   │   ├── projection.py               # Next-season PPG + age curves
│   │   └── value.py                    # Production-anchored fair value + dynasty
│   ├── viz/                            # Matplotlib + Plotly figures
│   ├── app/                            # Streamlit app
│   │   ├── Home.py                     # Over/under-valued board
│   │   └── pages/                      # Player Card · Market/Driver · Roster · Auction · Trade
│   └── research/                       # Exploratory work-in-progress
│       └── wr_weekly.py + wr_weekly_model.py  # WR weekly archetype discovery
├── docs/
│   ├── architecture.md
│   ├── data_sources.md
│   ├── data_dictionary.md              # Asset table + raw/derived stat glossary
│   ├── analysis_plan.md                # The ML roadmap
│   ├── rules.md                        # League cap rules
│   ├── figures.md
│   └── research/                       # Findings docs from exploratory work
├── figures/                            # Generated figures (git-ignored)
├── requirements.txt
├── .env.example                        # Template (real values in git-ignored .env)
└── CLAUDE.md                           # Working notes for fast session resume
```

## Getting Started

```bash
git clone https://github.com/JohnK-au/FF_Player_Value_Analysis.git
cd FF_Player_Value_Analysis

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |  Unix: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # fill in CONTRACTS_SHEET_ID, ESPN_S2, ESPN_SWID, etc.
```

### Build the data + run the app

```bash
# Build the full engine (cached parquet downloads on first run; later runs are fast)
python -m src.data.sheets         # cache the 3 relevant Google Sheet tabs
python -m src.data.contracts      # parse active rosters + extensions
python -m src.data.cap            # parse cap sections + reconcile CAP USED
python -m src.data.players        # build contract <-> ESPN crosswalk
python -m src.data.performance    # season + weekly (wk 1-13) fantasy points 2022-25
python -m src.data.population     # build all-NFL-skill-players x season training frame
python -m src.data.context        # per-player baseline/delta/z + year_type
python -m src.models.production   # production PPG model + VOR + replacement
python -m src.models.projection   # next-season projection + age curves
python -m src.models.value        # produces data/processed/player_value_2026.csv

# Launch the interactive app
streamlit run src/app/Home.py
```

### Configuration

League-specific identifiers (contract sheet ID, ESPN league/team ids, auth
cookies) are **not stored in this repo**. They live in a local, git-ignored
`.env` file (see `.env.example` for the template).

The contract sheet is read via Google's CSV-export endpoint, so it must be
shared as **"Anyone with the link can view."** The ESPN league is private,
which means the API requires your `ESPN_S2` and `ESPN_SWID` cookies (they
expire periodically — refresh from a logged-in browser when calls return 401).

## Roadmap

- [x] Project scaffolding + GitHub setup
- [x] Contract-sheet ingestion + parsing (active rosters + extensions)
- [x] ESPN performance ingestion (season + weekly)
- [x] Unified player dataset (performance + cost) + crosswalk
- [x] Advanced metrics (pbp / NGS / PFR / snaps) + team/offense context
- [x] Production model (OOF R² ≈ 0.81) + replacement levels / VOR
- [x] Fair-value engine — two lenses (production-anchored + market-fit),
      pricing-degeneracy fixed (deep baseline + multiplicative consistency)
- [x] Per-player longitudinal context layer (S2) — `year_type` ∈ {up, par, down, rookie, partial}
- [x] Next-season projection model + positional age curves (S3, S4)
- [x] Streamlit app (board + value card + market/driver explorer + roster + auction + trade)
- [ ] Roster optimization (Phase E) — integer-cap-constrained keep/cut/tag/extend/trade recommender
- [ ] Active research: WR week-level archetype discovery → extend to RB/TE/QB
      ([`docs/research/wr_weekly_archetypes.md`](docs/research/wr_weekly_archetypes.md))
- [ ] Reconstruct weekly skill scoring NFL-wide from nflverse so FAs get a real consistency factor
- [ ] Extend historical data beyond 2022–25 to stabilize age curves and projection

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — codebase layering and where features land
- [`docs/data_sources.md`](docs/data_sources.md) — data-scraping reference
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — every data asset + raw/derived stat glossary
- [`docs/analysis_plan.md`](docs/analysis_plan.md) — the ML roadmap with current state
- [`docs/rules.md`](docs/rules.md) — league cap rules
- [`docs/figures.md`](docs/figures.md) — figure catalog
- [`docs/research/`](docs/research/) — findings from exploratory work
- [`CLAUDE.md`](CLAUDE.md) — working notes for fast session resume

## Data Sources

- **ESPN league** — accessed via league/team/season identifiers + auth cookies in `.env`.
- **Contract spreadsheet** — Google Sheet referenced by `CONTRACTS_SHEET_ID`; link kept out of this public repo.
- **nflverse** — play-by-play, Next Gen Stats, PFR advanced stats, snap counts,
  schedule/Vegas lines, seasonal rosters, draft, combine (all via `nfl_data_py`).
