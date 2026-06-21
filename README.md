# FF Player Value Analysis

Machine learning analysis of player performance and contract value for a dynasty
fantasy football league that uses an offline, NFL-style salary cap and contract system.

> **Status:** 🚧 Early development. The league has a deep and evolving rule set, so
> contract/cap logic is being clarified and encoded incrementally. Expect the data
> model and scope to change.

## Purpose

This project combines two data sources that are normally kept separate in fantasy football:

1. **On-field performance** — weekly and seasonal stats from the ESPN league.
2. **Contract economics** — the league's homebrewed salary-cap and multi-year
   contract system, tracked in a Google Sheet.

The goal is to use machine learning to answer questions a real NFL front office would ask:

- Which players are **over- or under-valued** relative to their salary and contract length?
- How should **rookie contracts and franchise tags** be valued?
- What is a player's **projected performance** over the remaining years of their deal?
- How should **dead cap, retained salary, and cap space** factor into roster decisions?
- Which trades, extensions, cuts, or tags **maximize roster value** under the cap?

## The League

- **Format:** Dynasty (rosters carry over season to season).
- **Source:** ESPN Fantasy Football.
- **Teams:** 8 (Nate, Seeb, Silv, Kerr, Will, Drew, Couc, Haft).
- **Contracts:** Offline system designed to mimic real NFL practices.

### Contract & cap system (work in progress)

The contract data lives in a private Google Sheet and currently tracks:

| Concept | Description |
| --- | --- |
| **Active contracts** | Player, salary, years remaining (organized 5 years down to 1). |
| **Acquisition info** | Years left and season at acquisition. |
| **Franchise tags** | Designated tag slots with salary and league year. |
| **Rookie deals** | Draft year, drafted vs. true salary, years remaining. |
| **Cap summary** | Salary used, dead cap, and cap space per team across multiple seasons. |
| **Special designations** | IR, practice squad, amnesty, and cut players with retained salary. |

> ⚠️ These rules are still being documented. See [`docs/rules.md`](docs/rules.md)
> (to be added) for the authoritative, detailed rule set as it gets clarified.

## Tech Stack

- **Language:** Python 3
- **Core libraries:** `pandas`, `numpy`, `requests`, `python-dotenv`
- **Planned:** `scikit-learn` (modeling), `jupyter` / `matplotlib` (analysis & viz)

## Project Structure

```
.
├── data/            # Raw and processed data (git-ignored)
├── src/
│   └── data/
│       └── sheets.py   # Reads the contract sheet via CSV export
├── notebooks/       # Exploratory analysis (to be added)
├── docs/            # League rules and contract-system docs (to be added)
├── .env.example     # Template for local configuration
└── README.md
```

## Getting Started

```bash
git clone https://github.com/JohnK-au/FF_Player_Value_Analysis.git
cd FF_Player_Value_Analysis

# (recommended) create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

League-specific identifiers (like the contract spreadsheet ID) are **not stored
in this repo**. They live in a local `.env` file, which is git-ignored.

```bash
cp .env.example .env
# then edit .env and set CONTRACTS_SHEET_ID to your spreadsheet's ID
```

The contract sheet is read with no authentication via Google's CSV-export
endpoint ("Option A"), so the sheet must be shared as **"Anyone with the link
can view."**

### Fetch the contract data

```bash
python -m src.data.sheets          # caches the default tab to data/raw/
```

```python
from src.data.sheets import fetch_tab

df = fetch_tab()   # raw DataFrame of the contract sheet
```

## Roadmap

- [x] Project scaffolding and GitHub setup
- [x] Raw contract-sheet ingestion (CSV export)
- [ ] Document the full contract & cap rule set
- [ ] Parse the contract sheet into tidy per-player records
- [ ] Ingest ESPN performance data
- [ ] Build a unified player value dataset (performance + cost)
- [ ] Baseline player-value and performance-projection models
- [ ] Roster optimization under the salary cap

## Data Sources

- **ESPN league** — accessed via league/team/season identifiers (kept local).
- **Contract spreadsheet** — Google Sheet referenced by `CONTRACTS_SHEET_ID`
