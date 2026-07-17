# TabFM next-season forecasting (standalone research analysis)

A learning-first project: forecast next-season fantasy PPG from **raw** player
data with [TabFM](https://github.com/google-research/tabfm) (Google's tabular
foundation model), validated with season-forward backtests, then compared with
the V2 value engine as an independent **second opinion**. Nothing here touches
`src/models/`, the V2 CSVs, or pricing.

**Start here → [docs/research/tabfm/00_learning_plan.md](../../../docs/research/tabfm/00_learning_plan.md)**
(curriculum, collaboration model, and reading order for the other docs).
Unfamiliar term? → [05_ml_glossary.md](../../../docs/research/tabfm/05_ml_glossary.md).

## Two environments (why two: dependency conflict)

| venv | python | used for | why |
|---|---|---|---|
| `.venv` | 3.9 | `build_dataset.py` | ingestion imports `nfl_data_py` (pins pandas<2) |
| `.venv-tabfm` | 3.11 | everything else | TabFM requires python ≥3.11 |

Bridge between them: `data/processed/research/tabfm_transitions.parquet`
(the data contract; see design rationale §6).

```bash
# one-time setup of the model venv
/opt/homebrew/bin/python3.11 -m venv .venv-tabfm
.venv-tabfm/bin/pip install "tabfm[pytorch]"

# phase order
.venv-tabfm/bin/python -m src.research.tabfm.smoke_test      # Phase 0
.venv/bin/python       -m src.research.tabfm.build_dataset   # Phase 1
.venv-tabfm/bin/python -m src.research.tabfm.baselines       # Phase 2
.venv-tabfm/bin/python -m src.research.tabfm.run_tabfm       # Phase 3
.venv-tabfm/bin/python -m src.research.tabfm.compare_v2      # Phase 4
```

## License note

TabFM's **code** is Apache-2.0; its **pretrained weights** are under the TabFM
Non-Commercial License v1.0. Fine for this personal, non-commercial project.
Weights auto-download from HuggingFace to a cache outside the repo and must
never be committed (this repo is public).

## Data rules (public repo)

Committed data may contain raw public NFL stats and league fantasy point
totals (precedent: `wr_weekly_features.csv`). Never league/team identifiers,
ESPN cookies, or anything from `.env`.
