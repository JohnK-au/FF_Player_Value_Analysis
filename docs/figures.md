# Figures

Generated figures live under `figures/` (git-ignored — they contain player
names/salaries), organized by theme:

```
figures/
├── contracts/   team contract timelines
├── cap/         cap distribution / projection / salary-by-position
└── value/       value scatters (static + interactive)
```

| File | Shows | Regenerate |
| --- | --- | --- |
| `contracts/team_contracts_2026.png` | Per-team 2026 contracts as a timeline (length × salary) | `python -m src.viz.contracts` |
| `cap/cap_distribution_2026.png` | Per-team 2026 salary vs the 1500 cap, by component | `python -m src.viz.cap` |
| `cap/cap_projection_2025_2029.png` | Cap composition across seasons (space opening up) | `python -m src.viz.cap` |
| `cap/salary_by_position_2026.png` | Salary by position group, per team + league total | `python -m src.viz.cap` |
| `value/value_facets_ppg_2025_{log,linear}.png` | Salary vs 2025 PPG, faceted by position; colour = surplus, size = age, your players ringed | `python -m src.viz.value` |
| `value/value_facets_ppg_2yr_{log,linear}.png` | Same, but 2024–25 average PPG | `python -m src.viz.value` |
| `value/value_interactive.html` | Interactive (hover) value scatter; your players as stars | `python -m src.viz.value_interactive` |

## Viewing the interactive HTML
`value/value_interactive.html` is a **self-contained** Plotly page (works offline).
Open it in a browser:

```bash
open figures/value/value_interactive.html      # macOS
```

Hover any point for player / team / salary / PPG / age / fair salary / surplus;
zoom/pan; toggle the "Mine"/"Other" traces via the legend.

## Conventions
- Figure output paths are defined in [`src/config.py`](../src/config.py)
  (`FIG_CONTRACTS`, `FIG_CAP`, `FIG_VALUE`).
- The value scatters come in **log** and **linear** x-axis variants (salary is
  heavily skewed — log spreads the cheap players, linear reads as raw cap units).
