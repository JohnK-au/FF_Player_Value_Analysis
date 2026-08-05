# Syntax Cheatsheet

> Quick reference for the `# TODO(you)` blocks. Organized by task, not by
> library. Look here when you know *what* you want but not *how to type it*.
> (If you don't know what you want, that's a [solutions](04_solutions.md) or
> ask-Claude moment instead.)
>
> **Don't recognize a term below — Ridge, HistGBR, R², Spearman, bootstrap?**
> → [05_ml_glossary.md](05_ml_glossary.md) explains every one in plain English.
> This file assumes you know *what* the thing is and only shows the typing.

## Environments — which python runs what

```bash
# Dataset build (needs nfl_data_py; py3.9):
.venv/bin/python -m src.research.tabfm.build_dataset

# Everything model-side (needs tabfm; py3.11):
.venv-tabfm/bin/python -m src.research.tabfm.baselines
.venv-tabfm/bin/python -m src.research.tabfm.run_tabfm

# Which interpreter am I on?  Which packages?
.venv-tabfm/bin/python --version
.venv-tabfm/bin/pip list | grep -i tabfm
```

## Parquet (the data contract)

```python
df.to_parquet("data/processed/research/tabfm_transitions.parquet", index=False)
df = pd.read_parquet("data/processed/research/tabfm_transitions.parquet")
# Types survive parquet; they do NOT survive CSV. That's why parquet.
```

## Building transition pairs (the Phase-1 core move)

The trick: a table of player-seasons joined **to itself**, offset by one year.

```python
# left = season t ("now"), right = season t+1 ("the future we predict")
left  = seasons_df.copy()
right = seasons_df[["espn_id", "season", "ppg", "games"]].copy()
right["season"] = right["season"] - 1          # shift so t+1 lines up with t
right = right.rename(columns={"ppg": "target_ppg", "games": "target_games"})

pairs = left.merge(right, on=["espn_id", "season"], how="inner")
# inner join = players who exist in BOTH seasons -> rookies in t+1 drop out
# (that's the v1 rookie exclusion happening naturally)
```

Sanity idioms:

```python
pairs.groupby("season").size()          # transitions per season -- eyeball it
pairs[pairs.espn_id == SOME_ID]         # trace one player through the table
assert not any(c.endswith("_t1") for c in feature_cols)   # leakage tripwire
```

## Weekly points → consistency features (groupby-agg)

```python
weekly = pd.read_csv("data/processed/performance_weekly.csv")

def downside_dev(s, floor=None):
    """Std of only the below-mean weeks (bust risk; booms don't count against)."""
    m = s.mean() if floor is None else floor
    below = s[s < m]
    return ((below - m) ** 2).mean() ** 0.5 if len(below) else 0.0

cons = (
    weekly.groupby(["espn_id", "season"])["points"]
    .agg(
        weekly_std="std",
        weekly_mean="mean",
        boom_weeks=lambda s: (s >= 20).mean(),   # share of weeks >= 20
        bust_weeks=lambda s: (s < 5).mean(),
    )
    .reset_index()
)
cons["weekly_cv"] = cons["weekly_std"] / cons["weekly_mean"]   # watch mean~0!
```

## The backtest splitter (Phase 2)

```python
def backtest_split(pairs, test_season):
    """Context = all transitions strictly before test_season; test = test_season.
    `season` here is season t of the pair, so a pair (2023 -> 2024) has
    season == 2023 and belongs to the test set for test_season 2023."""
    train = pairs[pairs.season < test_season]
    test  = pairs[pairs.season == test_season]
    return train, test
```

## Models (all speak the same fit/predict contract)

```python
# Persistence needs no model at all:
pred = test["ppg"]                       # "next year = this year"

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ridge = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))
gbr   = HistGradientBoostingRegressor(random_state=0)    # NaN-native, no imputer

ridge.fit(X_train, y_train); pred = ridge.predict(X_test)
```

```python
# TabFM (py3.11 venv only). First load downloads weights from HuggingFace.
from tabfm import TabFMRegressor

from src.research.tabfm._weights import load_core   # NOT tabfm_v1.load() -- see note

core = load_core("regression")                      # handles the safetensors skew
reg = TabFMRegressor(model=core, random_state=0)    # n_estimators=32 default
reg.fit(X_train, y_train)        # instant -- just stores context
pred = reg.predict(X_test)       # the slow part; cache the output!
```

> Use `load_core`, not the package's `tabfm_v1_0_0_pytorch.load()`: the pip
> release hunts for `pytorch_model.bin`, but the hosted weights are
> `model.safetensors`, so the official loader errors. `_weights.py` wraps the
> fix. (A textbook library-vs-weights version skew — [01 §6](01_how_tabfm_works.md#6-what-the-sklearn-wrapper-looks-like-in-practice).)

Categorical columns (`team`, `position`): TabFM's wrapper encodes string
columns itself. sklearn's Ridge/HistGBR need them numeric — simplest is
`pd.get_dummies(X, columns=["position", "team"])`, or drop `team` for the
linear models and note it.

## Metrics (Phase 2)

```python
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
rho  = spearmanr(y_true, y_pred).statistic

# Calibration slope: regress ACTUAL on PREDICTED; want ~1.0
slope = np.polyfit(y_pred, y_true, 1)[0]
```

## Paired bootstrap over players (Phase 2)

```python
rng = np.random.default_rng(0)
players = test["espn_id"].unique()
deltas = []
for _ in range(2000):
    sample = rng.choice(players, size=len(players), replace=True)
    idx = test["espn_id"].isin(sample)          # see note below
    deltas.append(
        mean_absolute_error(y[idx], pred_a[idx])
        - mean_absolute_error(y[idx], pred_b[idx])
    )
lo, hi = np.percentile(deltas, [2.5, 97.5])
# CI excludes 0 -> the difference is probably real
# NOTE: isin() ignores duplicate draws (a player sampled twice counts once).
# Fine for a first pass; the solutions doc shows the strict resample-with-
# multiplicity version.
```

## Caching predictions (Phase 3 discipline)

```python
out = test[["espn_id", "season"]].copy()
out["pred_ppg"] = pred
out["model"], out["run"], out["seed"] = "tabfm", "A_full", 0
out.to_csv(f"data/processed/research/tabfm_preds_{tag}.csv", index=False)
# Evaluation then reads these CSVs -- never re-runs inference.
```

## Gotchas encountered on this machine

- The two venvs run **different pandas major versions** (1.5.3 vs 3.0.3).
  Write nothing version-clever; keep the contract file boring.
- `weekly_cv` divides by the mean — a player averaging ~0 points explodes it.
  Clip or floor the denominator.
- `spearmanr(...)` returns an object; you want `.statistic` (pandas-3-era
  scipy). Older snippets online use `[0]`.
- TabFM's first `load()` downloads weights (~network wait, once per machine).
