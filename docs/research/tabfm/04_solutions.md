# Solutions Cheat Sheet

> **Attempt the TODO first.** This file exists so being stuck never blocks you
> for more than a few minutes — not to replace the attempt. If you end up
> copying, close the file afterwards and retype from memory; if you can't,
> re-read the matching learning-plan module before moving on.
>
> Solutions are appended phase by phase, so this file only ever contains
> answers for TODOs that exist in the code.

---

## Phase 0 — `smoke_test.py`

### TODO(you) 0.1 — load TabFM, fit, predict

```python
from tabfm import TabFMRegressor

from src.research.tabfm._weights import load_core

core = load_core("regression")     # downloads weights on first run (~13 GB, cached)
reg = TabFMRegressor(model=core, random_state=0)

reg.fit(X_train, y_train)          # instant: stores rows as context
tabfm_pred = reg.predict(X_test)   # the actual forward pass
```

**Why it looks like this:**
- `load_core(...)` fetches the pretrained network (the "pilot" from
  [01 §2](01_how_tabfm_works.md)); `TabFMRegressor` wraps it in the sklearn
  `fit`/`predict` contract so it's interchangeable with Ridge.
- **Why `load_core` and not `tabfm_v1_0_0_pytorch.load()`?** The pip package
  looks for a `pytorch_model.bin` file, but Google ships the weights in the
  newer `safetensors` format — so the official loader raises FileNotFoundError.
  Our [`_weights.py`](../../../src/research/tabfm/_weights.py) helper tries the
  official path first and falls back to loading safetensors directly. This is a
  real, common failure mode: a library pinned to an older weight format than
  the hosted weights. Worth reading once.
- `model_type="regression"` — the same weights family serves classification
  and regression; you must say which head you want.
- `random_state=0` pins the ensemble-view shuffling so your run reproduces.
- Notice **which call is slow**. For Ridge, `fit` does the work; for TabFM,
  `fit` returns instantly and `predict` takes the time. If you observed that,
  you've seen in-context learning ([01 §3](01_how_tabfm_works.md)) with your
  own eyes — that's the whole point of the smoke test.

**Expected outcome:** on this synthetic nonlinear task TabFM should beat
Ridge's R² comfortably (Ridge can only draw straight lines through a curved
world). Exact numbers vary by machine/seed; direction is what matters.

---

## Phase 1 — `build_dataset.py`

### TODO(you) 1.1 — weekly points → consistency features

```python
def consistency_features(weekly: pd.DataFrame) -> pd.DataFrame:
    def downside_dev(s):
        m = s.mean()
        below = s[s < m]
        return ((below - m) ** 2).mean() ** 0.5 if len(below) else 0.0

    cons = (
        weekly.groupby(["espn_id", "season"])["points"]
        .agg(
            weekly_std="std",
            weekly_mean="mean",
            downside_dev=downside_dev,
            boom_weeks=lambda s: (s >= BOOM_POINTS).mean(),
            bust_weeks=lambda s: (s < BUST_POINTS).mean(),
            n_weeks="count",
        )
        .reset_index()
    )
    cons["weekly_cv"] = cons["weekly_std"] / cons["weekly_mean"]
    return cons.drop(columns=["weekly_mean"])
```

**Why it looks like this:**
- One `groupby(["espn_id", "season"])` does everything — each aggregation is a
  column in the output, named at the call site (pandas "named aggregation").
- `downside_dev` measures only below-mean weeks: a 40-point boom must not read
  as "inconsistency" the way a 2-point bust does. Same idea your V1 engine used.
- We **drop `weekly_mean`**: it's nearly the frame's `ppg` again (over a
  slightly different week range) and near-duplicate features add noise, not
  information. Keeping it wouldn't be *wrong* — it's a judgment call to note.
- Edge cases worth knowing you accepted: a 1-week season has `std = NaN`
  (pandas uses ddof=1), and `weekly_cv` explodes when the mean is near zero.
  `n_weeks` exists precisely so later phases can filter these if they distort.

### TODO(you) 1.2 — the transition self-join

```python
def build_transitions(season_table: pd.DataFrame) -> pd.DataFrame:
    outcomes = season_table[["espn_id", "season", "ppg", "games"]].copy()
    outcomes["season"] = outcomes["season"] - 1        # t+1 row lines up with t
    outcomes = outcomes.rename(columns={"ppg": "target_ppg",
                                        "games": "target_games"})
    return season_table.merge(outcomes, on=["espn_id", "season"], how="inner")
```

**Why it looks like this:**
- The mental model: take the same table, shift the *outcome* copy's season
  back one year, and merge — so 2023 features sit beside 2024 results.
- **Why `inner` is correct:** a rookie's first season has no season-t row → no
  pair; a player who left the league has no t+1 row → no pair. Both exclusions
  we *wanted* happen as a property of the join, with zero special-case code.
  (2025 feature rows also vanish naturally — there are no 2026 outcomes yet.)
- Only `ppg` and `games` cross the timeline, pre-renamed to `target_*` —
  nothing from t+1 can sneak in unlabeled, which is what 1.3 verifies.

### TODO(you) 1.3 — leakage checks

```python
def run_leakage_checks(transitions: pd.DataFrame,
                       season_table: pd.DataFrame) -> None:
    dup = transitions.duplicated(["espn_id", "season"]).sum()
    assert dup == 0, f"{dup} duplicate (espn_id, season) rows -- join fanned out"

    target_cols = sorted(c for c in transitions.columns
                         if c.startswith("target_"))
    assert target_cols == ["target_games", "target_ppg"], (
        f"unexpected future-info columns: {target_cols}")

    assert transitions["season"].between(FIRST_SEASON,
                                         LAST_FEATURE_SEASON).all()
    assert transitions["target_ppg"].notna().all(), "inner join should forbid this"

    lookup = season_table.set_index(["espn_id", "season"])["ppg"]
    for _, row in transitions.sample(3, random_state=0).iterrows():
        expected = lookup.loc[(row["espn_id"], row["season"] + 1)]
        assert row["target_ppg"] == expected, (
            f"{row['name']} {row['season']}: target_ppg {row['target_ppg']} "
            f"!= frame ppg {expected}")
    print("leakage checks passed")
```

**Why it looks like this:**
- (a) catches the classic silent killer: a merge key that isn't as unique as
  you believed *duplicates* rows, and every downstream metric quietly inflates.
- (b) is the naming-rule tripwire — if anyone later adds a t+1 column without
  the `target_` prefix, this won't catch it, but if they add one *with* it,
  the feature-selection code in Phase 2 excludes it mechanically. Convention +
  tripwire together are the defense.
- (d) is the one that matters most: shape checks pass on beautifully-wrong
  data. Only re-deriving a few values by an independent path (`set_index` +
  `.loc`, not the merge) proves the join grabbed the *right* numbers.

---

## Phase 2 — `evaluate.py` + `baselines.py`

### TODO(you) 2.1 — the four metric functions

```python
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred) -> float:
    return r2_score(y_true, y_pred)

def spearman(y_true, y_pred) -> float:
    return spearmanr(y_true, y_pred).statistic

def calibration_slope(y_true, y_pred) -> float:
    # slope of ACTUAL regressed on PREDICTED; 1.0 = calibrated
    return float(np.polyfit(y_pred, y_true, 1)[0])
```

**Why each exists:**
- `mae` — average miss in PPG. The number you can say out loud ("off by ~2.4").
- `r2` — variance explained; comparable across datasets, but rewards spreading
  predictions out, so never read it alone.
- `spearman` — did we get the *ranking* right? Your draft/auction decisions are
  orderings, so this is arguably the metric that matters most commercially.
- `calibration_slope` — regress actual on predicted; `<1` means predictions are
  compressed toward the mean (elites under-called, scrubs over-called). A model
  can win on R² and still be dangerous here — that's why we track it.
- Note the argument order in `np.polyfit(y_pred, y_true, 1)`: x first, then y.
  Calibration regresses actual **on** predicted, so predicted is x.

### TODO(you) 2.2 — the rolling-origin splitter

```python
def backtest_split(pairs, test_season):
    train = pairs[pairs["season"] < test_season]
    test  = pairs[pairs["season"] == test_season]
    return train, test
```

**Why it looks like this:**
- `season` is season *t* of the pair, so `season == 2023` is the "predict 2024"
  test set, and everything with `season < 2023` is legal training context.
- **Strictly less-than is the whole point.** A random K-fold would scatter 2024
  transitions into the training set used to predict 2023 — leaking the future.
  Time-ordered data demands time-ordered splits. This one line is the difference
  between an honest backtest and a fantasy of one.

### TODO(you) 2.3 — the persistence baseline

```python
def persistence(train, test) -> np.ndarray:
    return test["ppg"].to_numpy()
```

**Why it looks like this:**
- Ignores `train` entirely — there's nothing to learn. It predicts next year's
  PPG = this year's PPG.
- It is *the bar*. In fantasy, last season's PPG already bakes in talent, role,
  and offense, so persistence is a genuinely strong opponent. If a model can't
  beat it — after training, tuning, and 13 GB of weights — that model has
  learned nothing you couldn't get for free. Publishing this number *before*
  running TabFM is how you stop yourself moving the goalposts later.

---

*(Phase 3+ solutions are appended when those scaffolds exist.)*
