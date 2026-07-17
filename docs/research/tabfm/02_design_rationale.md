# Design Rationale — why each decision, and why this order

> Companion to [00_learning_plan.md](00_learning_plan.md). Every non-obvious
> choice in this project, with the reasoning spelled out — because "apply this
> to other datasets later" requires knowing *why*, not just *what*.

## §1 Why a standalone analysis (the "two experts" framing)

The V2 engine already produces a value opinion per player. We are deliberately
**not** swapping TabFM into it. Instead we build an independent second opinion
from raw data and compare.

- **Zero blast radius.** The V2 pricing constants (`USER_BASELINES`,
  `pool_scale`) are hand-calibrated to the current model's output
  distribution; changing the model silently mis-prices everything. A separate
  analysis can't break what it doesn't touch.
- **Disagreement is the product.** Two independent experts agreeing tells you
  a little; disagreeing tells you a lot — one of them is missing something,
  and finding out which is the interesting work (Phase 4).
- **Different features by design.** V2 sees engineered components (OFV,
  dynasty value); TabFM sees only raw stats. If they still agree, that's
  strong evidence the signal is real and not an artifact of either pipeline.

## §2 Why transition pairs, and what "as-of discipline" means

**The unit of analysis is a player-transition:** one row = (everything known
about a player at the end of season t) → (their PPG in season t+1). Roughly
3–4k such rows across 2016–2025.

**As-of discipline:** every feature must have been knowable at the moment of
prediction — the end of season t. No t+1 team (offseason moves leak outcome
information), no t+1 anything.

**Analogy:** an exam where the answer key is faintly printed on the back of
the page. A model trained with leaked future information gets a beautiful
score and is worthless the day you actually need it — because at real
prediction time, the back of the page is blank. Leakage is the #1 way
forecasting projects fail in industry, and it's almost always subtle (a
"current team" column that was refreshed after the season; a season aggregate
that includes the week you're predicting). Phase 1's assertions exist to make
leakage *loud*.

Why this beats the earlier idea of comparing models on same-season data: the
V2 Production model predicts season-t PPG from season-t usage — mostly
*reconstruction* (2023 receiving EPA nearly determines 2023 receiving points).
Predicting t+1 is a true forecast, which is the question dynasty decisions
actually ask.

**Rookies are excluded in v1** — they have no season-t row to condition on.
That's a real limitation (rookie pricing is where leverage lives), documented
and deferred, not hidden.

**The ≥4-games eval filter** (decided): we score predictions only for players
with ≥4 games in t+1, so models are judged on talent forecasting, not injury
luck. The price is survivorship bias — we never grade the wrecked seasons.
Every eval filter is a bias you *choose and disclose*; this is ours.

## §3 Why raw features only

The feature set is raw box-score + profile + league fantasy output — nothing
derived by V2 (no OFV, no dynasty value, no component scores). Three reasons:

1. Independence (§1): feeding V2's outputs into TabFM would collapse the two
   experts into one.
2. Transferability: "raw operational data in, forecast out" is the shape
   you'll meet everywhere else.
3. It lets TabFM do the thing it's built for — inferring structure itself
   rather than consuming ours.

Consistency features (std/CV/downside deviation/boom-bust of weekly points)
are the one "engineered" family we allow, because they're aggregations of raw
weekly scores, not modeled values — and week-to-week reliability is real
fantasy information a season total hides.

**The coverage problem and the ablation (decided):** weekly points exist only
from 2022, so pre-2022 rows have NaN consistency — which TabFM silently
mean-imputes ([01 §5](01_how_tabfm_works.md)). Rather than pick between
shrinking the context (2022+ only) or dropping the features, we run every
model twice: **Run A** (full features) vs **Run B** (no consistency columns).
The delta *measures* what the consistency family is worth instead of arguing
about it. This is an ablation study — the standard honest way to value a
feature group, and a technique worth having in your kit.

## §4 Why baselines come before TabFM, and which ones

**A number without a baseline is meaningless.** "TabFM hit R² 0.45" sounds
impressive until you learn that copying last year's PPG hits 0.4x. The
sequence — baselines first, headline model last — is deliberate: once the
bar is published (to yourself), you can't quietly move the goalposts.

1. **Persistence** (t+1 PPG := t PPG). Embarrassingly strong in fantasy; the
   bar. If TabFM can't beat this, the entire exercise ends with a useful
   negative result.
2. **Position-mean blend** — regression to the mean, the second-most-naive
   idea, and the classic fix for persistence's flaw (outlier seasons revert).
3. **Ridge** — the linear yardstick. If Ridge ≈ TabFM, the relationship is
   basically linear and no foundation model was needed.
4. **Small HistGBR** — the classical-ML yardstick (new code inside this
   analysis; the V2 component is untouched). This is what a competent
   practitioner would deploy in an afternoon; TabFM's value-add is measured
   against *it*, not against persistence.

A useful historical landmark, not a comparable baseline: the deleted V1
`projection.py` scored OOF R² 0.48 on this same next-season task — but under
leaky random KFold, which inflates. Expect honest season-forward numbers to be
lower *for every model*. Don't be discouraged by "worse than 0.48"; be
suspicious of anything much better.

## §5 Why rolling-origin backtests, and why these metrics

**Backtests** (the user's original instinct, formalized):

| Backtest | Context | Predict | Scored against |
|---|---|---|---|
| A | transitions 2016→17 … 2022→23 | 2024 from 2023 profiles | actual 2024 |
| B | transitions 2016→17 … 2023→24 | 2025 from 2024 profiles | actual 2025 |

Random K-fold CV would shuffle 2024 outcomes into the training folds —
information from the future. Rolling-origin ("walk-forward") is the standard
for time-structured problems: the model only ever stands at a point in time
and looks backward. Two test years also gives a stability read: a model that
wins 2024 and loses 2025 is noise; winning both is signal.

**Metrics, and why four of them:**
- **MAE (PPG)** — plain-English size of the miss ("off by 2.1 points per game").
- **R²** — comparability with everything else ever reported; variance
  explained.
- **Spearman rank correlation** — drafts and auctions are *rankings*; a model
  can have mediocre MAE and still order players well, which is most of the
  practical value.
- **Calibration slope** (regress actual on predicted) — a model can rank
  perfectly while compressing everything toward the mean; slope 0.8 means
  elite predictions are systematically 20% too tame. Analogy: a thermometer
  that always reads 10% too cold — useful ordering, dangerous numbers.

**Paired bootstrap** for model-vs-baseline deltas: resample players (not
rows — the same player's transitions are correlated), recompute the delta
each time, report the interval. No error bar → no claim; with ~500 test
players per backtest, plausible-looking differences are routinely noise.

## §6 Why two environments with a parquet handoff

- The project venv (py3.9) must build the dataset: ingestion imports
  `nfl_data_py`, which pins `pandas<2` and cannot meet TabFM's `python>=3.11`.
- The TabFM venv (`.venv-tabfm/`, py3.11) runs models. Live proof of the
  isolation, from this machine: the project venv runs pandas **1.5.3**, the
  TabFM venv pandas **3.0.3** — same laptop, same repo, both work, because
  they never share an interpreter.
- The bridge is one file: `tabfm_transitions.parquet` + a data dictionary.
  This is a **data contract** — freeze data on one side, analyze on the
  other. It's also how cross-team and cross-language work happens in
  industry, so the pattern itself is CV-relevant. (Parquet, not CSV: types
  survive the round trip; no "everything became a string" surprises.)

## §7 Why this implementation order

0. **Environment first** — prove the exotic tool runs before building
   anything on it. Cheapest possible failure.
1. **Dataset second** — every downstream number inherits this table's
   correctness; leakage discipline is the project's core skill, so it gets
   the most careful, most-reviewed phase.
2. **Baselines + harness third** — debug the evaluation machinery on cheap,
   deterministic models; publish the bar before the headline model runs.
3. **TabFM fourth** — by now it's a drop-in: same harness, same folds, same
   metrics. Any surprise is attributable to the model, not the plumbing.
4. **Capstone last** — analysis and narrative on top of cached predictions,
   so the notebook re-renders in seconds without re-running models.

The through-line: **each phase makes the next one's failures unambiguous.**
If TabFM's numbers look weird in Phase 3, the dataset was verified in Phase 1
and the harness in Phase 2 — so it's the model. That's what a good experiment
scaffold buys you.

## §8 Public-repo rules for this analysis

- Raw NFL stats + league fantasy points: **fine to commit** (established
  precedent: `wr_weekly_features.csv`).
- Never commit: league/team identifiers, ESPN cookies (`.env` stays
  ignored), TabFM weights (non-commercial license — auto-downloaded per
  machine, never vendored).
- `.venv-tabfm/` is gitignored like every other venv.
