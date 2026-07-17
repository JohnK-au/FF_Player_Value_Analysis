# How TabFM Works, Mechanistically

> Plain-English explainer with analogies. Sources: the
> [TabFM repo](https://github.com/google-research/tabfm) (including its
> `classifier_and_regressor.py` preprocessing source, which we verified
> directly), the
> [Google Research announcement](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/),
> and the [HF model card](https://huggingface.co/google/tabfm-1.0.0-pytorch).
> Released 2026-06-30 (v1.0.0).

## §1 Why tables needed a foundation model at all

Text and images got their foundation-model revolutions years ago. Tables
didn't — gradient-boosted trees (XGBoost, LightGBM, sklearn's HistGBR) kept
winning benchmarks. The reason: every table is its own tiny universe. A model
pretrained on *this* table's columns learns nothing transferable to a table
with different columns, scales, and meanings. There was nothing analogous to
"all text shares grammar" to pretrain on.

TabFM's answer (building on TabPFN's insight): don't pretrain facts about any
table. Pretrain **the skill of reading a table** — the general procedure of
looking at labeled rows and inferring the input→output relationship.

## §2 Pretraining: millions of simulated worlds

TabFM was trained on **hundreds of millions of synthetic datasets**, generated
on the fly from *structural causal models* (SCMs). An SCM is a little random
recipe: invent some variables, wire them together with random cause→effect
functions, add noise, sample rows. Each recipe produces one fake dataset —
one small "world" with its own rules.

Training task: show the transformer most of a fake dataset's rows *with*
labels, ask it to predict the held-out rows, grade it, repeat across millions
of different worlds. The model can't memorize any world — the next batch is a
new one. The only thing that helps it is getting better at the *general
procedure*: infer this world's rules from the labeled rows in front of you.

**Analogy — the flight simulator.** A pilot trained on millions of randomized
simulator scenarios has never flown *your* aircraft, but has internalized how
aircraft behave. Handed a new cockpit, they figure it out from the instruments
in front of them. TabFM has never seen NFL data; it has internalized how
columns relate to targets, and it reads your table on the spot.

(Why synthetic rather than scraping real tables? Coverage and hygiene: the SCM
generator produces unlimited variety, with no privacy or licensing problems —
and, importantly for us, no risk that our particular public NFL stats were
memorized during pretraining.)

## §3 In-context learning: `fit()` doesn't train anything

This is the single most disorienting fact if you're used to classical ML:

- **Classical model (e.g. HistGBR):** `fit()` is the expensive step — hundreds
  of trees get built, patterns get baked into parameters. `predict()` is then
  nearly free, and the training data can be thrown away.
- **TabFM:** `fit()` is nearly instant — verified from source, it just encodes
  and *stores your training rows* (plus sets up preprocessing). All the work
  happens in `predict()`: your training rows and your query rows go through
  the network **together, in one forward pass**, and the model infers the
  relationship right there. No weights change. Ever.

**Analogy — the open-book exam.** The GBDT studied the textbook for weeks and
walks in with memorized rules; fast on exam day, but everything it knows was
fixed at study time. TabFM is a brilliant generalist handed the open book *at
the exam* — it reads your table on the spot, every time. The cost profile
flips accordingly: studying (fit) is free, but every exam (predict) involves
re-reading the book. That's why TabFM inference is the slow part and why we'll
cache predictions to disk in Phase 3.

## §4 The architecture, top to bottom

Three stages (per the Google announcement — a hybrid of TabPFN-style and
TabICL-style designs):

1. **Alternating cell attention.** The table is treated as a grid of cells.
   Attention alternates between looking **down columns** (what's the
   distribution of this feature? what's normal, what's extreme?) and **across
   rows** (how do this player's features hang together?). Analogy: how you
   actually read a spreadsheet — scan a column to calibrate what "high" means,
   then scan a row to size up one player.
2. **Row compression.** Each row's cells are squeezed into one dense vector —
   a fixed-size "player profile embedding" — so the next stage scales with
   rows, not rows × columns.
3. **ICL transformer over rows.** A transformer attends across all the row
   embeddings — labeled context rows and unlabeled query rows together — and
   emits predictions for the queries. This is where "infer the rules from the
   labeled rows" happens.

**Ensembling:** by default the sklearn wrapper runs `n_estimators=32`
"views" — variants of your dataset (shuffled column/row orders, alternative
normalizations) — and averages the predictions. This buys stability (the model
has mild order-sensitivity) at the price of 32× the compute. It's the main
runtime knob we'll experiment with in Phase 3.

## §5 Preprocessing: what TabFM does to your data (verified from source)

From `tabfm/src/classifier_and_regressor.py`:

- **Missing numeric values are mean-imputed** (`SimpleImputer()`). TabFM
  *never sees* missingness. Consequence for us: our pre-2022 rows have no
  weekly-consistency data, and TabFM will silently fill those cells with the
  column mean — fabricated "average consistency." This is precisely why
  Phase 3 runs the ablation (with vs without consistency columns) instead of
  pretending the issue doesn't exist. Contrast: HistGBR treats "missing" as
  its own signal and can even exploit it ("missing cpoe" ≈ "pre-2018 season").
  The two models genuinely see different data where NaNs live.
- **Categoricals** are ordinal-encoded (missing → −1). Our `team` and
  `position` columns ride through this path.
- **Normalization is built in** (`norm_methods`) — we don't scale features
  ourselves.
- `fit()` also builds the ensemble views; `cache_context=True` can pre-encode
  the context for reuse.

## §6 What the sklearn wrapper looks like in practice

```python
from tabfm import TabFMRegressor

from src.research.tabfm._weights import load_core  # weights auto-download from HF

core = load_core("regression")     # the pretrained network (see note below for why not tabfm_v1.load)
reg = TabFMRegressor(model=core)   # sklearn-compatible wrapper
reg.fit(X_train, y_train)          # instant: stores context
preds = reg.predict(X_test)        # the real work happens here
```

Because it speaks sklearn's `fit`/`predict` contract, it drops into the same
harness as Ridge and HistGBR with no special-casing — the whole point of
building the harness first (Phase 2).

> **Why `load_core`, not the package's `tabfm_v1_0_0_pytorch.load()`?** The pip
> release (1.0.0) looks for a `pytorch_model.bin` checkpoint, but Google hosts
> the weights as `model.safetensors`, so the official loader raises
> FileNotFoundError. [`src/research/tabfm/_weights.py`](../../../src/research/tabfm/_weights.py)
> tries the official path first, then falls back to loading the safetensors
> weights directly (verified: keys match exactly). This is a textbook
> **library-vs-weights version skew** — the packaged code and the hosted
> weights drifted apart. You will hit this class of problem constantly with
> fast-moving ML libraries; it's worth reading the helper once.

Knobs worth knowing: `n_estimators` (ensemble views, default 32),
`random_state` (seed — we test 2–3 seeds because ICL ensembling has mild run-
to-run variance), `max_num_features` (default 500 — we're nowhere near it),
`max_num_rows` (per-view row cap — check it's not silently subsampling us).

## §7 Strengths, limits, and honest expectations for our task

**Where TabFM should shine:** small-to-medium tables (hundreds to low
thousands of rows — exactly us), zero tuning, mixed numeric/categorical, and
tasks where a flexible function-fitter beats a hand-tuned one.

**Where to be skeptical:**
- **Predicting football is mostly irreducible noise.** Next-season PPG depends
  on injuries, scheme changes, and target competition that no season-t stat
  encodes. The honest ceiling is low for *everyone* — which is why Phase 2's
  persistence baseline matters more than any model.
- **Inference cost:** every prediction re-reads the context ×32 views, on CPU.
  Minutes, not milliseconds. Fine for an annual forecast; wrong tool for a
  live app.
- **Mean-imputation** of informative missingness (§5) — measured, not assumed,
  via the ablation.
- **License:** code is Apache-2.0, but the **pretrained weights are TabFM
  Non-Commercial License v1.0**. Fine for this personal project; the weights
  are never committed to this public repo.

## §8 Glossary (plain English)

| Term | Meaning |
|---|---|
| Foundation model | A big pretrained network meant to be reused across many tasks without retraining |
| Zero-shot | Making predictions on a task it was never trained on, with no fine-tuning |
| In-context learning (ICL) | Learning from examples placed *in the input* at prediction time, with no weight updates |
| Context | The labeled rows you hand TabFM via `fit()` — its "open book" |
| SCM (structural causal model) | A random recipe wiring variables together with cause→effect functions; used to generate the synthetic pretraining worlds |
| Ensemble views (`n_estimators`) | Reshuffled/re-normalized copies of your dataset whose predictions get averaged |
| Imputation | Filling missing cells with a substitute value (TabFM: the column mean) |
| Calibration | Whether predicted magnitudes match reality on average, not just rank order |
