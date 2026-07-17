# TabFM Learning Plan

> **What this is:** the curriculum for a learning-first project — forecasting
> next-season fantasy performance with TabFM, a tabular foundation model, as a
> **standalone second opinion** alongside (never inside) the V2 value engine.
> You write the core code; the scaffolds mark each critical step with
> `# TODO(you)`. The goal is that afterwards you can rebuild this workflow on a
> completely different dataset without help.

## How to use these materials

1. Read the module for the phase you're on (below).
2. Open the phase's script in [`src/research/tabfm/`](../../../src/research/tabfm/)
   and attempt every `# TODO(you)` block yourself.
3. Hit a **word** you don't know (Ridge? calibration slope? bootstrap?) →
   [05_ml_glossary.md](05_ml_glossary.md). Plain English, analogies, and a
   "why it matters here" for each. Never push past an unfamiliar term — look
   it up; that's what it's for.
4. Stuck on *how to write it*? → [03_syntax_cheatsheet.md](03_syntax_cheatsheet.md).
5. Stuck on *what to write*? → [04_solutions.md](04_solutions.md) — but attempt
   first; the struggle is where the learning happens.
6. When a phase runs, we review the diff together, talk through anything shaky,
   and run the phase's verification before moving on.

**The five reference docs:**
[01](01_how_tabfm_works.md) how TabFM works (mechanism) ·
[02](02_design_rationale.md) design rationale (why) ·
[03](03_syntax_cheatsheet.md) syntax (how to type it) ·
[04](04_solutions.md) solutions (last resort) ·
[05](05_ml_glossary.md) ML glossary (what the words mean)

Rule of thumb: if you copy a solution, close the file and retype it from
memory. If you can't, you haven't got it yet — and that's fine, that's signal.

## The five phases

### Phase 0 — Environment & smoke test
**You will be able to:** stand up an isolated Python environment for a tool
whose dependencies conflict with your main project, and run a foundation-model
prediction end-to-end.

- Concepts: virtual environments as isolation boundaries; why TabFM (py≥3.11)
  cannot live in `.venv` (py3.9, pinned by `nfl_data_py`); what downloading
  "weights" means; the sklearn estimator API (`fit`/`predict`) as a universal
  contract.
- Your TODO: the toy fit/predict in `smoke_test.py` — your first TabFM call.
- Read first: [01_how_tabfm_works.md](01_how_tabfm_works.md) §1–§4.

### Phase 1 — The transitions dataset (the heart of the project)
**You will be able to:** turn longitudinal data into leak-free
(features-at-time-t → outcome-at-t+1) training rows — *the* transferable skill
of this whole project. Sales forecasting, churn, medical outcomes: same shape.

- Concepts: unit of analysis (player-transition pairs); **as-of discipline**
  (nothing the model couldn't have known at prediction time); temporal leakage
  and why it produces beautiful, worthless models; survivorship bias as a
  *chosen, disclosed* eval filter; a data dictionary as a contract.
- Your TODOs: the self-join that builds transition pairs; the weekly-points →
  consistency features aggregation; the leakage assertions.
- Ends with our **collaborative feature review** — we finalize the feature list
  together before any model sees it.
- Read first: [02_design_rationale.md](02_design_rationale.md) §2–§3.

### Phase 2 — Baselines & the evaluation harness
**You will be able to:** build an evaluation harness once and reuse it for any
model; and explain why a number without a baseline is meaningless.

- Concepts: **persistence** (naive "next year = this year") as the bar to
  clear; rolling-origin backtests vs random CV; MAE vs R² vs Spearman (when
  ranking is what you actually sell); calibration slope (accurate-but-
  compressed models silently misprice elites); paired bootstrap for "is this
  difference real?".
- Your TODOs: the backtest splitter; each metric function; the persistence,
  Ridge, and HistGBR baselines; the bootstrap.
- Read first: [02_design_rationale.md](02_design_rationale.md) §4–§5.

### Phase 3 — TabFM
**You will be able to:** run a tabular foundation model responsibly — seeds,
runtime, ensemble size, prediction caching — and slot it into an existing
harness without special-casing it.

- Concepts: in-context learning at predict-time (why `fit()` is instant and
  `predict()` is the expensive call — the *opposite* of classical ML);
  `n_estimators` as ensemble views; seed sensitivity as a mandatory check;
  the consistency **ablation** (Run A with, Run B without) as the honest way
  to value a feature group.
- Your TODOs: the TabFM fit/predict calls inside the harness; the seed loop;
  the ablation toggle.
- Read first: [01_how_tabfm_works.md](01_how_tabfm_works.md) §5–§7.

### Phase 4 — Agreement analysis & the capstone notebook
**You will be able to:** compare two independent models productively (where do
the experts disagree, and *why*?) and package an analysis so a stranger — or a
hiring manager — can follow it.

- Concepts: model agreement/disagreement as information; joining predictions
  across systems (`espn_id` as the key); narrative-first notebooks (scripts do
  the work, the notebook tells the story).
- Your TODOs: the V2 join; the disagreement table; the notebook narrative.

## The concepts map (where each big idea gets taught)

| Concept | Introduced | Exercised |
|---|---|---|
| Isolated environments & data contracts | Phase 0 | Phases 1→2 handoff |
| In-context learning / foundation models | Phase 0 reading | Phase 3 |
| Temporal leakage & as-of discipline | Phase 1 | Phase 1 assertions |
| Survivorship bias | Phase 1 | Phase 4 writeup |
| Baselines before headlines | Phase 2 | Phase 3 |
| Backtesting (rolling origin) | Phase 2 | Phases 2–3 |
| Metric choice (MAE/R²/Spearman/calibration) | Phase 2 | Phase 4 |
| Uncertainty (paired bootstrap) | Phase 2 | Phases 2–4 |
| Ablation studies | Phase 3 | Phase 3 |
| Informative missingness | Phase 1 reading | Phase 3 ablation |

## Definition of done

- All backtest results reproduce from cached predictions without re-running
  models.
- You can explain, unprompted: why persistence is the bar; why the backtests
  never use future data; why TabFM's `fit()` is instant; what the ablation
  measured; and what the biggest V2-vs-TabFM disagreement is and your best
  diagnosis of it.
- `notebooks/tabfm_report.ipynb` renders top to bottom on a fresh kernel.
