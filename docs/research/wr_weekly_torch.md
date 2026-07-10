# PyTorch Sequence Models — WR Weekly (Learning Project + Plan)

**Branch:** `pytorch-experiments` (stacked on `wr-weekly-archetypes`)
**Status:** planned — scaffolding in place, milestones not yet built
**Purpose of this doc:** a self-contained brief so this work can be resumed on
**any device** (including one with GitHub access but none of the local files).
Open this file, skim it, and paste the "Briefing Claude" block below to get a
fresh Claude Code session fully oriented.

---

## 1. Why this exists (read this first)

This is a **PyTorch upskilling project** built on top of the real FF value
engine. The motivation is the **job market**: PyTorch is the dominant
deep-learning framework, and "I built a sequence model and benchmarked it
honestly against a gradient-boosted baseline" is a strong portfolio +
interview talking point.

**Honest framing — important, keep it in the write-up.** For tabular regression
at ~3k rows, gradient-boosted trees usually beat neural networks. PyTorch is
**not** being added because the project needs it for predictive lift. It's for
**learning** and a **defensible comparison**. The honest conclusion may well be
"the GBM is still better" — that is an *acceptable and interview-worthy* result.
Do not fudge the evaluation to make the neural net look good.

**Teaching mode:** the user is *learning* PyTorch here and has asked for
**deep / first-principles** explanations — every new concept (tensors, autograd,
`nn.Module`, `Dataset`/`DataLoader`, padding/masking, attention, embeddings,
the training loop) gets a plain-English *what-it-is-and-why* **before** it is
used in code, with the football data as the running example. Explain, then
implement, then run, then checkpoint understanding. Do not just dump code.

---

## 2. The task & the baseline to beat

**Task:** predict a WR's **weekly** fantasy points (our league scoring) from
*pre-game* signals only.

**Data:** `data/processed/research/wr_weekly_features.csv` — one row per
`(season, week, espn_id)`, **2,905 WR-weeks**, 146 unique WRs, 2022–2025, ~30
pre-game features (player static, Vegas game environment, rolling opponent
defense, per-player rolling 4-week history). Built by
[`src/research/wr_weekly.py`](../../src/research/wr_weekly.py).

**The sequence framing (the whole point):** the baseline treats every row as
independent and relies on hand-built `_roll4` rolling-average columns. A
**sequence model** instead reads each player's weeks *in order* and *learns*
what to remember from the trajectory. Each WR's ordered weekly feature vectors
= one training sequence.

**Baseline to benchmark against** — `HistGradientBoostingRegressor` in
[`src/research/wr_weekly_model.py`](../../src/research/wr_weekly_model.py):

| Model | OOF R² | MAE (PPG) |
|---|---:|---:|
| **HistGradientBoosting** (the number to beat/match) | **+0.024** | **9.05** |
| Baseline: constant mean | +0.000 | 9.23 |
| Baseline: trailing-4-wk PPG | −0.084 | 9.44 |

Per-week fantasy points are **genuinely hard** (weekly std 11.7 on mean 12.4 —
single-game variance dominates). The descriptive *ceiling* with same-week
role/efficiency features is R² 0.68; the *predictive* pre-game floor is ~0.02.
See [`wr_weekly_archetypes.md`](wr_weekly_archetypes.md) for the full findings.

**Apples-to-apples rule:** reuse the **exact KFold split** from
`wr_weekly_model.py` (`KFold(5, shuffle=True, random_state=0)`) so the DL vs GBM
comparison is honest. The DL model must be evaluated OOF on the same folds.

---

## 3. The plan: LSTM → Transformer → compare

Decision (2026-07-10): **build the LSTM first**, then add the Transformer as a
second architecture that reuses the same Dataset + training loop, then produce a
**three-way comparison (GBM / LSTM / Transformer)** — the strongest portfolio
artifact.

Rationale: ~80% of the PyTorch fundamentals (Dataset, DataLoader, padding,
training loop, autograd, checkpointing) are identical for both. The LSTM lets
you learn all of that without also fighting positional-encoding and masking
bugs on day one. The Transformer is then a small, high-learning delta.

- **LSTM** — reads the sequence left-to-right, carrying a running "memory"
  (hidden state) it updates each week. Simple, robust, matches "time flows
  forward". Great fit for our short (~13–17 week) sequences.
- **Transformer** — sees all weeks at once and uses **attention** to learn which
  weeks matter for predicting each week. Needs **positional encoding** (it has no
  built-in sense of order) and **padding masks** (players have different lengths).
  More expressive, more portfolio cachet, more places to get subtly wrong.

---

## 4. Learning curriculum (8 milestones)

Every milestone: **concept first (plain English) → commented code → run it →
checkpoint understanding.** Milestones 0–6 get the LSTM working end-to-end; 7 is
the Transformer swap; 8 is the write-up.

| # | Milestone | PyTorch concepts | Football framing |
|---|---|---|---|
| **0** | Setup & tensors | `torch.Tensor`, dtypes, `device`, NumPy↔tensor, autograd (`requires_grad`, `.backward()`) via a tiny toy example | the object everything is made of |
| **1** | The `Dataset` | `torch.utils.data.Dataset`, `__len__`/`__getitem__`, grouping rows into per-player sequences | one player's game log = one example |
| **2** | `DataLoader` + padding | `DataLoader`, batching, variable-length sequences, `collate_fn`, padding + masks | players have different # of weeks |
| **3** | The model | `nn.Module`, `__init__` vs `forward`, `nn.LSTM`, `nn.Linear`, parameter registration | the LSTM that reads the sequence |
| **4** | Training loop | `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`, MSE loss, train vs eval mode, gradient clipping | teaching the net to predict points |
| **5** | Honest evaluation | reuse the GBM's exact KFold, OOF R²/MAE, masked loss (don't score padding) | apples-to-apples vs R² 0.024 |
| **6** | Checkpointing | `state_dict`, save/load weights | don't retrain every run |
| **7** | Transformer swap | `nn.TransformerEncoder`, positional encoding, attention masks (reuse M1–M6 unchanged) | the second contender |
| **8** | Write-up | three-way comparison (GBM / LSTM / Transformer) | the portfolio artifact |

**Stretch goals (after M8, each a self-contained new concept):**
1. Learned `nn.Embedding` for player / team / opponent → visualise with
   UMAP/t-SNE to surface archetype clusters.
2. Mixture-density network: output mean + variance (or mixture-of-Gaussians)
   with NLL loss → uncertainty estimates (probabilistic ML).
3. Multi-task head: jointly predict PPG + targets + air_yards + catch_rate from
   a shared representation.

Optional cross-cutting skill: **TensorBoard or Weights & Biases** for experiment
tracking (also a job-market skill).

---

## 5. Code organization

Self-contained subpackage under `src/research/` (matches the repo's `src/`
package layout; one responsibility per file):

```
src/research/wr_torch/
  __init__.py   # package doc + intended layout (already created)
  data.py       # M1–M2: SequenceDataset + collate_fn (padding/masking); reuses
                #         the KFold split from wr_weekly_model.py
  models.py     # M3, M7: WRSequenceLSTM and WRSequenceTransformer (nn.Module)
  train.py      # M4, M6: architecture-agnostic train/eval loop + checkpointing
  run.py        # M5, M8: CLI — loads data, runs KFold, prints GBM-comparable metrics
docs/research/wr_weekly_torch.md   # this doc (plan + M8 write-up lives here too)
```

Why this shape (not one big script):
- `data.py` **imports the split logic** from `wr_weekly_model.py` so the
  comparison can't silently drift.
- `models.py` holds **only architecture** — swapping LSTM→Transformer touches
  nothing else.
- `train.py` is **architecture-agnostic** — it trains any `nn.Module`, which is
  what makes milestone 7 cheap.
- `run.py` is the **only entry point**:
  `python -m src.research.wr_torch.run` (mirrors every other module here).

Only `__init__.py` exists so far; the rest are created milestone-by-milestone.

---

## 6. Working from a different device (GitHub access, no local files)

A fresh clone has the **code but not the secrets** — `.gitignore` blocks `.env`,
`.venv/`, `data/raw/`, and most `*.csv`. The training CSV, however, **is now
committed** (see §6b Path A), so the PyTorch work runs with no ESPN creds.

### 6a. Environment setup

```bash
git clone https://github.com/JohnK-au/FF_Player_Value_Analysis.git
cd FF_Player_Value_Analysis
git checkout pytorch-experiments        # the branch this work lives on

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt         # includes torch now
# If torch's default download is huge/slow (or you want the CPU-only build):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **Windows note:** use `.venv\Scripts\python.exe` to run modules, **not** the
> Unix `.venv/bin/python` shown in CLAUDE.md.

### 6b. Getting the training data

**Path A — it's committed (default; no action needed).** The feature table
`data/processed/research/wr_weekly_features.csv` is a static research snapshot
(public NFL stats + our scoring totals; no salaries/IDs/secrets), so it has been
**allowlisted in `.gitignore` and committed**. A fresh clone already has it — the
PyTorch milestones need nothing else. Confirm the baseline reproduces:

```bash
python -m src.research.wr_weekly_model     # expect OOF R² ≈ 0.024, MAE ≈ 9.05
```

**Path B — regenerate from source (only if you change the feature set).**
Requires a populated `.env` (ESPN cookies — see 6c), because the target column
`fantasy_points` comes from the ESPN pull; everything else is public nflverse
data that auto-downloads and caches locally.

```bash
python -m src.data.performance      # rebuild performance_weekly.csv (needs .env creds)
python -m src.research.wr_weekly    # rebuild wr_weekly_features.csv (nflverse pbp/NGS)
```

### 6c. Secrets (only needed for Path B / the wider project)

`.env` is git-ignored; copy `.env.example` → `.env` and fill in:
`CONTRACTS_SHEET_ID`, `ESPN_LEAGUE_ID`, `ESPN_TEAM_ID`, and the auth cookies
`ESPN_S2` / `ESPN_SWID` (from a logged-in browser: DevTools → Application →
Cookies → `.espn.com`). ESPN cookies **expire** — refresh when calls 401. The
DL milestones themselves need **no secrets** (Path A).

---

## 7. Briefing Claude on a fresh device

Paste this into a new Claude Code session in the repo:

> I'm continuing a PyTorch **learning** project on the `pytorch-experiments`
> branch. Read `docs/research/wr_weekly_torch.md` (the full plan + curriculum)
> and `CLAUDE.md` (project orientation), then `src/research/wr_weekly_model.py`
> (the HistGBR baseline: OOF R² 0.024, MAE 9.05) and
> `docs/research/wr_weekly_archetypes.md` (findings). I want **deep /
> first-principles** explanations — teach each PyTorch concept before using it.
> Tell me which milestone (0–8 in the doc) we're resuming from, confirm
> `data/processed/research/wr_weekly_features.csv` is present, and pick up from
> there.

---

## 8. Three-way comparison — results (filled in at Milestone 8)

_Pending — GBM vs LSTM vs Transformer on the shared KFold. Record OOF R²/MAE for
each here, plus the honest interpretation (did DL beat the tree? why / why not?)._

| Model | OOF R² | MAE (PPG) | Notes |
|---|---:|---:|---|
| HistGradientBoosting (baseline) | 0.024 | 9.05 | from `wr_weekly_model.py` |
| LSTM | _tbd_ | _tbd_ | |
| Transformer | _tbd_ | _tbd_ | |
