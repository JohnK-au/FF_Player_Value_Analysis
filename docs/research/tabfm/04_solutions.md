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

*(Phase 1+ solutions are appended when those scaffolds exist.)*
