"""Phase 0 smoke test — your first TabFM prediction.

GOAL
    Prove the isolated py3.11 environment works end-to-end: generate a toy
    regression problem, fit a Ridge baseline (pre-filled), then fit TabFM
    yourself and compare.

RUN WITH  (the TabFM venv, NOT the project .venv)
    .venv-tabfm/bin/python -m src.research.tabfm.smoke_test

READ FIRST
    docs/research/tabfm/01_how_tabfm_works.md sections 1-4 and 6.

WHAT TO WATCH FOR while it runs
    Time the two models' fit() and predict() calls in your head. Ridge: fit
    slow(ish), predict instant. TabFM: fit instant, predict slow. That
    reversal IS in-context learning -- see 01 section 3.

NOTE  The first TabFM load downloads pretrained weights from HuggingFace
    (network required, one-time per machine). The weights are under a
    non-commercial license and are cached outside the repo -- never commit them.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `from src.research.tabfm...` work whether this is run as a script
# (python src/research/tabfm/smoke_test.py) or a module (-m src.research...).
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabfm import TabFMRegressor

from src.research.tabfm._weights import load_core

# ---------------------------------------------------------------- toy problem
# Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise.
# Deliberately NONLINEAR -- a straight-line model (Ridge) cannot fully capture
# it, so there is headroom for TabFM to demonstrate value.
X, y = make_friedman1(n_samples=600, n_features=8, noise=1.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"toy data: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, "
      f"{X_train.shape[1]} features\n")

# ------------------------------------------------------------- ridge baseline
# Pre-filled: this is the "number without which TabFM's number is meaningless".
t0 = time.perf_counter()
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_train, y_train)
t_fit_ridge = time.perf_counter() - t0

t0 = time.perf_counter()
ridge_pred = ridge.predict(X_test)
t_pred_ridge = time.perf_counter() - t0

print(f"Ridge  R2 = {r2_score(y_test, ridge_pred):.3f}   "
      f"(fit {t_fit_ridge:.3f}s, predict {t_pred_ridge:.3f}s)")

# -------------------------------------------------------------------- TabFM
# Loading the pretrained network is plumbing, not your concern -- `load_core`
# handles the HuggingFace download + the safetensors format. See _weights.py
# for why that helper exists.
core = load_core("regression")

# =========================================================================
# TODO(you) 0.1 -- produce `tabfm_pred` for X_test using TabFM.
#
# Steps (see 03_syntax_cheatsheet.md "Models" if the API is unfamiliar):
#   1. Wrap `core` in TabFMRegressor(model=..., random_state=0).
#   2. fit on (X_train, y_train); predict on X_test into `tabfm_pred`.
#   3. Time the fit and predict calls with time.perf_counter() pairs, like
#      the Ridge block above -- the timing reversal is the whole lesson.
#
# Stuck after a real attempt? -> docs/research/tabfm/04_solutions.md (0.1)
# =========================================================================

t0 = time.perf_counter()
reg = TabFMRegressor(model=core, random_state=0)
reg.fit(X_train, y_train)
t_fit_tabfm = time.perf_counter() - t0

t0 = time.perf_counter()
tabfm_pred = reg.predict(X_test)
t_pred_tabfm = time.perf_counter() - t0

print(f"TabFM  R2 = {r2_score(y_test, tabfm_pred):.3f}   "
      f"(fit {t_fit_tabfm:.3f}s, predict {t_pred_tabfm:.3f}s)")

# ------------------------------------------------------------------- verdict
print("\nSmoke test complete. Expected: TabFM beats Ridge on this nonlinear "
      "toy task,\nand TabFM's cost lives in predict() not fit(). If both held, "
      "Phase 0 is done.")
