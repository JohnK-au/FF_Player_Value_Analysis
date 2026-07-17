# Codebase Review — July 2026

**Date:** 2026-07-10 · **Scope:** four dimensions — (1) codebase health & risks,
(2) ML methodology soundness, (3) PyTorch plan critique, (4) portfolio/interview
readiness. **Method:** 3 parallel exploration agents (methodology docs / value
engine / data-assembly leakage surface); the 4 highest-severity findings were
hand-verified against source before reporting.

**Status:** the **Now tier is DONE** — merged to main via PR #15 (findings #3,
#4, #12, #13, #16; see that PR for details). This doc preserves the full
findings list and the outstanding **Next / Later** tiers.

**TL;DR:** Architecture, documentation culture, and honest-metrics framing are
genuinely strong — but the V2 Production component (the foundation of the value
chain) scores players **in-sample** on features **contemporaneous with the
target**, so Dynasty Value is closer to smoothed current-form than projection.
The PyTorch plan is sound; its one real gap is that the shared KFold split leaks
player-time across folds and no sequence-level OOF protocol is defined yet.
(Verified clean, for the record: `context.py`'s `shift(1).rolling` pattern and
`wr_weekly.py`'s opponent-defense roll.)

## Findings

> Severity H/M/L · Dim 1 = code health, 2 = ML methodology, 3 = PyTorch plan,
> 4 = portfolio. Line numbers are as of review date (main @ `474a1d3`).
> ✅ = fixed in PR #15.

1. **[H][2]** `src/models/components/production.py:147,181-191` — `production_value`
   is an **in-sample** prediction: the model is fit on the full frame, then predicts
   the same rows it trained on; the OOF machinery (`:156-164`) is diagnostic-print
   only. Veterans' scores are overfit-flattered, and this feeds OFV → DV → pricing.
   *(verified)* → Score from `cross_val_predict` OOF predictions (or
   leave-one-season-out) instead of the refit model.
2. **[H][2]** `src/data/population.py:57-77` — season-*t* advanced features
   (target_share, EPA, snap_pct…) are joined to season-*t* PPG target, nothing
   shifted; no projection harness exists (V1 `projection.py` was deleted).
   "Production" therefore *describes* season t rather than projecting; using 2025
   rows to value 2026 is a train/serve mismatch. → Build a transition frame
   (features ≤ t, target t+1) or explicitly re-document the component as a
   current-form score, not a projection.
3. ✅ **[H][1]** `framework.py` surplus columns — subtracted cap units from a 0–100
   score with the opposite sign convention from pricing. Dropped; pricing owns surplus.
4. ✅ **[H][2]** `pricing.py` — missing age zeroed fair value (player read as
   maximally overpaid) while `age_mult` showed neutral 1.0. Now neutral.
5. **[H][2]** `src/models/components/production.py:122,126-130` — per-season
   normalization anchor `top = observed target max` — the scale is drawn from the
   target distribution (compounds #1). *(verified)* → Fixed anchors or prior-season
   percentiles.
6. **[H][3]** `src/research/wr_weekly_model.py:124` + `docs/research/wr_weekly_torch.md`
   §2 — the shared `KFold(5, shuffle=True)` is row-level: a player's weeks span
   folds, so training sees the **same player's future weeks** when predicting past
   ones — absolute OOF R² is inflated for both models (fair as A/B, not as a
   claim). The plan also never defines the sequence-model OOF protocol (a
   per-player sequence spans folds: which timesteps may receive gradient? which
   get scored?). → Keep the shared split for the A/B; add GroupKFold(player) +
   season-holdout as secondary numbers; specify the timestep protocol in the doc
   **before** Milestone 1 (score only test-fold weeks; no training loss on
   test-fold timesteps).
7. **[M][2]** `src/models/pricing.py:27-30,220-243` — age enters fair value three
   times: DV weight 0.20, the year-1 age band, and multi-year decay — and
   `above_baseline_dv ** 1.25` non-linearly amplifies the age already in the
   basis. → Sensitivity-check once, then remove Age from the pricing basis or
   shrink the band.
8. **[M][2]** `docs/methodology/team.md` + `combination.md` — OFV multiplier bands
   widened beyond evidence: WR ±12.5% on a component with hold-out R² −0.04; QB
   kept at ±5% with negative OOF R². Band-width rationale is inconsistent
   (effect-size for WR/RB, signal-strength for TE). → Shrink bands toward the
   evidence or gate on OOF R² > 0.
9. **[M][2]** `src/data/population.py:64-69,116-121` — `adv`/`draft`/`comb` merged
   on `espn_id` **without dedup** (unlike `dataset.py:101-104`) → duplicate ids fan
   out rows; `context.py:80-89` silently drops dups `keep="first"`, masking it. →
   Apply the dataset.py dedup pattern.
10. **[M][2]** Target-basis inconsistency: `performance.py:30-43` uses ESPN
    **full-season** PPG as the population target while `dataset.py:67,77` prefers
    13-week `fpts` (violates the documented PPG-basis policy), and
    `scoring.py:114-144` gives 2016-24 a reconstructed basis (~0.17 PPG bias) vs
    ESPN 2025 — a step-change at the serve year. → Unify on the 13-week basis;
    document the historical discontinuity.
11. **[M][2]** `src/data/performance.py:90-98` — `points != 0` filter drops genuine
    0-point games → `games` undercounts, `ppg`/`stdev` inflate; benched weeks also
    count as games. → Use a played-game indicator, not points≠0.
12. ✅ **[M][1]** `pricing.py` — `build_pricing(alpha=1.7)` default contradicted the
    locked `USER_ALPHA=1.25`. Fixed.
13. ✅ **[M][1]** `intangibles.py` — reviewed dtype risk was a false alarm, but
    verification exposed a real bug: CWD-relative `OVERRIDES_PATH` silently loaded
    nothing unless run from repo root. Root-anchored via `config.DATA_DIR`.
14. **[M][2]** `src/models/pricing.py:196-197` — Stage-2 `max(0, DV−baseline)`
    collapses every sub-baseline player to fair 0 → surplus = full salary.
    Documented as "mid-tier collapse" but degenerate for the mid-tier roster
    decisions the app recommends. → Add a small floor value or flag collapsed rows
    in the app.
15. **[M][4]** Portfolio: DL/research work is branch-only — visitors landing on
    `main` never see it; README has zero screenshots (figures/ git-ignored —
    league names/salaries, so needs a redacted/synthetic shot); no tests or CI
    (2-3 leakage-regression tests would fix real risk *and* read well in
    interviews). → Link research branches from README; add one redacted app
    screenshot; add a minimal test suite.
16. ✅ **[L][1]** `wr_weekly.py` docstring overclaimed "no look-ahead leakage" —
    scoped to defense-roll/Vegas + explicit LEAKAGE WARNING added.
17. **[L][1]** `src/data/advanced.py:91-103` — traded players get a single modal
    `posteam` for the season → team context blends two offenses. → Acknowledge or
    weight by weeks.
18. **[L][1]** Minor bundle: `production.py:194-204` re-reads draft/combine tables
    per rookie inside the scoring loop; `injury.py:153` `__main__` passes
    `LATEST_SEASON` into the `position` arg (masked by `.get` default); dead
    "Phases 3-4" fallbacks (`production.py:228`, `age.py:75`, `team.py:366`). →
    Hoist, fix arg order, delete dead branches.

## Action items

### ✅ Now (correctness) — DONE, merged via PR #15
- [x] NaN-age zeroing → neutral multiplier (#4)
- [x] Drop framework surplus columns; pricing owns surplus (#3)
- [x] `build_pricing` alpha default → `USER_ALPHA` (#12)
- [x] Reword `wr_weekly.py` docstring leakage claim (#16)
- [x] Intangibles overrides join verified; relative-path bug fixed (#13)

### Next (methodology — the real work, in order)
- [ ] Rescore Production from OOF predictions; replace target-max anchors (#1, #5)
      — re-run the framework and diff the master CSV to see how much DV moves
- [ ] Decide: build a true transition frame (features ≤ t → target t+1) or
      re-document Production as current-form (#2)
- [ ] Unify the PPG target basis per the documented policy (#10); fix the
      0-point-game filter (#11)
- [ ] Dedup the population.py merges using the dataset.py pattern (#9)
- [ ] **Before PyTorch Milestone 1:** amend `wr_weekly_torch.md` with the
      sequence-OOF timestep protocol + add GroupKFold(player)/season-holdout as
      secondary splits (#6)

### Later (calibration + polish)
- [ ] Age-influence sensitivity check → simplify triple-counting (#7)
- [ ] Revisit Team/OFV band widths against hold-out evidence (#8)
- [ ] Mitigate or surface the sub-baseline collapse in the app (#14)
- [ ] Portfolio: README screenshot (redacted), link research branches, minimal
      test suite + CI (#15)
- [ ] Minor bundle cleanup (#17, #18)
