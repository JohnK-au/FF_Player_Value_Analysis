# Age Component

> **Status:** Phase 1E complete for WR (this commit). RB / QB / TE land in
> Phases 2-4 with per-position sigmoid parameters.

## Intent

Score each player in [0, 100] based on **years of productive runway ahead**
for a multi-year dynasty contract, not where they sit on the typical
performance curve right now. A 22-year-old beats a 29-year-old on Age even
if the 29-year-old is in their prime production-wise — because the
22-year-old has more remaining productive seasons under a multi-year deal.

Current performance level is captured by the [Production component](production.md).
Age answers a different question: *how many years of NOW are still ahead?*

## Locked WR design

Hand-parameterized logistic decay:

```
age_value = 100 / (1 + exp((age − 28) / 2))
```

| Age | age_value | Reads as |
|---:|---:|---|
| 22 | 95 | lots of runway |
| 25 | 82 | prime years still ahead |
| 28 | 50 | decline 50/50 (the sigmoid center) |
| 30 | 27 | clear decline |
| 32 | 12 | late career |
| 34 | 5 | typically retired or replaceable |

`center = 28` (decline midpoint), `steepness = 2` (cliff sharpness). Hand-set
per dynasty intuition because the naive empirical median-PPG-by-age curve is
**survivorship-biased**: WRs who survive to age 31+ are selection-biased
toward elites (their bad-WR peers washed out by age 27-28), so using the
empirical median directly would *reward* old age — the opposite of what
dynasty value wants.

## Why hand-parameterized vs data-driven

The median-PPG-by-age curve looked like this in the data:

| Age | n | Median PPG | Mean PPG |
|---:|---:|---:|---:|
| 21 | 57 | 7.67 | 7.72 |
| 25 | 218 | 6.74 | 8.17 |
| 28 | 141 | 5.80 | 7.50 |
| 31 | 44 | 8.28 | 9.44 |

That's wrong-shaped for our purpose — 31-year-old WRs in the data outperform
25-year-olds *because the data only contains the survivors*. The sigmoid
sidesteps this and encodes the dynasty intuition directly.

## Empirical validation

The right outcome to validate against is **4-year forward cumulative PPG**
(dynasty horizon), not next-season production:

| Metric | Result |
|---|---|
| Pearson correlation `age_value` vs future-4yr total PPG | **+0.18** |
| Regression z-coef (age_value alone) | **+4.03** PPG per z-unit |
| Coefficient stability (5-fold std) | 0.25 (very stable, mean 4.03) |
| OOF R² (age alone) | **2.9%** |
| OOF R² (age + current PPG together) | 50% (current PPG dominates; age adds incrementally) |

The data confirms both the *shape* (sharp dynasty cliff around 28-29) and
the *direction* (younger = more total future production). Validated as a
dynasty-horizon metric.

**Important caveat**: when validated against *next-season* PPG (rather than
4-year forward), age_value has near-zero predictive signal. This is correct
behavior — a 30-year-old in their prime may still produce well next year;
Age isn't about next year, it's about the *next 4-5 years*. Don't expect
Age to predict near-term production.

## Elite-aging boost (deferred)

For players defying the typical curve (Brady, Rice, Hopkins) we'd want a
detector that boosts age_value beyond the sigmoid for established elites
showing no decline. Deferred to v2:
- For 32-year-old Diggs / Tyreek / Mike Evans, the sigmoid scores them low
- The Production component already rewards their actual recent production
- That's the right division of labor for v1 — Age stays honest about typical
  decline, Production rewards individual exceptionalism

When we add the elite-aging boost, we should validate it specifically:
do tagged "elite agers" actually outperform the age curve in their 4-yr
forward window?

## Reused infrastructure

- [`src/data/population.py::extended_training_frame`](../../src/data/population.py) — used in validation for transition pairs

## Phase plan

- **Phase 1E** ✅ — WR Age sigmoid (center 28, steepness 2); empirically validated for dynasty horizon
- **Phase 2-4**: per-position sigmoid parameters (RB / QB / TE)
  - RB: likely earlier center (e.g., 26-27) due to faster decline + injury attrition
  - QB: later center (e.g., 32-33) due to longer prime
  - TE: likely between WR and QB
- **Later**: elite-aging detector + parameter re-tuning from richer multi-position data

## TODOs

- Re-tune sigmoid `center` and `steepness` empirically once we have more validation runs
- Build elite-aging detector (Phase 2+ or later)
