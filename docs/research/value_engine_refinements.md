# Value-Engine Refinements — Session Findings

**Date:** 2026-06-27 · **Branch:** `value-engine-refinements`

Three of the menu-B refinements explored this session. Headline: **DISCOUNT_RATE
is set-and-forget**, **DEEP_FACTOR and RISK_LAMBDA are real choices that move ~20%
of borderline calls**, **historical extension is blocked on scoring reconstruction**,
and **the deep-threat penalty from the WR weekly research is NOT warranted at the
season level** (the model has already learned aDOT is uninformative).

---

## 1. Tunable sensitivity sweep (`src/research/value_sensitivity.py`)

Each of `DEEP_FACTOR`, `RISK_LAMBDA`, `DISCOUNT_RATE` was swept across a reasonable
range with the other two held at default, on 155 priced players.

| Tunable | Range | Stability | Effect on rankings |
|---|---|---|---|
| `DISCOUNT_RATE` | 0.05 → 0.25 | **Spearman 1.000**, 0 sign flips | Watch list moves < 5 pts |
| `DEEP_FACTOR` | 0.3 → 0.7 | Spearman 0.80–0.96, **29–38 sign flips** | Puka 80→−70, Bijan 1→−199, Jefferson 42→−100 |
| `RISK_LAMBDA` | 0.0 → 1.0 | Spearman 0.81–0.95, **25–32 sign flips** | Puka 76→−31, Jefferson 40→−73, Mahomes 4→84 |

**Interpretation**
- **`DISCOUNT_RATE = 0.10` is robust.** Most contracts are short enough that the
  per-year discount barely changes dynasty rankings. Don't bother tuning.
- **`DEEP_FACTOR` and `RISK_LAMBDA` matter a lot.** Either moved by ±0.2 flips
  ~20% of player sign-of-surplus calls. The current 0.5/0.5 produces sensible
  central output, but **borderline calls (|surplus| < ~30) should be taken with
  this uncertainty in mind** — they could flip on a different reasonable setting.
- This isn't a "fix needed" outcome. It's a *known property* of the recipe and
  a useful caveat to surface in the app (e.g., a tooltip on small-surplus values).

---

## 2. Historical extension (deferred)

**Question:** can we extend training data back to 2018–2021 to stabilize the
projection model + age curves?

**Finding:** our ESPN league doesn't exist before 2022 (league created mid-2022),
so direct fantasy-scoring pulls fail with `ESPNInvalidLeague`. BUT nflverse advanced
metrics ARE available for 2018–2021 (NGS receiving ~1,400/season, snaps ~24,000/season,
PFR available, pbp parquet HTTP 200).

**Conclusion:** to actually use the older data, we'd need to **reconstruct fantasy
scoring from nflverse weekly box scores** using the league's custom formula
(passing +0.18/yd, +0.35/comp, −1/att, −0.66/inc, 6 pt pass-TD, plus more standard
rushing/receiving rules). That's a focused 1–2 hour task — worth doing, but as its
own session, not jammed into this one.

**Status:** deferred to a future "scoring-reconstruction" task. Once done, it
unlocks NFL-wide weekly fantasy scoring (also gives FAs a real consistency factor)
AND extending the training window to 2018+ for both the production and projection
models. Both are wins.

---

## 3. The aDOT "deep-threat penalty" — actually NOT warranted

**Motivation:** the WR weekly research (`docs/research/wr_weekly_archetypes.md`)
showed aDOT_roll4 has a slightly *negative* marginal effect on weekly fantasy
points (Q1 13.5 → Q4 11.4 PPG). The proposed B-item: bake a deep-threat penalty
into the value engine.

**But we checked at the season level — and it's a *target_share confound*.**

| Slice | Pearson r (aDOT vs season PPG) |
|---|---:|
| All WRs ≥ 8 games (n=457) | **−0.062** (essentially zero) |
| Within Low target-share stratum (n=153) | −0.076 |
| Within Mid target-share stratum (n=153) | **+0.043** |
| Within High target-share stratum (n=151) | **+0.001** |

The aDOT effect **vanishes once you control for target_share**. Why: deep
specialists tend to be lower-target WRs (they run a lot of go routes, get fewer
short-area looks). So aDOT is correlated with low target share, and *target share*
is what drives PPG — not aDOT itself.

**The production model already knows this.**

| | `adot` |
|---|---:|
| Permutation importance | **+0.0019** |
| Rank of 32 features | #28 |
| PDP across the full aDOT range (others at median) | predicted PPG moves 5.62 → 6.13 (essentially flat) |

The model already weights aDOT at near zero. Manually adding a deep-threat
penalty would **double-count** (the model has correctly learned it's
non-informative) and **introduce a spurious penalty** for high-aDOT players who
also happen to have high target share / good context (the cases where they're
actually valuable).

**Decision:** ❌ do not add a deep-threat penalty. The model has it right.

**Methodological lesson worth keeping:** a one-variable-at-a-time exploration
(the weekly heatmap finding) can mislead. Always verify against the
multivariable model's actual feature usage before hard-coding a rule.

---

## What's next from Menu B

- **Reconstruct NFL-wide weekly skill scoring from nflverse** (1–2 hr focused
  task) — unlocks (a) historical training data back to 2018, (b) real
  consistency factors for free agents in the auction view. Highest leverage
  remaining item.
- The aDOT-penalty item is **closed: no change**.
- The DEEP_FACTOR/RISK_LAMBDA findings could feed a Streamlit tooltip /
  uncertainty band on small-surplus calls — low-effort UI addition.
