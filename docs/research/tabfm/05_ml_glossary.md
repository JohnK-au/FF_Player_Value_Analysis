# ML Glossary — plain English

> Reference doc: the classical-ML vocabulary this project uses, explained
> without jargon, with analogies where they help and a **"why it matters
> here"** note tying each term to our actual task. Consult it anytime.
>
> TabFM-specific terms (in-context learning, context, SCM, ensemble views)
> live in [01_how_tabfm_works.md §8](01_how_tabfm_works.md) instead.

## The shape of the problem

**Feature** — an input column; something you know. For us: age, receptions,
last season's PPG. Also called a predictor or independent variable.

**Target** (or label) — the output column; the thing you're predicting. For
us: next season's PPG. Also called the response or dependent variable.

**Row / observation / sample** — one example. For us: one player-transition
(what we knew about a player after season t, plus what they actually did in
t+1).

**Regression vs classification** — regression predicts a *number* (next
season's PPG: 14.7); classification predicts a *category* (will this player
be a top-12 WR: yes/no). We're doing regression.

**Model** — a function from features to a target, whose specifics are learned
from data rather than written by hand.

**Hyperparameter** — a setting you choose *before* training, which the model
does not learn itself. Ridge's `alpha`, HistGBR's `max_depth`, TabFM's
`n_estimators`. Analogy: the oven temperature, not the cake.

**Seed / `random_state`** — a fixed starting point for anything random inside
a model, so that re-running gives identical results. Setting seeds is what
makes an experiment reproducible; *not* setting them is how people
accidentally report noise as a finding.

---

## The models we use

**Linear regression** — draws the best straight line (or flat plane, in many
dimensions) through the data: `ppg = 2.3×receptions + 0.8×age + ...`. Each
feature gets one weight; predictions are a weighted sum. Simple, fast,
completely interpretable — and blind to anything curved.

**Ridge regression** — *linear regression with a brake on it.*

The problem it fixes: with correlated features — and ours are badly
correlated, since receptions, targets and receiving yards all move together —
plain linear regression goes unstable. It might settle on "+50×receptions,
−49×targets", which nearly cancels and fits the training data beautifully but
swings wildly on new data. Ridge adds a penalty for large weights, so the
model must justify every unit of weight it spends. Coefficients come out
smaller and steadier: slightly worse on training data, usually better on new
data.

> **Analogy:** plain linear regression is the student who memorizes every
> practice question including the typos — perfect on the mock, lost on the
> real exam. Ridge is that student told "your answer must be simple enough to
> explain in one sentence."

`alpha` is the brake strength: `alpha=0` *is* plain linear regression; larger
alpha forces flatter, simpler models.

*Why it matters here:* Ridge is our **linear yardstick**. If Ridge ≈ TabFM,
the relationship is basically a straight line and the foundation model bought
us nothing — a genuinely useful negative result.

**Decision tree** — a flowchart of yes/no questions learned from data:
"targets > 100? → yes: age > 28? → yes: predict 11.2 PPG." Captures curves and
interactions naturally; a single tree is unstable and overfits easily.

**Gradient boosting / HistGBR** — hundreds of *shallow* trees built in
sequence, where each new tree is trained to fix the running total's remaining
errors. The final prediction is the sum of all of them.

> **Analogy:** a committee of specialists, hired one at a time. The first gives
> a rough estimate; the second is hired specifically to correct the first's
> mistakes; the third corrects what's left. Individually weak, collectively
> excellent.

`HistGradientBoostingRegressor` is sklearn's fast version ("Hist" = it buckets
continuous features into histogram bins for speed). It's **NaN-native**: it
learns which direction missing values should go, so it can exploit
missingness as information.

*Why it matters here:* HistGBR is the **classical-ML yardstick** — what a
competent practitioner would deploy in an afternoon, and the real bar TabFM
must clear. (It's also what the V2 Production component uses, though this
analysis builds its own and never touches that one.)

**Persistence baseline** — "next year = this year." No model, no fitting; just
copy last season's PPG forward.

*Why it matters here:* it's the bar. In fantasy football, persistence is
embarrassingly strong, because last season's PPG already encodes talent, role,
and offense. **If a model can't beat persistence, it has learned nothing
useful.** Publishing this number before running TabFM is how we stop ourselves
from moving the goalposts later.

**Regression to the mean** — the tendency of extreme results to drift back
toward average next time. A WR who posts a freak 22 PPG season is partly good
and partly lucky, and luck doesn't repeat. Our "position-mean blend" baseline
exploits exactly this: predict something between the player's own PPG and
their position's average.

---

## Measuring performance

**MAE (mean absolute error)** — average size of the miss, in the target's own
units. MAE of 2.1 = "typically off by 2.1 PPG." The most human-readable
metric, and it treats all errors proportionally.

**RMSE (root mean squared error)** — like MAE but squares errors before
averaging, so big misses hurt disproportionately. Use when one catastrophic
error is worse than several small ones.

**R² (r-squared, "coefficient of determination")** — the share of variance
explained, on a scale where **1.0 = perfect** and **0.0 = no better than
always guessing the average**. (It can go *negative*: worse than guessing the
average.) Its virtue is comparability across datasets; its vice is that it
rewards spreading predictions out, and it's inflated by any leakage.

**Spearman rank correlation** — how well the *ordering* matches, ignoring
magnitudes. 1.0 = perfect ranking; 0 = no relationship.

> **Analogy:** MAE asks "how close was each guess?"; Spearman asks "did you
> get them in the right order?"

*Why it matters here:* your league decisions are **rankings** — who to draft,
who to bid on, who to trade. A model with unimpressive MAE that still ranks
players correctly is commercially useful. This is why R² alone would mislead
us.

**Calibration slope** — regress *actual* values on *predicted* ones; the ideal
slope is 1.0. A slope of 0.8 means predictions are systematically compressed
toward the middle: the model ranks fine but under-calls the extremes.

> **Analogy:** a thermometer that always reads 10% closer to room temperature
> than reality. It'll still tell you which day was hotter, but you can't trust
> the numbers.

*Why it matters here:* a compressed forecaster silently misprices elites —
the exact players your cap decisions turn on. A model can look good on R² and
still be dangerous on calibration.

**Overfitting** — learning the training data's noise as if it were signal.
Great training scores, poor real-world scores. **Underfitting** is the
opposite: the model is too simple to capture what's actually there.

**Regularization** — any technique that deliberately constrains a model to
prevent overfitting. Ridge's weight penalty is a textbook example.

---

## Validating honestly

**Train / test split** — fit on one portion, evaluate on data the model has
never seen. Testing on training data measures memorization, not skill.

**Cross-validation (K-fold)** — split into K parts; each part takes a turn as
the test set while the rest train. Uses all data for both purposes, and gives
K scores instead of one.

*Why we DON'T use it here:* K-fold shuffles randomly, which would put 2024
outcomes into the training data used to predict 2023 — information from the
future. Fine for static data, wrong for anything time-ordered.

**Backtest / rolling-origin (walk-forward) validation** — the time-aware
alternative. Stand at the end of 2023, train only on what existed then,
predict 2024. Then stand at the end of 2024, predict 2025. The model never
sees its own future.

> **Analogy:** testing a trading strategy by replaying history day by day, in
> order, rather than shuffling the calendar into a deck of cards.

**Leakage** — when information that wouldn't be available at real prediction
time sneaks into training. The cardinal sin of forecasting.

- **Temporal leakage:** a feature secretly containing future information (a
  "current team" column refreshed *after* the season you're predicting).
- **Player leakage:** the same player appearing in both train and test, so the
  model has effectively already met them. (This is what inflates the V2
  engine's recorded R² numbers — its CV splits player-*seasons* randomly, so
  a player's 2022 and 2023 land on opposite sides.)

> **Analogy:** an exam where the answer key is faintly printed on the back of
> the page. Beautiful score, worthless the day it counts — because in the real
> world, the back of the page is blank.

**Survivorship bias** — evaluating only on the subjects that "survived" to be
measurable. We score only players with ≥4 games in t+1, so seasons wrecked by
injury never count against any model. That's a deliberate trade (we're testing
talent forecasting, not injury prediction) — and every eval filter is a bias
you **choose and disclose**, never one you hide.

**Ablation study** — remove one component and re-measure, to find out what it
was actually worth.

> **Analogy:** the way you find which ingredient matters is to bake the cake
> again without it.

*Why it matters here:* our consistency features only exist from 2022 onward.
Rather than argue about whether to include them, we run with (**Run A**) and
without (**Run B**) and let the delta answer.

---

## Handling messy data

**NaN** ("not a number") — a missing value.

**Informative missingness** — when *the fact that a value is missing* carries
signal. In our data it does: `cpoe` is missing precisely for pre-2018 seasons
(the NGS tracking era hadn't started), so "missing" ≈ "older season."

*Why it matters here:* HistGBR can use that as a free feature. **TabFM
cannot** — it mean-imputes before it ever sees the data
([01 §5](01_how_tabfm_works.md)). The two models genuinely see different
information wherever NaNs live, which is a real, measured caveat rather than a
flaw to hide.

**Imputation** — filling missing cells with substitutes (the column mean, the
median, a model's guess). Convenient, but it *invents data* and erases the
missingness signal.

**Standardization / scaling** — rescaling features to comparable ranges
(typically mean 0, standard deviation 1). Necessary when a model treats raw
magnitudes as importance — `draft_pick` (1–260) would otherwise dwarf
`catch_pct` (0–1) in a linear model. Ridge needs it; tree models don't
(they only care about order, not units).

**One-hot encoding** — turning a category into one 0/1 column per value
(`team` → 32 columns). No false ordering implied. **Ordinal encoding** maps
categories to integers instead (ARI=0, ATL=1, …), which is compact but implies
an order that isn't real — acceptable for tree-based and attention-based
models that don't assume numeric meaning. TabFM's wrapper does this internally.

---

## Knowing whether a result is real

**Standard deviation** — typical distance from the average. For weekly fantasy
points, high std = a boom/bust player.

**Coefficient of variation (CV)** — standard deviation ÷ mean; spread relative
to size. Lets you compare consistency between a 20-PPG WR1 and a 6-PPG WR4
fairly. (Watch out: it explodes when the mean approaches zero.)

**Downside deviation** — like standard deviation, but only counting results
*below* a threshold. Punishes bust weeks without punishing booms.

*Why it matters here:* in weekly head-to-head, a 30-point ceiling week can't
lose you a matchup, but a 2-point floor week can. Your V1 engine already used
this idea; we're rebuilding it from raw weekly scores as a feature.

**Bootstrap** — estimate uncertainty by resampling your own data with
replacement, thousands of times, and watching how much the answer moves.

> **Analogy:** you can't replay the 2024 season 2,000 times — but you can
> repeatedly redraw random subsets of the players you *did* observe, and see
> how much your conclusion wobbles. If it survives every reshuffle, it's
> probably real.

**Paired bootstrap** — bootstrap the *difference* between two models, scoring
both on the identical resampled players each time. Removes "we happened to
draw an easy set of players" from the comparison.

*Why it matters here:* we resample **players, not rows**, because one player's
transitions are correlated — treating them as independent would make our
confidence intervals falsely narrow.

**Confidence interval** — the plausible range for a result. "TabFM beats
persistence by 0.4 PPG (95% CI: 0.1 to 0.7)" excludes zero, so the win is
probably real. "(95% CI: −0.3 to 1.1)" includes zero — that's a coin flip
dressed up as a finding.

> **The rule for this project: no error bar, no claim.** With ~500 test
> players, differences that look meaningful are routinely noise.
