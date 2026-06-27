# Age Component

> **Status:** Phase 0 (foundation). Returns neutral 50. Real curves land in
> Phase 1 (WR) and extend position-by-position.

## Intent

Score each player in [0, 100] based on **where they sit on the positional
age curve** for their position. A player in mid-prime gets a high score; a
player past their typical decline gets a low score; rookies sit near a
moderate default with upside.

## Curve to fit (per position)

From historical NFL production data (nflverse, extend beyond 2022–25 if
needed for stability), fit an empirical age vs PPG curve per position. The
curve should expose:

- **Prime onset** — typical age range where production peaks
- **Prime length** — how long the peak typically lasts at this position
- **Regression onset** — age decline typically begins
- **Curve shape** — bell, plateau-then-cliff, gradual decline, etc.

Open-source dynasty age curves exist as a reference; for v1 we derive ours
empirically from the data we have so the methodology is transparent.

## Elite-aging detection

Some elite players have **longer primes** that defy the median curve — Brady,
Rice, Hopkins, etc. A flat application of the positional median curve would
under-value them. The component should detect elite-aging candidates (e.g.
players still posting top-quartile production after typical decline age) and
**boost their age score** rather than dock it.

## Reused infrastructure

- `src/models/projection.py::positional_age_curves` — already computes median
  YoY PPG ratio by (position, age) from historical transition pairs. This is
  the seed for the v2 age curve.
- `src/data/population.py::training_frame()` — provides multi-season production
  for curve fitting

## Phase plan

- **Phase 1 (WR)**: fit WR age curve; design elite-aging detector; validate
  scores against intuition (Tyreek at 32, Diggs at 32, Puka at 25 should
  separate cleanly)
- **Phases 2–4**: fit RB / QB / TE curves (RB cliff age is well-known; QB
  late prime; TE late-blooming)

## Known gaps

- Multi-season history is limited to 2022–25 — may yield noisy curves at the
  age extremes. Extending nflverse coverage further back is a candidate
  later-phase task.
- Injury-driven early decline confounds the curve (the player's age suggests
  prime but injury says otherwise). The [injury component](injury.md)
  captures this separately so we shouldn't double-count.
