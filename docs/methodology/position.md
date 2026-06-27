# Position Component

> **Status:** Phase 0 (foundation). Returns neutral 50. Real scoring lands in
> Phase 1 (WR) and extends position-by-position.

## Intent

Score each player in [0, 100] based on **how much value their position
contributes** to a dynasty roster — driven by scarcity at that position.

## Drivers

### 1. Replacement gap

How big is the drop-off from a starter at this position to a replacement?
Bigger gap → higher position value at that position.

Already computed by [`src/models/production.py::replacement_levels`](../../src/models/production.py)
— uses the 8-team lineup-fill algorithm against league rules §4 (1 QB, 2 RB,
2 WR + WR/TE flex, 1 TE, RB/WR/TE flex).

### 2. Elite-tier concentration

When a position has very few elite producers (3–4 standouts vs a flat field),
those elites are *more* valuable because you must start someone at that
position and the alternatives drop off fast.

**Formula (v1 default):** coefficient of variation of the top-N PPG
distribution at the position. Higher CV → more bifurcated → higher value
for the elites.

### 3. Roster-slot demand

How many starters of this position the league lineup requires (rules §4):

| Position | Required starters | Flex eligibility |
|---|---|---|
| QB | 1 | — |
| RB | 2 | RB/WR/TE flex |
| WR | 2 | WR/TE flex + RB/WR/TE flex |
| TE | 1 | WR/TE flex + RB/WR/TE flex |

Higher required starters + flex eligibility → higher position value.

## Reused infrastructure

- `src/models/production.py::replacement_levels`, `vor_table` — replacement
  PPG per position and lineup-fill VOR pool

## Phase plan

- **Phase 1 (WR)**: WR-specific position score using all three drivers;
  validate against league context (WR demand is highest in our format)
- **Phases 2–4**: extend to RB / QB / TE with position-appropriate weighting
  (e.g. 1-QB league makes QB position value lower than 2-QB; RB injury
  attrition affects elite-tier concentration dynamically)

## Known gaps

- "Elite-tier concentration" formula is a default choice (CV of top-N) — may
  want to iterate after seeing v1 outputs
- Roster-slot demand is static (a league setting); we hardcode rules §4
