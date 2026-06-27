# Production Component

> **Status:** Phase 0 (foundation). Returns neutral 50. Real scoring lands in
> Phase 1 (WR) and extends position-by-position.

## Intent

Score each player in [0, 100] based on what they have actually produced on the
field, weighted appropriately for **where they are in their career**.

## Tiered inputs

Per the user's spec, the inputs that feed Production differ by experience tier:

| Tier | years_exp | Inputs |
|---|---|---|
| **Rookie** | 0 | NFL draft position + combine measurables + *our dynasty league's rookie draft slot* (deferred — needs data) |
| **Early (1–3)** | 1, 2, 3 | Blend of NFL draft + combine + accumulating NFL production. Draft/combine weight decays as production accumulates. |
| **Veteran (4+)** | 4+ | Production history only. Likely the existing `projected_ppg` from the projection model (which is built on production + advanced metrics + S2 context). Draft and combine no longer carry signal. |

## Subjective override (planned)

The user reserves the right to nudge production for a player based on
non-data signal (scouting, role notes). The default path is an optional
`production_value` column in `data/research/intangibles_overrides.csv` —
mechanics TBD; covered in the [intangibles doc](intangibles.md).

## Reused infrastructure

- `src/models/projection.py::projected_production` — next-season PPG (v1
  scoring backbone for veterans)
- `src/data/context.py::add_player_context` — rolling baselines for early-tier
  blending
- `src/data/population.py::training_frame_with_context` — assembled feature
  frame

## Phase plan

- **Phase 1 (WR)**: implement WR-specific production scoring; pick the
  rookie/early/veteran formulas; validate top/median/bottom
- **Phases 2–4**: adapt for RB / QB / TE (carries vs targets, etc.)
- **Later**: enable the subjective override pathway

## Known gaps

- **Dynasty league rookie draft slot data** — not yet pulled. Source TBD
  (likely a new column on the Master Cap Sheet or a separate sheet).
- **Subjective override mechanism** — design deferred; default rookie/veteran
  scoring should work without it.
