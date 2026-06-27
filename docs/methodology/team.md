# Team Component

> **Status:** Phase 0 (foundation). Returns neutral 50. Real scoring lands in
> Phase 1 (WR) and extends position-by-position.

## Intent

Score each player in [0, 100] based on the **quality of the offense they
operate within**. A great player on a bad offense is limited; a mediocre
player on a great offense gets a boost.

## Position-specific recipe

| Position | Primary signals | Source |
|---|---|---|
| WR | team_pass_epa (efficiency) + team_cpoe (accuracy) + team_pass_rate (volume) | `src/data/advanced.py` (already pulled) |
| TE | Same as WR but weight pass_rate higher (TE production is target-volume-bound) | `src/data/advanced.py` |
| RB | team_rush_epa + offensive line proxy + game script (pass rate inverse) | partial — O-line proxy needs sourcing |
| QB | Supporting cast quality: WR separation, O-line pressure allowed, run-game help | `src/data/advanced.py` partial; some signals TBD |

## Reused infrastructure

- `src/data/advanced.py` — already aggregates `team_pass_epa`, `team_cpoe`,
  `team_rush_epa`, `team_pass_rate` per team-season from pbp data
- `src/data/context.py::add_diagnostic_rollups` — produces `qb_context` which
  is a per-player z-score of their team's pass EPA vs their own history (a
  *delta* view); the team component is the *level* view (this team's
  absolute quality)

## Phase plan

- **Phase 1 (WR)**: WR-specific scoring from team_pass_epa + team_cpoe; verify
  the top of the league (KC, BUF, LAR pass offenses) score high
- **Phase 2 (RB)**: source an O-line proxy (PFF/ESPN aren't free; nflverse has
  some derived metrics; consider rolling team_rush_epa + adjusted line yards
  if available)
- **Phase 3 (QB)**: supporting cast — receiver separation, pressure allowed
- **Phase 4 (TE)**: WR-like with pass_rate weighting tweak

## Subjective override (deferred)

The user mentioned wanting a future feature for **standardised subjective
team-context notes** — e.g. "Packers play from the lead and primarily run."
This is deferred until the data-only version is stable. Implementation
sketch: a small enum of standardised tags applied per team-season that
emit ±score multipliers.

## Known gaps

- No clean free O-line quality data source
- Play-calling tendency is hard to quantify cleanly (rush rate is a partial proxy)
