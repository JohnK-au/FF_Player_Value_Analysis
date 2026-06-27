# Team Component

> **Status:** Phase 1C complete for WR (this commit). RB / QB / TE land in
> Phases 2-4 with their own per-position recipes.

## Intent

Score each player in [0, 100] based on the **quality of the offense they
operate within**. A great WR on a bad offense is limited; a mediocre WR on
a great offense gets a boost.

## WR locked design

| Decision | Choice | Source |
|---|---|---|
| **Feature set** | `team_pass_rate`, `team_pass_epa`, `top_2_target_share_excl_self` | Empirical residual regression |
| **Dropped feature** | `team_cpoe` | Wrong sign (-0.06 z-coef) + near-zero R² contribution (0.02%) — likely noise |
| **Feature weights** | +0.38 × pass_rate_z + 0.37 × pass_epa_z − 0.25 × top_2_excl_self_z | Empirical regression coefficients |
| **Score mapping** | Composite → min-max normalised within season to [0, 100] | Mirrors Production's per-season anchoring |
| **Team assignment** | Modal pbp team from `advanced.py::_pbp_weekly` for the season being scored (currently 2025) | Existing infrastructure |

### Why these features

- **`team_pass_rate`** (+): more pass plays per game = more target chances per WR. Largest single-feature contributor in the regression (R² = 0.94% alone).
- **`team_pass_epa`** (+): better QB play lifts every receiver's outcomes. Stable across folds (CV 0.07).
- **`top_2_target_share_excl_self`** (−): captures competition for targets from OTHER team WRs. Higher concentration on other receivers = fewer targets left for this player. The user's intuition (spread-vs-concentrated passing attacks affect a WR's ceiling) is empirically validated.
- **`team_cpoe` dropped**: regression coefficient came out negative (wrong sign — accuracy should *help* receivers) and tiny (0.02% R² alone). Likely just noise correlating with no real effect; removing it tightens the model.

### Empirical tuning methodology (residual regression)

1. Fit the WR Production model on player features only
2. Compute per-WR-season residual = actual_PPG − predicted_PPG
3. Regress residual on team features
4. Anything that explains the leftover residual is genuine team effect beyond what player features already captured

**Diagnostics** (1,691 WR-seasons with complete team features, 2016-2025):

| Metric | Result |
|---|---|
| OOF R² (5-fold) | **0.0118** (team explains 1.2% of residual variance) |
| Coefficient stability (max CV across folds) | 0.19 (pass_epa 0.07, pass_rate 0.18, top_2 0.19; cpoe rejected at 0.38) |
| Hold-out 2025 R² | **−0.04** (yellow flag — effect may not generalize cleanly across eras) |
| Max worst-to-best PPG swing | ~1 PPG (≈8% on a 13 PPG WR baseline) |

**Honest read:** the Team effect on WR residual is small. The Production
model has absorbed most team context through correlated player features
(target_share, receiving_epa). Team is a real but modest tilt. This
empirical reality drove the multiplier-band conservatism (see
[combination.md](combination.md)) — though the user chose to widen the
band beyond the data-driven recommendation for design reasons.

## Per-player team_value can vary on the same team

Because `top_2_target_share_excl_self` is computed *excluding the player*,
two WRs on the same NFL team can have different `team_value`. Example: in
Cincinnati, Ja'Marr Chase's team_value sees Higgins + #3 as competition;
Higgins's team_value sees Chase + #3 as competition. Chase is the bigger
target hog, so Higgins's "competition" score is higher → Higgins has a
slightly *lower* team_value than Chase even though they're on the same
offense. This is intentional and matches the user's spread-vs-concentrated
intuition.

## Reused infrastructure

- [`src/data/advanced.py::_pbp_weekly`](../../src/data/advanced.py) — team-level pbp aggregates (`team_pass_epa`, `team_pass_rate`)
- [`src/data/population.py::extended_training_frame`](../../src/data/population.py) — assembled per-season feature frames
- [`src/models/components/production.py`](../../src/models/components/production.py) — used as the residual basis during tuning

## Phase plan

- **Phase 1C** ✅ — WR Team component implemented + empirically tuned
- **Phase 1D** ✅ — **On-Field Value** = Production × Team locked at multiplier band [0.875, 1.125]; written to `on_field_value` column in master CSV (see [combination.md](combination.md))
- **Phase 2 (RB)**: empirically tune RB-specific team features (likely `team_rush_epa`, plus O-line proxy if a clean source can be found)
- **Phase 3 (QB)**: supporting-cast quality (receiver separation pool, O-line pressure-allowed)
- **Phase 4 (TE)**: WR-like with TE-specific weight rebalancing
- **Later**: subjective override mechanism for team-context notes (e.g. "Packers play from the lead and primarily run") — deferred

## Known gaps

- **2026 team override**: the modal-pbp-team logic uses *2025* team assignments. Players traded in the 2026 offseason are scored against their old team's context until the override CSV pathway is built (deferred). The Jordan Addison ↔ NYG misattribution noted during Phase 1D validation may be a related case (mid-2025 trade affecting his modal team).
- **No clean free O-line quality data source** for the RB Team component (Phase 2 will surface a workaround).
- **`team_cpoe` exclusion**: re-evaluate after extending training data to 15+ years — coefficient may stabilize with more sample.
