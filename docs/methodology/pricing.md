# Cap Pricing Engine

> **Status:** Phase 5.5 v1 locked. Multi-stage cap-unit pricing pipeline
> that layers ON TOP of the V2 quality framework. V2 outputs (Dynasty Value,
> On-Field Value) are inputs; V2 master CSV is never modified.

## Intent

Translate V2's [0, 100] quality scores into **cap-unit fair values** that
reflect four league-market dynamics:

1. **Elite premium** — top producers command a disproportionate share of the
   cap pool (Chase, Puka, JSN willing to pay $150+ because alternatives are
   scarce).
2. **Mid-tier collapse to baseline** — if similar production is reachable
   from the FA pool, teams don't commit big money.
3. **Explicit age emphasis** — pricing itself discounts older players ON TOP
   OF DV's 20% Age weight. Same DV prices lower for a 32yo than a 24yo.
4. **Multi-year age decay** — long deals on aging players discount future
   years because year-5 fair reflects a projected 34yo, not the current 30yo.

Output columns join back to the V2 master via `espn_id`:
- `fair_value_2026` — single-season cap-unit fair value
- `surplus_2026 = salary_2026 - fair_value_2026` — positive = overpaid
- `fair_value_dynasty` — multi-year total with per-year age decay
- `surplus_dynasty = dynasty_total_salary - fair_value_dynasty`

Plus provenance columns: `pricing_basis`, `pricing_pool`, `pricing_rate`,
`replacement_dv`, `above_baseline_dv`, `scarcity_value`, `base_fair`,
`age_mult`.

## 4-stage pipeline

Inputs per player from V2 master: `dynasty_value` (basis), `age`,
`position_group`, `years_2026`, `salary_2026`, `dynasty_total_salary`,
`roster_status`.

### Stage 1 — Per-position replacement baseline

`replacement_dv[pos]` — the DV threshold below which a player is considered
"replaceable from FA" and drops to fair=0.

Two ways to set it in code:
- **Tier-based** (`per_position_baseline(basis, tier, master)`): computes
  mean DV of the "replacement tier" per position using v2 Position VORP-Deep
  methodology. Tiers = ranks N+1..2N (backup), 2N+1..3N (FA), 3N+1..4N (deep),
  where N = 8 × S_p (v2 Position slot count).
- **User override** (`baseline_override={"QB": 41, "RB": 29, ...}`): direct
  per-position values, bypassing the tier calculation.

**Locked v1 baselines (user-picked, 2026-07-03):**

| Position | replacement_dv | Interpretation |
|---|---:|---|
| QB | 41 | Roughly Sam Darnold / Baker Mayfield tier |
| RB | 29 | Deep-FA RB2/RB3 tier |
| WR | 34 | Streaming-WR tier |
| TE | 31 | Streaming-TE tier |

### Stage 2 — Above-baseline quality

```
above_baseline_dv = max(0, dynasty_value - replacement_dv[pos])
```

Players at or below their position's baseline collapse to `above_baseline_dv = 0`
and receive fair_value = 0. This encodes the "not worth committing big money to
someone the FA pool can approximate" insight.

### Stage 3 — Non-linear elite premium

```
scarcity_value = above_baseline_dv ^ alpha
```

`alpha > 1` concentrates the pool on elites: an elite player with 2× the
above-baseline gap gets `2^alpha` times the scarcity value, not 2×. Encodes
"Chase-tier commands premium because there are only a few of them."

**Locked v1: alpha = 1.25** (user preference — 1.3 was "too aggressive").

### Stage 4 — Redistribute pool + age multiplier + multi-year decay

**Rate**:
```
rate = pool / sum(scarcity_value for rostered skill players)
base_fair = scarcity_value × rate
```

**Pool** options:
- `empirical`: sum of actual 2026 salary for rostered skill players (default
  base). Self-balancing — mean rostered surplus ≈ 0 by construction.
- `rule`: `N_TEAMS × cap_per_team × SKILL_SHARE_RULE` (8 × 1500 × 0.80 = 9600).
  Prescriptive.

**Pool scaling** (`pool_scale`): multiplies the chosen pool. `pool_scale > 1`
means the model believes teams collectively UNDERSPEND on skill (sum of fair
values > actual pool). All fair values scale linearly.

**Locked v1: pool_method = empirical, pool_scale = 1.45.** Implies the league
underspends skill by ~30% relative to intrinsic value.

**Age multiplier + multi-year decay**:
```
For each contract year y in [1, years_2026]:
    projected_age    = age + (y - 1)
    year_age_value   = age_curve(projected_age, position)   # reuses v2 Age
    year_age_mult    = band_lo + (year_age_value / 100) × (band_hi - band_lo)
    year_fair        = base_fair × year_age_mult

fair_value_2026    = year 1's fair
fair_value_dynasty = sum(year_fair for y in years_2026)
```

Age curve is reused from [`components/age.py`](../../src/models/components/age.py):
per-position logistic decay (RB center=25 sharpest, QB center=33 latest).
The multiplier here is applied ON TOP OF DV (which already contains age at 20%
weight) — this is the "more emphasis" the pricing layer adds beyond quality
scoring.

**Locked v1: age_band = [0.85, 1.15]** (moderate ±15% swing).

## Sign convention

Matches V1 app's `style_surplus()`:

```
surplus_2026    = salary_2026          - fair_value_2026        # + = overpaid (red)
surplus_dynasty = dynasty_total_salary - fair_value_dynasty     # + = overpaid (red)
```

Positive surplus reads as **overpaid** (paying more than fair, cut candidate),
negative reads as **bargain** (paying less than fair, extend candidate).

## Locked v1 parameters (2026-07-03)

Preset `C_user_targets`:

| Parameter | Value | Notes |
|---|---|---|
| `basis` | `dynasty_value` | V2 headline quality score |
| `pool_method` | `empirical` | Sum of rostered skill salary |
| `pool_scale` | **1.45** | Model implies 30% league underspend |
| `baseline_override` | QB=41, RB=29, WR=34, TE=31 | User-picked per-position |
| `alpha` | **1.25** | Modest non-linear elite premium |
| `age_band` | [0.85, 1.15] | ±15% multiplicative swing |

**Sanity checks** on the locked spec:
- Elite RB (Gibbs) fair ≈ $240 (matches user target 230-250)
- Elite WR cluster (JSN/Puka/Chase) fair ≈ $180-195 (target 200-225, slightly
  under because their DVs are 8-9 pts below Gibbs)
- Elite TE (McBride) fair ≈ $175 (top of user target 150-175)
- Elite QB (Maye/Allen) fair ≈ $121-134 (matches user target ~130-160)
- Mid-tier receivers on multi-year deals (DJ Moore $52, McLaurin $26) get
  meaningful fair value, up from prior iterations where they collapsed to $0
- Massive overpays still stand out: AJ Brown +$143 single-season surplus,
  Josh Jacobs +$135, Derrick Henry +$103 (age decay)

## Rookie handling (no special-casing)

Rookies flow through the same pipeline. Their salaries are fixed by draft slot
per rules.md §7 (R1 picks 1-8: 109/64/50/45/34/30/26/24; R2:20, R3:15, R4:10;
+25% raise for years 3-4 if 4th-year option picked up; year 5 via franchise tag).

Their `years_2026` in the master CSV reflects remaining rookie-deal years
correctly. Expected outputs:

- **Well-drafted elite rookies** (Gibbs #12 → salary $50 vs fair $240, McBride)
  show as huge bargains. Captures the arbitrage of drafting well.
- **Late-round productive rookies**: massive negative surplus.
- **Under-performing high draft picks**: fair collapses to baseline; high
  draft-slot salary shows as overpay.

This is INFORMATION, not a bug. It's the intrinsic-value lens the framework
targets. The Streamlit rebuild (Phase 6) surfaces this alongside "actionable"
status (extension eligibility, tag windows).

## Reproducibility

```python
from src.models.pricing import build_pricing, PRESETS

# Recommended locked preset
preset = PRESETS["C_user_targets"]
df = build_pricing(
    basis="dynasty_value",
    pool_method="empirical",
    pool_scale=preset["pool_scale"],
    baseline_override=preset["baseline_override"],
    alpha=preset["alpha"],
    age_band=preset["age_band"],
)
# 490 rows joined to V2 master by espn_id.
# Writes to data/processed/player_pricing_2026.csv
```

Or via CLI:
```bash
.venv/Scripts/python.exe -m src.models.pricing
```

Regenerate workshop HTMLs:
```bash
.venv/Scripts/python.exe -m src.viz.pricing_variants
# Renders A_market_match / B_moderate_premium / C_user_targets / D_high_pool
# to figures/pricing_*.html + pricing_index.html.
```

## Reused infrastructure

- [`src/models/components/position.py::SLOT_COUNTS`](../../src/models/components/position.py) — S_p per position
- [`src/models/components/age.py`](../../src/models/components/age.py) — per-position logistic age curve (`AGE_PARAMS`, `_logistic_age_value`)
- [`data/processed/player_value_v2_2026.csv`](../../data/processed/player_value_v2_2026.csv) — V2 master (never modified)

## Design decisions + open items

- **Redistribution vs anchoring**: chose redistribution (sum of fair values
  scales with pool) rather than anchoring (fix a reference like "top player
  = $200"). Redistribution preserves the "pool sums to something meaningful"
  property; pool_scale > 1 lets us reject the empirical pool as the reference
  when we believe the league underspends.
- **"Pedigree" signal not included** — user's observation that Alec Pierce
  ≠ AJ Brown at similar DV points to a missing consistency/history component.
  This belongs in the V2 Intangibles component (currently stub) — [[multi-tier-position-overlay]]
  logs a related note about tier concentration as an overlay. Future work.
- **Consistency/variance multiplier** — V1 had `consistency_factor` from
  weekly variance. Not adding now; can revisit if pricing feels wrong on
  volatile producers.
- **Contract-length scarcity premium** — longer deals on elite young players
  might command a premium beyond linear year sum (locked-in supply is
  valuable). Not modeled; current uses linear year sum with age decay.
- **Rule-based pool** — kept as an option in the engine but locked preset
  uses empirical + pool_scale.
- **Alpha overshoot at extreme values** — during workshop iteration, α=1.7
  with FA baseline produced Gibbs $513 (way above market). Locked α=1.25
  keeps elite premium modest.

## Version history

| Version | Date | Preset | Locked params | Notes |
|---|---|---|---|---|
| Phase 5.5 iter 1 | 2026-07-03 | (4 presets across α/tier) | — | Linear + non-linear sweep; identified pool-compression issue |
| Phase 5.5 iter 2 | 2026-07-03 | `C_balanced` | α=1.2, FA-tier + offset -10 | Added `baseline_offset` after user asked for lower per-position baselines |
| **Phase 5.5 v1 locked** | 2026-07-03 | **`C_user_targets`** | α=1.25, baseline_override QB=41/RB=29/WR=34/TE=31, pool_scale=1.45, age_band=[0.85, 1.15] | User-picked baselines + pool scaled 1.45x to hit stated elite target ranges (Gibbs $240, Chase $179, McBride $175, Maye $134, McLaurin $26). Model implies league underspends skill by ~30%. |
