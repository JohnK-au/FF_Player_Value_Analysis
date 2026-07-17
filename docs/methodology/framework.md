# Framework — 6-Component Player Value (v2)

> **Status:** Phase 0 (foundation). Skeleton + neutral stubs only. Real
> component scoring lands per phase: WR → RB → QB → TE.

## Why this framework

The legacy value engine ([`src/models/value.py`](../../src/models/value.py)) blends
every input (production, age, team context, consistency, positional scarcity)
into a single `value_2026` number via a fixed VOR-based pricing recipe. It
works, but you can't ask it "**why** is this player valued this way" — the
ingredients are entangled inside the pricing function.

The v2 framework replaces this with **6 explicit components**, each in [0, 100]
and each computed by an independent module. The components are then combined
into the headline **Dynasty Value** (intrinsic worth) and a derived **Contract
Value** (generic v1 derivation; roster-aware later).

## The 6 components

| Component | What it captures | Doc |
|---|---|---|
| Production | Past on-field production, tiered by years_exp | [production.md](production.md) |
| Age | Positional age curve + elite-aging detection | [age.md](age.md) |
| Team | Offensive environment quality (position-specific) | [team.md](team.md) |
| Injury | Durability / floor risk (consistency proxy v1) | [injury.md](injury.md) |
| Position | Positional scarcity (replacement + elite-tier + roster demand) | [position.md](position.md) |
| Intangibles | User-supplied subjective overrides | [intangibles.md](intangibles.md) |

## Architecture

```
component scores (0-100):
   production_value  ──┐
   age_value         ─┤
   team_value        ─┤  ┐
                        ├─ On-Field Value (Production × Team multiplier; Phase 1D)
   production_value  ─┘  ┘
                       ──┼── combine() ──> Dynasty Value
   injury_value      ───┤                      │
   position_value    ───┤                      └─> Contract Value (generic v1)
   intangibles_value ───┘                            │
                                                     └─> Contract Surplus = Contract Value − salary
                                                     └─> Dynasty Surplus  = Dynasty Value − total remaining salary
```

Each component module exports:

```python
def score(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Returns the input frame + `{component}_value` column in [0, 100]."""
```

Combination is pluggable — see [combination.md](combination.md).

## Output

Master CSV written to `data/processed/player_value_v2_2026.csv`. Columns:

- **Identity**: espn_id, player, team, position_group, age, years_exp
- **Contract**: salary_2026, years_2026, dynasty_total_salary
- **Component scores (0-100)**: production_value, age_value, team_value, injury_value, position_value, intangibles_value
- **On-Field Value** (Phase 1D): on_field_value = production_value × team multiplier; the "what they deliver on the field given their environment" intermediate
- **Final outputs**: dynasty_value, contract_value. (Surplus is deliberately *not*
  in this table: dynasty_value is a 0-100 quality score, so score-minus-salary is
  dimensionally meaningless. Surplus is owned by the pricing layer —
  `surplus = salary − fair_value`, positive = overpaid, in cap units; see
  [pricing.md](pricing.md).)

The legacy [`player_value_2026.csv`](../../data/processed/player_value_2026.csv)
is still produced by the old engine; the two coexist until v2 is validated
across all 4 positions (then the old engine is archived in Phase 7).

## Rollout phases

See the plan at [`docs/research/v2_validation/`](../research/v2_validation/) for
per-phase findings.

| Phase | Scope |
|---|---|
| 0 | Foundation: package skeleton, neutral stubs, framework wired |
| 1 | WR component implementations + validation |
| 2 | RB extension |
| 3 | QB extension |
| 4 | TE extension |
| 5 | Combination method workshop + lock-in |
| 6 | Streamlit app integration |
| 7 | Archive old engine + repo cleanup |

## Discipline

- Every component module has a matching `docs/methodology/{component}.md`
- Updates to a component formula update the doc in the same commit
- [`combination.md`](combination.md) keeps a version history of combination methods used
