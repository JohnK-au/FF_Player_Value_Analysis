"""6-Component Player Value Framework (v2).

Each player is represented by 6 explicit component scores (each in [0, 100])
that combine into a single Dynasty Value (the headline) and a derived Contract
Value. The components are computed independently so each can be iterated on
without touching the others.

Components:
    production    - past on-field production (tiered by years_exp)
    age           - positional age-curve scoring (with elite-aging detection)
    team          - quality of player's offense (position-specific)
    injury        - durability / floor risk (consistency proxy v1)
    position      - positional scarcity (replacement + elite-tier concentration)
    intangibles   - user-supplied subjective overrides (neutral default)

Each module exports a canonical:
    score(players: pd.DataFrame, position: str) -> pd.DataFrame
returning the input frame + a single column `{component}_value` in [0, 100].

The combine() function (combine.py) is pluggable; v1 default = uniform weighted
sum. The orchestrator (framework.py) runs all components, combines them, and
writes the master CSV at data/processed/player_value_v2_2026.csv.

Documentation lives in docs/methodology/.
"""
from . import age, injury, intangibles, position, production, team

__all__ = ["production", "age", "team", "injury", "position", "intangibles"]
