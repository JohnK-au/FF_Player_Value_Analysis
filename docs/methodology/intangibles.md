# Intangibles Component

> **Status:** Phase 0 (foundation). Loader implemented; default neutral 50 for
> every player.

## Intent

Every player starts at a **neutral baseline of 50** (mid-scale on [0, 100]).
The user manually overrides specific players based on non-data information —
examples the user cited:

- "Rashee Rice spent 30 days in prison for a car accident" → score below 50
- "This player wants to be traded" → score below 50
- "Reported coaching change favors his role" → score above 50

This component is the explicit pathway for **subjective knowledge** the
engine can't infer from numbers.

## Input mechanism

An editable CSV at:

```
data/research/intangibles_overrides.csv
```

With the schema:

| Column | Type | Description |
|---|---|---|
| espn_id | int | Join key (authoritative) |
| player_name | str | For human review only; not used by the loader |
| intangibles_value | float in [0, 100] | The override score |
| note | str | Free-text rationale (visible in app later) |
| updated_at | ISO date | When this entry was last edited |

The loader [`src/models/components/intangibles.py::score`](../../src/models/components/intangibles.py)
reads this file, joins on `espn_id`, clips values to [0, 100], and fills
neutral 50 for any player not in the file.

## Editing workflow

1. Open `data/research/intangibles_overrides.csv` in any editor
2. Add or update a row with the player's `espn_id` (lookup in
   `data/processed/player_value_v2_2026.csv` if needed) and a score 0–100
3. Add a one-line `note` so the rationale survives across sessions
4. Save; re-run the framework

The CSV is **checked in** (so the overrides survive across sessions and
machines) but contains nothing sensitive — espn_ids are public and notes are
personal opinions.

## Future enhancements (Phase 6+)

- Streamlit UI for entering overrides directly (with the `note` shown as a
  tooltip in the Player Card)
- Override expiry / staleness flag (an override from 6 months ago may not
  apply anymore)
- Multiple weighted dimensions (e.g. coaching change vs character vs role
  certainty) — for v1, one number per player is sufficient
