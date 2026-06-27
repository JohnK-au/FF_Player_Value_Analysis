# Injury Component

> **Status:** Phase 0 (foundation). Returns neutral 50. v1 proxy implementation
> lands in Phase 1 (WR).

## Intent

Score each player in [0, 100] based on **durability and floor risk** — the
likelihood they miss games or post zero-or-near-zero weeks due to injury.

## v1: consistency proxy (deferred to Phase 1)

We don't currently have a per-player injury history dataset on disk. As a
v1 proxy we use the existing **weekly consistency** signal from
[`src/models/production.py::weekly_consistency`](../../src/models/production.py):

```
downside_deviation = sqrt(mean[(min(0, weekly_pts - player_mean))^2])
```

This is the RMS of weekly *shortfalls below the player's own mean*. A
player with high downside has frequent bust weeks — which in a weekly H2H
league matters more than upside weeks. **Boom weeks do NOT hurt the score**
(by construction).

**Limitation:** this is a *consistency* signal, not strictly an *injury*
signal. It picks up bust weeks for any reason — bad matchups, weather,
benching, role changes — not just injury. As a v1 proxy it captures the
right concept (floor risk) but conflates causes.

## v2: real injury data (later)

The nflverse `import_injuries()` function exposes weekly injury report
status (Out, Doubtful, Questionable, IR placements). Pulling this gives us:

- Games missed per season per player
- Injury severity profile (IR vs game-time decision)
- Multi-season durability trend

The scoring interface stays the same; the inputs get richer. This is the
biggest single follow-up after the v1 framework is validated.

## Reused infrastructure

- `src/models/production.py::weekly_consistency` — already computes the
  proxy signal

## Phase plan

- **Phase 1 (WR)**: wire `weekly_consistency` into the WR injury score;
  validate that chronically-banged-up players (Diggs, Aiyuk pre-injury) get
  lower scores than iron-men
- **Later**: pull nflverse injury data, replace the proxy

## Known gaps

- **No real injury data** — current downside proxy is the only signal
- **FA pool has no downside data** — only rostered players have weekly fantasy
  scores in our dataset. FAs default to neutral until scoring reconstruction
  from nflverse gets weekly skill scoring league-wide
