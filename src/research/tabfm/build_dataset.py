"""Phase 1 — build the leak-free transitions dataset (the heart of the project).

GOAL
    One row per player per (season t -> season t+1):
      - FEATURES: only things knowable at the end of season t
        (profile + raw box score + league fantasy output + weekly consistency)
      - TARGETS:  season t+1 outcomes, clearly suffixed: target_ppg, target_games
    Written to data/processed/research/tabfm_transitions.parquet -- the data
    contract the py3.11 venv consumes in Phases 2-3.

RUN WITH  (the PROJECT venv -- this file touches src/data/, which needs
    pandas<2. The TabFM venv would fail. Yes, the opposite of smoke_test.py.)
        .venv/bin/python src/research/tabfm/build_dataset.py --check-inputs
        .venv/bin/python src/research/tabfm/build_dataset.py
        .venv/bin/python src/research/tabfm/build_dataset.py --spot "Justin Jefferson"

    First full run downloads seasonal box scores for 2016-2023 from nflverse
    (a few minutes); each season caches to data/processed/ and is instant after.

READ FIRST
    docs/research/tabfm/02_design_rationale.md sections 2-3 (as-of discipline,
    why transition pairs). Cheatsheet sections "Building transition pairs" and
    "Weekly points -> consistency features" contain the two core moves.

NAMING RULE (this IS the leakage defense -- internalize it)
    Feature columns describe season t and carry NO suffix. Anything from
    season t+1 carries the target_ prefix. If a target_ column ever appears in
    a model's feature list, the tripwire in TODO 1.3 must catch it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `from src...` imports work when run as a script.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.version_info >= (3, 10):  # the TabFM venv is 3.11; this file needs 3.9
    print("WARNING: you look to be in the wrong venv -- use .venv/bin/python "
          "(project, py3.9) for Phase 1, not .venv-tabfm.\n", file=sys.stderr)

PROCESSED = _ROOT / "data" / "processed"
OUT_PARQUET = PROCESSED / "research" / "tabfm_transitions.parquet"
OUT_DICT = _ROOT / "docs" / "research" / "tabfm" / "data_dictionary.md"

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
FIRST_SEASON = 2016          # earliest feature season
LAST_FEATURE_SEASON = 2024   # 2024 features -> 2025 targets is the last pair
IDENTITY_COLS = ["season", "espn_id", "name", "position_group", "team",
                 "age", "years_exp", "games", "points", "ppg"]
BOX_STATS = ["completions", "attempts", "passing_yards", "passing_tds",
             "interceptions", "carries", "rushing_yards", "rushing_tds",
             "targets", "receptions", "receiving_yards", "receiving_tds"]

# --- Feature TIERS (for the Phase-2/3 ablation) ------------------------------
# Two toggleable bundles let us MEASURE what each is worth, rather than assume:
#   Run "full"       = core + consistency + advanced
#   Run "no-advanced"= core + consistency         (isolates the advanced tier)
#   Run "no-consist" = core + advanced            (isolates consistency)
# Phase 2 imports these names to build the arms; keep them as the single source
# of truth for which columns belong to which bundle.

# Curated advanced-efficiency stats. Chosen for COVERAGE first: at the feature-
# review checkpoint we saw catch_pct(9%) / cpoe(3%) / passing_epa(5%) are so
# sparse that TabFM's mean-imputation would make them ~90% fabricated constant,
# so they're held back despite being interesting. team_* stats are excluded on
# principle -- they're offense-environment (the V2 Team component's territory),
# not raw player signal. Revisit position-specific sparse stats only if the
# ablation shows the advanced tier earns its place.
ADVANCED_STATS = ["snap_pct", "target_share", "wopr",
                  "receiving_epa", "rushing_epa"]

# Consistency columns produced by TODO 1.1 (named here so Phase 2 can toggle
# them as a bundle without re-listing).
CONSISTENCY_STATS = ["weekly_std", "weekly_cv", "downside_dev",
                     "boom_weeks", "bust_weeks", "n_weeks"]

# Missingness indicator (added at assembly, NOT in 1.1): 1 when a player-season
# has no weekly consistency data (pre-2022, or absent from the weekly table).
# Its own toggle so Phase 3 can ablate "does knowing-we-don't-know carry signal?"
# independently of the consistency values themselves. See design rationale s3.
CONSISTENCY_MISSING_FLAG = "consistency_missing"

# Provisional thresholds -- revisit at the feature-review checkpoint.
BOOM_POINTS = 20.0
BUST_POINTS = 5.0


# ------------------------------------------------------------------ loaders
# Pre-filled: IO plumbing, not the lesson.

def _dedup_key(df: pd.DataFrame,
               key: tuple[str, ...] = ("espn_id", "season")) -> pd.DataFrame:
    """One row per `key`, keeping the most complete (fewest-NaN) copy.

    The upstream training frame + nflverse advanced tables carry a few duplicate
    player-seasons -- mostly exact-identical rows, plus a few that differ only by
    a missing `team` (e.g. a mid-season trade recorded twice). Left over, they
    make the self-join fan out (leakage check (a) catches it). Collapsing to the
    most-complete row is safe here because the differing copies share identical
    stats; only sparse identity fields like `team` vary.
    """
    df = df.assign(_na=df.isna().sum(axis=1))
    df = (df.sort_values("_na")
            .drop_duplicates(list(key), keep="first")
            .drop(columns="_na")
            .reset_index(drop=True))
    return df


def load_season_frame() -> pd.DataFrame:
    """Skill player-seasons 2016-2025 with identity + league fantasy output."""
    path = PROCESSED / "training_frame_extended.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- regenerate with "
                         ".venv/bin/python -m src.data.population")
    df = pd.read_csv(path)[IDENTITY_COLS]
    df = df[df["position_group"].isin(SKILL_POSITIONS)].reset_index(drop=True)
    assert df["espn_id"].notna().all(), "null espn_id would poison the self-join"
    return _dedup_key(df)  # a few duplicate player-seasons upstream; collapse them


def load_boxscores() -> pd.DataFrame:
    """Raw per-season counting stats 2016-2024, keyed by (espn_id, season).

    2025 box scores are deliberately NOT needed: 2025 rows only ever serve as
    the *outcome* side of a transition (target_ppg/target_games come from the
    season frame), never as a feature row.
    """
    from src.data.nflverse import player_boxscore_stats  # heavy import, deferred

    frames = []
    for season in range(FIRST_SEASON, LAST_FEATURE_SEASON + 1):
        b = player_boxscore_stats(season)  # cached after first pull
        b = b.copy()
        b["season"] = season
        frames.append(b)
    out = pd.concat(frames, ignore_index=True)
    return out[["espn_id", "season"] + [c for c in BOX_STATS if c in out.columns]]


def load_weekly() -> pd.DataFrame:
    """ESPN weekly league points, 2022-2025, weeks 1-13 (fantasy regular season)."""
    w = pd.read_csv(PROCESSED / "performance_weekly.csv")
    return w[["espn_id", "season", "week", "points"]]


def load_advanced() -> pd.DataFrame:
    """Curated advanced-efficiency stats 2016-2024, keyed by (espn_id, season).

    The advanced ablation tier (see ADVANCED_STATS). Coverage is far thinner
    than box score -- these columns are heavily NaN and will be mean-imputed by
    TabFM, which is exactly why we ablate them rather than trust them blind.
    """
    from src.data.advanced import advanced_features  # heavy import, deferred

    frames = []
    for season in range(FIRST_SEASON, LAST_FEATURE_SEASON + 1):
        a = advanced_features(season)  # cached parquet after first build
        a = a.copy()
        a["season"] = season
        keep = ["espn_id", "season"] + [c for c in ADVANCED_STATS if c in a.columns]
        frames.append(a[keep])
    return _dedup_key(pd.concat(frames, ignore_index=True))  # traded players dup'd


# ------------------------------------------------------- consistency features
# =========================================================================
# TODO(you) 1.1 -- weekly points -> per (espn_id, season) consistency features.
#
# Return ONE row per (espn_id, season) with these columns:
#   weekly_std    std of that season's weekly points
#   weekly_cv     std / mean  (relative volatility -- beware mean near 0)
#   downside_dev  std computed over BELOW-MEAN weeks only (bust risk;
#                 boom weeks must not count against a player)
#   boom_weeks    share of weeks with points >= BOOM_POINTS
#   bust_weeks    share of weeks with points <  BUST_POINTS
#   n_weeks       how many weekly rows the stats are computed from
#
# The cheatsheet section "Weekly points -> consistency features" has the
# groupby-agg skeleton and a downside_dev helper. Adapt, don't copy blindly:
# check what its `bust_weeks` threshold is vs BUST_POINTS here.
#
# Think about (and note for the checkpoint): players with 1-2 weeks have
# meaningless std -- leave them in (n_weeks lets us filter later) or drop?
#
# Stuck after a real attempt? -> 04_solutions.md (1.1)
# =========================================================================
def consistency_features(weekly: pd.DataFrame) -> pd.DataFrame:
    def downside_dev(s, floor=None):
        m = s.mean() if floor is None else floor
        below = s[s<m]
        return ((m-below) ** 2).mean() ** 0.5 if len(below) else 0.0

    result = weekly.groupby(['espn_id', 'season'])['points'].agg(
        weekly_std='std',
        weekly_mean='mean',
        n_weeks='count',
        boom_weeks=lambda s: (s >= BOOM_POINTS).mean(),
        bust_weeks=lambda s: (s < BUST_POINTS).mean(),
        downside_dev=downside_dev,
    )

    result['weekly_cv'] = result['weekly_std'] / result['weekly_mean']
    # Near-zero (or negative) weekly means blow weekly_cv up to +/-inf, which
    # errors sklearn models outright. Coerce to NaN so it flows through the same
    # imputation path as any other missing value (and gets flagged by the
    # consistency_missing indicator added at assembly). See design rationale s3.
    result['weekly_cv'] = result['weekly_cv'].replace([np.inf, -np.inf], np.nan)

    return result.drop(columns = 'weekly_mean').reset_index()

# --------------------------------------------------------- season-level table
# Pre-filled: two left joins. NaN semantics are deliberate and documented:
#   - box stats NaN        -> player-season missing from nflverse crosswalk
#   - consistency NaN      -> season < 2022 (no weekly data) -- the ablation
#     in Phase 3 measures what these columns are worth.

def assemble_season_table(frame: pd.DataFrame, box: pd.DataFrame,
                          cons: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    t = frame.merge(box, on=["espn_id", "season"], how="left")
    t = t.merge(cons, on=["espn_id", "season"], how="left")
    t = t.merge(adv, on=["espn_id", "season"], how="left")
    # Missingness indicator: the consistency left-join yields all-NaN rows for
    # player-seasons with no weekly data (n_weeks becomes NaN). Flag them so the
    # model can learn that "unknown consistency" is itself informative, rather
    # than silently reading a mean-imputed value as if it were observed.
    t[CONSISTENCY_MISSING_FLAG] = t["n_weeks"].isna().astype(int)
    return t


# ------------------------------------------------------------ transition pairs
# =========================================================================
# TODO(you) 1.2 -- the self-join that turns seasons into transitions.
#
# Input: season_table (one row per player-season, 2016-2025).
# Output: one row per (player, season t) that ALSO played in t+1, carrying
#   every season-t column unchanged, plus exactly two new columns:
#     target_ppg    = that player's ppg in season t+1
#     target_games  = games in t+1 (Phase 2 filters evaluation to >= 4)
#
# The cheatsheet section "Building transition pairs" shows the core move:
# join the table to itself with the outcome side's season shifted by one.
# Use how="inner" -- and be ready to say in review WHY inner is correct here
# (hint: it is the rookie exclusion and the retirement filter, happening
# naturally rather than by special-case code).
#
# Stuck after a real attempt? -> 04_solutions.md (1.2)
# =========================================================================
def build_transitions(season_table: pd.DataFrame) -> pd.DataFrame:
    outcomes = season_table[["espn_id", "season", "ppg", "games"]].copy()
    outcomes["season"] = outcomes["season"] - 1
    outcomes = outcomes.rename(columns={ "ppg" : "target_ppg",
                                        "games" : "target_games"})
    return season_table.merge(outcomes, on=["espn_id", "season"], how="inner")

# ------------------------------------------------------------- leakage checks
# =========================================================================
# TODO(you) 1.3 -- assertions that would catch the bugs that ruin projects.
#
# Write asserts (with messages) for at least:
#   a) (espn_id, season) is unique -- a bad join silently DUPLICATES rows,
#      and duplicated rows inflate every downstream metric.
#   b) The only columns starting with "target_" are target_ppg, target_games
#      -- the naming-rule tripwire from the module docstring.
#   c) Feature seasons span FIRST_SEASON..LAST_FEATURE_SEASON and target_ppg
#      is never null (inner join should guarantee it -- verify, don't trust).
#   d) VALUE-LEVEL spot check: pick 2-3 espn_ids, look up their season t+1
#      row in season_table by hand, and assert target_ppg matches it exactly.
#      (a/b/c check shape; only this one checks the join grabbed the RIGHT
#      values -- shape checks pass on beautifully-wrong data.)
#
# Stuck after a real attempt? -> 04_solutions.md (1.3)
# =========================================================================
def run_leakage_checks(transitions: pd.DataFrame,
                       season_table: pd.DataFrame) -> None:
    dup = transitions.duplicated(["espn_id", "season"]).sum()
    assert dup == 0, f"{dup} duplicate (espn_id, season) rows -- join fanned out"

    # (b) The self-join must add EXACTLY the two outcome columns and nothing
    # else. Check what the merge ADDED (set difference vs season_table) rather
    # than scanning a "target_" prefix: `target_share` is a season-t FEATURE
    # that also starts with target_, so a prefix scan false-positives. Set
    # difference is both more precise and collision-proof.
    added = sorted(set(transitions.columns) - set(season_table.columns))
    assert added == ["target_games", "target_ppg"], (
        f"self-join added unexpected columns (expected only the 2 outcomes): {added}")

    assert transitions["season"].between(FIRST_SEASON,
                                         LAST_FEATURE_SEASON).all()
    assert transitions["target_ppg"].notna().all(), "inner join should forbid this"

    lookup = season_table.set_index(["espn_id", "season"])["ppg"]
    for _, row in transitions.sample(3, random_state=0).iterrows():
        expected = lookup.loc[(row["espn_id"], row["season"] + 1)]
        assert row["target_ppg"] == expected, (
            f"{row['name']} {row['season']}: target_ppg {row['target_ppg']} "
            f"!= frame ppg {expected}")
    print("leakage checks passed")


# ------------------------------------------------------------ data dictionary
# Pre-filled: the data dictionary is a CONTRACT -- Phases 2-3 read the parquet
# in a different venv and must not have to guess what a column means.

COL_DOCS = {
    "season": ("Feature season t; the row predicts t+1", "frame", "2016-2024"),
    "espn_id": ("Stable player key across all sources", "frame", "all"),
    "name": ("Player name (identification only -- never a model feature)", "frame", "all"),
    "position_group": ("QB/RB/WR/TE", "frame", "all"),
    "team": ("NFL team in season t (pre-offseason-moves by construction)", "frame", "all"),
    "age": ("Age in season t", "frame/nflverse", "all"),
    "years_exp": ("Seasons of NFL experience at t", "frame/nflverse", "all"),
    "games": ("Games played in season t", "frame", "all"),
    "points": ("Total league fantasy points, season t", "frame (league scoring)", "all"),
    "ppg": ("League fantasy points per game, season t", "frame (league scoring)", "all"),
    **{c: (f"Raw season-t counting stat: {c.replace('_', ' ')}", "nflverse seasonal", "2016-2024")
       for c in BOX_STATS},
    "weekly_std": ("Std of weekly league points, season t", "ESPN weekly wk1-13", "2022+ else NaN"),
    "weekly_cv": ("weekly_std / weekly mean", "derived", "2022+ else NaN"),
    "downside_dev": ("Std over below-mean weeks only (bust risk)", "derived", "2022+ else NaN"),
    "boom_weeks": (f"Share of weeks >= {BOOM_POINTS} pts", "derived", "2022+ else NaN"),
    "bust_weeks": (f"Share of weeks < {BUST_POINTS} pts", "derived", "2022+ else NaN"),
    "n_weeks": ("Weekly rows behind the consistency stats", "derived", "2022+ else NaN"),
    "consistency_missing": ("1 if no weekly consistency data for this player-season (pre-2022 etc.)", "derived", "all"),
    "snap_pct": ("ADVANCED tier: share of team snaps played", "nflverse advanced", "~69%, sparse"),
    "target_share": ("ADVANCED tier: share of team targets", "nflverse advanced", "~23%, sparse"),
    "wopr": ("ADVANCED tier: weighted opportunity rating", "nflverse advanced", "~23%, sparse"),
    "receiving_epa": ("ADVANCED tier: receiving expected points added", "nflverse advanced", "~23%, sparse"),
    "rushing_epa": ("ADVANCED tier: rushing expected points added", "nflverse advanced", "~14%, sparse"),
    "target_ppg": ("TARGET: league PPG in season t+1", "frame", "2017-2025"),
    "target_games": ("Games in t+1; evaluation filters to >= 4 (survivorship: disclosed)", "frame", "2017-2025"),
}


def write_data_dictionary(transitions: pd.DataFrame) -> None:
    lines = [
        "# TabFM transitions — data dictionary",
        "",
        "> Generated by `src/research/tabfm/build_dataset.py`. One row per",
        "> player per (season t -> t+1). **Feature columns describe season t",
        "> only; the only future information is in the `target_` columns.**",
        "",
        f"Rows: **{len(transitions)}** · file: `data/processed/research/{OUT_PARQUET.name}`",
        "",
        "| column | meaning | source | coverage |",
        "|---|---|---|---|",
    ]
    for col in transitions.columns:
        desc, src, cov = COL_DOCS.get(col, ("UNDOCUMENTED -- add to COL_DOCS", "?", "?"))
        lines.append(f"| `{col}` | {desc} | {src} | {cov} |")
    lines += [
        "",
        "Known, disclosed properties: rookie seasons cannot appear as rows",
        "(no season-t stats); players who left the league after t drop out",
        "(inner join); weekly consistency covers weeks 1-13 only; `team` is",
        "the season-t team, so offseason moves are unknown to the model.",
        "",
        "## Feature tiers (for the Phase-2/3 ablation)",
        "",
        "- **Core** — profile (age/team/years_exp), 12 raw box-score counting",
        "  stats, league output (points/ppg). Near-complete coverage.",
        "- **Consistency** — the six `weekly_*`/`*_weeks`/`n_weeks` columns.",
        "  Real for 2022+, NaN before (no weekly data).",
        "- **Advanced** — the five efficiency columns tagged ADVANCED above",
        "  (snap_pct, target_share, wopr, receiving_epa, rushing_epa). Sparse",
        "  and mean-imputed by TabFM, so we **measure** their worth by running",
        "  models with and without this tier rather than trusting them blind.",
        "  Sparser stats (catch_pct, cpoe, passing_epa) are held back on purpose",
        "  -- at 3-9% coverage they'd be ~90% fabricated constant post-imputation.",
        "",
        "## Coverage gap — 2023",
        "",
        "2023 carries ~15% fewer players than its neighbours (457 vs ~550) not",
        "because of any rule change here, but because nflverse's 2023 seasonal",
        "roster crosswalk assigns an `espn_id` to far fewer players that year",
        "(34% missing, vs 16% in 2024), and a row with no `espn_id` cannot join",
        "to the league data so it is dropped. **Audited, not assumed:** of the",
        "~330 missing 2023 players, ~74% are cut / practice-squad / reserve /",
        "inactive / retired (status CUT/DEV/RES/INA/RET) — players who would be",
        "filtered out downstream regardless. The remaining ~26% are active-roster",
        "but overwhelmingly rookies and backups (e.g. Marvin Mims, Jalin Hyatt).",
        "So the effect on fantasy-relevant contributors is small; the main",
        "collateral is a handful of 2023 rookies missing from the feature side of",
        "2023->2024 transitions. Lesson worth keeping: a row-count anomaly is a",
        "question, not a verdict — the answer is in *which* rows.",
    ]
    OUT_DICT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DICT.relative_to(_ROOT)}")


# -------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-inputs", action="store_true",
                    help="preview the three input tables and exit (no downloads)")
    ap.add_argument("--spot", metavar="NAME",
                    help="after building, print this player's transition rows")
    args = ap.parse_args()

    if args.check_inputs:
        frame = load_season_frame()
        weekly = load_weekly()
        print("season frame:", frame.shape)
        print(frame.groupby("season").size().rename("players/season"), "\n")
        print("weekly:", weekly.shape, "| seasons:",
              sorted(weekly["season"].unique()))
        cached = [s for s in range(FIRST_SEASON, LAST_FEATURE_SEASON + 1)
                  if (PROCESSED / f"player_boxscore_stats_{s}.csv").exists()]
        missing = [s for s in range(FIRST_SEASON, LAST_FEATURE_SEASON + 1)
                   if s not in cached]
        print(f"box scores cached: {cached}")
        print(f"box scores to download on first full run: {missing}")
        return

    frame = load_season_frame()
    box = load_boxscores()
    weekly = load_weekly()
    adv = load_advanced()

    cons = consistency_features(weekly)                    # TODO 1.1
    season_table = assemble_season_table(frame, box, cons, adv)
    transitions = build_transitions(season_table)          # TODO 1.2
    run_leakage_checks(transitions, season_table)          # TODO 1.3

    # Attrition report: how many feature rows found no t+1 partner.
    eligible = season_table[season_table["season"] <= LAST_FEATURE_SEASON]
    lost = len(eligible) - len(transitions)
    print(f"\ntransitions: {len(transitions)} rows "
          f"({lost} of {len(eligible)} eligible player-seasons had no t+1 "
          f"season -- retirements/out-of-league; disclosed, not hidden)")
    print(transitions.groupby("season").size().rename("transitions/season"))

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET.relative_to(_ROOT)}")
    write_data_dictionary(transitions)

    if args.spot:
        rows = transitions[transitions["name"].str.contains(args.spot, case=False, na=False)]
        cols = ["season", "name", "position_group", "team", "age", "games",
                "ppg", "weekly_std", "target_ppg", "target_games"]
        print(f"\nspot check -- {args.spot}:")
        print(rows[cols].to_string(index=False) if len(rows) else "  (no rows matched)")


if __name__ == "__main__":
    main()
