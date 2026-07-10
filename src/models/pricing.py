"""Cap-unit pricing engine (v2 -- 4-stage pipeline).

Layer ON TOP of the V2 quality framework. V2 (src/models/components/) produces
quality scores in [0, 100]; this module translates them into cap-unit fair
values that reflect four league-market dynamics:

  1. Elite premium -- top producers command disproportionate share (non-linear
     concentration via alpha > 1)
  2. Mid-tier collapse to baseline -- if similar production is reachable from
     the FA pool, teams don't pay much for it (per-position replacement floor)
  3. Explicit age emphasis -- pricing discounts older players ON TOP OF DV's
     20% age weight (multiplicative age band)
  4. Multi-year age decay -- long deals on aging players discount future years
     (per-year age curve projection)

V2 master CSV is NEVER modified. This module reads it, derives pricing, and
writes `data/processed/player_pricing_2026.csv` joined on espn_id.

4-stage pipeline
----------------
Stage 1: replacement_dv[pos] = mean DV of the "replacement tier" per position
         Tiers (from v2 Position VORP-Deep methodology; N = 8 * S_p):
           backup: ranks N+1  to 2N
           fa    : ranks 2N+1 to 3N  (default -- realistic FA grab)
           deep  : ranks 3N+1 to 4N  (deep FA / desperate replacement)

Stage 2: above_baseline_dv = max(0, dynasty_value - replacement_dv[pos])
         Mid-tier and below collapse to 0.

Stage 3: scarcity_value = above_baseline_dv ** alpha
         Alpha > 1 concentrates pool on elites.

Stage 4: rate = pool / sum(scarcity_value for rostered skill)
         base_fair = scarcity_value * rate
         For each contract year:
             projected_age = age + (y - 1)
             year_age_mult = band_lo + (age_curve(proj_age)/100) * (band_hi - band_lo)
             year_fair     = base_fair * year_age_mult
         fair_value_2026    = year 1's fair
         fair_value_dynasty = sum(year_fair for y in years_2026)

Sign convention (matches V1 app):
    surplus_2026    = salary_2026 - fair_value_2026        # + = overpaid
    surplus_dynasty = dynasty_total_salary - fair_value_dynasty
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.models.components.age import AGE_PARAMS, _logistic_age_value
from src.models.components.position import SLOT_COUNTS

V2_MASTER_CSV = Path(PROCESSED_DIR) / "player_value_v2_2026.csv"
PRICING_CSV   = Path(PROCESSED_DIR) / "player_pricing_2026.csv"

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
TOTAL_CAP_PER_TEAM = 1500
N_TEAMS = 8
SKILL_SHARE_RULE = 0.80

VALID_BASES = ("dynasty_value", "on_field_value")
VALID_POOLS = ("empirical", "rule")
VALID_TIERS = ("backup", "fa", "deep")

# Tier -> (lo_multiplier, hi_multiplier) applied to N = 8 * S_p
TIER_RANGES: dict[str, tuple[int, int]] = {
    "backup": (1, 2),   # ranks N+1  to 2N
    "fa":     (2, 3),   # ranks 2N+1 to 3N
    "deep":   (3, 4),   # ranks 3N+1 to 4N
}

# Preset parameter combinations for the workshop (v3 -- final calibration).
# User feedback iterations (2026-07-03):
#   1. alpha=1.3 too aggressive -> settled at alpha=1.25
#   2. Repl DV should drop 10-20 pts -> settled on user-picked per-pos baselines:
#      QB=41, RB=29, WR=34, TE=31
#   3. Elite fair values still too low vs user's intuition -> add pool_scale to
#      allow model to reflect that teams collectively underspend on skill
#      (sum of fair > actual pool)
#
# All presets fix alpha=1.25 and user's baselines; vary only pool_scale to let
# user pick between "empirical self-balancing" (scale=1.0) and "elites hit
# intuition targets" (scale=1.45).
USER_BASELINES = {"QB": 41.0, "RB": 29.0, "WR": 34.0, "TE": 31.0}
USER_ALPHA = 1.25
USER_AGE_BAND = (0.85, 1.15)

PRESETS: dict[str, dict] = {
    "A_market_match": {
        "alpha": USER_ALPHA, "baseline_tier": "fa",
        "baseline_override": USER_BASELINES,
        "pool_scale": 1.0,
        "age_band": USER_AGE_BAND,
        "label": "A -- Market Match (pool_scale=1.0)",
        "desc": ("Empirical self-balancing pool. Gibbs ~\\$165 (below current market "
                 "top of \\$200), Chase ~\\$123, McBride ~\\$121, McLaurin ~\\$18. "
                 "Mean rostered surplus ≈ 0 by construction (fair prices sum to "
                 "actual league spend).")
    },
    "B_moderate_premium": {
        "alpha": USER_ALPHA, "baseline_tier": "fa",
        "baseline_override": USER_BASELINES,
        "pool_scale": 1.3,
        "age_band": USER_AGE_BAND,
        "label": "B -- Moderate Premium (pool_scale=1.3)",
        "desc": ("Pool scaled 1.3x. Gibbs ~\\$215, JSN ~\\$175, Chase ~\\$160, "
                 "McBride ~\\$157, Maye ~\\$121, McLaurin ~\\$24. Model implies "
                 "teams underspend on skill by ~15% collectively.")
    },
    "C_user_targets": {
        "alpha": USER_ALPHA, "baseline_tier": "fa",
        "baseline_override": USER_BASELINES,
        "pool_scale": 1.45,
        "age_band": USER_AGE_BAND,
        "label": "C -- User Targets (pool_scale=1.45) [Recommended]",
        "desc": ("Pool scaled 1.45x. Hits user's stated elite target ranges: "
                 "Gibbs/Bijan ~\\$230-240, JSN/Puka/Chase ~\\$180-195, "
                 "McBride ~\\$175, Bowers ~\\$154, Maye ~\\$134, Allen ~\\$121. "
                 "McLaurin ~\\$26, DJ Moore ~\\$52. Model implies teams underspend "
                 "on skill by ~30% collectively.")
    },
    "D_high_pool": {
        "alpha": USER_ALPHA, "baseline_tier": "fa",
        "baseline_override": USER_BASELINES,
        "pool_scale": 1.5,
        "age_band": USER_AGE_BAND,
        "label": "D -- High Pool (pool_scale=1.5)",
        "desc": ("Pool scaled 1.5x. Gibbs \\$248, Bijan \\$239, JSN \\$201, "
                 "McBride \\$181. Elite RBs top of target range; WRs finally hit "
                 "\\$200. Model implies teams underspend by ~33%.")
    },
}


# ---- Stage 1: per-position replacement baseline ------------------------------

def per_position_baseline(
    basis_col: str,
    tier: str,
    master_df: pd.DataFrame,
    offset: float = 0.0,
) -> dict[str, float]:
    """For each skill position, compute the mean of `basis_col` in the given tier.

    Uses the entire priced universe (rostered + FAs) to compute ranks, then averages
    the DV over ranks (N * lo_mult + 1) to (N * hi_mult), where N = 8 * S_p.

    `offset` is subtracted from each per-position baseline (e.g., offset=15 lowers all
    baselines by 15 DV, letting more mid-tier players qualify as above-baseline).
    """
    if tier not in TIER_RANGES:
        raise ValueError(f"unknown tier {tier!r}; valid: {VALID_TIERS}")
    lo_mult, hi_mult = TIER_RANGES[tier]

    baselines: dict[str, float] = {}
    for pos in SKILL_POSITIONS:
        pool = (
            master_df[master_df["position_group"] == pos]
            .dropna(subset=[basis_col])
            .sort_values(basis_col, ascending=False)
            .reset_index(drop=True)
        )
        S = SLOT_COUNTS.get(pos, 1.0)
        N = int(round(N_TEAMS * S))
        lo = int(N * lo_mult)   # 0-indexed: rank (N * lo_mult + 1) -> index N * lo_mult
        hi = int(N * hi_mult)

        if len(pool) > hi:
            tier_slice = pool.iloc[lo:hi]
        elif len(pool) > lo:
            tier_slice = pool.iloc[lo:]
        else:
            tier_slice = pool.tail(1)

        baselines[pos] = float(tier_slice[basis_col].mean()) - offset
    return baselines


# ---- Stage 2 + 3: scarcity value --------------------------------------------

def scarcity_value(
    basis_col: str,
    baseline_map: dict[str, float],
    alpha: float,
    master_df: pd.DataFrame,
) -> pd.Series:
    """Above-baseline DV raised to alpha, per player. Non-skill positions -> 0."""
    def _one(row):
        pos = row["position_group"]
        if pos not in SKILL_POSITIONS:
            return 0.0
        bl = baseline_map.get(pos, 0.0)
        above = max(0.0, float(row[basis_col]) - bl)
        return above ** alpha
    return master_df.apply(_one, axis=1)


# ---- Stage 4: age multiplier + multi-year decay ------------------------------

def age_multiplier(
    age: float,
    position: str,
    band: tuple[float, float],
) -> float:
    """Multiplicative age adjustment in `band`. Uses v2 Age component's per-position
    logistic curve (see components/age.AGE_PARAMS). NaN age -> 1.0 (neutral)."""
    if pd.isna(age):
        return 1.0
    lo, hi = band
    params = AGE_PARAMS.get(position)
    if params is None:
        return 1.0  # non-skill / no curve -> neutral
    age_val = _logistic_age_value(float(age), params["center"], params["steepness"])
    return lo + (age_val / 100.0) * (hi - lo)


def fair_value_dynasty(
    base_fair: float,
    age: float,
    years: float,
    position: str,
    band: tuple[float, float],
) -> tuple[float, float]:
    """Return (fair_value_2026, fair_value_dynasty) with per-year age decay.

    fair_value_2026    = base_fair * age_multiplier(current age)
    fair_value_dynasty = sum over years [1..years_2026] with projected_age = age + (y-1)
    """
    if pd.isna(age) or pd.isna(base_fair) or base_fair <= 0:
        return 0.0, 0.0
    yrs = int(max(1, round(float(years)))) if not pd.isna(years) else 1
    yr1_mult = age_multiplier(age, position, band)
    fair_1 = base_fair * yr1_mult

    total = 0.0
    for y in range(1, yrs + 1):
        proj_age = float(age) + (y - 1)
        m = age_multiplier(proj_age, position, band)
        total += base_fair * m
    return fair_1, total


# ---- Pools ------------------------------------------------------------------

def empirical_pool(master_df: pd.DataFrame) -> float:
    """Sum of actual 2026 salary for rostered (non-FA) skill players."""
    rostered = master_df[
        master_df["position_group"].isin(SKILL_POSITIONS)
        & (master_df["roster_status"] != "fa")
    ]
    return float(rostered["salary_2026"].sum())


def rule_based_pool() -> float:
    """N_TEAMS * cap_per_team * SKILL_SHARE_RULE = 8 * 1500 * 0.80 = 9600."""
    return TOTAL_CAP_PER_TEAM * N_TEAMS * SKILL_SHARE_RULE


def pool_for(method: str, master_df: pd.DataFrame) -> float:
    if method == "empirical":
        return empirical_pool(master_df)
    if method == "rule":
        return rule_based_pool()
    raise ValueError(f"unknown pool method {method!r}; valid: {VALID_POOLS}")


# ---- Orchestrator -----------------------------------------------------------

def build_pricing(
    basis: str = "dynasty_value",
    pool_method: str = "empirical",
    pool_scale: float = 1.0,
    baseline_tier: str = "fa",
    baseline_offset: float = 0.0,
    baseline_override: dict[str, float] | None = None,
    alpha: float = 1.7,
    age_band: tuple[float, float] = (0.85, 1.15),
    master_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the V2 master extended with 4-stage pricing columns.

    `pool_scale` multiplies the chosen pool (e.g., 1.4 = 40% larger than empirical/rule).
    A pool_scale > 1 means the model believes teams collectively UNDERSPEND on skill
    (sum of fair values > actual/prescribed pool). All fair values scale linearly.

    `baseline_offset` shifts the per-position replacement DV down by this many points
    (e.g., 15 → more mid-tier players get non-zero fair value while elite gap widens).

    `baseline_override` (optional) directly sets per-position baselines, bypassing
    the tier+offset derivation. Format: {"QB": 41, "RB": 29, "WR": 34, "TE": 31}.
    When provided, `baseline_tier` and `baseline_offset` are recorded for provenance
    but the actual baselines come from the override dict.

    Output columns added on top of master:
      pricing_basis, pricing_pool_method, pricing_pool, pricing_pool_scale,
      pricing_baseline_tier, pricing_baseline_offset, pricing_alpha,
      pricing_age_band_lo, pricing_age_band_hi, pricing_rate,
      replacement_dv        (per-position baseline for the player's position),
      above_baseline_dv, scarcity_value, base_fair, age_mult,
      fair_value_2026, surplus_2026, fair_value_dynasty, surplus_dynasty
    """
    if basis not in VALID_BASES:
        raise ValueError(f"unknown basis {basis!r}; valid: {VALID_BASES}")
    if pool_method not in VALID_POOLS:
        raise ValueError(f"unknown pool method {pool_method!r}; valid: {VALID_POOLS}")
    if baseline_tier not in VALID_TIERS:
        raise ValueError(f"unknown baseline_tier {baseline_tier!r}; valid: {VALID_TIERS}")
    if master_df is None:
        master_df = pd.read_csv(V2_MASTER_CSV)

    # Stage 1
    if baseline_override is not None:
        missing = [p for p in SKILL_POSITIONS if p not in baseline_override]
        if missing:
            raise ValueError(f"baseline_override missing positions: {missing}")
        baseline_map = {p: float(baseline_override[p]) for p in SKILL_POSITIONS}
    else:
        baseline_map = per_position_baseline(basis, baseline_tier, master_df, offset=baseline_offset)

    # Stages 2 + 3
    sv = scarcity_value(basis, baseline_map, alpha, master_df)
    # Rate uses rostered skill sum of scarcity_value
    rostered_mask = (
        master_df["position_group"].isin(SKILL_POSITIONS)
        & (master_df["roster_status"] != "fa")
    )
    total_scarcity_rostered = float(sv[rostered_mask].sum())
    base_pool = pool_for(pool_method, master_df)
    pool = base_pool * pool_scale
    rate = pool / total_scarcity_rostered if total_scarcity_rostered > 0 else 0.0

    # Stage 4
    base_fair_series = (sv * rate).astype(float)
    fair_1 = []
    fair_dyn = []
    age_mult_yr1 = []
    for base_fair, age, years, pos in zip(
        base_fair_series,
        master_df["age"],
        master_df["years_2026"],
        master_df["position_group"],
    ):
        f1, fd = fair_value_dynasty(base_fair, age, years, pos, age_band)
        fair_1.append(f1)
        fair_dyn.append(fd)
        age_mult_yr1.append(age_multiplier(age, pos, age_band))

    out = master_df.copy()
    out["pricing_basis"] = basis
    out["pricing_pool_method"] = pool_method
    out["pricing_pool_scale"] = pool_scale
    out["pricing_pool"] = round(pool, 2)
    out["pricing_baseline_tier"] = baseline_tier
    out["pricing_baseline_offset"] = baseline_offset
    out["pricing_alpha"] = alpha
    out["pricing_age_band_lo"] = age_band[0]
    out["pricing_age_band_hi"] = age_band[1]
    out["pricing_rate"] = round(rate, 6)
    out["replacement_dv"] = out["position_group"].map(baseline_map).fillna(0.0).round(2)
    out["above_baseline_dv"] = (
        (out[basis] - out["replacement_dv"]).clip(lower=0)
    ).round(2)
    out["scarcity_value"] = sv.round(4)
    out["base_fair"] = base_fair_series.round(2)
    out["age_mult"] = pd.Series(age_mult_yr1, index=out.index).round(4)
    out["fair_value_2026"] = pd.Series(fair_1, index=out.index).round(2)
    out["surplus_2026"] = (out["salary_2026"] - out["fair_value_2026"]).round(2)
    out["fair_value_dynasty"] = pd.Series(fair_dyn, index=out.index).round(2)
    total_sal = pd.to_numeric(out.get("dynasty_total_salary", 0), errors="coerce")
    out["surplus_dynasty"] = (total_sal - out["fair_value_dynasty"]).round(2)
    return out


def main() -> None:
    """Build pricing with the recommended C_user_targets preset and write CSV."""
    preset = PRESETS["C_user_targets"]
    df = build_pricing(
        basis="dynasty_value",
        pool_method="empirical",
        pool_scale=preset.get("pool_scale", 1.0),
        baseline_tier=preset["baseline_tier"],
        baseline_offset=preset.get("baseline_offset", 0.0),
        baseline_override=preset.get("baseline_override"),
        alpha=preset["alpha"],
        age_band=preset["age_band"],
    )
    df.to_csv(PRICING_CSV, index=False)

    baselines = {
        pos: float(df[df["position_group"] == pos]["replacement_dv"].iloc[0])
        for pos in SKILL_POSITIONS if (df["position_group"] == pos).any()
    }
    print(f"Preset: {preset['label']}")
    print(f"  basis=dynasty_value  pool_method=empirical  baseline_tier={preset['baseline_tier']}  "
          f"alpha={preset['alpha']}  age_band={preset['age_band']}")
    print(f"  Pool: {df['pricing_pool'].iloc[0]:.0f} cap units  Rate: {df['pricing_rate'].iloc[0]:.4f}")
    print(f"  Per-position replacement_dv: " + "  ".join(f"{p}={v:.1f}" for p, v in baselines.items()))
    print(f"wrote {len(df)} rows -> {PRICING_CSV}")

    cols = ["player", "team", "position_group", "roster_status", "age", "salary_2026",
            "years_2026", "dynasty_value", "fair_value_2026", "surplus_2026",
            "fair_value_dynasty", "surplus_dynasty"]
    print(f"\nTop 10 by dynasty_value:")
    print(df.nlargest(10, "dynasty_value")[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nTop 5 most overpaid (rostered, highest positive surplus_2026):")
    ros = df[df["roster_status"] != "fa"]
    print(ros.nlargest(5, "surplus_2026")[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nTop 5 biggest bargains (rostered, most negative surplus_2026):")
    print(ros.nsmallest(5, "surplus_2026")[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
