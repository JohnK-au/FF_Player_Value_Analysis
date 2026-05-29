"""Fair-value model — two complementary lenses on the same ``surplus = actual − fair``.

**Lens 1 (PRIMARY): production-anchored.** Fair value comes from on-field value,
not from prices. Each player's value-over-replacement (VOR, in our scoring) is
computed against league-wide positional replacement levels (``models.production``),
then VOR is priced into cap units by redistributing the league's *total*
skill-position spend in proportion to VOR. Because the benchmark is production,
``surplus_prod`` flags mispricing even when the *whole* market is systematically
off (e.g. overpaying elite QBs, whose VOR is low in a 1-QB league). This is the
lens to trust for "who's over/under-valued".

**Lens 2 (secondary): market-fit.** A model that *predicts* the league's actual
salaries from production + age + position + advanced metrics + draft + combine
(HistGradientBoosting, out-of-fold so ``surplus`` isn't circular). Its R² measures
how reproducible the market's own pricing is — we deliberately do **not** maximize
it, since a perfect fit would call everyone fairly valued. Kept as a "what the
market pays" comparison against the production-anchored value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import PROCESSED_DIR
from ..data.dataset import build_model_features
from .production import ADV_NUM, COMBINE_NUM, DRAFT_NUM, vor_table, weekly_consistency

SKILL = ["QB", "RB", "WR", "TE"]
TARGET = "salary_2026"
CAT_FEATURES = ["position_group"]

# Advanced / draft / combine feature names live in models.production (single
# source of truth, shared with the production model); only BASE_NUM is specific
# to the market-fit model below.
BASE_NUM = ["age", "ppg_2025", "ppg_2024", "games_2025", "stdev_2025", "years_exp"]
NUM_FEATURES = BASE_NUM + ADV_NUM + DRAFT_NUM + COMBINE_NUM  # full set (viz imports this)

# Consistency factor: scale a player's value down by RELATIVE downside (volatility/mean)
# as a bounded multiplicative factor in [MIN_CONSISTENCY_FACTOR, 1.0]. Volatile-but-startable
# players are penalized but never zeroed out — no asymmetric subtraction.
RISK_LAMBDA = 0.5
MIN_CONSISTENCY_FACTOR = 0.5

# Deep-baseline pricing: redistribute spend over value above a DEEPER baseline than the
# starter replacement (a fraction of it), so most rostered players have positive fair
# value. Replaces an earlier degenerate scheme that floored ~84% of rostered players to
# fair=$1 with a single player absorbing nearly the whole cap pool.
DEEP_FACTOR = 0.5

# Dynasty discount: how much each future contract year is worth relative to year 1.
# At 0.10/yr, a year-2 dollar is worth 0.91, year-3 is worth 0.83, year-5 ≈ 0.68.
DISCOUNT_RATE = 0.10


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["position_group"].isin(SKILL)].copy()
    d = d[d[TARGET].notna() & d["ppg_2025"].notna()]
    d["ppg_2024"] = d["ppg_2024"].fillna(d["ppg_2025"])  # no prior year → assume similar
    # advanced/combine NaNs are left for the model to handle natively.
    return d.reset_index(drop=True)


def _pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
        remainder="passthrough",
    )
    gbm = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def _oof(d: pd.DataFrame, num_features: list[str]) -> np.ndarray:
    X = d[CAT_FEATURES + num_features]
    cv = KFold(5, shuffle=True, random_state=0)
    return cross_val_predict(_pipeline(), X, d[TARGET], cv=cv)


def fair_value_table(num_features: list[str] | None = None) -> tuple[pd.DataFrame, Pipeline]:
    """Per-player fair salary + surplus (out-of-fold) and the fitted model."""
    num_features = num_features or NUM_FEATURES
    d = _prepare(build_model_features())
    d["fair_salary"] = np.clip(_oof(d, num_features), 1, None).round(1)
    d["surplus"] = (d[TARGET] - d["fair_salary"]).round(1)  # + overpaid / − bargain
    model = _pipeline().fit(d[CAT_FEATURES + num_features], d[TARGET])
    return d, model


# --- Lens 1: production-anchored value (the one to trust) ---------------------
def production_value_table(
    season: int = 2025,
    risk_lambda: float = RISK_LAMBDA,
    deep_factor: float = DEEP_FACTOR,
) -> tuple[pd.DataFrame, float, dict]:
    """Our 2026 contract skill players priced by production-anchored fair value.

    Pricing recipe (replaces an earlier degenerate scheme — subtractive risk penalty
    + $1 floor — that floored ~84% of rostered players and concentrated nearly the
    whole cap pool on one player):

    1. **Consistency factor** (bounded, multiplicative): ``prod_adj = ppg_full ×
       max(MIN_CONSISTENCY_FACTOR, 1 − risk_lambda · downside/ppg_full)``. A
       volatile-but-startable player is penalized but never zeroed (``downside`` =
       RMS of weekly shortfalls below the player's own mean, from the 13-week box
       scores). Players without weekly data take factor 1.0.
    2. **Deep baseline** per position: ``deep_baseline = deep_factor × starter
       replacement`` PPG. Most rostered players have positive ``deep_vor =
       prod_adj − deep_baseline``.
    3. **Fair value** redistributes the league's total skill-cap spend in proportion
       to each player's positive ``deep_vor``. ``surplus_prod = salary − prod_fair``
       flags over/under-payment. Sub-baseline players get ``prod_fair = 0`` (the whole
       salary registers as surplus / overpaid).

    Kept as diagnostics (not used for pricing): ``vor``/``vor_adj`` (starter-VOR view),
    ``replacement`` (starter PPG threshold). PPG basis = full-season (the only NFL-wide,
    position-comparable measure).
    """
    from ..data.dataset import build_player_dataset

    vor, repl = vor_table(season)
    base = build_player_dataset()
    base = base[base["position_group"].isin(SKILL)].copy()
    base["espn_id"] = pd.to_numeric(base["espn_id"], errors="coerce").astype("Int64")
    # The contract sheet lists a few players twice (typo'd name variants, e.g.
    # "Shadeur"/"sheduer"); collapse to one row per player so VOR isn't double-counted.
    base = base.dropna(subset=["espn_id"]).drop_duplicates("espn_id")

    keep = vor[["espn_id", "ppg", "replacement", "vor"]].rename(columns={"ppg": "ppg_full"})
    m = base.merge(keep, on="espn_id", how="left")
    m = m.merge(weekly_consistency(season), on="espn_id", how="left")

    # Consistency: bounded multiplicative factor on relative downside (volatility/mean).
    rel = (risk_lambda * m["downside"] / m["ppg_full"]).fillna(0.0)
    m["consistency_factor"] = (1 - rel).clip(lower=MIN_CONSISTENCY_FACTOR, upper=1.0).round(3)
    m["prod_adj"] = (m["ppg_full"] * m["consistency_factor"]).round(2)
    # Legacy starter-VOR diagnostics (not used for pricing, retained for reporting).
    m["vor_adj"] = (m["vor"] - risk_lambda * m["downside"].fillna(0.0)).round(2)

    # Deep baseline per position (fraction of starter replacement) → deep VOR.
    deep_bl = {pos: round(deep_factor * v, 2) for pos, v in repl.items()}
    m["deep_baseline"] = m["position_group"].map(deep_bl)
    m["deep_vor"] = (m["prod_adj"] - m["deep_baseline"]).round(2).clip(lower=0)

    priced_pool = m[m["prod_adj"].notna() & (m["deep_vor"] > 0)]
    total_spend = m.loc[m["prod_adj"].notna(), "salary_2026"].sum()
    pool_vor = priced_pool["deep_vor"].sum()
    rate = float(total_spend / pool_vor) if pool_vor > 0 else 0.0
    m["prod_fair"] = (m["deep_vor"] * rate).round(1)
    m["surplus_prod"] = (m["salary_2026"] - m["prod_fair"]).round(1)
    return m, rate, repl


def projected_value_table(
    target_season: int = 2026,
    risk_lambda: float = RISK_LAMBDA,
    deep_factor: float = DEEP_FACTOR,
) -> tuple[pd.DataFrame, float, dict]:
    """Same pricing recipe as ``production_value_table`` but on **projected** PPG.

    Replaces single-season 2025 PPG with the next-season projection (S3) as the value
    basis — de-flukes single-season distortions (Jefferson's down 2025 → projected
    rebound). Consistency factor still applies (the volatility risk doesn't go away).
    Replacement levels are computed NFL-wide on projected PPG.
    """
    from ..data.dataset import build_player_dataset
    from .projection import projected_vor_table

    pvor, repl = projected_vor_table(target_season)
    base = build_player_dataset()
    base = base[base["position_group"].isin(SKILL)].copy()
    base["espn_id"] = pd.to_numeric(base["espn_id"], errors="coerce").astype("Int64")
    base = base.dropna(subset=["espn_id"]).drop_duplicates("espn_id")

    keep = pvor[["espn_id", "projected_ppg", "replacement", "vor"]].rename(
        columns={"replacement": "replacement_proj", "vor": "vor_proj"}
    )
    m = base.merge(keep, on="espn_id", how="left")
    m = m.merge(weekly_consistency(target_season - 1), on="espn_id", how="left")

    rel = (risk_lambda * m["downside"] / m["projected_ppg"]).fillna(0.0)
    m["consistency_factor"] = (1 - rel).clip(lower=MIN_CONSISTENCY_FACTOR, upper=1.0).round(3)
    m["prod_adj_proj"] = (m["projected_ppg"] * m["consistency_factor"]).round(2)

    deep_bl = {pos: round(deep_factor * v, 2) for pos, v in repl.items()}
    m["deep_baseline_proj"] = m["position_group"].map(deep_bl)
    m["deep_vor_proj"] = (m["prod_adj_proj"] - m["deep_baseline_proj"]).round(2).clip(lower=0)

    priced_pool = m[m["prod_adj_proj"].notna() & (m["deep_vor_proj"] > 0)]
    total_spend = m.loc[m["prod_adj_proj"].notna(), "salary_2026"].sum()
    pool_vor = priced_pool["deep_vor_proj"].sum()
    rate = float(total_spend / pool_vor) if pool_vor > 0 else 0.0
    m["value_2026"] = (m["deep_vor_proj"] * rate).round(1)
    m["surplus_2026"] = (m["salary_2026"] - m["value_2026"]).round(1)
    return m, rate, repl


def dynasty_value_table(
    target_season: int = 2026,
    risk_lambda: float = RISK_LAMBDA,
    deep_factor: float = DEEP_FACTOR,
    discount_rate: float = DISCOUNT_RATE,
) -> tuple[pd.DataFrame, float, dict]:
    """Multi-year dynasty value: discounted sum of projected per-year fair values.

    For each contract player, project PPG year-by-year via positional age curves
    (`projection.project_multiyear`), apply the same consistency factor + deep
    baseline + redistribution rate per year, discount future years by
    ``discount_rate``, and sum. ``dynasty_surplus`` compares the discounted total
    value to the remaining cap cost (``salary_2026 × years_2026``).
    """
    from ..data.contracts import build_2026_contracts
    from .projection import positional_age_curves, project_multiyear

    base, rate, repl = projected_value_table(target_season, risk_lambda, deep_factor)
    deep_bl = {pos: round(deep_factor * v, 2) for pos, v in repl.items()}

    contracts, _ = build_2026_contracts()
    yc = contracts[["team", "player", "years_2026"]].drop_duplicates(["team", "player"])
    m = base.merge(yc, on=["team", "player"], how="left")
    m["years_2026"] = m["years_2026"].fillna(1).clip(lower=1, upper=5).astype(int)

    curves = positional_age_curves()
    dyn_values = []
    for _, r in m.iterrows():
        if pd.isna(r["projected_ppg"]) or pd.isna(r["age"]):
            dyn_values.append(np.nan)
            continue
        per_year = project_multiyear(
            float(r["projected_ppg"]), r["position_group"],
            float(r["age"]), int(r["years_2026"]), curves,
        )
        cf = float(r["consistency_factor"]) if pd.notna(r["consistency_factor"]) else 1.0
        dl = float(deep_bl.get(r["position_group"], 0.0))
        total = 0.0
        for y, ppg_y in enumerate(per_year, start=1):
            prod_adj_y = ppg_y * cf
            deep_vor_y = max(0.0, prod_adj_y - dl)
            fair_y = deep_vor_y * rate
            total += fair_y / ((1.0 + discount_rate) ** (y - 1))
        dyn_values.append(round(total, 1))
    m["dynasty_value"] = dyn_values
    m["dynasty_total_salary"] = (m["salary_2026"] * m["years_2026"]).round(1)
    m["dynasty_surplus"] = (m["dynasty_total_salary"] - m["dynasty_value"]).round(1)
    return m, rate, repl


def player_value_2026(
    target_season: int = 2026,
    risk_lambda: float = RISK_LAMBDA,
    deep_factor: float = DEEP_FACTOR,
    discount_rate: float = DISCOUNT_RATE,
) -> tuple[pd.DataFrame, float, dict]:
    """Consolidated master value table — current-season + dynasty + market + context.

    Combines:
      - Projection-based current-season value (``value_2026``, ``surplus_2026``)
      - Dynasty multi-year value (``dynasty_value``, ``dynasty_surplus``)
      - Market-fit lens (``market_fair``, ``surplus_market``)
      - S2 longitudinal context (``year_type``, ``usage_trend``, ``results_trend``, ``qb_context``)

    This is the table the Streamlit app reads. Saved by the CLI to
    ``data/processed/player_value_2026.csv``.
    """
    from ..data.context import build_context_2026

    dyn, rate, repl = dynasty_value_table(target_season, risk_lambda, deep_factor, discount_rate)
    market, _ = fair_value_table()
    mk = (
        market[["espn_id", "fair_salary", "surplus"]]
        .dropna(subset=["espn_id"])
        .drop_duplicates("espn_id")
        .rename(columns={"fair_salary": "market_fair", "surplus": "surplus_market"})
    )
    ctx = build_context_2026()[
        ["espn_id", "year_type", "usage_trend", "results_trend", "qb_context",
         "ppg_baseline", "ppg_z"]
    ]
    out = dyn.merge(mk, on="espn_id", how="left").merge(ctx, on="espn_id", how="left")
    return out, rate, repl


def combined_value_table(
    season: int = 2025, risk_lambda: float = RISK_LAMBDA
) -> tuple[pd.DataFrame, float, dict]:
    """Both lenses side by side: production-anchored value + market-fit price."""
    prod, rate, repl = production_value_table(season, risk_lambda)
    market, _ = fair_value_table()
    mk = (
        market[["espn_id", "fair_salary", "surplus"]]
        .dropna(subset=["espn_id"])
        .drop_duplicates("espn_id")  # one market price per player (avoid cross-join)
        .rename(columns={"fair_salary": "market_fair", "surplus": "surplus_market"})
    )
    return prod.merge(mk, on="espn_id", how="left"), rate, repl


if __name__ == "__main__":
    tbl, rate, repl = combined_value_table()
    priced = tbl[tbl["prod_fair"].notna()]
    cols = ["player", "team", "position_group", "salary_2026", "ppg_full", "downside",
            "consistency_factor", "prod_adj", "deep_vor", "prod_fair", "surplus_prod"]

    print("=== PRODUCTION-ANCHORED fair value (PRIMARY lens) ===")
    print(f"Starter replacement PPG: "
          + ", ".join(f"{p} {v:.1f}" for p, v in repl.items())
          + f"  |  deep baseline = {DEEP_FACTOR} x replacement (pricing floor)")
    floored = int((priced["prod_fair"] == 0).sum())
    print(f"League rate: {rate:.2f} cap units per deep-VOR point  |  "
          f"{floored} of {len(priced)} priced players sub-baseline (prod_fair=0)\n")
    print("Most UNDER-valued by production (bargains):")
    print(priced.nsmallest(10, "surplus_prod")[cols].to_string(index=False))
    print("\nMost OVER-valued by production:")
    print(priced.nlargest(10, "surplus_prod")[cols].to_string(index=False))

    movers = priced[priced["downside"].notna() & (priced["ppg_full"] > 0)].copy()
    movers["markdown_ppg"] = (movers["ppg_full"] - movers["prod_adj"]).round(2)
    print("\nBiggest consistency markdowns (highest relative downside — PPG shaved off):")
    print(movers.nlargest(8, "markdown_ppg")[
        ["player", "team", "position_group", "ppg_full", "downside",
         "consistency_factor", "markdown_ppg", "prod_adj"]
    ].to_string(index=False))

    # Lens 2 diagnostic: how reproducible is the market's own pricing? (not maximized)
    d = _prepare(build_model_features())
    base_oof = _oof(d, BASE_NUM)
    full_oof = np.clip(_oof(d, NUM_FEATURES), 1, None)
    print(f"\n=== MARKET-FIT model (secondary lens) — n={len(d)} skill players ===")
    print(f"  out-of-fold R²: baseline {r2_score(d[TARGET], base_oof):.3f} | "
          f"+advanced {r2_score(d[TARGET], full_oof):.3f} "
          f"(MAE {mean_absolute_error(d[TARGET], full_oof):.1f})")
    print("  (measures how well the market's pricing is reproduced — deliberately "
          "NOT maximized; the gap between this and production value is the signal)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(PROCESSED_DIR / "fair_value_2026.csv", index=False)
    print(f"\nSaved single-season both-lens table -> {PROCESSED_DIR / 'fair_value_2026.csv'}")

    # ============================================================================
    # PROJECTION-BASED CURRENT VALUE + DYNASTY (the consolidated master table)
    # ============================================================================
    print("\n\n=== PROJECTION-BASED current + DYNASTY value ===")
    master, p_rate, p_repl = player_value_2026()
    priced = master[master["value_2026"].notna() & (master["projected_ppg"].notna())].copy()
    print(f"Replacement (projected PPG): "
          + ", ".join(f"{p} {v:.1f}" for p, v in p_repl.items()))
    floored_p = int((priced["value_2026"] == 0).sum())
    print(f"League rate: {p_rate:.2f} cap units per deep-VOR point  |  "
          f"{floored_p} of {len(priced)} sub-baseline (value_2026=0)  |  "
          f"discount = {DISCOUNT_RATE}/yr (dynasty)\n")

    cur_cols = ["player", "team", "position_group", "salary_2026", "ppg_2025",
                "projected_ppg", "value_2026", "surplus_2026", "year_type"]
    print("Most UNDER-valued (current 2026, projection-based):")
    print(priced.nsmallest(10, "surplus_2026")[cur_cols].to_string(index=False))
    print("\nMost OVER-valued (current 2026):")
    print(priced.nlargest(10, "surplus_2026")[cur_cols].to_string(index=False))

    dyn_cols = ["player", "team", "position_group", "salary_2026", "years_2026",
                "dynasty_total_salary", "dynasty_value", "dynasty_surplus", "year_type"]
    dyn_priced = priced[priced["dynasty_value"].notna()].copy()
    print("\nMost UNDER-valued (dynasty, multi-year discounted):")
    print(dyn_priced.nsmallest(10, "dynasty_surplus")[dyn_cols].to_string(index=False))
    print("\nMost OVER-valued (dynasty):")
    print(dyn_priced.nlargest(10, "dynasty_surplus")[dyn_cols].to_string(index=False))

    # Headline: did the projection rehabilitate Jefferson, AJ Brown, etc.?
    notable = ["Justin Jefferson", "AJ Brown", "Puka Nacua", "Bijan Robinson",
               "Mark Andrews", "TJ Hockenson", "Drake Maye", "Jahmyr Gibbs"]
    cmp_cols = ["player", "team", "position_group", "salary_2026",
                "ppg_2025", "projected_ppg", "value_2026", "surplus_2026",
                "dynasty_surplus", "year_type"]
    notable_rows = priced[priced["player"].isin(notable)].copy()
    if len(notable_rows):
        notable_rows["_p"] = notable_rows["position_group"].map({"QB":0,"RB":1,"WR":2,"TE":3})
        notable_rows = notable_rows.sort_values(["_p", "player"])
        print("\nNotable players — single-season vs projection-based vs dynasty surplus:")
        print(notable_rows[cmp_cols].to_string(index=False))

    out_master = PROCESSED_DIR / "player_value_2026.csv"
    master.to_csv(out_master, index=False)
    print(f"\nSaved consolidated master table -> {out_master}")
