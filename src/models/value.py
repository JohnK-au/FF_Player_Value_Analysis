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

# Consistency penalty for the production lens: value a player at mean − λ·(weekly
# stdev), so steady scorers outrank boom-bust ones with the same mean (this is a
# weekly head-to-head league). λ=0 → pure mean; ~1 ≈ a one-stdev "floor". Tunable.
RISK_LAMBDA = 0.5


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
    season: int = 2025, risk_lambda: float = RISK_LAMBDA
) -> tuple[pd.DataFrame, float, dict]:
    """Our 2026 contract skill players priced by (risk-adjusted) value-over-replacement.

    ``prod_fair`` redistributes the league's *total* skill-position cap spend
    across players in proportion to their positive VOR — answering "given what the
    league spends in total on skill players, how SHOULD it be allocated by on-field
    value?" rather than reproducing the market's actual prices. ``surplus_prod =
    actual − prod_fair`` (+ overpaid / − bargain) then flags mispricing against
    production, catching even league-wide systematic bias.

    VOR is **risk-adjusted** (``vor_adj = vor − risk_lambda · downside``): in a weekly
    head-to-head league a steady scorer is worth more than a boom-bust one with the
    same mean. The penalty uses **downside deviation** (shortfalls below a player's
    own average — bust weeks lose matchups), not symmetric stdev, so big ceiling
    weeks aren't punished. ``downside`` is from the 13-week box scores (rostered
    players); players without it take no penalty.

    VOR uses full-season PPG (the only NFL-wide, position-comparable basis), so it
    differs slightly from the 13-week ``ppg_2025`` the market-fit model uses.
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

    # Consistency penalty: mark down floor risk via downside deviation (points/game,
    # comparable to PPG). No in-league weekly data → no penalty.
    m = m.merge(weekly_consistency(season), on="espn_id", how="left")
    m["vor_adj"] = (m["vor"] - risk_lambda * m["downside"].fillna(0.0)).round(2)

    priced = m[m["vor"].notna()]
    rate = priced["salary_2026"].sum() / priced["vor_adj"].clip(lower=0).sum()
    m["prod_fair"] = (m["vor_adj"].clip(lower=0) * rate).round(1).clip(lower=1)
    m["surplus_prod"] = (m["salary_2026"] - m["prod_fair"]).round(1)
    return m, float(rate), repl


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
            "vor", "vor_adj", "prod_fair", "surplus_prod"]

    print("=== PRODUCTION-ANCHORED fair value (PRIMARY lens) ===")
    print(f"Replacement PPG by position: "
          + ", ".join(f"{p} {v:.1f}" for p, v in repl.items())
          + f"  |  consistency penalty lambda={RISK_LAMBDA} x weekly downside deviation")
    print(f"League rate: {rate:.2f} cap units per risk-adjusted VOR point — "
          f"redistributes {priced['salary_2026'].sum():.0f} skill-cap units across "
          f"{len(priced)} producing players.\n")
    print("Most UNDER-valued by production (bargains):")
    print(priced.nsmallest(10, "surplus_prod")[cols].to_string(index=False))
    print("\nMost OVER-valued by production:")
    print(priced.nlargest(10, "surplus_prod")[cols].to_string(index=False))

    movers = priced[priced["downside"].notna()].copy()
    movers["markdown"] = (RISK_LAMBDA * movers["downside"]).round(1)
    print("\nBiggest consistency markdowns (worst floor risk — lambda*downside shaved off VOR):")
    print(movers.nlargest(8, "markdown")[
        ["player", "team", "position_group", "ppg_full", "downside", "stdev", "markdown", "vor", "vor_adj"]
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
    print(f"\nSaved both-lens table -> {PROCESSED_DIR / 'fair_value_2026.csv'}")
