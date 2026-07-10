"""V2 Player Comparison Page ("This or That").

Presents two players side by side. User picks who is more valuable across three
lenses: 2026 (single-season), Dynasty (contract-length), Real-life NFL (on-field).

Mechanics:
- Winner-stays king-of-the-hill: after each pick, the loser's slot cycles to a
  new random challenger; the winner remains for streak comparisons.
- Random shuffle default with manual dropdown overrides at any point.
- Anonymous mode hides name, teams, and all salary-derived fields for
  stat-blind evaluation.
- Position filter constrains the challenger pool.

Every selection persists to data/research/user_comparisons.csv. Per-player
comment textareas persist to data/research/player_comments.csv. Both files
are allowlisted in .gitignore.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for p in (_THIS.parents[1], _THIS.parents[3]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from _lib import (  # noqa: E402
    COMPONENT_COLS, POS_ORDER,
    fmt_float_or_dash, fmt_int_or_dash, latest_valuation, load_boxscore_stats,
    load_comparisons, load_comments, load_master, load_season_stats,
    player_headshot_url, save_comment, save_comparison, save_valuation,
    team_logo_url,
)

st.set_page_config(page_title="Compare (V2)", layout="wide")
st.title("This or That -- Player Comparison")

master = load_master()
_stats_df = load_season_stats()
_stats_lookup = {
    int(r["espn_id"]): r.to_dict()
    for _, r in _stats_df.dropna(subset=["espn_id"]).iterrows()
}
_box_lookup = load_boxscore_stats(2024)  # last complete season's counting stats


def _stats_for(espn_id) -> dict:
    if pd.isna(espn_id):
        return {}
    return _stats_lookup.get(int(espn_id), {})


def _box_for(espn_id) -> dict:
    if pd.isna(espn_id):
        return {}
    return _box_lookup.get(int(espn_id), {})

# --- Session state init -----------------------------------------------------

def _random_pair(pool: pd.DataFrame, exclude: set[int] | None = None) -> tuple[int, int] | None:
    ids = pool["espn_id"].dropna().astype(int).tolist()
    if exclude:
        ids = [i for i in ids if i not in exclude]
    if len(ids) < 2:
        return None
    a, b = random.sample(ids, 2)
    return int(a), int(b)


def _random_challenger(pool: pd.DataFrame, exclude: set[int]) -> int | None:
    ids = [int(i) for i in pool["espn_id"].dropna().astype(int).tolist() if int(i) not in exclude]
    if not ids:
        return None
    return random.choice(ids)


POOL_OPTIONS = [
    "All (rostered + FAs)",
    "Rostered only",
    "Rostered + top 50 FAs by OFV",
]


def _pool_espn_ids(pool_scope: str) -> set[int]:
    """Compute the set of eligible espn_ids for the current pool scope.
    Applied to the master AFTER the position filter; both filters intersect."""
    skill = master[master["position_group"].isin(POS_ORDER)]
    if pool_scope == "Rostered only":
        base = skill[skill["roster_status"] != "fa"]
    elif pool_scope == "Rostered + top 50 FAs by OFV":
        rostered = skill[skill["roster_status"] != "fa"]
        fas = skill[skill["roster_status"] == "fa"]
        top_fas = fas.nlargest(50, "on_field_value")
        base = pd.concat([rostered, top_fas], ignore_index=True)
    else:  # "All (rostered + FAs)"
        base = skill
    return set(base["espn_id"].dropna().astype(int).tolist())


def _filtered_pool(pos_filter: list[str], pool_scope: str) -> pd.DataFrame:
    eligible_ids = _pool_espn_ids(pool_scope)
    if not pos_filter:
        return master[
            master["position_group"].isin(POS_ORDER)
            & master["espn_id"].astype("Int64").isin(eligible_ids)
        ]
    return master[
        master["position_group"].isin(pos_filter)
        & master["espn_id"].astype("Int64").isin(eligible_ids)
    ]


# --- Top controls -----------------------------------------------------------

c1, c2, c3, c4 = st.columns([1.6, 1.0, 1.6, 1.0])
category = c1.radio(
    "Evaluation lens",
    options=["2026 value", "Dynasty value", "Real-life NFL"],
    horizontal=True,
    help="Determines which model column the reveal panel highlights. "
         "All 3 comparisons are logged regardless of which is 'active'."
)
anonymous = c2.toggle(
    "Anonymous mode",
    value=False,
    help="Hide names, teams, salaries, and photos. Stat-blind evaluation.",
)
pos_filter = c3.multiselect(
    "Position filter (empty = cross-position)",
    POS_ORDER,
    default=[],
    help="Only affects the challenger slot. Winner-stays overrides position.",
)
shuffle_clicked = c4.button("Shuffle both", use_container_width=True)

# Second control row for pool scope + pool-size caption
p1, p2 = st.columns([2.0, 3.0])
pool_scope = p1.radio(
    "Player pool",
    options=POOL_OPTIONS,
    index=0,
    horizontal=True,
    help="Constrains BOTH slots. 'Rostered only' skips the FA universe entirely. "
         "'Rostered + top 50 FAs' surfaces the best FAs for realistic comparisons.",
)

# Name <-> espn_id lookup (needed before state transitions)
all_names = sorted(master["player"].dropna().unique().tolist())
name_to_id = master.dropna(subset=["player", "espn_id"]).set_index("player")["espn_id"].astype(int).to_dict()
id_to_name = {v: k for k, v in name_to_id.items()}


def _sync_override_widgets_to_current() -> None:
    """Push current_a / current_b names into the override_a/b selectbox state.

    MUST be called BEFORE the selectbox widgets are instantiated on the current
    run -- Streamlit forbids modifying a widget's session_state key once the
    widget has been rendered. See the top-level sync block below for the
    correct call site after a pick/skip/shuffle rerun.
    """
    a_name = id_to_name.get(int(st.session_state.current_a))
    b_name = id_to_name.get(int(st.session_state.current_b))
    if a_name:
        st.session_state["override_a"] = a_name
    if b_name:
        st.session_state["override_b"] = b_name


# Initialize / re-shuffle the pair -------------------------------------------
pool = _filtered_pool(pos_filter, pool_scope)
p2.caption(f"Pool size: **{len(pool)}** players match the current filters.")

if "current_a" not in st.session_state or "current_b" not in st.session_state:
    pair = _random_pair(pool)
    if pair:
        st.session_state.current_a, st.session_state.current_b = pair
        _sync_override_widgets_to_current()

if shuffle_clicked:
    pair = _random_pair(pool)
    if pair:
        st.session_state.current_a, st.session_state.current_b = pair
        _sync_override_widgets_to_current()

# --- Top-level resync (runs BEFORE the selectbox renders) --------------------
# After a pick/skip rerun, current_a/current_b may have been rotated to new
# players but the widget's session_state ["override_a"] still holds the OLD
# name. We sync here (widget not yet instantiated -> setting is legal).
if "current_a" in st.session_state and "override_a" in st.session_state:
    expected_a = id_to_name.get(int(st.session_state["current_a"]))
    if expected_a is not None and st.session_state["override_a"] != expected_a:
        st.session_state["override_a"] = expected_a
if "current_b" in st.session_state and "override_b" in st.session_state:
    expected_b = id_to_name.get(int(st.session_state["current_b"]))
    if expected_b is not None and st.session_state["override_b"] != expected_b:
        st.session_state["override_b"] = expected_b

# Guard for empty pool
if "current_a" not in st.session_state:
    st.warning("Not enough players in the filtered pool. Broaden the position filter.")
    st.stop()

# Row lookup ----------------------------------------------------------------

def _row(espn_id: int) -> pd.Series | None:
    r = master[master["espn_id"] == espn_id]
    if not len(r):
        return None
    return r.iloc[0]


row_a = _row(st.session_state.current_a)
row_b = _row(st.session_state.current_b)

if row_a is None or row_b is None:
    st.error("Could not resolve one of the current players. Try Shuffle.")
    st.stop()

# --- Manual override dropdowns (on_change updates current_a / current_b) ---

def _on_override_a_changed() -> None:
    new_name = st.session_state.get("override_a")
    new_id = name_to_id.get(new_name)
    if new_id is not None:
        st.session_state.current_a = int(new_id)


def _on_override_b_changed() -> None:
    new_name = st.session_state.get("override_b")
    new_id = name_to_id.get(new_name)
    if new_id is not None:
        st.session_state.current_b = int(new_id)


sel1, sel2 = st.columns(2)
# Seed session_state override_a/b if they're not present yet (first render)
if "override_a" not in st.session_state:
    st.session_state["override_a"] = id_to_name.get(int(st.session_state.current_a), all_names[0])
if "override_b" not in st.session_state:
    st.session_state["override_b"] = id_to_name.get(int(st.session_state.current_b), all_names[1])

sel1.selectbox("Override Player A", all_names, key="override_a", on_change=_on_override_a_changed)
sel2.selectbox("Override Player B", all_names, key="override_b", on_change=_on_override_b_changed)

# --- Side-by-side panels ---------------------------------------------------

CATEGORY_TO_COL = {
    "2026 value": "fair_value_2026",
    "Dynasty value": "fair_value_dynasty",
    "Real-life NFL": "on_field_value",
}


def _render_side(row: pd.Series, label: str, side_key: str) -> str | None:
    """Return 'a'|'b'|None depending on whether user clicked pick. label = 'A' or 'B'."""
    with st.container(border=True):
        # Header: name + logo + photo (or anonymized)
        head_cols = st.columns([1, 3, 1])
        with head_cols[0]:
            if not anonymous:
                url = player_headshot_url(row.get("espn_id"))
                if url:
                    st.image(url, width=80)
        with head_cols[1]:
            if anonymous:
                st.subheader(f"Player {label}")
                st.caption(f"{row.get('position_group', '-')}  ·  age {fmt_float_or_dash(row.get('age'), n=1)}")
            else:
                st.subheader(str(row.get("player", "?")))
                st.caption(
                    f"{row.get('position_group', '-')}  ·  age "
                    f"{fmt_float_or_dash(row.get('age'), n=1)}  ·  "
                    f"NFL {row.get('nfl_team_2025', '-')}  ·  league team "
                    f"{row.get('team') or 'FA'}"
                )
        with head_cols[2]:
            if not anonymous:
                lu = team_logo_url(row.get("nfl_team_2025"))
                if lu:
                    st.image(lu, width=60)

        # Contract row (hidden in anonymous)
        if not anonymous:
            c1, c2, c3 = st.columns(3)
            c1.metric("Salary 2026", fmt_int_or_dash(row.get("salary_2026")))
            c2.metric("Yrs remaining", fmt_int_or_dash(row.get("years_2026")))
            c3.metric("Roster status", str(row.get("roster_status", "-")))

        # --- 2025 season stats (the "real" numbers to base your judgement on) --
        stats = _stats_for(row.get("espn_id"))
        pos = row.get("position_group", "-")
        games_25 = stats.get("games_2025")
        ppg_25 = stats.get("ppg_2025")
        ppg_24 = stats.get("ppg_2024")
        fpts_25 = stats.get("points_2025")
        snap_25 = stats.get("snap_pct_2025")

        st.markdown("**2025 season**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Games", fmt_int_or_dash(games_25))
        if pd.notna(ppg_25) and pd.notna(ppg_24):
            s2.metric("PPG", f"{float(ppg_25):.1f}",
                      delta=f"{float(ppg_25) - float(ppg_24):+.1f} vs 2024",
                      delta_color="normal")
        elif pd.notna(ppg_25):
            s2.metric("PPG", f"{float(ppg_25):.1f}")
        else:
            s2.metric("PPG", "-")
        s3.metric("Total FP", fmt_int_or_dash(fpts_25))
        s4.metric("Snap %", f"{float(snap_25) * 100:.0f}%" if pd.notna(snap_25) else "-")

        # Position-adaptive rate line
        if pos in ("WR", "TE"):
            ts = stats.get("target_share_2025")
            wopr = stats.get("wopr_2025")
            bits = []
            if pd.notna(ts):
                bits.append(f"Target share: **{float(ts) * 100:.1f}%**")
            if pd.notna(wopr):
                bits.append(f"WOPR: **{float(wopr):.2f}**")
            if bits:
                st.caption("  ·  ".join(bits))
        elif pos == "RB":
            rush_epa = stats.get("rushing_epa_2025")
            if pd.notna(rush_epa):
                st.caption(f"Rushing EPA: **{float(rush_epa):.1f}**")

        # --- 2024 box-score counting stats (last complete season) ------------
        box = _box_for(row.get("espn_id"))
        if box:
            st.markdown("**2024 box-score (counting stats)**")
            if pos == "QB":
                # passing + rushing
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Pass yds", fmt_int_or_dash(box.get("passing_yards")))
                b2.metric("Pass TDs", fmt_int_or_dash(box.get("passing_tds")))
                b3.metric("INTs", fmt_int_or_dash(box.get("interceptions")))
                cmp_ = box.get("completions")
                att = box.get("attempts")
                cmp_pct = (100.0 * float(cmp_) / float(att)) if (pd.notna(cmp_) and pd.notna(att) and float(att) > 0) else None
                b4.metric("Comp %", f"{cmp_pct:.1f}%" if cmp_pct is not None else "-")
                r1, r2, r3 = st.columns(3)
                r1.metric("Rush yds", fmt_int_or_dash(box.get("rushing_yards")))
                r2.metric("Rush TDs", fmt_int_or_dash(box.get("rushing_tds")))
                r3.metric("Carries", fmt_int_or_dash(box.get("carries")))
            elif pos == "RB":
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Carries", fmt_int_or_dash(box.get("carries")))
                b2.metric("Rush yds", fmt_int_or_dash(box.get("rushing_yards")))
                b3.metric("Rush TDs", fmt_int_or_dash(box.get("rushing_tds")))
                carries = box.get("carries")
                rush_y = box.get("rushing_yards")
                ypc = (float(rush_y) / float(carries)) if (pd.notna(rush_y) and pd.notna(carries) and float(carries) > 0) else None
                b4.metric("YPC", f"{ypc:.2f}" if ypc is not None else "-")
                r1, r2, r3 = st.columns(3)
                r1.metric("Receptions", fmt_int_or_dash(box.get("receptions")))
                r2.metric("Rec yds", fmt_int_or_dash(box.get("receiving_yards")))
                r3.metric("Rec TDs", fmt_int_or_dash(box.get("receiving_tds")))
            else:  # WR / TE / others
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Receptions", fmt_int_or_dash(box.get("receptions")))
                b2.metric("Targets", fmt_int_or_dash(box.get("targets")))
                b3.metric("Rec yds", fmt_int_or_dash(box.get("receiving_yards")))
                b4.metric("Rec TDs", fmt_int_or_dash(box.get("receiving_tds")))
                # Rushing line for pass-catchers who also carry (WR jet sweeps, etc.)
                car = box.get("carries")
                if pd.notna(car) and float(car) >= 5:
                    st.caption(
                        f"Rushing: **{fmt_int_or_dash(car)}** carries · "
                        f"**{fmt_int_or_dash(box.get('rushing_yards'))}** yds · "
                        f"**{fmt_int_or_dash(box.get('rushing_tds'))}** TDs"
                    )

        # --- User's 1-year valuation (optional, non-anonymous only) ---
        if not anonymous:
            val_cur, val_ts = latest_valuation(int(row["espn_id"]))
            vc1, vc2 = st.columns([2, 1])
            with vc1:
                val_key = f"valuation_input_{side_key}"
                default_val = float(val_cur) if val_cur is not None else 0.0
                v_new = st.number_input(
                    "Your 1-yr valuation (cap units)",
                    min_value=0.0, max_value=500.0,
                    value=default_val, step=1.0,
                    key=val_key,
                    help="Your subjective single-season value for this player. "
                         "Compared to the model's fair_value_2026 in the reveal panel.",
                )
            with vc2:
                if st.button("Save valuation", key=f"save_v_{side_key}"):
                    save_valuation(int(row["espn_id"]), float(v_new))
                    st.success(f"Saved: {v_new:.0f}")
            if val_cur is not None:
                st.caption(f"Latest saved: **{val_cur:.0f}** (as of `{val_ts}`).")

        # Comment box (only in non-anonymous mode; would leak identity via comments)
        if not anonymous:
            comment_key = f"comment_{side_key}"
            if comment_key not in st.session_state:
                st.session_state[comment_key] = ""
            comment_txt = st.text_area(
                "Add a note about this player",
                value=st.session_state[comment_key],
                key=f"ta_{side_key}",
                height=80,
                placeholder="e.g., traded to Buffalo this offseason",
            )
            if st.button(f"Save note ({row.get('player', label)})", key=f"save_c_{side_key}"):
                if comment_txt.strip():
                    save_comment(int(row["espn_id"]), comment_txt)
                    st.session_state[comment_key] = ""
                    st.success("Note saved.")
                else:
                    st.info("Empty note ignored.")

            # Show recent comments for this player
            prior = load_comments(int(row["espn_id"]))
            if len(prior):
                with st.expander(f"Prior notes ({len(prior)})", expanded=False):
                    for _, r in prior.tail(5).iterrows():
                        st.caption(f"**{r['timestamp']}** — {r['comment']}")

        # Selection button
        pick = st.button(
            f"{'Player ' + label if anonymous else str(row.get('player', label))} is more valuable",
            key=f"pick_{side_key}",
            use_container_width=True,
            type="primary",
        )
        return "a" if (pick and label == "A") else ("b" if (pick and label == "B") else None)


col_a, col_b = st.columns(2)
with col_a:
    picked_a = _render_side(row_a, "A", "a")
with col_b:
    picked_b = _render_side(row_b, "B", "b")

# Middle: skip/even -------------------------------------------------------
skip_clicked = st.button("Skip / even (both cycle)", use_container_width=True)

# --- Handle selection logic ------------------------------------------------

active_choice: str | None = None
if picked_a:
    active_choice = "a"
elif picked_b:
    active_choice = "b"
elif skip_clicked:
    active_choice = "skip"

# --- Model reveal (before rotating so we can compute with current pair) ----

def _model_pick(row_a: pd.Series, row_b: pd.Series, col: str) -> tuple[str, float]:
    """Return ('a'|'b'|'tie', abs_gap)."""
    a_val = float(row_a.get(col) or 0)
    b_val = float(row_b.get(col) or 0)
    diff = a_val - b_val
    if abs(diff) < 0.5:
        return "tie", 0.0
    return ("a" if diff > 0 else "b"), abs(diff)


if active_choice is not None:
    # Save
    category_key = {"2026 value": "season", "Dynasty value": "dynasty", "Real-life NFL": "reallife"}[category]
    save_comparison(
        category=category_key,
        player_a_espn_id=int(row_a["espn_id"]),
        player_b_espn_id=int(row_b["espn_id"]),
        choice=active_choice,
        anonymous_mode=anonymous,
    )

    # Track alignment (skip counts as no-info)
    if active_choice != "skip":
        model_col = CATEGORY_TO_COL[category]
        m_choice, _gap = _model_pick(row_a, row_b, model_col)
        agree = (m_choice == active_choice) if m_choice != "tie" else None
        st.session_state.setdefault("alignment_log", []).append(
            {"category": category_key, "agreed": agree}
        )

    # Snapshot the just-compared pair so the reveal panel on the NEXT render
    # shows what the model thought about it (real names, regardless of anon).
    st.session_state["last_comparison"] = {
        "row_a": row_a.to_dict(),
        "row_b": row_b.to_dict(),
        "category_label": category,
        "category_key": category_key,
        "user_choice": active_choice,
        "was_anonymous": bool(anonymous),
    }

    # Rotate: winner stays; loser cycles to a new random challenger.
    if active_choice == "a":
        new_b = _random_challenger(pool, exclude={int(st.session_state.current_a)})
        if new_b:
            st.session_state.current_b = int(new_b)
    elif active_choice == "b":
        # B moves to A slot; new challenger goes into B.
        st.session_state.current_a = int(st.session_state.current_b)
        new_b = _random_challenger(pool, exclude={int(st.session_state.current_a)})
        if new_b:
            st.session_state.current_b = int(new_b)
    else:  # skip
        new_pair = _random_pair(pool)
        if new_pair:
            st.session_state.current_a, st.session_state.current_b = new_pair

    # NOTE: do NOT call _sync_override_widgets_to_current() here -- the selectbox
    # widget has already been instantiated on this run (rendered above the pick
    # buttons), so Streamlit forbids writing to its session_state key. The
    # top-level resync block runs on the next rerun and updates the widget
    # state before the selectbox re-instantiates.
    st.rerun()

# --- Model reveal panel (only after a pick; shows the LAST comparison) ------

last = st.session_state.get("last_comparison")
if last is not None:
    st.markdown("---")
    prev_a = pd.Series(last["row_a"])
    prev_b = pd.Series(last["row_b"])
    name_a = str(prev_a.get("player", "A"))
    name_b = str(prev_b.get("player", "B"))
    st.subheader(f"Last comparison -- {name_a} vs {name_b}")
    user_choice_label = {"a": name_a, "b": name_b, "skip": "Skip / even"}[last["user_choice"]]
    st.caption(
        f"Category picked: **{last['category_label']}**  ·  "
        f"You said: **{user_choice_label}**"
        + ("  ·  _(anonymous mode was on)_" if last.get("was_anonymous") else "")
    )
    reveal_cols = st.columns(3)
    for i, (label, col) in enumerate([
        ("2026 value (fair_value_2026)", "fair_value_2026"),
        ("Dynasty value (fair_value_dynasty)", "fair_value_dynasty"),
        ("Real-life NFL (on_field_value)", "on_field_value"),
    ]):
        with reveal_cols[i]:
            m_choice, gap = _model_pick(prev_a, prev_b, col)
            val_a_raw = prev_a.get(col)
            val_b_raw = prev_b.get(col)
            val_a_num = float(val_a_raw) if pd.notna(val_a_raw) else 0.0
            val_b_num = float(val_b_raw) if pd.notna(val_b_raw) else 0.0

            active = (col == CATEGORY_TO_COL[last["category_label"]])
            # Agreement badge only for the active category (user only picked in one lens)
            badge = ""
            if active and last["user_choice"] in ("a", "b"):
                agrees = (m_choice == last["user_choice"])
                if m_choice != "tie":
                    badge = " ✓" if agrees else " ✗"
                else:
                    badge = " (tie)"
            elif active:
                badge = " (skipped)"

            st.markdown(f"**{label}**{badge}")

            # Two per-player metric cards side by side. Winner gets the delta
            # showing the size of the gap; loser gets nothing (cleaner than
            # a negative delta on the losing side).
            sub = st.columns(2)
            trophy_a = "🏆 " if m_choice == "a" else ""
            trophy_b = "🏆 " if m_choice == "b" else ""
            sub[0].metric(
                f"{trophy_a}{name_a}",
                f"{val_a_num:.1f}",
                delta=f"+{gap:.1f}" if m_choice == "a" else None,
                delta_color="normal",
            )
            sub[1].metric(
                f"{trophy_b}{name_b}",
                f"{val_b_num:.1f}",
                delta=f"+{gap:.1f}" if m_choice == "b" else None,
                delta_color="normal",
            )
            if m_choice == "tie":
                st.caption("Effective tie (< 0.5 gap)")

    # --- User valuation vs model fair_value_2026 (if the user has saved one) --
    val_a_user, _ = latest_valuation(int(prev_a.get("espn_id") or 0))
    val_b_user, _ = latest_valuation(int(prev_b.get("espn_id") or 0))
    if val_a_user is not None or val_b_user is not None:
        st.markdown("**Your 1-year valuations vs model's Fair 2026**")
        vv1, vv2 = st.columns(2)
        for side_col, name, prev_row, val_user in [
            (vv1, name_a, prev_a, val_a_user),
            (vv2, name_b, prev_b, val_b_user),
        ]:
            with side_col:
                model_fair = prev_row.get("fair_value_2026")
                if val_user is None:
                    st.metric(f"{name}: your value", "-",
                              help="No saved valuation yet. Enter one on the player's card.")
                else:
                    if pd.notna(model_fair):
                        gap = float(val_user) - float(model_fair)
                        st.metric(
                            f"{name}: your value",
                            f"{val_user:.0f}",
                            delta=f"{gap:+.0f} vs model ({float(model_fair):.0f})",
                            delta_color="normal",
                            help="Positive: you value them higher than the model. "
                                 "Negative: you value them lower.",
                        )
                    else:
                        st.metric(f"{name}: your value", f"{val_user:.0f}")

# --- Sidebar: session stats -------------------------------------------------

with st.sidebar:
    st.subheader("Session progress")
    all_comps = load_comparisons()
    st.metric("Total comparisons logged", len(all_comps))
    if len(all_comps):
        breakdown = all_comps["category"].value_counts().reindex(
            ["season", "dynasty", "reallife"]).fillna(0).astype(int)
        st.caption(
            f"season: **{breakdown.get('season', 0)}**  ·  "
            f"dynasty: **{breakdown.get('dynasty', 0)}**  ·  "
            f"reallife: **{breakdown.get('reallife', 0)}**"
        )
    st.markdown("---")
    log = st.session_state.get("alignment_log", [])
    if log:
        df = pd.DataFrame(log)
        st.subheader("Model alignment (this session)")
        for cat in ["season", "dynasty", "reallife"]:
            sub = df[(df["category"] == cat) & (df["agreed"].notna())]
            if len(sub):
                agree_pct = 100.0 * sub["agreed"].sum() / len(sub)
                st.metric(cat, f"{agree_pct:.0f}%",
                          delta=f"{int(sub['agreed'].sum())}/{len(sub)}", delta_color="off")

    st.markdown("---")
    st.caption(
        f"Filtered pool: **{len(pool)}** players.  "
        "Selections and comments persist to `data/research/`."
    )
