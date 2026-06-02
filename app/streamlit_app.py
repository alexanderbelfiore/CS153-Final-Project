"""NFL Defensive Playcaller Assistant — Streamlit demo UI.

Lets you pick any 3rd/4th-down conversion attempt from the held-out test set
(2025 weeks 16+), runs the trained RF policy + LLM coordinator on it live,
and shows: game state, RF candidate rankings, injury context, LLM rationale,
and what actually happened in the real game.

Run from the project root with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import streamlit as st

# Make `src/` and `app/` importable when launched via `streamlit run`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import inference  # noqa: E402
from src.context.injury_context import build_injury_context  # noqa: E402
from src.llm.coordinator import call_coordinator  # noqa: E402


# ---------------------------------------------------------------------------
# Page config + global styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NFL Defensive Playcaller Assistant",
    page_icon="🏈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_artifacts():
    ok, missing = inference.artifacts_exist()
    if not ok:
        return None, missing
    # Touch each loader once so subsequent calls are no-ops.
    inference.load_rf_model()
    inference.load_candidates()
    inference.load_feature_cols()
    inference.load_pbp_history()
    return inference.load_test_scenarios(), []


@st.cache_data(show_spinner=False)
def _game_options(test_df: pd.DataFrame) -> list[str]:
    return sorted(test_df["game_id"].unique().tolist())


@st.cache_data(show_spinner=False)
def _plays_for_game(test_df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    sub = test_df[test_df["game_id"] == game_id].copy()
    return sub.sort_values("play_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_clock(seconds_remaining) -> str:
    """`game_seconds_remaining` -> 'Q3 7:42'."""
    try:
        secs = int(seconds_remaining)
    except (TypeError, ValueError):
        return "—"
    if secs < 0:
        secs = 0
    # NFL game: 4 quarters of 15:00. game_seconds_remaining counts down from 3600.
    elapsed = 3600 - secs
    quarter = min(4, elapsed // 900 + 1)
    q_remaining = secs - (4 - quarter) * 900
    if q_remaining < 0:
        q_remaining = 0
    mm, ss = divmod(q_remaining, 60)
    return f"Q{quarter} {mm}:{ss:02d}"


def _field_position(row: pd.Series) -> str:
    """yardline_100 -> 'OPP 38' or 'OWN 22' style label."""
    y = int(row["yardline_100"])
    if y == 50:
        return "50"
    if y < 50:
        return f"OPP {y}"
    return f"OWN {100 - y}"


def _format_play_label(row) -> str:
    down = int(row["down"])
    suffix = {1: "st", 2: "nd", 3: "rd", 4: "th"}.get(down, "th")
    return (
        f"play {int(row['play_id'])} — "
        f"{down}{suffix} & {int(row['ydstogo'])} | "
        f"{_format_clock(row['game_seconds_remaining'])} | "
        f"{row['posteam']} vs {row['defteam']}"
    )


def _format_game_label(game_id: str) -> str:
    """`2025_17_BUF_NE` -> '2025 Wk 17 — BUF @ NE'."""
    parts = game_id.split("_")
    if len(parts) >= 4:
        season, week, away, home = parts[0], parts[1], parts[2], parts[3]
        return f"{season} Wk {int(week)} — {away} @ {home}"
    return game_id


# ---------------------------------------------------------------------------
# UI: header / artifact check
# ---------------------------------------------------------------------------

st.title("🏈 NFL Defensive Playcaller Assistant")
st.caption(
    "CS 153 final project · pick any 3rd/4th-down play from the 2025 wk 16+ test set "
    "and watch the RF policy + LLM coordinator make a recommendation."
)

test_df, missing = _load_artifacts()

if test_df is None:
    st.error("Model artifacts are missing. Run the export cell at the end of `defense_asst.ipynb` §3 first.")
    with st.expander("Missing files"):
        for p in missing:
            st.code(p)
    st.stop()


# ---------------------------------------------------------------------------
# UI: selectors
# ---------------------------------------------------------------------------

games = _game_options(test_df)

col_sel1, col_sel2, col_sel3 = st.columns([2, 3, 1])
with col_sel1:
    selected_game = st.selectbox(
        "Game",
        games,
        format_func=_format_game_label,
        index=0,
    )

plays_df = _plays_for_game(test_df, selected_game)
play_options = list(range(len(plays_df)))

with col_sel2:
    selected_play_idx = st.selectbox(
        "Play",
        play_options,
        format_func=lambda i: _format_play_label(plays_df.iloc[i]),
        index=0,
    )

with col_sel3:
    st.write("")  # vertical alignment
    st.write("")
    run = st.button("Analyze ▶", type="primary", use_container_width=True)

st.divider()

# Pull selected row (this is the source of truth for everything below).
row = plays_df.iloc[selected_play_idx]


# ---------------------------------------------------------------------------
# UI: game state + recent history (always shown, no need for Run)
# ---------------------------------------------------------------------------

left, right = st.columns([2, 3])

with left:
    st.subheader("Game state")

    down = int(row["down"])
    down_suffix = {1: "st", 2: "nd", 3: "rd", 4: "th"}.get(down, "th")
    score_diff = int(row["score_differential"])
    score_str = (
        f"{row['posteam']} +{score_diff}" if score_diff > 0
        else f"{row['posteam']} {score_diff}" if score_diff < 0
        else "Tied"
    )

    gs_col1, gs_col2 = st.columns(2)
    with gs_col1:
        st.metric("Down & Distance", f"{down}{down_suffix} & {int(row['ydstogo'])}")
        st.metric("Field position", _field_position(row))
        st.metric("Offense personnel", str(row.get("off_personnel", "—")))
    with gs_col2:
        st.metric("Score", score_str)
        st.metric("Clock", _format_clock(row["game_seconds_remaining"]))
        st.metric("Defense", row["defteam"])

    st.markdown("**Recent 3rd/4th-down looks (this defense, this game)**")
    hist = inference.history_for_row(row)
    if not hist:
        st.caption("No prior 3rd/4th-down attempts — cold-start.")
    else:
        hist_df = pd.DataFrame(hist)
        hist_df.columns = ["Personnel", "Man/Zone", "Coverage"]
        st.dataframe(hist_df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Run inference on click
# ---------------------------------------------------------------------------

if run:
    st.session_state["last_result"] = None  # clear stale
    with right:
        with st.spinner("Scoring RF candidates…"):
            t0 = time.time()
            try:
                scoring = inference.score_row(row)
            except Exception as e:
                st.exception(e)
                st.stop()
            rf_time = time.time() - t0

        with st.spinner("Building injury context…"):
            try:
                injury_ctx = build_injury_context(
                    season=inference.season_from_game_id(row["game_id"]),
                    week=int(row["week"]),
                    posteam=row["posteam"],
                    game_id=row["game_id"],
                    offense_players=row.get("offense_players"),
                    offense_names=row.get("offense_names"),
                    offense_positions=row.get("offense_positions"),
                )
            except Exception as e:
                st.warning(f"Injury context unavailable: {e}")
                injury_ctx = {
                    "meta": {"mode": "unavailable"},
                    "out": [], "questionable": [], "on_field": [], "qa_flags": [],
                }

        with st.spinner("Calling LLM coordinator (~2–5s)…"):
            t0 = time.time()
            try:
                game_state = inference.game_state_for_row(row)
                llm_result = call_coordinator(
                    game_state=game_state,
                    candidates=scoring["top_candidates_llm"],
                    history=scoring["recent_history"],
                    injury_context=injury_ctx,
                    game_id=row["game_id"],
                    play_id=int(row["play_id"]),
                )
                llm_time = time.time() - t0
                llm_error = None
            except Exception as e:
                llm_result = None
                llm_error = str(e)
                llm_time = time.time() - t0

    st.session_state["last_result"] = {
        "row_key": (selected_game, int(row["play_id"])),
        "scoring": scoring,
        "injury_ctx": injury_ctx,
        "llm_result": llm_result,
        "llm_error": llm_error,
        "timings": (rf_time, llm_time),
    }


# ---------------------------------------------------------------------------
# Render results (if we have a result for this row)
# ---------------------------------------------------------------------------

result = st.session_state.get("last_result")
current_key = (selected_game, int(row["play_id"]))
have_result = result is not None and result["row_key"] == current_key

with right:
    st.subheader("RF candidate rankings")
    if not have_result:
        st.info("Click **Analyze ▶** to score this play.")
    else:
        scoring = result["scoring"]
        chart_df = pd.DataFrame([
            {
                "Action": f"{'★ ' if c.is_rf_pick else ''}{c.def_personnel} / {c.defense_coverage_type}",
                "Predicted EPA": c.predicted_epa,
                "Policy score (λ)": c.policy_score,
                "RF pick": "★" if c.is_rf_pick else "",
            }
            for c in scoring["top_candidates"]
        ])
        st.bar_chart(
            chart_df.set_index("Action")["Predicted EPA"],
            horizontal=True,
            height=240,
        )
        st.dataframe(
            chart_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted EPA": st.column_config.NumberColumn(format="%+.3f"),
                "Policy score (λ)": st.column_config.NumberColumn(format="%+.3f"),
            },
        )


# Lower section: injury panel + LLM recommendation + ground truth
if have_result:
    st.divider()
    inj_col, rec_col = st.columns([1, 1])

    # ----- Injury context -----
    with inj_col:
        st.subheader("Injury context")
        ctx = result["injury_ctx"]
        mode = ctx.get("meta", {}).get("mode", "—")
        st.caption(f"Mode: `{mode}` · opposing offense: **{row['posteam']}**")

        out_list = ctx.get("out", [])
        material_out = [e for e in out_list if e.get("is_material")]
        nonmat_out = [e for e in out_list if not e.get("is_material")]
        q_list = ctx.get("questionable", [])
        material_q = [e for e in q_list if e.get("is_material")]

        if material_out:
            st.markdown("**OUT (material starters)**")
            for e in material_out:
                inj = e.get("primary_injury") or "—"
                src = e.get("source", "report")
                st.markdown(f"• **{e.get('name', '?')}** ({e.get('position', '?')}) — {inj} _[{src}]_")
        if material_q:
            st.markdown("**Questionable (material starters)**")
            for e in material_q:
                inj = e.get("primary_injury") or "—"
                st.markdown(f"• **{e.get('name', '?')}** ({e.get('position', '?')}) — {inj}")
        if nonmat_out:
            with st.expander(f"Other out/inactive ({len(nonmat_out)})"):
                for e in nonmat_out:
                    inj = e.get("primary_injury") or "—"
                    st.markdown(f"• {e.get('name', '?')} ({e.get('position', '?')}) — {inj}")
        if not (material_out or material_q or nonmat_out):
            st.caption("No reported absences for this team-week.")

    # ----- LLM recommendation -----
    with rec_col:
        st.subheader("LLM recommendation")
        llm = result["llm_result"]
        if llm is None:
            st.error(f"LLM call failed: {result['llm_error']}")
            scoring = result["scoring"]
            rf_pick = next(c for c in scoring["top_candidates"] if c.is_rf_pick)
            st.markdown(
                f"_Falling back to RF pick:_ **{rf_pick.def_personnel} / {rf_pick.defense_coverage_type}**  "
                f"(pred EPA {rf_pick.predicted_epa:+.3f})"
            )
        else:
            rec = llm.recommended_action
            badge = "✅ AFFIRMED" if llm.affirmed_policy else "⚠️ OVERRIDE"
            st.markdown(f"### {rec['def_personnel']} / {rec['defense_coverage_type']}")
            st.markdown(
                f"{badge} · pred EPA **{rec['predicted_epa']:+.3f}** · "
                f"man/zone: {rec.get('defense_man_zone_type', '—')}"
            )
            if llm.override_reason_codes:
                st.caption("Reason codes: " + ", ".join(f"`{c}`" for c in llm.override_reason_codes))
            st.markdown("**Rationale**")
            st.markdown(f"> {llm.rationale}")

    # ----- Ground truth -----
    st.divider()
    st.subheader("What actually happened")
    gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)
    with gt_col1:
        st.metric("Called personnel", str(row.get("def_personnel", "—")))
    with gt_col2:
        st.metric("Called coverage", str(row.get("defense_coverage_type", "—")))
    with gt_col3:
        epa = float(row["epa"])
        st.metric("Realized EPA (offense)", f"{epa:+.3f}",
                  delta=None, help="Negative is good for the defense.")
    with gt_col4:
        success = row.get("def_success")
        if pd.notna(success):
            st.metric("Defense success?", "Yes ✅" if int(success) == 1 else "No ❌")
        else:
            st.metric("Defense success?", "—")

    # Timing footer
    rf_t, llm_t = result["timings"]
    st.caption(f"⏱  RF scoring: {rf_t*1000:.0f} ms · LLM: {llm_t:.1f} s")
