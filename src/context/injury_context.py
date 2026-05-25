"""InjuryContext builder.

Joins nfl_data_py weekly injury reports, weekly rosters (gameday inactive flag),
and depth charts onto play-by-play snap-level player lists to emit structured
JSON describing who is out / questionable / on the field for a given
(season, week, posteam).

The output is consumed by the Week 6 LLM coordinator. See the project plan
section "Week 6: Qualitative Layer" for the design rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import nfl_data_py as nfl
import pandas as pd

CACHE_DIR = os.path.join("data", "context_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# NFL offensive position codes used in injury reports and weekly rosters.
OFFENSE_POSITIONS = {
    "QB", "RB", "HB", "FB",
    "WR", "TE",
    "OL", "OT", "OG", "C", "G", "T", "LT", "RT", "LG", "RG",
}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

def _cache_path(table: str, season: int) -> str:
    return os.path.join(CACHE_DIR, f"{table}_{season}.parquet")


def _load_or_pull(table: str, season: int, puller) -> pd.DataFrame:
    path = _cache_path(table, season)
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = puller(years=[season])
    df.to_parquet(path, index=False)
    return df


@lru_cache(maxsize=8)
def load_injuries(season: int) -> pd.DataFrame:
    return _load_or_pull("injuries", season, nfl.import_injuries)


@lru_cache(maxsize=8)
def load_weekly_rosters(season: int) -> pd.DataFrame:
    return _load_or_pull("weekly_rosters", season, nfl.import_weekly_rosters)


@lru_cache(maxsize=8)
def load_depth_charts(season: int) -> pd.DataFrame:
    return _load_or_pull("depth_charts", season, nfl.import_depth_charts)


@lru_cache(maxsize=8)
def load_schedules(season: int) -> pd.DataFrame:
    return _load_or_pull("schedules", season, nfl.import_schedules)


def _game_date(season: int, game_id: str) -> pd.Timestamp | None:
    """Return the gameday for a given game_id, or None if not found."""
    if not game_id:
        return None
    sch = load_schedules(season)
    hit = sch[sch["game_id"] == game_id]
    if hit.empty:
        return None
    return pd.to_datetime(hit["gameday"].iloc[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_gsis(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s and s.lower() != "nan" else None


def _split_semis(s) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [tok.strip() for tok in str(s).split(";") if tok.strip()]


def _align_snap_lists(
    players: Iterable[str] | None,
    names: Iterable[str] | None,
    positions: Iterable[str] | None,
) -> tuple[list[dict], list[str]]:
    """Zip the three semicolon-delimited snap fields into per-player dicts.

    Returns (on_field, qa_flags). Length mismatches go to QA flags rather than
    raising — the LLM should never see partially-aligned rows.
    """
    p = list(players) if players is not None else []
    n = list(names) if names is not None else []
    pos = list(positions) if positions is not None else []
    qa: list[str] = []
    if p and n and len(p) != len(n):
        qa.append(f"len(offense_players)={len(p)} != len(offense_names)={len(n)}")
    if p and pos and len(p) != len(pos):
        qa.append(f"len(offense_players)={len(p)} != len(offense_positions)={len(pos)}")
    out = []
    for i, gid in enumerate(p):
        out.append({
            "gsis_id": _norm_gsis(gid),
            "name": n[i] if i < len(n) else None,
            "position": pos[i] if i < len(pos) else None,
        })
    return out, qa


def _depth_label(rank: str | None) -> str:
    if rank == "1":
        return "starter"
    if rank in {"2", "3"}:
        return "backup"
    return "unknown"


# ---------------------------------------------------------------------------
# Core lookups for a (season, week, team)
# ---------------------------------------------------------------------------

@dataclass
class _TeamWeekTables:
    injuries: pd.DataFrame
    roster: pd.DataFrame
    depth: pd.DataFrame


def _normalize_depth_chart(
    df: pd.DataFrame, season: int, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize the depth chart to a common internal schema across nfl_data_py versions.

    The 2024 (and earlier) schema has ``club_code``/``week``/``depth_team``/
    ``depth_position``/``formation`` columns. The 2025+ schema is a stream of
    daily snapshots with ``team``/``pos_rank``/``pos_abb``/``pos_grp``/``dt``
    and no weekly slicing. For 2025+ snapshots, ``as_of`` (the gameday) is used
    to pick the latest snapshot at or before the game — depth charts change a
    lot during the season (injuries, trades), so using the latest snapshot
    available for every week would conflate post-season state with Week 1.
    """
    cols = set(df.columns)
    if {"club_code", "week", "depth_team", "depth_position", "formation"}.issubset(cols):
        out = df.rename(columns={"club_code": "team"}).copy()
        if "season" not in out.columns:
            out["season"] = season
        return out
    if {"team", "pos_rank", "pos_abb", "pos_grp", "dt"}.issubset(cols):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
        if as_of is not None:
            df = df[df["dt"] <= pd.Timestamp(as_of)]
        df = df.sort_values("dt").drop_duplicates(
            subset=["team", "gsis_id", "pos_abb"], keep="last"
        )
        pg = df["pos_grp"].fillna("")
        formation = pd.Series("Offense", index=df.index)
        formation[pg.str.contains(r" D$", regex=True)] = "Defense"
        formation[pg == "Special Teams"] = "Special Teams"
        df["formation"] = formation
        df["depth_team"] = df["pos_rank"].astype(str)
        df["depth_position"] = df["pos_abb"]
        df["full_name"] = df["player_name"]
        df["position"] = df["pos_abb"]
        df["season"] = season
        df["week"] = -1  # sentinel: snapshot applies to its as_of date
        return df
    return pd.DataFrame(columns=[
        "season", "team", "week", "formation", "depth_team",
        "depth_position", "gsis_id", "position", "full_name",
    ])


def _team_week_tables(
    season: int, week: int, team: str, game_id: str | None = None,
) -> _TeamWeekTables:
    inj = load_injuries(season)
    inj_tw = inj[(inj["season"] == season) & (inj["week"] == week) & (inj["team"] == team)].copy()
    inj_tw["gsis_id"] = inj_tw["gsis_id"].map(_norm_gsis)

    ros = load_weekly_rosters(season)
    ros_tw = ros[(ros["season"] == season) & (ros["week"] == week) & (ros["team"] == team)].copy()
    ros_tw["gsis_id"] = ros_tw["player_id"].map(_norm_gsis)

    gameday = _game_date(season, game_id) if game_id else None
    dc = _normalize_depth_chart(load_depth_charts(season), season, as_of=gameday)
    if dc.empty:
        dc_tw = dc
    else:
        week_match = (dc["week"] == week) | (dc["week"] == -1)
        dc_tw = dc[
            (dc["season"] == season)
            & week_match
            & (dc["team"] == team)
            & (dc["formation"] == "Offense")
        ].copy()
        dc_tw["gsis_id"] = dc_tw["gsis_id"].map(_norm_gsis)
    return _TeamWeekTables(inj_tw, ros_tw, dc_tw)


def _depth_lookup(depth: pd.DataFrame) -> dict[str, dict]:
    """Map gsis_id -> {depth_position, depth_team} (first row per id)."""
    out: dict[str, dict] = {}
    for _, row in depth.iterrows():
        gid = row.get("gsis_id")
        if gid and gid not in out:
            out[gid] = {
                "depth_position": str(row.get("depth_position") or "").strip() or None,
                "depth_team": str(row.get("depth_team") or "").strip() or None,
                "position": row.get("position"),
                "name": row.get("full_name"),
            }
    return out


def _is_material(depth_team: str | None) -> bool:
    return depth_team == "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_injury_context(
    season: int,
    week: int,
    posteam: str,
    *,
    game_id: str | None = None,
    offense_players: str | Iterable[str] | None = None,
    offense_names: str | Iterable[str] | None = None,
    offense_positions: str | Iterable[str] | None = None,
) -> dict:
    """Build a structured injury / on-field context for an offense snap.

    Parameters
    ----------
    season, week, posteam
        Identify the team-week slice of injury / roster / depth data.
    game_id
        Echoed into the output for traceability; not used for joins.
    offense_players, offense_names, offense_positions
        Semicolon-delimited PBP snap fields (or pre-parsed iterables of GSIS
        IDs / names / positions). When provided, the output includes the
        on-field lineup and flags QA mismatches against the injury report.
        When None (scenario mode), only out/questionable/expected-starters
        sections are populated.

    Returns
    -------
    dict
        JSON-serializable context framed for the defensive coordinator.
        Top-level keys: ``meta`` (traceability), ``out`` / ``questionable``
        (injury/inactive status of the opposing offense, each entry tagged with
        ``is_material: bool``), ``on_field`` (replay mode only), ``qa_flags``
        (pipeline-internal — do not forward to the LLM prompt).
    """
    # Accept either raw semicolon strings or pre-split iterables.
    if isinstance(offense_players, str):
        offense_players = _split_semis(offense_players)
    if isinstance(offense_names, str):
        offense_names = _split_semis(offense_names)
    if isinstance(offense_positions, str):
        offense_positions = _split_semis(offense_positions)

    tables = _team_week_tables(season, week, posteam, game_id=game_id)
    depth_by_id = _depth_lookup(tables.depth)

    # --- Out / Questionable from injury report -----------------------------
    out_list: list[dict] = []
    questionable_list: list[dict] = []
    for _, row in tables.injuries.iterrows():
        gid = row.get("gsis_id")
        status = row.get("report_status")
        if not gid or not status:
            continue
        entry = {
            "gsis_id": gid,
            "name": row.get("full_name"),
            "position": row.get("position"),
            "report_status": status,
            "primary_injury": row.get("report_primary_injury") or None,
            "source": "report",
        }
        d = depth_by_id.get(gid)
        if d:
            entry["depth_position"] = d["depth_position"]
            entry["depth_team"] = d["depth_team"]
        if entry.get("position") not in OFFENSE_POSITIONS:
            continue
        if status == "Out":
            out_list.append(entry)
        elif status in {"Questionable", "Doubtful"}:
            questionable_list.append(entry)

    # --- Augment with gameday-inactive flag from weekly rosters (status=INA)
    inactive_ids_in_report = {e["gsis_id"] for e in out_list}
    for _, row in tables.roster.iterrows():
        if row.get("status") != "INA":
            continue
        gid = _norm_gsis(row.get("player_id"))
        if not gid or gid in inactive_ids_in_report:
            continue
        entry = {
            "gsis_id": gid,
            "name": row.get("player_name"),
            "position": row.get("position"),
            "report_status": "Inactive",
            "primary_injury": None,
            "source": "inactive",
        }
        d = depth_by_id.get(gid)
        if d:
            entry["depth_position"] = d["depth_position"]
            entry["depth_team"] = d["depth_team"]
        if entry.get("position") not in OFFENSE_POSITIONS:
            continue
        out_list.append(entry)

    # --- Tag is_material inline on out / questionable -----------------------
    # Material = starter-tier absence (depth_team == "1"). Marked on each entry
    # so the LLM coordinator can scan a single list rather than cross-referencing
    # a separate material_absences key.
    for entry in out_list:
        d = depth_by_id.get(entry["gsis_id"], {})
        entry["is_material"] = _is_material(d.get("depth_team"))
    for entry in questionable_list:
        d = depth_by_id.get(entry["gsis_id"], {})
        entry["is_material"] = _is_material(d.get("depth_team"))

    # --- On-field lineup (replay mode) --------------------------------------
    on_field, qa_flags = _align_snap_lists(offense_players, offense_names, offense_positions)
    on_field_ids = {p["gsis_id"] for p in on_field if p["gsis_id"]}
    out_ids = {e["gsis_id"] for e in out_list}
    # QA: listed Out but appears on the field — downgrade is_material for that snap.
    for gid in on_field_ids & out_ids:
        qa_flags.append(f"player {gid} listed Out/Inactive but appears in offense_players")
        for entry in out_list:
            if entry["gsis_id"] == gid:
                entry["is_material"] = False

    # Enrich on-field with depth info where we have it.
    for p in on_field:
        d = depth_by_id.get(p["gsis_id"]) if p["gsis_id"] else None
        if d:
            p["depth_position"] = d["depth_position"]
            p["depth_team"] = d["depth_team"]
            p["depth"] = _depth_label(d["depth_team"])
        else:
            p["depth_position"] = None
            p["depth_team"] = None
            p["depth"] = "unknown"

    mode = "replay" if on_field else "scenario"

    return {
        "meta": {
            "season": season,
            "week": week,
            "opposing_offense": posteam,
            "game_id": game_id,
            "mode": mode,
        },
        "out": out_list,
        "questionable": questionable_list,
        "on_field": on_field,
        # qa_flags: pipeline-internal QA only — do not pass to the LLM prompt.
        "qa_flags": qa_flags,
    }
