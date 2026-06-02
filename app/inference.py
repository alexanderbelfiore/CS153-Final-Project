"""Inference wrapper for the Streamlit defensive playcaller demo.

Loads the frozen RF policy model + candidate set + test scenarios that were
exported from the notebook (§3 end), and re-implements the candidate scoring /
action selection / history lookup logic that lives in the notebook as a clean
importable module.

The function bodies here mirror the notebook one-to-one so the UI stays in
lockstep with the trained model. If anything in the notebook changes
(`ALL_FEATURES`, `candidates`, the RF), re-run the export cell and the UI
picks up the new artifacts on next launch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, regardless of CWD)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RF_PATH = os.path.join(MODELS_DIR, "rf_policy.joblib")
CANDIDATES_PATH = os.path.join(MODELS_DIR, "candidate_set.parquet")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_cols.json")
TEST_DF_PATH = os.path.join(DATA_DIR, "test_scenarios.parquet")
PBP_HISTORY_PATH = os.path.join(DATA_DIR, "pbp_history.parquet")

# Defaults matching the notebook's tuned λ-sweep result (Section 3.6).
HISTORY_WINDOW = 5
DEFAULT_LAMBDA = 0.033
DEFAULT_NARROW_BAND = 0.10


# ---------------------------------------------------------------------------
# Artifact loaders (cached so Streamlit reruns are free)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_rf_model():
    return joblib.load(RF_PATH)


@lru_cache(maxsize=1)
def load_candidates() -> pd.DataFrame:
    return pd.read_parquet(CANDIDATES_PATH)


@lru_cache(maxsize=1)
def load_feature_cols() -> dict:
    with open(FEATURE_COLS_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_test_scenarios() -> pd.DataFrame:
    df = pd.read_parquet(TEST_DF_PATH)
    return df.sort_values(["game_id", "play_id"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_pbp_history() -> pd.DataFrame:
    return pd.read_parquet(PBP_HISTORY_PATH)


def artifacts_exist() -> tuple[bool, list[str]]:
    """Return (all_present, list_of_missing_paths)."""
    paths = [RF_PATH, CANDIDATES_PATH, FEATURE_COLS_PATH, TEST_DF_PATH, PBP_HISTORY_PATH]
    missing = [p for p in paths if not os.path.exists(p)]
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Core scoring (ported from notebook §3.5 / §2.3)
# ---------------------------------------------------------------------------

def score_all_candidates(df: pd.DataFrame, model, candidates_df: pd.DataFrame) -> np.ndarray:
    """Return (n_rows, n_candidates) RF EPA predictions.

    Mirrors notebook cell 83 exactly: replicate each state row n_candidates
    times, substitute each candidate's action components into the row, and
    predict in a single batch.
    """
    cols = load_feature_cols()
    NUMERIC_STATE_FEATURES = cols["NUMERIC_STATE_FEATURES"]
    ALL_CATEGORICAL = cols["ALL_CATEGORICAL"]
    ALL_FEATURES = cols["ALL_FEATURES"]
    ACTION_COMPONENTS = cols["ACTION_COMPONENTS"]
    CANDIDATE_KEYS = cols["CANDIDATE_KEYS"]

    n_state = len(df)
    n_act = len(candidates_df)
    rep = df.loc[df.index.repeat(n_act)].reset_index(drop=True)
    act_block = pd.concat([candidates_df[CANDIDATE_KEYS]] * n_state, ignore_index=True)
    rep[ACTION_COMPONENTS] = act_block[ACTION_COMPONENTS].values
    X = rep[ALL_FEATURES].copy()
    for c in NUMERIC_STATE_FEATURES:
        X[c] = X[c].astype(float).fillna(0.0)
    for c in ALL_CATEGORICAL:
        X[c] = X[c].astype("string").fillna("NA")
    return model.predict(X).reshape(n_state, n_act)


def policy_score(epa_pred: float, candidate, recent_history, lambda_pred: float) -> float:
    """Composite score (lower is better for defense). Notebook cell 26."""
    repeat = sum(
        1 for h in recent_history
        if h.get("defense_coverage_type") == candidate["defense_coverage_type"]
    )
    return float(epa_pred) + lambda_pred * repeat


def select_action(
    epa_predictions: np.ndarray,
    candidates_df: pd.DataFrame,
    recent_history: list[dict],
    lambda_pred: float = DEFAULT_LAMBDA,
    narrow_band: float = DEFAULT_NARROW_BAND,
) -> int:
    """Pick the candidate index with the best composite score. Notebook cell 26."""
    epa_predictions = np.asarray(epa_predictions, dtype=float)
    if len(epa_predictions) != len(candidates_df):
        raise ValueError("epa_predictions length must match number of candidates")

    best = epa_predictions.min()
    in_band = np.where(epa_predictions <= best + narrow_band)[0]

    if len(in_band) == 1:
        return int(in_band[0])

    scored = [
        (i, policy_score(epa_predictions[i], candidates_df.iloc[i], recent_history, lambda_pred))
        for i in in_band
    ]
    return int(min(scored, key=lambda x: x[1])[0])


def history_for_row(row: pd.Series, pbp_history: pd.DataFrame | None = None) -> list[dict]:
    """Return the last HISTORY_WINDOW prior 3rd/4th attempts for this defense in this game.

    Notebook cell 86. Pass the exported `pbp_history.parquet` as a DataFrame.
    """
    src = load_pbp_history() if pbp_history is None else pbp_history
    prior = (
        src[(src["game_id"] == row["game_id"])
            & (src["defteam"] == row["defteam"])
            & (src["play_id"] < row["play_id"])]
        .sort_values("play_id")
        .tail(HISTORY_WINDOW)
    )
    return prior[
        ["def_personnel", "defense_man_zone_type", "defense_coverage_type"]
    ].to_dict("records")


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper for the Streamlit app
# ---------------------------------------------------------------------------

def season_from_game_id(game_id: str) -> int:
    """Extract season int from nflverse game_id format `YYYY_WW_AWY_HOM`."""
    return int(game_id.split("_")[0])


@dataclass
class ScoredCandidate:
    rank: int
    def_personnel: str
    defense_coverage_type: str
    defense_man_zone_type: str
    predicted_epa: float
    policy_score: float
    is_rf_pick: bool


def score_row(
    row: pd.Series,
    lambda_pred: float = DEFAULT_LAMBDA,
    narrow_band: float = DEFAULT_NARROW_BAND,
    top_k: int = 5,
) -> dict:
    """Run the full RF policy pipeline for one test row.

    Returns a dict with:
      - `predictions`: full EPA array, one entry per candidate (raw RF output)
      - `policy_scores`: λ-adjusted scores for every candidate
      - `recent_history`: prior-down history list (LLM input)
      - `rf_pick_idx`: integer index into candidates of the selected action
      - `top_candidates`: list[ScoredCandidate] sorted by policy_score (top-k)
      - `top_candidates_llm`: list of dicts in the exact shape `call_coordinator`
        expects (top-3 by policy_score; index 0 = RF pick)
    """
    model = load_rf_model()
    candidates = load_candidates()

    preds = score_all_candidates(row.to_frame().T.reset_index(drop=True), model, candidates).flatten()
    hist = history_for_row(row)

    ps_all = np.array([
        policy_score(preds[i], candidates.iloc[i], hist, lambda_pred)
        for i in range(len(candidates))
    ])
    rf_pick_idx = select_action(preds, candidates, hist, lambda_pred, narrow_band)

    # Display table: top-k by policy_score
    sort_idx = np.argsort(ps_all)
    top_idx = sort_idx[:top_k]
    top_candidates = [
        ScoredCandidate(
            rank=r + 1,
            def_personnel=str(candidates.iloc[i]["def_personnel"]),
            defense_coverage_type=str(candidates.iloc[i]["defense_coverage_type"]),
            defense_man_zone_type=str(candidates.iloc[i]["defense_man_zone_type"]),
            predicted_epa=float(preds[i]),
            policy_score=float(ps_all[i]),
            is_rf_pick=(i == rf_pick_idx),
        )
        for r, i in enumerate(top_idx)
    ]

    # LLM coordinator input: top-3 by policy_score, RF pick forced to index 0
    K = 3
    topk_idx = list(sort_idx[:K])
    if rf_pick_idx in topk_idx:
        topk_idx.remove(rf_pick_idx)
    topk_idx = [rf_pick_idx] + topk_idx[: K - 1]
    top_candidates_llm = [
        {
            "def_personnel": str(candidates.iloc[i]["def_personnel"]),
            "defense_coverage_type": str(candidates.iloc[i]["defense_coverage_type"]),
            "defense_man_zone_type": str(candidates.iloc[i]["defense_man_zone_type"]),
            "predicted_epa": round(float(preds[i]), 4),
            "policy_score": round(float(ps_all[i]), 4),
        }
        for i in topk_idx
    ]

    return {
        "predictions": preds,
        "policy_scores": ps_all,
        "recent_history": hist,
        "rf_pick_idx": rf_pick_idx,
        "top_candidates": top_candidates,
        "top_candidates_llm": top_candidates_llm,
    }


def game_state_for_row(row: pd.Series) -> dict:
    """Build the `game_state` dict in the shape `call_coordinator` expects."""
    return {
        "game_id": row["game_id"],
        "play_id": int(row["play_id"]),
        "down": int(row["down"]),
        "ydstogo": int(row["ydstogo"]),
        "yardline_100": int(row["yardline_100"]),
        "score_differential": int(row["score_differential"]),
        "off_personnel": row.get("off_personnel"),
        "defteam": row["defteam"],
        "posteam": row["posteam"],
    }
