"""LLM defensive coordinator (Week 6).

Consumes top-k RF candidates (with predicted EPA and lambda-adjusted policy
scores), recent in-game history, and the InjuryContext JSON, then either
AFFIRMS the policy pick or RE-RANKS within the same candidate set when a
material offensive absence warrants it.

The constrained output schema is enforced via OpenAI function-calling (tool
use): the model must call ``submit_defensive_recommendation`` exactly once,
and the chosen action is selected by integer index into the candidate list
(so the LLM cannot invent actions outside the candidate set).

Backend: DigitalOcean inference endpoint (OpenAI-compatible API).
See project plan section "Week 6: Qualitative Layer" for design rationale.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import openai
from dotenv import load_dotenv

MODEL_NAME = "llama3.3-70b-instruct"  # DigitalOcean model ID
BASE_URL = "https://inference.do-ai.run/v1"
MAX_TOKENS = 1024

# Reason codes the LLM may attach when overriding the RF pick. Closed set so
# downstream evaluation can aggregate cleanly.
OVERRIDE_REASON_CODES = [
    "qb_out",
    "wr1_out",
    "rb1_out",
    "te1_out",
    "ol_starter_out",
    "backup_on_field",
    "narrow_epa_tie",
    "no_material_change",
    "other",
]


def _load_api_key() -> str:
    load_dotenv()
    key = os.getenv("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY not set in environment / .env")
    return key


SYSTEM_PROMPT = """You are an NFL defensive coordinator's assistant.

A Random Forest policy has already scored a finite set of candidate defensive
configurations by predicted offensive EPA (lower is better for the defense) and
applied a small predictability penalty when candidates are within a narrow EPA
band. The candidates are presented to you ordered by RF rank: index 0 is the
RF policy pick.

Your job: either AFFIRM the policy's top pick (index 0) or RE-RANK among the
SAME candidates when a material absence on the opposing offense warrants it.

Hard rules:
- Pick exactly one candidate by index from the provided list. Never invent
  actions outside the candidate set.
- Cite injury facts ONLY from the `injury_context` JSON. Never fabricate
  statuses, injuries, or player names.
- Default to AFFIRM. The RF pick is the default; override only when a
  material absence (entries with `is_material: true` in `out` or
  `questionable`) plausibly shifts the offensive threat profile.
- Examples of justified re-ranks:
  * starting QB out / backup QB on field -> consider pressure / disguised
    zone looks over the RF pick if a candidate in the band fits.
  * WR1 (or multiple starting WRs) out -> lean run-friendlier shells when
    a candidate in the band fits.
  * starting OL out -> interior rush / blitz looks may be more attractive.
- The `on_field` list (replay mode only) is the actual snap personnel; use
  it to confirm whether a listed-out player is actually absent.
- Prefer the smallest justified change. If no material absence is in play,
  AFFIRM and explain briefly why.

Output is via the `submit_defensive_recommendation` tool. Set
`affirmed_policy=true` iff `recommended_candidate_index == 0`.
Set `override=true` iff you change away from index 0.
The rationale is 2-4 sentences, grounded in the provided facts."""


def _build_tool(num_candidates: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_defensive_recommendation",
            "description": (
                "Submit the defensive coordinator's recommendation. "
                "Must pick exactly one candidate from the provided list by index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recommended_candidate_index": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max(0, num_candidates - 1),
                        "description": (
                            "0-based index of the chosen candidate in the input "
                            "`candidates` list (index 0 = RF policy pick)."
                        ),
                    },
                    "affirmed_policy": {
                        "type": "boolean",
                        "description": (
                            "True iff the chosen candidate index is 0 "
                            "(i.e. matches the RF policy pick)."
                        ),
                    },
                    "override": {
                        "type": "boolean",
                        "description": (
                            "True iff the chosen candidate index is NOT 0 "
                            "(re-rank away from the RF pick)."
                        ),
                    },
                    "override_reason_codes": {
                        "type": "array",
                        "items": {"type": "string", "enum": OVERRIDE_REASON_CODES},
                        "description": (
                            "Short reason codes. Use `no_material_change` when "
                            "affirming; one or more situational codes when "
                            "overriding."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "2-4 sentences. Cite only facts from the provided "
                            "injury_context and candidate scores."
                        ),
                    },
                },
                "required": [
                    "recommended_candidate_index",
                    "affirmed_policy",
                    "override",
                    "override_reason_codes",
                    "rationale",
                ],
            },
        },
    }


def _strip_qa(injury_context: dict) -> dict:
    """Strip pipeline-internal qa_flags before forwarding to the LLM."""
    return {k: v for k, v in injury_context.items() if k != "qa_flags"}


def _material_absences(injury_context: dict) -> list[dict]:
    return [
        e
        for e in injury_context.get("out", []) + injury_context.get("questionable", [])
        if e.get("is_material")
    ]


def _user_payload(
    game_state: dict,
    candidates: list[dict],
    history: list[dict],
    injury_context: dict,
) -> str:
    payload = {
        "game_state": game_state,
        "candidates": candidates,
        "recent_history": history,
        "injury_context": _strip_qa(injury_context),
    }
    return (
        "Decide whether to affirm the RF pick (index 0) or re-rank within "
        "the provided candidate set. Call "
        "`submit_defensive_recommendation` exactly once.\n\n"
        f"INPUT:\n```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )


@dataclass
class CoordinatorResult:
    recommended_action: dict
    affirmed_policy: bool
    override: bool
    override_reason_codes: list[str]
    rationale: str
    override_log: dict
    raw_tool_input: dict = field(repr=False)


def call_coordinator(
    *,
    game_state: dict,
    candidates: list[dict],
    history: list[dict],
    injury_context: dict,
    game_id: str,
    play_id: int | str | None = None,
    client: openai.OpenAI | None = None,
    model: str = MODEL_NAME,
    max_tokens: int = MAX_TOKENS,
) -> CoordinatorResult:
    """Call the LLM coordinator via DigitalOcean's OpenAI-compatible endpoint.

    Parameters
    ----------
    game_state
        Dict of human-readable game state fields (down, ydstogo, yardline_100,
        score_differential, off_personnel, etc.).
    candidates
        Ordered top-k candidates. Index 0 must be the RF policy pick. Each
        entry should at least include `def_personnel`, `defense_coverage_type`,
        `predicted_epa`, and `policy_score`.
    history
        Up to 5 most recent prior 3rd/4th conversion attempts on this defense
        in the same game.
    injury_context
        Output of `build_injury_context(...)`. `qa_flags` is stripped before
        sending to the LLM.
    game_id, play_id
        Echoed into the override log entry for downstream evaluation.

    Returns
    -------
    CoordinatorResult
        Includes the chosen candidate dict, the affirm/override booleans,
        reason codes, rationale, and a complete override-log entry.
    """
    if not candidates:
        raise ValueError("candidates must be a non-empty list")

    if client is None:
        client = openai.OpenAI(api_key=_load_api_key(), base_url=BASE_URL)

    tool = _build_tool(len(candidates))
    user_content = _user_payload(game_state, candidates, history, injury_context)

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "submit_defensive_recommendation"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    msg = response.choices[0].message
    if not msg.tool_calls:
        raise RuntimeError(
            "LLM did not call submit_defensive_recommendation; "
            f"finish_reason={response.choices[0].finish_reason}, "
            f"content={msg.content!r}"
        )

    args: dict[str, Any] = json.loads(msg.tool_calls[0].function.arguments)
    idx = int(args["recommended_candidate_index"])
    if not (0 <= idx < len(candidates)):
        raise RuntimeError(
            f"LLM returned out-of-range candidate index {idx} "
            f"(have {len(candidates)} candidates)"
        )

    rec = candidates[idx]
    policy_choice = candidates[0]
    override_log = {
        "policy_choice": policy_choice,
        "llm_choice": rec,
        "override": bool(args["override"]),
        "reason_codes": list(args["override_reason_codes"]),
        "material_absences": _material_absences(injury_context),
        "game_id": game_id,
        "play_id": play_id,
    }
    return CoordinatorResult(
        recommended_action=rec,
        affirmed_policy=bool(args["affirmed_policy"]),
        override=bool(args["override"]),
        override_reason_codes=list(args["override_reason_codes"]),
        rationale=str(args["rationale"]),
        override_log=override_log,
        raw_tool_input=args,
    )
