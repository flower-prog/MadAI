from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent.tools.execution_tools import maybe_load_json


_SYSTEM_PROMPT = """You are a conservative clinical-trial eligibility reviewer.
Use only the supplied patient evidence card and trial cards.
Do not invent patient facts. If evidence is missing, mark it as missing.
Judge both trial-topic relevance and eligibility fit, but treat explicit eligibility
criteria as stronger evidence than the trial summary.
Do not reduce the score because a trial is Withdrawn, Completed, Terminated,
Unknown status, or otherwise not currently recruiting; trial lifecycle status is
only a note for the final report, not a patient-match criterion.
Return only a JSON object matching the requested schema."""


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _compact_trial_for_prompt(card: Mapping[str, Any]) -> dict[str, Any]:
    eligibility = dict(card.get("eligibility") or {})
    return {
        "nct_id": _normalize_text(card.get("nct_id")),
        "title": _normalize_text(card.get("title")),
        "overall_status": _normalize_text(card.get("overall_status") or card.get("status")),
        "study_type": _normalize_text(card.get("study_type")),
        "phase": _normalize_text(card.get("phase")),
        "conditions": list(card.get("conditions") or []),
        "interventions": list(card.get("interventions") or []),
        "brief_summary": _normalize_text(card.get("brief_summary")),
        "gender": _normalize_text(eligibility.get("gender")),
        "minimum_age": _normalize_text(eligibility.get("minimum_age")),
        "maximum_age": _normalize_text(eligibility.get("maximum_age")),
        "inclusion_text": _normalize_text(eligibility.get("inclusion_text")),
        "exclusion_text": _normalize_text(eligibility.get("exclusion_text")),
        "raw_eligibility_text": _normalize_text(eligibility.get("raw_text")),
    }


def _build_user_prompt(patient_evidence_card: Mapping[str, Any], trial_cards: list[Mapping[str, Any]]) -> str:
    payload = {
        "task": (
            "Assess whether the patient appears suitable for each trial based on the supplied "
            "trial card. Use title, conditions, interventions, and brief summary to judge whether "
            "the trial topic is relevant. Use age, sex, inclusion text, exclusion text, and raw "
            "eligibility text to judge eligibility fit. This is a screening review, not a final "
            "enrollment decision. Ignore overall_status when assigning score, trial_relevance, "
            "and eligibility_assessment; overall_status is only a lifecycle note."
        ),
        "decision_labels": [
            "likely_eligible",
            "possible_with_missing_info",
            "likely_excluded",
            "not_relevant",
        ],
        "score_rubric": {
            "score": "integer 0-100; higher means this patient is a better candidate for this trial",
            "90_100": "topic is highly relevant and all major eligibility criteria appear met",
            "70_89": "topic is relevant and eligibility is plausible with only minor missing details",
            "40_69": "possibly relevant but important eligibility information is missing or mixed",
            "10_39": "probably excluded by one or more important criteria",
            "0_9": "not clinically relevant to the patient's condition/intervention context",
        },
        "patient_evidence_card": dict(patient_evidence_card or {}),
        "trials": [_compact_trial_for_prompt(card) for card in trial_cards],
        "output_schema": {
            "results": [
                {
                    "nct_id": "string",
                    "score": "integer 0-100",
                    "trial_relevance": "high | medium | low",
                    "eligibility_assessment": (
                        "likely_eligible | possible_with_missing_info | likely_excluded | not_relevant"
                    ),
                    "inclusion_matches": [
                        {
                            "criterion_or_text": "string",
                            "patient_evidence": "string",
                            "status": "met | unknown | not_met",
                        }
                    ],
                    "exclusion_risks": [
                        {
                            "criterion_or_text": "string",
                            "patient_evidence": "string",
                            "status": "present | absent | unknown",
                        }
                    ],
                    "missing_information": ["string"],
                    "short_reason": "string",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fallback_score(*, assessment: str, relevance: str) -> int:
    base_by_assessment = {
        "likely_eligible": 82,
        "possible_with_missing_info": 55,
        "likely_excluded": 20,
        "not_relevant": 5,
    }
    relevance_adjustment = {
        "high": 10,
        "medium": 0,
        "low": -10,
    }
    score = base_by_assessment.get(assessment, 55) + relevance_adjustment.get(relevance, 0)
    if assessment == "likely_excluded":
        score = min(score, 35)
    if assessment == "not_relevant":
        score = min(score, 9)
    return max(min(int(score), 100), 0)


def _normalize_score(value: Any, *, assessment: str, relevance: str) -> int:
    try:
        if isinstance(value, str):
            value = value.strip().rstrip("%")
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = _fallback_score(assessment=assessment, relevance=relevance)
    return max(min(score, 100), 0)


def _trial_status_note(status: str) -> str:
    normalized = _normalize_text(status)
    if not normalized:
        return ""
    lowered = normalized.casefold()
    active_statuses = {
        "recruiting",
        "not yet recruiting",
        "enrolling by invitation",
        "available",
    }
    if lowered in active_statuses:
        return ""
    return (
        f"Trial lifecycle status is {normalized}; this status was not used to lower "
        "the patient-match score."
    )


def _normalize_result(item: Mapping[str, Any], *, overall_status: str = "") -> dict[str, Any] | None:
    nct_id = _normalize_text(item.get("nct_id"))
    if not nct_id:
        return None
    assessment = _normalize_text(item.get("eligibility_assessment"))
    if assessment not in {"likely_eligible", "possible_with_missing_info", "likely_excluded", "not_relevant"}:
        assessment = "possible_with_missing_info"
    relevance = _normalize_text(item.get("trial_relevance"))
    if relevance not in {"high", "medium", "low"}:
        relevance = "medium"
    score = _normalize_score(item.get("score"), assessment=assessment, relevance=relevance)
    return {
        "nct_id": nct_id,
        "score": score,
        "trial_relevance": relevance,
        "eligibility_assessment": assessment,
        "overall_status": _normalize_text(overall_status),
        "trial_status_note": _trial_status_note(overall_status),
        "inclusion_matches": [
            dict(row)
            for row in list(item.get("inclusion_matches") or [])
            if isinstance(row, Mapping)
        ],
        "exclusion_risks": [
            dict(row)
            for row in list(item.get("exclusion_risks") or [])
            if isinstance(row, Mapping)
        ],
        "missing_information": [
            _normalize_text(value)
            for value in list(item.get("missing_information") or [])
            if _normalize_text(value)
        ],
        "short_reason": _normalize_text(item.get("short_reason")),
    }


def judge_trials_with_llm(
    *,
    patient_evidence_card: Mapping[str, Any],
    trial_cards: list[Mapping[str, Any]],
    chat_client: Any,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if chat_client is None:
        return {
            "schema_version": 1,
            "status": "skipped",
            "results": [],
            "warnings": ["llm_eligibility_chat_client_not_configured"],
        }
    if not trial_cards:
        return {"schema_version": 1, "status": "completed", "results": [], "warnings": []}

    answer = chat_client.complete(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(patient_evidence_card, trial_cards)},
        ],
        model=model,
        temperature=temperature,
    )
    payload = maybe_load_json(answer)
    if not isinstance(payload, Mapping):
        return {
            "schema_version": 1,
            "status": "failed",
            "results": [],
            "warnings": ["llm_eligibility_response_not_json"],
            "raw_response": str(answer or "")[:2000],
        }

    status_by_nct = {
        _normalize_text(card.get("nct_id")): _normalize_text(card.get("overall_status") or card.get("status"))
        for card in list(trial_cards or [])
        if isinstance(card, Mapping) and _normalize_text(card.get("nct_id"))
    }

    results: list[dict[str, Any]] = []
    for item in list(payload.get("results") or []):
        if not isinstance(item, Mapping):
            continue
        nct_id = _normalize_text(item.get("nct_id"))
        normalized = _normalize_result(item, overall_status=status_by_nct.get(nct_id, ""))
        if normalized is not None:
            results.append(normalized)

    score_summary = [
        {
            "nct_id": item["nct_id"],
            "score": item["score"],
            "trial_relevance": item["trial_relevance"],
            "eligibility_assessment": item["eligibility_assessment"],
            "overall_status": item["overall_status"],
            "trial_status_note": item["trial_status_note"],
        }
        for item in sorted(results, key=lambda row: int(row.get("score") or 0), reverse=True)
    ]
    return {
        "schema_version": 1,
        "status": "completed",
        "results": results,
        "score_summary": score_summary,
        "warnings": [],
    }
