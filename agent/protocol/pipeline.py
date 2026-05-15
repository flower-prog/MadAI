from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .criteria_parser import parse_trial_criteria
from .criterion_judge import judge_criterion
from .eligibility_aggregator import aggregate_trial_eligibility
from .evidence_retriever import build_patient_evidence_index, find_evidence_for_criterion
from .llm_eligibility import judge_trials_with_llm
from .missing_data import generate_missing_questions
from .patient_card import build_patient_evidence_card
from .trial_card import build_trial_card, build_trial_card_text
from .types import to_plain_dict


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _candidate_trials(trial_bundle: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(trial_bundle.get("candidate_ranking") or []):
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        nct_id = _normalize_text(candidate.get("nct_id"))
        if not nct_id or nct_id in seen:
            continue
        seen.add(nct_id)
        selected.append(candidate)
        if len(selected) >= max(int(limit), 1):
            break
    return selected


def _resolve_trial_record(trial_retriever: Any, candidate: Mapping[str, Any]) -> dict[str, Any]:
    nct_id = _normalize_text(candidate.get("nct_id"))
    get_trial_record = getattr(trial_retriever, "get_trial_record", None)
    if callable(get_trial_record):
        record = get_trial_record(nct_id)
        if isinstance(record, Mapping):
            return {**dict(candidate), **dict(record), "nct_id": nct_id}

    catalog = getattr(trial_retriever, "catalog", None)
    get_record = getattr(catalog, "get_record", None)
    if callable(get_record):
        record = get_record(nct_id)
        if isinstance(record, Mapping):
            return {**dict(candidate), **dict(record), "nct_id": nct_id}

    get_document = getattr(catalog, "get", None)
    if callable(get_document):
        document = get_document(nct_id)
        if document is not None:
            payload = dict(getattr(document, "source_payload", {}) or {})
            payload.setdefault("nct_id", nct_id)
            payload.setdefault("title", getattr(document, "title", ""))
            payload.setdefault("eligibility_text", getattr(document, "eligibility", ""))
            payload.setdefault("overall_status", getattr(document, "overall_status", ""))
            payload.setdefault("enrollment_open", getattr(document, "enrollment_open", False))
            return {**dict(candidate), **payload, "nct_id": nct_id}

    return dict(candidate)


def _parse_warning_stats(assessed_trials: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for trial in list(assessed_trials or []):
        status = _normalize_text(trial.get("eligibility_section_parse_status")) or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        for warning in list(trial.get("eligibility_section_parse_warnings") or []):
            text = _normalize_text(warning)
            if not text:
                continue
            warning_counts[text] = warning_counts.get(text, 0) + 1
    return {
        "section_parse_status_counts": status_counts,
        "section_parse_warning_counts": warning_counts,
    }


def assess_trial_eligibility_candidates(
    *,
    structured_case: Mapping[str, Any],
    calculation_results: list[Any] | None,
    calculator_matches: list[Any] | None,
    calculator_evidence_bundle: Mapping[str, Any] | None = None,
    trial_bundle: Mapping[str, Any],
    trial_retriever: Any = None,
    limit: int = 3,
    trial_search_intent: Mapping[str, Any] | None = None,
    patient_evidence_bundle: Mapping[str, Any] | None = None,
    eligibility_chat_client: Any = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    evidence_index = build_patient_evidence_index(
        structured_case,
        calculation_results=list(calculation_results or []),
        calculator_matches=list(calculator_matches or []),
        calculator_evidence_bundle=calculator_evidence_bundle,
    )
    resolved_patient_evidence_bundle = dict(patient_evidence_bundle or {})
    if not resolved_patient_evidence_bundle:
        resolved_patient_evidence_bundle = {
            "schema_version": 1,
            "status": "completed",
            "evidence_spans": [to_plain_dict(item) for item in evidence_index],
            "evidence_span_count": len(evidence_index),
        }
    resolved_trial_search_intent = dict(
        trial_search_intent
        or (structured_case.get("trial_search_intent") if isinstance(structured_case, Mapping) else {})
        or {}
    )
    patient_evidence_card = build_patient_evidence_card(
        structured_case,
        trial_search_intent=resolved_trial_search_intent,
        patient_evidence_bundle=resolved_patient_evidence_bundle,
    )

    assessed_trials = []
    trial_cards: list[dict[str, Any]] = []
    for candidate in _candidate_trials(trial_bundle, limit=limit):
        trial_record = _resolve_trial_record(trial_retriever, candidate)
        trial_card = build_trial_card(trial_record, candidate=candidate)
        trial_card["card_text"] = build_trial_card_text(trial_card)
        trial_cards.append(trial_card)
        criteria = parse_trial_criteria(trial_record)
        assessments = []
        for criterion in criteria:
            evidence_spans = find_evidence_for_criterion(evidence_index, criterion)
            assessments.append(judge_criterion(criterion, evidence_spans))
        missing_questions = generate_missing_questions(assessments)
        trial_assessment = aggregate_trial_eligibility(
            trial_record,
            assessments,
            missing_questions=missing_questions,
        )
        assessment_payload = to_plain_dict(trial_assessment)
        assessment_payload["eligibility_section_parse_status"] = _normalize_text(
            trial_record.get("eligibility_section_parse_status")
        )
        assessment_payload["eligibility_section_parse_warnings"] = [
            _normalize_text(item)
            for item in list(trial_record.get("eligibility_section_parse_warnings") or [])
            if _normalize_text(item)
        ]
        assessment_payload["eligibility_unsplit_text_present"] = bool(
            _normalize_text(trial_record.get("eligibility_unsplit_text"))
        )
        assessment_payload["trial_card"] = trial_card
        assessed_trials.append(assessment_payload)

    llm_bundle = judge_trials_with_llm(
        patient_evidence_card=patient_evidence_card,
        trial_cards=trial_cards,
        chat_client=eligibility_chat_client,
        model=llm_model,
    )
    llm_by_nct = {
        _normalize_text(item.get("nct_id")): dict(item)
        for item in list(llm_bundle.get("results") or [])
        if isinstance(item, Mapping) and _normalize_text(item.get("nct_id"))
    }
    if llm_by_nct:
        for trial in assessed_trials:
            llm_result = llm_by_nct.get(_normalize_text(trial.get("nct_id")))
            if llm_result:
                trial["llm_eligibility_assessment"] = llm_result

    return {
        "schema_version": 1,
        "assessed_trial_count": len(assessed_trials),
        "assessed_trials": assessed_trials,
        "parse_warning_stats": _parse_warning_stats(assessed_trials),
        "patient_evidence_card": patient_evidence_card,
        "llm_eligibility_bundle": llm_bundle,
    }
