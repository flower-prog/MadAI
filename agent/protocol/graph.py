from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .config import ProtocolGraphConfig
from .evidence_retriever import build_patient_evidence_index
from .medical_knowledge import retrieve_medical_knowledge_for_protocol
from .medical_phrase_parser import parse_medical_phrases_for_protocol
from .pipeline import assess_trial_eligibility_candidates
from .state import ProtocolGraphState
from .trial_query_planner import build_trial_search_intent
from .types import to_plain_dict


def _plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return dict(asdict(value))
    if isinstance(value, dict):
        return dict(value)
    return {}


def _build_calculator_evidence_bundle(
    *,
    calculation_results: list[Any],
    calculator_matches: list[Any],
    calculation_bundle: dict[str, Any],
) -> dict[str, Any]:
    risk_evidence_items: list[dict[str, Any]] = []
    for item in list(calculation_results or []):
        payload = _plain_dict(item)
        calculator = str(payload.get("linked_calculator") or payload.get("calculator") or "").strip()
        if not calculator:
            continue
        risk_evidence_items.append(
            {
                "calculator": calculator,
                "category": str(payload.get("category") or "risk_score"),
                "value": payload.get("value"),
                "unit": payload.get("unit") or "",
                "status": str(payload.get("status") or ""),
                "rationale": str(payload.get("rationale") or ""),
                "usable_for": [
                    "trial_query_profile",
                    "eligibility_evidence",
                    "treatment_recommendation",
                ],
                "missing_inputs": list(payload.get("missing_inputs") or []),
            }
        )

    return {
        "schema_version": 1,
        "status": "completed",
        "completed_results": [
            _plain_dict(item)
            for item in list(calculation_results or [])
            if str(_plain_dict(item).get("status") or "") == "completed"
        ],
        "partial_results": [
            _plain_dict(item)
            for item in list(calculation_results or [])
            if str(_plain_dict(item).get("status") or "") == "partial"
        ],
        "estimated_results": [
            _plain_dict(item)
            for item in list(calculation_results or [])
            if str(_plain_dict(item).get("status") or "") == "estimated"
        ],
        "calculator_matches": [_plain_dict(item) for item in list(calculator_matches or [])],
        "calculation_bundle": dict(calculation_bundle or {}),
        "risk_evidence_items": risk_evidence_items,
    }


def _build_patient_evidence_bundle(
    *,
    structured_case: dict[str, Any],
    calculation_results: list[Any],
    calculator_matches: list[Any],
    calculator_evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    evidence_spans = build_patient_evidence_index(
        structured_case,
        calculation_results=list(calculation_results or []),
        calculator_matches=list(calculator_matches or []),
        calculator_evidence_bundle=dict(calculator_evidence_bundle or {}),
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "evidence_spans": [to_plain_dict(item) for item in evidence_spans],
        "evidence_span_count": len(evidence_spans),
        "warnings": [],
    }


def _empty_calculator_evidence_bundle(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "completed_results": [],
        "partial_results": [],
        "estimated_results": [],
        "calculator_matches": [],
        "calculation_bundle": {},
        "risk_evidence_items": [],
        "warnings": [reason],
    }


def _empty_patient_evidence_bundle(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "evidence_spans": [],
        "evidence_span_count": 0,
        "warnings": [reason],
    }


def _empty_missing_data_bundle(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "missing_questions": [],
        "warnings": [reason],
    }


def _empty_trial_retrieval_bundle(state: ProtocolGraphState) -> dict[str, Any]:
    return {
        "query_text": "",
        "backend_used": "bm25",
        "available_backends": ["bm25"],
        "department_tags": list(state.department_tags),
        "fallback_to_full_catalog": False,
        "coarse_candidate_ids": [],
        "bm25_top5": [],
        "vector_top5": [],
        "candidate_ranking": [],
    }


def _skipped_trial_retrieval_bundle(state: ProtocolGraphState, reason: str) -> dict[str, Any]:
    bundle = _empty_trial_retrieval_bundle(state)
    bundle["status"] = "skipped"
    bundle["warnings"] = [reason]
    return bundle


def _retrieve_trial_candidates(state: ProtocolGraphState, config: ProtocolGraphConfig) -> dict[str, Any]:
    structured_case = dict(state.structured_case_json or {})
    if not structured_case:
        return _empty_trial_retrieval_bundle(state)
    if state.trial_search_intent:
        structured_case["trial_search_intent"] = dict(state.trial_search_intent)

    trial_retriever = state.trial_retriever
    if trial_retriever is None:
        raise TypeError("protocol trial_retriever is required for trial retrieval.")

    if hasattr(trial_retriever, "retrieve_from_structured_case"):
        return trial_retriever.retrieve_from_structured_case(
            structured_case,
            top_k=config.fine_top_k,
            coarse_top_k=config.coarse_top_k,
            department_tags=list(state.department_tags),
            backend=state.retriever_backend,
        )

    candidate_tool = getattr(trial_retriever, "retrieve_candidates", None)
    if callable(candidate_tool):
        return candidate_tool(
            structured_case=structured_case,
            top_k=config.fine_top_k,
            coarse_top_k=config.coarse_top_k,
            department_tags=list(state.department_tags),
            backend=state.retriever_backend,
        )

    raise TypeError(
        "trial_retriever registry entry must expose retrieve_from_structured_case(...) or retrieve_candidates(...)."
    )


def _assess_trial_eligibility(
    state: ProtocolGraphState,
    *,
    trial_bundle: dict[str, Any],
    config: ProtocolGraphConfig,
) -> dict[str, Any]:
    if not list(trial_bundle.get("candidate_ranking") or []):
        return {
            "schema_version": 1,
            "assessed_trial_count": 0,
            "assessed_trials": [],
        }

    return assess_trial_eligibility_candidates(
        structured_case=dict(state.structured_case_json or {}),
        calculation_results=list(state.calculation_results or []),
        calculator_matches=list(state.calculator_matches or []),
        calculator_evidence_bundle=dict(state.calculator_evidence_bundle or {}),
        trial_bundle=dict(trial_bundle or {}),
        trial_retriever=state.trial_retriever,
        limit=config.eligibility_limit,
        trial_search_intent=dict(state.trial_search_intent or {}),
        patient_evidence_bundle=dict(state.patient_evidence_bundle or {}),
        eligibility_chat_client=state.eligibility_chat_client,
        llm_model=state.llm_model,
    )


def _extract_missing_data_bundle(eligibility_assessment_bundle: dict[str, Any]) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trial in list(eligibility_assessment_bundle.get("assessed_trials") or []):
        if not isinstance(trial, dict):
            continue
        nct_id = str(trial.get("nct_id") or "").strip()
        for item in list(trial.get("missing_questions") or []):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            key = f"{nct_id}:{question.casefold()}"
            if not question or key in seen:
                continue
            seen.add(key)
            payload = dict(item)
            payload.setdefault("nct_id", nct_id)
            questions.append(payload)
    return {
        "schema_version": 1,
        "status": "completed",
        "missing_questions": questions,
        "missing_question_count": len(questions),
        "warnings": [],
    }


def _skipped_medical_knowledge_bundle(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "backend_used": "none",
        "queries": [],
        "retrieved_items": [],
        "knowledge_gaps": [],
        "warnings": [reason],
    }


def _build_medical_phrase_bundle(
    state: ProtocolGraphState,
    *,
    config: ProtocolGraphConfig,
) -> dict[str, Any]:
    return parse_medical_phrases_for_protocol(
        structured_case=dict(state.structured_case_json or {}),
        trial_retrieval_bundle=dict(state.trial_retrieval_bundle or {}),
        eligibility_assessment_bundle=dict(state.eligibility_assessment_bundle or {}),
        parser=state.medical_phrase_parser,
        limit=config.eligibility_limit,
    )


def _build_trial_search_intent(state: ProtocolGraphState) -> dict[str, Any]:
    return build_trial_search_intent(
        structured_case=dict(state.structured_case_json or {}),
        calculation_bundle=dict(state.calculation_bundle or {}),
        calculator_matches=list(state.calculator_matches or []),
        department_tags=list(state.department_tags or []),
        planner=state.trial_query_planner,
    )


def _build_protocol_decision_bundle(
    state: ProtocolGraphState,
    *,
    config: ProtocolGraphConfig,
) -> dict[str, Any]:
    trial_candidates = list((state.trial_retrieval_bundle or {}).get("candidate_ranking") or [])
    assessed_trials = list((state.eligibility_assessment_bundle or {}).get("assessed_trials") or [])
    return {
        "schema_version": 1,
        "status": "completed",
        "branch_status": {
            "trial_agent": str((state.trial_retrieval_bundle or {}).get("status") or "completed"),
            "medical_knowledge_agent": str((state.medical_knowledge_bundle or {}).get("status") or "unknown"),
            "risk_evidence_agent": str((state.calculator_evidence_bundle or {}).get("status") or "completed"),
            "patient_calculator_evidence_agent": str((state.patient_evidence_bundle or {}).get("status") or "unknown"),
            "medical_phrase_parser": str((state.medical_phrase_bundle or {}).get("status") or "unknown"),
        },
        "skip_flags": {
            "skip_trial_agent": config.skip_trial_agent,
            "skip_medical_knowledge_agent": config.skip_medical_knowledge_agent,
            "skip_patient_calculator_evidence_agent": config.skip_patient_calculator_evidence_agent,
        },
        "counts": {
            "trial_candidates": len(trial_candidates),
            "assessed_trials": len(assessed_trials),
            "patient_evidence_spans": int((state.patient_evidence_bundle or {}).get("evidence_span_count") or 0),
            "medical_concepts": int((state.medical_phrase_bundle or {}).get("concept_count") or 0),
            "medical_knowledge_items": len(list((state.medical_knowledge_bundle or {}).get("retrieved_items") or [])),
            "missing_questions": int((state.missing_data_bundle or {}).get("missing_question_count") or 0),
        },
        "warnings": [
            *list((state.trial_retrieval_bundle or {}).get("warnings") or []),
            *list((state.medical_phrase_bundle or {}).get("warnings") or []),
            *list((state.medical_knowledge_bundle or {}).get("warnings") or []),
            *list((state.patient_evidence_bundle or {}).get("warnings") or []),
            *list((state.missing_data_bundle or {}).get("warnings") or []),
        ],
    }


def run_protocol_subgraph(
    state: ProtocolGraphState,
    *,
    config: ProtocolGraphConfig | None = None,
) -> ProtocolGraphState:
    resolved_config = config or ProtocolGraphConfig.from_env()
    if resolved_config.skip_patient_calculator_evidence_agent:
        reason = "patient_calculator_evidence_agent skipped by protocol config"
        state.calculator_evidence_bundle = _empty_calculator_evidence_bundle(reason)
        state.patient_evidence_bundle = _empty_patient_evidence_bundle(reason)
        state.missing_data_bundle = _empty_missing_data_bundle(reason)
    else:
        state.calculator_evidence_bundle = _build_calculator_evidence_bundle(
            calculation_results=list(state.calculation_results or []),
            calculator_matches=list(state.calculator_matches or []),
            calculation_bundle=dict(state.calculation_bundle or {}),
        )
        state.patient_evidence_bundle = _build_patient_evidence_bundle(
            structured_case=dict(state.structured_case_json or {}),
            calculation_results=list(state.calculation_results or []),
            calculator_matches=list(state.calculator_matches or []),
            calculator_evidence_bundle=dict(state.calculator_evidence_bundle or {}),
        )

    if resolved_config.skip_trial_agent:
        reason = "trial_agent skipped by protocol config"
        state.trial_search_intent = _build_trial_search_intent(state)
        state.trial_retrieval_bundle = _skipped_trial_retrieval_bundle(state, reason)
        state.eligibility_assessment_bundle = {
            "schema_version": 1,
            "status": "skipped",
            "assessed_trial_count": 0,
            "assessed_trials": [],
            "warnings": [reason],
        }
    else:
        state.trial_search_intent = _build_trial_search_intent(state)
        state.trial_retrieval_bundle = _retrieve_trial_candidates(state, resolved_config)
        state.eligibility_assessment_bundle = _assess_trial_eligibility(
            state,
            trial_bundle=state.trial_retrieval_bundle,
            config=resolved_config,
        )

    if not state.missing_data_bundle:
        state.missing_data_bundle = _extract_missing_data_bundle(state.eligibility_assessment_bundle)

    state.medical_phrase_bundle = _build_medical_phrase_bundle(state, config=resolved_config)

    if resolved_config.skip_medical_knowledge_agent:
        state.medical_knowledge_bundle = _skipped_medical_knowledge_bundle(
            "medical_knowledge_agent skipped by protocol config"
        )
    else:
        state.medical_knowledge_bundle = retrieve_medical_knowledge_for_protocol(
            structured_case=dict(state.structured_case_json or {}),
            trial_retrieval_bundle=dict(state.trial_retrieval_bundle or {}),
            eligibility_assessment_bundle=dict(state.eligibility_assessment_bundle or {}),
            calculator_evidence_bundle=dict(state.calculator_evidence_bundle or {}),
            medical_phrase_bundle=dict(state.medical_phrase_bundle or {}),
            retriever=state.medical_knowledge_retriever,
            limit=resolved_config.eligibility_limit,
        )
    state.protocol_decision_bundle = _build_protocol_decision_bundle(state, config=resolved_config)
    return state
