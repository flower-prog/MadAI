from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _append_query(
    queries: list[dict[str, Any]],
    *,
    query: str,
    purpose: str,
    source: str,
) -> None:
    text = _normalize_text(query)
    if not text:
        return
    key = text.casefold()
    if any(str(item.get("query") or "").casefold() == key for item in queries):
        return
    queries.append(
        {
            "query": text,
            "purpose": purpose,
            "source": source,
        }
    )


def _case_problem_terms(structured_case: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in list(structured_case.get("problem_list") or []):
        text = _normalize_text(item)
        if text and text not in terms:
            terms.append(text)
    if not terms:
        summary = _normalize_text(structured_case.get("case_summary"))
        if summary:
            terms.append(summary[:120])
    return terms


def _top_trial_terms(trial_retrieval_bundle: Mapping[str, Any], *, limit: int) -> list[str]:
    terms: list[str] = []
    for candidate in list(trial_retrieval_bundle.get("candidate_ranking") or [])[: max(int(limit), 1)]:
        if not isinstance(candidate, Mapping):
            continue
        for item in [*list(candidate.get("conditions") or []), *list(candidate.get("interventions") or [])]:
            text = _normalize_text(item)
            if text and text not in terms:
                terms.append(text)
    return terms


def _unknown_criteria(eligibility_assessment_bundle: Mapping[str, Any] | None) -> list[str]:
    criteria: list[str] = []
    for trial in list((eligibility_assessment_bundle or {}).get("assessed_trials") or []):
        if not isinstance(trial, Mapping):
            continue
        for criterion in list(trial.get("criteria") or []):
            if not isinstance(criterion, Mapping):
                continue
            if _normalize_text(criterion.get("label")) != "unknown":
                continue
            raw_text = _normalize_text(criterion.get("raw_text"))
            if raw_text and raw_text not in criteria:
                criteria.append(raw_text)
    return criteria


def _calculator_terms(calculator_evidence_bundle: Mapping[str, Any] | None) -> list[str]:
    terms: list[str] = []
    for item in list((calculator_evidence_bundle or {}).get("risk_evidence_items") or []):
        if not isinstance(item, Mapping):
            continue
        for key in ("calculator", "category"):
            text = _normalize_text(item.get(key))
            if text and text not in terms:
                terms.append(text)
    return terms


def retrieve_medical_knowledge_for_protocol(
    *,
    structured_case: Mapping[str, Any],
    trial_retrieval_bundle: Mapping[str, Any],
    eligibility_assessment_bundle: Mapping[str, Any] | None = None,
    calculator_evidence_bundle: Mapping[str, Any] | None = None,
    retriever: Any | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    problem_terms = _case_problem_terms(structured_case)
    trial_terms = _top_trial_terms(trial_retrieval_bundle, limit=limit)
    calculator_terms = _calculator_terms(calculator_evidence_bundle)

    for criterion in _unknown_criteria(eligibility_assessment_bundle)[: max(int(limit), 1)]:
        prefix = problem_terms[0] if problem_terms else ""
        _append_query(
            queries,
            query=f"{prefix} trial eligibility {criterion}",
            purpose="criterion_explanation",
            source="unknown_criterion",
        )

    for term in trial_terms[: max(int(limit), 1)]:
        _append_query(
            queries,
            query=f"{term} clinical trial eligibility treatment guideline",
            purpose="trial_context",
            source="trial_candidate",
        )

    for term in calculator_terms[: max(int(limit), 1)]:
        prefix = problem_terms[0] if problem_terms else "clinical trial eligibility"
        _append_query(
            queries,
            query=f"{prefix} {term} trial eligibility interpretation",
            purpose="calculator_context",
            source="calculator_evidence",
        )

    if retriever is None:
        return {
            "schema_version": 1,
            "status": "not_configured",
            "backend_used": "none",
            "queries": queries,
            "retrieved_items": [],
            "knowledge_gaps": [
                {
                    "query": item["query"],
                    "reason": "medical knowledge retriever is not configured",
                }
                for item in queries
            ],
            "warnings": [],
        }

    retrieve = getattr(retriever, "retrieve", None)
    if not callable(retrieve):
        return {
            "schema_version": 1,
            "status": "not_configured",
            "backend_used": "invalid_retriever",
            "queries": queries,
            "retrieved_items": [],
            "knowledge_gaps": [
                {
                    "query": item["query"],
                    "reason": "medical knowledge retriever does not expose retrieve(...)",
                }
                for item in queries
            ],
            "warnings": ["medical_knowledge_retriever_missing_retrieve"],
        }

    retrieved_items: list[dict[str, Any]] = []
    knowledge_gaps: list[dict[str, Any]] = []
    for item in queries[: max(int(limit), 1)]:
        rows = retrieve(item["query"], top_k=max(int(limit), 1))
        if not rows:
            knowledge_gaps.append({"query": item["query"], "reason": "no results"})
            continue
        for row in list(rows):
            if isinstance(row, Mapping):
                payload = dict(row)
            else:
                payload = {"text": _normalize_text(row)}
            payload.setdefault("query", item["query"])
            payload.setdefault("purpose", item["purpose"])
            retrieved_items.append(payload)

    return {
        "schema_version": 1,
        "status": "completed",
        "backend_used": retriever.__class__.__name__,
        "queries": queries,
        "retrieved_items": retrieved_items,
        "knowledge_gaps": knowledge_gaps,
        "warnings": [],
    }
