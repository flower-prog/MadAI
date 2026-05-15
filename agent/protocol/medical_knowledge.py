from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any

from .query_planner import build_protocol_medical_queries


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _retrieve_rows(retrieve: Any, query_item: Mapping[str, Any], *, top_k: int) -> Any:
    query = str(query_item.get("query") or "")
    concepts = list(query_item.get("concepts") or [])
    filters = dict(query_item.get("filters") or {})
    try:
        signature = inspect.signature(retrieve)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        kwargs: dict[str, Any] = {"top_k": top_k}
        if "concepts" in params:
            kwargs["concepts"] = concepts
        if "filters" in params:
            kwargs["filters"] = filters
        return retrieve(query, **kwargs)
    try:
        return retrieve(query, top_k=top_k, concepts=concepts, filters=filters)
    except TypeError:
        try:
            return retrieve(query, top_k=top_k)
        except TypeError:
            return retrieve(query)


def retrieve_medical_knowledge_for_protocol(
    *,
    structured_case: Mapping[str, Any],
    trial_retrieval_bundle: Mapping[str, Any],
    eligibility_assessment_bundle: Mapping[str, Any] | None = None,
    calculator_evidence_bundle: Mapping[str, Any] | None = None,
    medical_phrase_bundle: Mapping[str, Any] | None = None,
    retriever: Any | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    query_bundle = build_protocol_medical_queries(
        structured_case=structured_case,
        trial_retrieval_bundle=trial_retrieval_bundle,
        eligibility_assessment_bundle=eligibility_assessment_bundle,
        calculator_evidence_bundle=calculator_evidence_bundle,
        medical_phrase_bundle=medical_phrase_bundle,
        limit=limit,
    )
    queries = list(query_bundle.get("queries") or [])

    if retriever is None:
        return {
            "schema_version": 1,
            "status": "not_configured",
            "backend_used": "none",
            "queries": queries,
            "query_bundle": query_bundle,
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
            "query_bundle": query_bundle,
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
        rows = _retrieve_rows(retrieve, item, top_k=max(int(limit), 1))
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
        "query_bundle": query_bundle,
        "retrieved_items": retrieved_items,
        "knowledge_gaps": knowledge_gaps,
        "warnings": [],
    }
