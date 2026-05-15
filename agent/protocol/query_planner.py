from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _append_unique(items: list[str], value: Any) -> None:
    text = _normalize_text(value)
    if not text:
        return
    key = text.casefold()
    if any(item.casefold() == key for item in items):
        return
    items.append(text)


def _append_query(
    queries: list[dict[str, Any]],
    *,
    query: str,
    purpose: str,
    source: str,
    linked_nct_id: str = "",
    linked_criterion: str = "",
    concepts: list[dict[str, Any]] | None = None,
    filters: dict[str, Any] | None = None,
    required_patient_evidence: list[str] | None = None,
) -> None:
    text = _normalize_text(query)
    if not text:
        return
    key = (
        text.casefold(),
        _normalize_text(purpose).casefold(),
        _normalize_text(linked_nct_id).casefold(),
        _normalize_text(linked_criterion).casefold(),
    )
    existing = {
        (
            str(item.get("query") or "").casefold(),
            str(item.get("purpose") or "").casefold(),
            str(item.get("linked_nct_id") or "").casefold(),
            str(item.get("linked_criterion") or "").casefold(),
        )
        for item in queries
    }
    if key in existing:
        return
    queries.append(
        {
            "query": text,
            "purpose": purpose,
            "source": source,
            "linked_nct_id": linked_nct_id,
            "linked_criterion": linked_criterion,
            "concepts": list(concepts or []),
            "filters": dict(filters or {}),
            "required_patient_evidence": list(required_patient_evidence or []),
        }
    )


def _case_problem_terms(structured_case: Mapping[str, Any]) -> list[str]:
    payload = dict(structured_case or {})
    if isinstance(payload.get("structured_case"), Mapping):
        payload = dict(payload["structured_case"])

    terms: list[str] = []
    for key in ("problem_list", "known_facts", "risk_hints"):
        for item in list(payload.get(key) or []):
            _append_unique(terms, item)
    for key in ("case_summary", "raw_text", "raw_request"):
        text = _normalize_text(payload.get(key))
        if text and not terms:
            _append_unique(terms, text[:160])
    return terms


def _calculator_terms(calculator_evidence_bundle: Mapping[str, Any] | None) -> list[str]:
    terms: list[str] = []
    for item in list((calculator_evidence_bundle or {}).get("risk_evidence_items") or []):
        if not isinstance(item, Mapping):
            continue
        for key in ("calculator", "category"):
            _append_unique(terms, item.get(key))
    return terms


def _concepts_by_name(medical_phrase_bundle: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    concepts: dict[str, dict[str, Any]] = {}
    for item in list((medical_phrase_bundle or {}).get("concepts") or []):
        if not isinstance(item, Mapping):
            continue
        canonical = _normalize_text(item.get("canonical_name") or item.get("name") or item.get("surface"))
        semantic_type = _normalize_text(item.get("semantic_type"))
        if not canonical:
            continue
        key = f"{canonical.casefold()}::{semantic_type.casefold()}"
        payload = dict(item)
        payload.setdefault("canonical_name", canonical)
        payload.setdefault("semantic_type", semantic_type)
        concepts[key] = payload
    return concepts


def _phrase_context_by_text(medical_phrase_bundle: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for item in list((medical_phrase_bundle or {}).get("parsed_phrases") or []):
        if not isinstance(item, Mapping):
            continue
        raw_text = _normalize_text(item.get("raw_text") or item.get("normalized_text"))
        if raw_text:
            contexts[raw_text.casefold()] = dict(item)
    return contexts


def _candidate_trials(trial_retrieval_bundle: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list((trial_retrieval_bundle or {}).get("candidate_ranking") or []):
        if not isinstance(item, Mapping):
            continue
        nct_id = _normalize_text(item.get("nct_id"))
        if nct_id and nct_id in seen:
            continue
        if nct_id:
            seen.add(nct_id)
        trials.append(dict(item))
        if len(trials) >= max(int(limit), 1):
            break
    return trials


def _trial_terms(candidate: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    conditions: list[str] = []
    interventions: list[str] = []
    for item in list(candidate.get("conditions") or []):
        _append_unique(conditions, item)
    for item in list(candidate.get("interventions") or []):
        _append_unique(interventions, item)
    return conditions, interventions


def _assessed_criteria(eligibility_assessment_bundle: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    criteria: list[dict[str, Any]] = []
    for trial in list((eligibility_assessment_bundle or {}).get("assessed_trials") or []):
        if not isinstance(trial, Mapping):
            continue
        nct_id = _normalize_text(trial.get("nct_id"))
        for criterion in list(trial.get("criteria") or []):
            if not isinstance(criterion, Mapping):
                continue
            raw_text = _normalize_text(criterion.get("raw_text"))
            if not raw_text:
                continue
            payload = dict(criterion)
            payload.setdefault("nct_id", nct_id)
            criteria.append(payload)
    return criteria


def _concept_filters(concept: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    semantic_type = _normalize_text(concept.get("semantic_type"))
    if semantic_type in {"drug", "treatment_or_medication"}:
        return "drug_or_intervention_context", {"source_type": ["drug_label", "guideline", "ontology"]}
    if semantic_type == "calculator":
        return "calculator_context", {"source_type": ["calculator_reference", "guideline", "ontology"]}
    if semantic_type == "measurement":
        return "criterion_explanation", {"source_type": ["guideline", "ontology", "trial_reference"]}
    return "condition_context", {"source_type": ["guideline", "ontology", "pubmed"]}


def _concept_query_text(concept: Mapping[str, Any]) -> str:
    canonical = _normalize_text(concept.get("canonical_name") or concept.get("surface"))
    semantic_type = _normalize_text(concept.get("semantic_type"))
    if semantic_type in {"drug", "treatment_or_medication"}:
        return f"{canonical} clinical trial eligibility contraindication treatment guideline"
    if semantic_type == "calculator":
        return f"{canonical} interpretation threshold clinical trial eligibility"
    if semantic_type == "measurement":
        return f"{canonical} clinical trial eligibility cutoff required evidence"
    return f"{canonical} clinical trial eligibility treatment guideline"


def _query_terms(values: list[str], *, limit: int) -> str:
    return " ".join(values[: max(int(limit), 1)])


def build_protocol_medical_queries(
    *,
    structured_case: Mapping[str, Any],
    trial_retrieval_bundle: Mapping[str, Any],
    eligibility_assessment_bundle: Mapping[str, Any] | None = None,
    calculator_evidence_bundle: Mapping[str, Any] | None = None,
    medical_phrase_bundle: Mapping[str, Any] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    case_terms = _case_problem_terms(structured_case)
    calculator_terms = _calculator_terms(calculator_evidence_bundle)
    concepts = list(_concepts_by_name(medical_phrase_bundle).values())
    phrase_contexts = _phrase_context_by_text(medical_phrase_bundle)
    trials = _candidate_trials(trial_retrieval_bundle, limit=limit)

    if case_terms and calculator_terms:
        _append_query(
            queries,
            query=f"{_query_terms(case_terms, limit=4)} {_query_terms(calculator_terms, limit=2)} treatment threshold interpretation",
            purpose="calculator_interpretation",
            source="case_calculator_overlap",
            filters={"source_type": ["calculator_reference", "guideline"]},
        )

    for candidate in trials:
        nct_id = _normalize_text(candidate.get("nct_id"))
        conditions, interventions = _trial_terms(candidate)
        overlap_terms = list(case_terms[:3])
        if conditions:
            overlap_terms.extend(conditions[:2])
        if interventions:
            overlap_terms.extend(interventions[:2])
        if overlap_terms:
            _append_query(
                queries,
                query=f"{_query_terms(overlap_terms, limit=7)} clinical trial eligibility patient selection",
                purpose="case_trial_overlap",
                source="case_trial_candidate",
                linked_nct_id=nct_id,
                filters={"source_type": ["guideline", "trial_reference", "pubmed"]},
            )

    for criterion in _assessed_criteria(eligibility_assessment_bundle):
        raw_text = _normalize_text(criterion.get("raw_text"))
        nct_id = _normalize_text(criterion.get("nct_id"))
        label = _normalize_text(criterion.get("label"))
        context = phrase_contexts.get(raw_text.casefold(), {})
        criterion_concepts = [
            dict(item)
            for item in list(context.get("concepts") or [])
            if isinstance(item, Mapping)
        ]
        required_evidence = [
            _normalize_text(item)
            for item in list(context.get("required_patient_evidence") or [])
            if _normalize_text(item)
        ]
        prefix = _query_terms(case_terms, limit=2)
        query = f"{prefix} trial eligibility {raw_text}".strip()
        _append_query(
            queries,
            query=query,
            purpose="criterion_explanation" if label == "unknown" else "criterion_context",
            source="trial_criterion",
            linked_nct_id=nct_id,
            linked_criterion=raw_text,
            concepts=criterion_concepts,
            filters={"source_type": ["guideline", "ontology", "trial_reference", "drug_label"]},
            required_patient_evidence=required_evidence,
        )

    for concept in concepts:
        canonical = _normalize_text(concept.get("canonical_name") or concept.get("surface"))
        if not canonical:
            continue
        purpose, filters = _concept_filters(concept)
        _append_query(
            queries,
            query=_concept_query_text(concept),
            purpose=purpose,
            source="medical_phrase_parser",
            concepts=[concept],
            filters=filters,
        )

    for term in calculator_terms[: max(int(limit), 1)]:
        prefix = _query_terms(case_terms, limit=2) or "clinical trial eligibility"
        _append_query(
            queries,
            query=f"{prefix} {term} trial eligibility interpretation",
            purpose="calculator_context",
            source="calculator_evidence",
            filters={"source_type": ["calculator_reference", "guideline"]},
        )

    return {
        "schema_version": 1,
        "status": "completed",
        "queries": queries[: max(int(limit), 1)],
        "candidate_query_count": len(queries),
        "warnings": [],
    }
