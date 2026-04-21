from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from agent.retrieval import (
    DEFAULT_QDRANT_COLLECTION_NAME,
    FieldedBM25Index,
    MedCPTRetriever,
    TrialChunkCatalog,
    TrialChunkDocument,
    build_structured_query_text,
    create_qdrant_trial_chunk_retriever,
    create_structured_retriever,
    resolve_trial_qdrant_runtime_config,
    resolve_trial_vector_output_root,
)

from .execution_tools import tool


_CACHE_LOCK = threading.RLock()
_TRIAL_CHUNK_RETRIEVER_CACHE: dict[tuple[str, str, str, str, str, str], "TrialChunkRetrievalTool"] = {}
_DEFAULT_PROTOCOL_CHUNK_TOP_K = 40
_OPEN_TRIAL_STATUSES = {
    "Recruiting",
    "Not yet recruiting",
    "Enrolling by invitation",
}
_EVIDENCE_TRIAL_STATUSES = {
    "Active, not recruiting",
    "Completed",
}
_ABANDONED_TRIAL_STATUSES = {
    "Terminated",
    "Withdrawn",
    "Suspended",
    "No longer available",
    "Temporarily not available",
}
_TRIAL_RERANK_PRIORITY_CHUNK_TYPES = (
    "eligibility_inclusion",
    "eligibility_exclusion",
    "overview",
    "arms_interventions",
)
_TRIAL_RERANK_CHUNK_SUPPORT_WEIGHTS = (1.0, 0.35, 0.2)
_NEGATION_CUE_PATTERN = re.compile(
    r"\b(?:"
    r"no|not|without|denies|deny|negative for|absence of|free of|"
    r"does not have|doesn't have|is not|was not|not currently|not prescribed"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_STYLE_CUE_PATTERN = re.compile(
    r"\b(?:what is|estimated|estimate|patient-years|based on these factors|stroke rate)\b",
    re.IGNORECASE,
)
_AGE_PATTERN = re.compile(r"\b(?P<age>\d{1,3})\s*[- ]?\s*year[- ]old\b", re.IGNORECASE)
_TRIAL_TERM_DEFINITIONS: tuple[tuple[str, tuple[re.Pattern[str], ...], str], ...] = (
    (
        "atrial fibrillation",
        (
            re.compile(r"\batrial fibrillation\b", re.IGNORECASE),
            re.compile(r"\ba[\s-]?fib\b", re.IGNORECASE),
        ),
        "condition",
    ),
    (
        "transient ischemic attack",
        (
            re.compile(r"\btransient ischemic attack\b", re.IGNORECASE),
            re.compile(r"\bTIA\b"),
        ),
        "condition",
    ),
    (
        "stroke",
        (
            re.compile(r"\bprior stroke\b", re.IGNORECASE),
            re.compile(r"\bischemic stroke\b", re.IGNORECASE),
            re.compile(r"\bstroke\b", re.IGNORECASE),
        ),
        "condition",
    ),
    (
        "hypertension",
        (
            re.compile(r"\bhypertension\b", re.IGNORECASE),
            re.compile(r"\bhigh blood pressure\b", re.IGNORECASE),
            re.compile(r"\bHTN\b"),
        ),
        "condition",
    ),
    (
        "diabetes",
        (
            re.compile(r"\bdiabetes(?: mellitus)?\b", re.IGNORECASE),
            re.compile(r"\bDM\b"),
            re.compile(r"\bdiabetic\b", re.IGNORECASE),
        ),
        "condition",
    ),
    (
        "congestive heart failure",
        (
            re.compile(r"\bcongestive heart failure\b", re.IGNORECASE),
            re.compile(r"\bheart failure\b", re.IGNORECASE),
            re.compile(r"\bCHF\b"),
        ),
        "condition",
    ),
    (
        "warfarin",
        (
            re.compile(r"\bwarfarin\b", re.IGNORECASE),
            re.compile(r"\bcoumadin\b", re.IGNORECASE),
        ),
        "intervention",
    ),
    (
        "anticoagulation",
        (
            re.compile(r"\banticoag(?:ulation|ulant)?\b", re.IGNORECASE),
            re.compile(r"\boral anticoagulant\b", re.IGNORECASE),
            re.compile(r"\bDOAC\b"),
        ),
        "intervention",
    ),
    (
        "antithrombotic therapy",
        (
            re.compile(r"\bantithrombotic therapy\b", re.IGNORECASE),
            re.compile(r"\bantithrombotic\b", re.IGNORECASE),
        ),
        "intervention",
    ),
    (
        "stroke prevention",
        (
            re.compile(r"\bstroke prevention\b", re.IGNORECASE),
            re.compile(r"\bprevent(?:ion|ive)\b", re.IGNORECASE),
        ),
        "intent",
    ),
    (
        "secondary prevention",
        (re.compile(r"\bsecondary prevention\b", re.IGNORECASE),),
        "intent",
    ),
)


def _normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Mapping):
        raw_items = list(value.values())
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = _normalize_whitespace(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _append_unique(target: list[str], value: Any) -> None:
    normalized = _normalize_whitespace(value)
    if not normalized:
        return
    normalized_key = normalized.casefold()
    if normalized_key in {item.casefold() for item in target}:
        return
    target.append(normalized)


def _extract_case_text_fragments(structured_case: Mapping[str, Any]) -> list[str]:
    case_payload = dict(structured_case or {})
    if isinstance(case_payload.get("structured_case"), Mapping):
        case_payload = dict(case_payload["structured_case"])

    fragments: list[str] = []
    for item in (
        case_payload.get("case_summary"),
        case_payload.get("raw_text"),
        case_payload.get("raw_request"),
    ):
        _append_unique(fragments, item)
    for item in _coerce_text_list(case_payload.get("problem_list")):
        _append_unique(fragments, item)
    for item in _coerce_text_list(case_payload.get("known_facts")):
        _append_unique(fragments, item)
    return fragments


def _extract_age_years(structured_case: Mapping[str, Any]) -> int | None:
    case_payload = dict(structured_case or {})
    structured_inputs = dict(case_payload.get("structured_inputs") or {})
    for key in ("age", "age_years", "patient_age", "age_in_years"):
        raw_value = structured_inputs.get(key)
        if raw_value in {None, ""}:
            continue
        try:
            age_value = int(float(raw_value))
        except (TypeError, ValueError):
            continue
        if 0 < age_value < 130:
            return age_value

    combined_text = " ".join(_extract_case_text_fragments(case_payload))
    age_match = _AGE_PATTERN.search(combined_text)
    if not age_match:
        return None
    try:
        age_value = int(age_match.group("age"))
    except (TypeError, ValueError):
        return None
    return age_value if 0 < age_value < 130 else None


def _extract_gender(structured_case: Mapping[str, Any]) -> str:
    case_payload = dict(structured_case or {})
    structured_inputs = dict(case_payload.get("structured_inputs") or {})
    for key in ("sex", "gender"):
        normalized = _normalize_whitespace(structured_inputs.get(key))
        lowered = normalized.casefold()
        if lowered in {"male", "m", "man"}:
            return "Male"
        if lowered in {"female", "f", "woman"}:
            return "Female"

    combined_text = " ".join(_extract_case_text_fragments(case_payload)).casefold()
    if any(token in combined_text for token in (" male ", " male,", " male.", " man ", " man,")) or combined_text.startswith("male "):
        return "Male"
    if any(token in combined_text for token in (" female ", " female,", " female.", " woman ", " woman,")) or combined_text.startswith("female "):
        return "Female"
    return ""


def _term_match_flags(
    *,
    patterns: tuple[re.Pattern[str], ...],
    text_fragments: list[str],
) -> tuple[bool, bool, bool]:
    referenced = False
    positive = False
    negative = False
    for fragment in list(text_fragments or []):
        clauses = [item.strip() for item in re.split(r"[.;\n,]+", str(fragment or "")) if item.strip()]
        if not clauses:
            clauses = [str(fragment or "").strip()]
        for clause in clauses:
            has_match = any(pattern.search(clause) for pattern in patterns)
            if not has_match:
                continue
            referenced = True
            if _NEGATION_CUE_PATTERN.search(clause) and not _QUESTION_STYLE_CUE_PATTERN.search(clause):
                negative = True
            else:
                positive = True
    return referenced, positive, negative


def _infer_trial_focus_terms(
    *,
    combined_text: str,
    trial_condition_terms: list[str],
    trial_intervention_terms: list[str],
    trial_intent_terms: list[str],
    derivation_notes: list[str],
) -> None:
    lowered_text = str(combined_text or "").casefold()
    has_cerebrovascular_cue = any(
        term in {item.casefold() for item in trial_condition_terms}
        for term in {"transient ischemic attack", "stroke"}
    )
    has_anticoagulation_cue = any(
        cue in lowered_text
        for cue in ("warfarin", "antithrombotic", "anticoag")
    )
    has_stroke_risk_cue = any(
        cue in lowered_text
        for cue in ("stroke rate", "patient-years", "stroke prevention")
    )

    if has_cerebrovascular_cue and (has_anticoagulation_cue or has_stroke_risk_cue):
        if "atrial fibrillation" not in {item.casefold() for item in trial_condition_terms}:
            trial_condition_terms.insert(0, "atrial fibrillation")
            _append_unique(
                derivation_notes,
                "Inferred atrial fibrillation as the trial-search anchor from stroke-risk and anticoagulation wording.",
            )
        _append_unique(trial_intent_terms, "stroke prevention")
        _append_unique(trial_intent_terms, "secondary prevention")

    if has_anticoagulation_cue:
        _append_unique(trial_intervention_terms, "anticoagulation")
    if "antithrombotic" in lowered_text:
        _append_unique(trial_intervention_terms, "antithrombotic therapy")


def build_protocol_trial_query_profile(
    structured_case: Mapping[str, Any] | None = None,
    *,
    raw_text: Any = "",
    case_summary: Any = None,
    problem_list: Any = None,
    known_facts: Any = None,
    structured_inputs: Any = None,
) -> dict[str, Any]:
    case_payload = dict(structured_case or {})
    if isinstance(case_payload.get("structured_case"), Mapping):
        case_payload = dict(case_payload["structured_case"])
    if raw_text is not None and raw_text != "":
        case_payload.setdefault("raw_text", raw_text)
    if case_summary is not None and case_summary != "":
        case_payload.setdefault("case_summary", case_summary)
    if problem_list is not None and problem_list != "":
        case_payload.setdefault("problem_list", list(problem_list) if not isinstance(problem_list, str) else [problem_list])
    if known_facts is not None and known_facts != "":
        case_payload.setdefault("known_facts", list(known_facts) if not isinstance(known_facts, str) else [known_facts])
    if isinstance(structured_inputs, Mapping) and structured_inputs:
        case_payload.setdefault("structured_inputs", dict(structured_inputs))

    text_fragments = _extract_case_text_fragments(case_payload)
    combined_text = " ".join(text_fragments)
    problem_items = _coerce_text_list(case_payload.get("problem_list"))
    known_fact_items = _coerce_text_list(case_payload.get("known_facts"))
    patient_positive_terms: list[str] = []
    patient_negative_terms: list[str] = []
    referenced_intervention_terms: list[str] = []
    referenced_intent_terms: list[str] = []
    trial_condition_terms: list[str] = []

    for term, patterns, field_name in _TRIAL_TERM_DEFINITIONS:
        referenced, positive, negative = _term_match_flags(
            patterns=patterns,
            text_fragments=text_fragments,
        )
        if referenced and field_name == "intervention":
            _append_unique(referenced_intervention_terms, term)
        elif referenced and field_name == "intent":
            _append_unique(referenced_intent_terms, term)

        if positive and field_name == "condition":
            _append_unique(patient_positive_terms, term)
            _append_unique(trial_condition_terms, term)
        elif positive and field_name == "intervention":
            _append_unique(patient_positive_terms, term)

        if negative and field_name in {"condition", "intervention"}:
            _append_unique(patient_negative_terms, term)

    derivation_notes: list[str] = []
    trial_intervention_terms = list(referenced_intervention_terms)
    trial_intent_terms = list(referenced_intent_terms)
    _infer_trial_focus_terms(
        combined_text=combined_text,
        trial_condition_terms=trial_condition_terms,
        trial_intervention_terms=trial_intervention_terms,
        trial_intent_terms=trial_intent_terms,
        derivation_notes=derivation_notes,
    )
    fallback_condition_terms: list[str] = []
    for item in problem_items:
        _append_unique(fallback_condition_terms, item)
    if not trial_condition_terms and fallback_condition_terms:
        trial_condition_terms = list(fallback_condition_terms)
        _append_unique(
            derivation_notes,
            "Fell back to problem_list phrases as trial condition anchors because no domain-specific condition term was detected.",
        )

    fallback_intervention_terms: list[str] = []
    for item in known_fact_items:
        if len(str(item).split()) <= 8:
            _append_unique(fallback_intervention_terms, item)
    if not trial_intervention_terms and fallback_intervention_terms:
        trial_intervention_terms = list(fallback_intervention_terms)

    age_years = _extract_age_years(case_payload)
    gender = _extract_gender(case_payload)
    demographic_terms: list[str] = []
    if age_years is not None:
        _append_unique(demographic_terms, f"{age_years} year old")
        if age_years >= 75:
            _append_unique(demographic_terms, "older adult")
    if gender:
        _append_unique(demographic_terms, gender.lower())

    focus_terms: list[str] = []
    for item in (
        *trial_condition_terms,
        *trial_intent_terms,
        *trial_intervention_terms,
        *demographic_terms,
        *[term for term in patient_positive_terms if term not in {"warfarin", "anticoagulation", "antithrombotic therapy"}],
        *fallback_condition_terms,
        *fallback_intervention_terms,
    ):
        _append_unique(focus_terms, item)

    positive_profile_terms: list[str] = list(demographic_terms)
    for term in patient_positive_terms:
        _append_unique(positive_profile_terms, term)
    negative_profile_terms: list[str] = []
    for term in patient_negative_terms:
        if term == "warfarin":
            _append_unique(negative_profile_terms, "not currently on warfarin")
        else:
            _append_unique(negative_profile_terms, f"no {term}")

    query_lines: list[str] = []
    if focus_terms:
        query_lines.append("clinical trial eligibility search: " + " ; ".join(focus_terms))
    if positive_profile_terms:
        query_lines.append("patient profile: " + " ; ".join(positive_profile_terms))
    if negative_profile_terms:
        query_lines.append("screening constraints: " + " ; ".join(negative_profile_terms))
    query_text = "\n".join(query_lines)
    if not query_text:
        query_text = build_structured_query_text(
            raw_text=case_payload.get("raw_text") or case_payload.get("raw_request") or "",
            case_summary=case_payload.get("case_summary"),
            problem_list=problem_items,
            known_facts=known_fact_items,
        )

    payload_filters: dict[str, Any] = {
        "must": [],
        "should": [],
        "must_not": [],
    }
    if trial_condition_terms:
        payload_filters["must"].append(
            {
                "field": "condition_terms",
                "values": [trial_condition_terms[0]],
            }
        )
    if len(trial_condition_terms) > 1:
        payload_filters["should"].append(
            {
                "field": "condition_terms",
                "values": list(trial_condition_terms[1:]),
            }
        )
    if trial_intervention_terms:
        payload_filters["should"].append(
            {
                "field": "intervention_terms",
                "values": list(trial_intervention_terms),
            }
        )
    if "stroke prevention" in {item.casefold() for item in trial_intent_terms} or "secondary prevention" in {
        item.casefold() for item in trial_intent_terms
    }:
        payload_filters["should"].append(
            {
                "field": "primary_purpose",
                "values": ["Prevention"],
            }
        )
    payload_filters["should"].append(
        {
            "field": "study_type",
            "values": ["Interventional"],
        }
    )
    negative_condition_terms = [
        term
        for term in patient_negative_terms
        if term in {"diabetes", "congestive heart failure"}
    ]
    if negative_condition_terms:
        payload_filters["must_not"].append(
            {
                "field": "condition_terms",
                "values": negative_condition_terms,
            }
        )
    if age_years is not None:
        payload_filters["age_years"] = int(age_years)
    if gender:
        payload_filters["gender"] = gender

    return {
        "query_text": query_text,
        "focus_terms": focus_terms,
        "patient_positive_terms": patient_positive_terms,
        "patient_negative_terms": patient_negative_terms,
        "trial_condition_terms": trial_condition_terms,
        "trial_intervention_terms": trial_intervention_terms,
        "trial_intent_terms": trial_intent_terms,
        "positive_profile_terms": positive_profile_terms,
        "negative_profile_terms": negative_profile_terms,
        "age_years": age_years,
        "gender": gender,
        "payload_filters": payload_filters,
        "derivation_notes": derivation_notes,
    }


def build_protocol_trial_query_text(
    *,
    raw_text: Any = "",
    case_summary: Any = None,
    problem_list: Any = None,
    known_facts: Any = None,
) -> str:
    return str(
        build_protocol_trial_query_profile(
            raw_text=raw_text,
            case_summary=case_summary,
            problem_list=problem_list,
            known_facts=known_facts,
        ).get("query_text")
        or ""
    )


def _normalize_backend_name(value: str | None) -> str:
    normalized = str(value or "hybrid").strip().lower()
    if not normalized:
        normalized = "hybrid"
    return {
        "keyword": "bm25",
        "auto": "hybrid",
        "medcpt": "vector",
    }.get(normalized, normalized)


def _normalize_vector_store_name(value: str | None) -> str:
    normalized = str(value or "faiss").strip().lower()
    if not normalized:
        normalized = "faiss"
    return {
        "default": "auto",
        "automatic": "auto",
        "local": "faiss",
        "medcpt": "faiss",
        "qdrant_local": "qdrant",
    }.get(normalized, normalized)


def _status_priority(status: str, enrollment_open: bool) -> int:
    normalized_status = str(status or "").strip().lower()
    if normalized_status == "trial_matched":
        return 3 if enrollment_open else 2
    if normalized_status == "manual_review":
        return 1
    return 0


def _map_protocol_trial_status(record: Mapping[str, Any]) -> dict[str, Any]:
    overall_status = _normalize_whitespace(record.get("overall_status"))
    if overall_status in _OPEN_TRIAL_STATUSES:
        return {
            "status": "trial_matched",
            "enrollment_open": True,
            "status_reason": (
                "The trial is actively open or preparing to open enrollment, so it can be surfaced as a direct trial candidate."
            ),
            "actions": [
                "Review detailed inclusion and exclusion criteria against the current case.",
                "Confirm site-level enrollment availability before surfacing this as a live trial option.",
            ],
        }
    if overall_status in _EVIDENCE_TRIAL_STATUSES:
        return {
            "status": "trial_matched",
            "enrollment_open": False,
            "status_reason": (
                "The trial is not currently open for enrollment, but it remains useful as protocol or evidence support."
            ),
            "actions": [
                "Use the trial as protocol or evidence support instead of assuming active enrollment.",
                "If a current trial option is needed, search for an active successor or related study before surfacing.",
            ],
        }
    if overall_status in _ABANDONED_TRIAL_STATUSES:
        return {
            "status": "abandoned",
            "enrollment_open": False,
            "status_reason": (
                "The trial lifecycle indicates that it is halted or unavailable, so it should not be treated as a current treatment-trial option."
            ),
            "actions": [
                "Do not surface this trial as a direct option for the current patient.",
                "Keep it only as historical context if the protocol remains clinically informative.",
            ],
        }
    if overall_status == "Unknown status":
        return {
            "status": "manual_review",
            "enrollment_open": False,
            "status_reason": (
                "The trial lifecycle is unclear, so manual review is required before using it in protocol recommendations."
            ),
            "actions": [
                "Review the full XML and the current ClinicalTrials.gov record manually before use.",
                "Do not promote this trial to a direct recommendation until the lifecycle status is confirmed.",
            ],
        }
    return {
        "status": "manual_review",
        "enrollment_open": False,
        "status_reason": (
            "The trial status does not map cleanly to a stable MedAI recommendation state, so it should remain under manual review."
        ),
        "actions": [
            "Review the trial lifecycle manually before using it in MedAI.",
            "Document any case-specific rationale before surfacing this trial to clinicians.",
        ],
    }


def _default_qdrant_runtime_config() -> dict[str, Any]:
    return resolve_trial_qdrant_runtime_config()


def _normalize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    raw_scores = {
        str(row.get("document_id") or row.get("chunk_id") or "").strip(): float(row.get("score") or 0.0)
        for row in list(rows or [])
        if str(row.get("document_id") or row.get("chunk_id") or "").strip()
    }
    if not raw_scores:
        return {}

    values = list(raw_scores.values())
    max_score = max(values)
    min_score = min(values)
    if max_score == min_score:
        if max_score <= 0.0:
            return {document_id: 0.0 for document_id in raw_scores}
        return {document_id: 1.0 for document_id in raw_scores}

    denominator = max_score - min_score
    return {
        document_id: (score - min_score) / denominator
        for document_id, score in raw_scores.items()
    }


def _coerce_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_match_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _collect_trial_match_text(
    record: Mapping[str, Any],
    chunk_rows: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for item in (
        record.get("display_title"),
        record.get("brief_title"),
        record.get("official_title"),
        record.get("brief_summary"),
        record.get("detailed_description"),
        record.get("eligibility_text"),
        record.get("primary_purpose"),
        record.get("study_type"),
        record.get("phase"),
    ):
        normalized = _normalize_whitespace(item)
        if normalized:
            parts.append(normalized)

    for list_field in (
        "conditions",
        "condition_terms",
        "interventions",
        "intervention_terms",
        "keywords",
        "condition_mesh_terms",
        "intervention_mesh_terms",
    ):
        for item in _coerce_text_list(record.get(list_field)):
            parts.append(item)

    for row in list(chunk_rows or [])[:5]:
        normalized_text = _normalize_whitespace(row.get("text"))
        if normalized_text:
            parts.append(normalized_text)
    return _normalize_match_text(" ".join(parts))


def _match_query_terms(
    terms: list[str],
    *,
    record: Mapping[str, Any],
    candidate_text: str,
    field_names: tuple[str, ...],
) -> list[str]:
    normalized_pools = {
        field_name: {
            _normalize_match_text(item)
            for item in _coerce_text_list(record.get(field_name))
            if _normalize_match_text(item)
        }
        for field_name in field_names
    }
    matched_terms: list[str] = []
    for term in list(terms or []):
        normalized_term = _normalize_match_text(term)
        if not normalized_term:
            continue
        if any(
            normalized_term in item or item in normalized_term
            for pool in normalized_pools.values()
            for item in pool
        ):
            matched_terms.append(str(term))
            continue
        if normalized_term in candidate_text:
            matched_terms.append(str(term))
    return matched_terms


def _match_intent_terms(
    terms: list[str],
    *,
    record: Mapping[str, Any],
    candidate_text: str,
) -> list[str]:
    normalized_primary_purpose = _normalize_match_text(record.get("primary_purpose"))
    matched_terms: list[str] = []
    for term in list(terms or []):
        normalized_term = _normalize_match_text(term)
        if not normalized_term:
            continue
        if normalized_term in candidate_text:
            matched_terms.append(str(term))
            continue
        if normalized_primary_purpose == "prevention" and normalized_term in {"stroke prevention", "secondary prevention"}:
            matched_terms.append(str(term))
    return matched_terms


def _must_not_conflicts(
    query_profile: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    candidate_text: str,
) -> list[str]:
    payload_filters = dict(query_profile.get("payload_filters") or {})
    conflict_terms: list[str] = []
    for raw_entry in list(payload_filters.get("must_not") or []):
        if not isinstance(raw_entry, Mapping):
            continue
        if str(raw_entry.get("field") or "").strip() != "condition_terms":
            continue
        for term in _coerce_text_list(raw_entry.get("values")):
            normalized_term = _normalize_match_text(term)
            if not normalized_term:
                continue
            if normalized_term in candidate_text:
                _append_unique(conflict_terms, term)
    return conflict_terms


def _trial_level_rerank_payload(
    *,
    record: Mapping[str, Any],
    chunk_rows: list[dict[str, Any]],
    query_profile: Mapping[str, Any] | None,
    stage: str,
) -> dict[str, Any]:
    top_chunks = list(chunk_rows or [])[: len(_TRIAL_RERANK_CHUNK_SUPPORT_WEIGHTS)]
    best_chunk_score = float(top_chunks[0].get("score") or 0.0) if top_chunks else 0.0
    chunk_support_score = 0.0
    for row, weight in zip(top_chunks, _TRIAL_RERANK_CHUNK_SUPPORT_WEIGHTS, strict=False):
        chunk_support_score += float(row.get("score") or 0.0) * float(weight)

    priority_chunk_types = {
        str(row.get("chunk_type") or "").strip()
        for row in top_chunks
        if str(row.get("chunk_type") or "").strip() in _TRIAL_RERANK_PRIORITY_CHUNK_TYPES
    }
    coverage_bonus = 0.0
    if stage == "fine":
        coverage_bonus += 0.05 * len(priority_chunk_types)
        if "eligibility_inclusion" in priority_chunk_types and "overview" in priority_chunk_types:
            coverage_bonus += 0.05
        if "arms_interventions" in priority_chunk_types and (
            "eligibility_inclusion" in priority_chunk_types or "eligibility_exclusion" in priority_chunk_types
        ):
            coverage_bonus += 0.04
    else:
        coverage_bonus += 0.03 * len(priority_chunk_types)

    final_score = float(chunk_support_score) + float(coverage_bonus)
    matched_condition_terms: list[str] = []
    matched_intervention_terms: list[str] = []
    matched_intent_terms: list[str] = []
    must_not_conflicts: list[str] = []
    eligibility_conflicts: list[str] = []
    eligibility_signals: list[str] = []
    status_meta = _map_protocol_trial_status(record)
    status_bonus = 0.0
    query_signal_bonus = 0.0
    eligibility_bonus = 0.0
    eligibility_penalty = 0.0

    if stage == "fine" and isinstance(query_profile, Mapping):
        candidate_text = _collect_trial_match_text(record, top_chunks)
        query_condition_terms = list(query_profile.get("trial_condition_terms") or [])
        query_intervention_terms = list(query_profile.get("trial_intervention_terms") or [])
        matched_condition_terms = _match_query_terms(
            query_condition_terms,
            record=record,
            candidate_text=candidate_text,
            field_names=("conditions", "condition_terms", "keywords", "condition_mesh_terms"),
        )
        matched_intervention_terms = _match_query_terms(
            query_intervention_terms,
            record=record,
            candidate_text=candidate_text,
            field_names=("interventions", "intervention_terms", "intervention_mesh_terms"),
        )
        matched_intent_terms = _match_intent_terms(
            list(query_profile.get("trial_intent_terms") or []),
            record=record,
            candidate_text=candidate_text,
        )
        query_signal_bonus += min(len(matched_condition_terms), 3) * 0.08
        query_signal_bonus += min(len(matched_intervention_terms), 3) * 0.06
        query_signal_bonus += min(len(matched_intent_terms), 2) * 0.07
        if query_condition_terms and not matched_condition_terms:
            eligibility_penalty += 0.55
            eligibility_conflicts.append(
                "No trial condition term matched the case disease focus."
            )
        if query_intervention_terms and not matched_intervention_terms:
            eligibility_penalty += 0.45
            eligibility_conflicts.append(
                "No trial intervention term matched the case treatment focus."
            )
        if (
            query_condition_terms
            and query_intervention_terms
            and (not matched_condition_terms or not matched_intervention_terms)
        ):
            eligibility_penalty += 0.2
            eligibility_conflicts.append(
                "Disease and intervention anchors were not both satisfied for this trial."
            )

        must_not_conflicts = _must_not_conflicts(
            query_profile,
            record=record,
            candidate_text=candidate_text,
        )
        if must_not_conflicts:
            eligibility_penalty += min(len(must_not_conflicts), 2) * 0.45
            eligibility_conflicts.extend(
                f"Conflicts with must_not condition term: {term}."
                for term in must_not_conflicts
            )

        age_years = _coerce_optional_float(query_profile.get("age_years"))
        age_floor = _coerce_optional_float(record.get("age_floor_years"))
        age_ceiling = _coerce_optional_float(record.get("age_ceiling_years"))
        if age_years is not None:
            if age_floor is not None and age_years < age_floor:
                eligibility_penalty += 0.75
                eligibility_conflicts.append(
                    f"Case age {int(age_years)} is below trial minimum age {int(age_floor)}."
                )
            elif age_floor is not None:
                eligibility_bonus += 0.06
                eligibility_signals.append(f"Case age clears minimum age {int(age_floor)}.")
            if age_ceiling is not None and age_years > age_ceiling:
                eligibility_penalty += 0.75
                eligibility_conflicts.append(
                    f"Case age {int(age_years)} is above trial maximum age {int(age_ceiling)}."
                )
            elif age_ceiling is not None:
                eligibility_bonus += 0.06
                eligibility_signals.append(f"Case age fits within maximum age {int(age_ceiling)}.")

        patient_gender = _normalize_whitespace(query_profile.get("gender"))
        trial_gender = _normalize_whitespace(record.get("gender"))
        if patient_gender:
            if trial_gender and trial_gender not in {"All", patient_gender}:
                eligibility_penalty += 0.75
                eligibility_conflicts.append(
                    f"Trial gender restriction is {trial_gender}, but the case is {patient_gender}."
                )
            elif trial_gender in {"All", patient_gender}:
                eligibility_bonus += 0.08
                eligibility_signals.append(
                    "Trial gender eligibility is compatible with the case."
                )

        normalized_status = str(status_meta.get("status") or "").strip().lower()
        if normalized_status == "trial_matched":
            status_bonus = 0.18 if bool(status_meta.get("enrollment_open")) else 0.08
        elif normalized_status == "abandoned":
            status_bonus = -0.45

    final_score += float(query_signal_bonus)
    final_score += float(eligibility_bonus)
    final_score -= float(eligibility_penalty)
    final_score += float(status_bonus)
    return {
        "score": float(final_score),
        "best_chunk_score": float(best_chunk_score),
        "chunk_support_score": float(chunk_support_score),
        "coverage_bonus": float(coverage_bonus),
        "query_signal_bonus": float(query_signal_bonus),
        "eligibility_bonus": float(eligibility_bonus),
        "eligibility_penalty": float(eligibility_penalty),
        "status_bonus": float(status_bonus),
        "coverage_chunk_types": sorted(priority_chunk_types),
        "matched_condition_terms": matched_condition_terms,
        "matched_intervention_terms": matched_intervention_terms,
        "matched_intent_terms": matched_intent_terms,
        "must_not_conflicts": must_not_conflicts,
        "eligibility_conflicts": eligibility_conflicts,
        "eligibility_signals": eligibility_signals,
        "score_breakdown": {
            "stage": stage,
            "best_chunk_score": float(best_chunk_score),
            "chunk_support_score": float(chunk_support_score),
            "coverage_bonus": float(coverage_bonus),
            "query_signal_bonus": float(query_signal_bonus),
            "eligibility_bonus": float(eligibility_bonus),
            "eligibility_penalty": float(eligibility_penalty),
            "status_bonus": float(status_bonus),
            "final_score": float(final_score),
        },
    }


class TrialChunkKeywordRetriever:
    FIELD_WEIGHTS: dict[str, float] = {
        "title": 2.8,
        "retrieval_text": 4.0,
        "purpose": 0.4,
        "eligibility": 0.8,
        "summary": 1.2,
    }

    def __init__(self, catalog: TrialChunkCatalog) -> None:
        self.catalog = catalog
        self._search_rows: list[dict[str, Any]] = []
        self._row_index_by_chunk_id: dict[str, int] = {}
        bm25_documents: list[dict[str, str]] = []
        for document in catalog.documents():
            row_index = len(self._search_rows)
            self._search_rows.append(
                {
                    "document": document,
                    "title": document.title,
                    "retrieval_text": document.retrieval_text,
                    "purpose": document.purpose,
                    "eligibility": document.eligibility,
                    "summary": document.summary,
                }
            )
            self._row_index_by_chunk_id[str(document.chunk_id)] = row_index
            bm25_documents.append(
                {
                    "title": document.title,
                    "retrieval_text": document.retrieval_text,
                    "purpose": document.purpose,
                    "eligibility": document.eligibility,
                    "summary": document.summary,
                }
            )
        self._bm25 = FieldedBM25Index(bm25_documents, field_weights=self.FIELD_WEIGHTS)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        candidate_pmids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        raw_candidate_ids = candidate_ids if candidate_ids is not None else candidate_pmids
        normalized_candidate_ids = {
            str(item).strip()
            for item in list(raw_candidate_ids or [])
            if str(item).strip()
        }
        if not str(query or "").strip():
            return []
        if raw_candidate_ids is not None and not normalized_candidate_ids:
            return []

        candidate_row_indexes = None
        if raw_candidate_ids is not None:
            candidate_row_indexes = [
                self._row_index_by_chunk_id[chunk_id]
                for chunk_id in sorted(normalized_candidate_ids)
                if chunk_id in self._row_index_by_chunk_id
            ]
            if not candidate_row_indexes:
                return []

        score_by_index = self._bm25.score_subset(
            query,
            document_indexes=candidate_row_indexes,
        )

        scored_rows: list[tuple[float, TrialChunkDocument]] = []
        for row_index, score in score_by_index.items():
            if score <= 0.0:
                continue
            if row_index >= len(self._search_rows):
                continue
            scored_rows.append((float(score), self._search_rows[row_index]["document"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].chunk_id))
        return [self._serialize(document, score) for score, document in scored_rows[: max(int(top_k), 1)]]

    @staticmethod
    def _serialize(document: TrialChunkDocument, score: float) -> dict[str, Any]:
        payload = document.to_brief()
        payload["score"] = float(score)
        return payload


class TrialChunkRetrievalTool:
    def __init__(
        self,
        catalog: TrialChunkCatalog,
        *,
        vector_retriever: Any | None = None,
        backend: str = "hybrid",
    ) -> None:
        self.catalog = catalog
        self.vector_retriever = vector_retriever
        self.keyword_retriever = TrialChunkKeywordRetriever(catalog)
        self._retriever = create_structured_retriever(
            catalog,
            bm25_retriever=self.keyword_retriever,
            vector_retriever=vector_retriever,
            query_builder=build_protocol_trial_query_text,
            default_backend=backend,
            id_field="chunk_id",
        )
        self.default_backend = self._retriever.resolve_backend(backend)

    @property
    def available_backends(self) -> tuple[str, ...]:
        return self._retriever.available_backends

    def resolve_backend(self, backend: str | None = None) -> str:
        return self._retriever.resolve_backend(backend or self.default_backend)

    def _retrieve_query_bundle(
        self,
        query_text: str,
        *,
        top_k: int,
        candidate_chunk_ids: set[str] | None,
        backend: str,
        retriever_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=candidate_chunk_ids,
                backend=backend,
                include_scores=True,
                retriever_options=retriever_options,
            )
        except Exception:
            if backend == "bm25":
                raise
            return self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=candidate_chunk_ids,
                backend="bm25",
                include_scores=True,
                retriever_options=retriever_options,
            )

    def _channel_rows(
        self,
        query_text: str,
        *,
        top_k: int,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None,
        backend: str,
        retriever_options: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_chunk_ids = self._resolve_candidate_chunk_ids(candidate_nct_ids)
        if backend == "vector" and self.vector_retriever is None:
            return []
        try:
            bundle = self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=candidate_chunk_ids,
                backend=backend,
                include_scores=True,
                retriever_options=retriever_options,
            )
        except Exception:
            return []
        return [dict(row) for row in list(bundle.get("candidate_ranking") or [])]

    def _build_query_profile(self, structured_case: Mapping[str, Any]) -> dict[str, Any]:
        return build_protocol_trial_query_profile(structured_case)

    @staticmethod
    def _vector_retriever_options(query_profile: Mapping[str, Any]) -> dict[str, Any]:
        payload_filters = dict(query_profile.get("payload_filters") or {})
        if not payload_filters:
            return {}
        return {"payload_filters": payload_filters}

    def _enrich_protocol_candidates(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for row in list(rows or []):
            nct_id = str(row.get("nct_id") or "").strip()
            if not nct_id:
                continue
            record = self.catalog.get_record(nct_id) or {"nct_id": nct_id}
            status_meta = _map_protocol_trial_status(record)
            display_title = _normalize_whitespace(
                record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
            )
            enriched.append(
                {
                    **dict(row),
                    "nct_id": nct_id,
                    "title": display_title,
                    "name": display_title,
                    "display_title": display_title,
                    "brief_summary": _normalize_whitespace(record.get("brief_summary")),
                    "status": str(status_meta.get("status") or ""),
                    "enrollment_open": bool(status_meta.get("enrollment_open")),
                    "status_reason": _normalize_whitespace(status_meta.get("status_reason")),
                    "actions": list(status_meta.get("actions") or []),
                    "overall_status": _normalize_whitespace(record.get("overall_status") or row.get("overall_status")),
                    "study_type": _normalize_whitespace(record.get("study_type") or row.get("study_type")),
                    "phase": _normalize_whitespace(record.get("phase") or row.get("phase")),
                    "primary_purpose": _normalize_whitespace(record.get("primary_purpose")),
                    "conditions": list(record.get("conditions") or row.get("conditions") or []),
                    "interventions": list(record.get("interventions") or row.get("interventions") or []),
                    "gender": _normalize_whitespace(record.get("gender") or row.get("gender")),
                    "age_floor_years": record.get("age_floor_years") if record.get("age_floor_years") is not None else row.get("age_floor_years"),
                    "age_ceiling_years": record.get("age_ceiling_years") if record.get("age_ceiling_years") is not None else row.get("age_ceiling_years"),
                    "source_url": _normalize_whitespace(record.get("source_url") or row.get("source_url")),
                }
            )
        enriched.sort(
            key=lambda row: (
                -float(row.get("score") or 0.0),
                -_status_priority(str(row.get("status") or ""), bool(row.get("enrollment_open"))),
                -len(list(row.get("matched_chunks") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("nct_id") or ""),
            )
        )
        return enriched

    @tool(
        name="trial_chunk_retriever",
        description=(
            "Retrieve protocol-stage trial chunks from the local XML-derived trial vector KB. "
            "Use this when the caller wants chunk evidence instead of department payload retrieval."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
            "candidate_nct_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_chunks(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 20,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_chunks_from_structured_case(
            structured_case,
            top_k=top_k,
            candidate_nct_ids=candidate_nct_ids,
            backend=backend,
        )

    @tool(
        name="trial_vector_candidate_retriever",
        description=(
            "Aggregate chunk-level retrieval over the XML-derived trial KB into trial-level candidates "
            "with matched_chunks and best_evidence_text."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
            "chunk_top_k": "int",
            "candidate_nct_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_trials(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        chunk_top_k: int = 40,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_trials_from_structured_case(
            structured_case,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            candidate_nct_ids=candidate_nct_ids,
            backend=backend,
        )

    def retrieve_chunks_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 20,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        query_profile = self._build_query_profile(case_payload)
        query_text = str(query_profile.get("query_text") or "")
        resolved_backend = self.resolve_backend(backend)
        candidate_chunk_ids = self._resolve_candidate_chunk_ids(candidate_nct_ids)
        if not query_text:
            return {
                "query_text": "",
                "query_profile": query_profile,
                "backend_used": resolved_backend,
                "available_backends": list(self.available_backends),
                "hits": [],
            }

        payload = self._retrieve_query_bundle(
            query_text,
            top_k=max(int(top_k), 1),
            candidate_chunk_ids=candidate_chunk_ids,
            backend=resolved_backend,
            retriever_options=self._vector_retriever_options(query_profile),
        )
        hits = self._hydrate_chunk_rows(list(payload.get("candidate_ranking") or []))
        return {
            "query_text": query_text,
            "query_profile": query_profile,
            "backend_used": str(payload.get("backend_used") or resolved_backend),
            "available_backends": list(payload.get("available_backends") or self.available_backends),
            "hits": hits,
        }

    def retrieve_trials_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        chunk_top_k: int = 40,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        query_profile = self._build_query_profile(case_payload)
        query_text = str(query_profile.get("query_text") or "")
        resolved_backend = self.resolve_backend(backend)

        if not query_text:
            return {
                "query_text": "",
                "query_profile": query_profile,
                "backend_used": resolved_backend,
                "available_backends": list(self.available_backends),
                "bm25_chunks": [],
                "vector_chunks": [],
                "candidate_ranking": [],
            }

        bm25_rows = self._channel_rows(
            query_text,
            top_k=max(int(chunk_top_k), 1),
            candidate_nct_ids=candidate_nct_ids,
            backend="bm25",
        )

        vector_rows: list[dict[str, Any]] = []
        if self.vector_retriever is not None and resolved_backend != "bm25":
            vector_rows = self._channel_rows(
                query_text,
                top_k=max(int(chunk_top_k), 1),
                candidate_nct_ids=candidate_nct_ids,
                backend="vector",
                retriever_options=self._vector_retriever_options(query_profile),
            )

        candidate_ranking = self._aggregate_trials(
            bm25_rows=bm25_rows,
            vector_rows=vector_rows,
            limit=max(int(top_k), 1),
            query_profile=query_profile,
            stage="fine",
        )

        return {
            "query_text": query_text,
            "query_profile": query_profile,
            "backend_used": "bm25" if not vector_rows else "hybrid",
            "available_backends": list(self.available_backends),
            "bm25_chunks": self._hydrate_chunk_rows(bm25_rows[:5]),
            "vector_chunks": self._hydrate_chunk_rows(vector_rows[:5]),
            "candidate_ranking": candidate_ranking,
        }

    def retrieve_coarse_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        del department_tags
        case_payload = dict(structured_case or {})
        query_profile = self._build_query_profile(case_payload)
        query_text = str(query_profile.get("query_text") or "")
        resolved_backend = self.resolve_backend(backend)
        if not query_text:
            return {
                "query_text": "",
                "query_profile": query_profile,
                "backend_used": "bm25" if resolved_backend == "bm25" or self.vector_retriever is None else "vector",
                "available_backends": list(self.available_backends),
                "department_tags": list(case_payload.get("department_tags") or []),
                "fallback_to_full_catalog": False,
                "candidate_ranking": [],
                "coarse_candidate_ids": [],
            }
        normalized_candidate_ids = {
            str(item).strip()
            for item in list(candidate_ids or [])
            if str(item).strip()
        } if candidate_ids is not None else None
        coarse_backend = "vector" if self.vector_retriever is not None and resolved_backend != "bm25" else "bm25"
        coarse_payload = self._retrieve_query_bundle(
            query_text,
            top_k=max(int(top_k) * 4, int(top_k), 1),
            candidate_chunk_ids=self._resolve_candidate_chunk_ids(normalized_candidate_ids),
            backend=coarse_backend,
            retriever_options=self._vector_retriever_options(query_profile) if coarse_backend == "vector" else None,
        )
        actual_backend = str(coarse_payload.get("backend_used") or coarse_backend).strip() or coarse_backend
        chunk_rows = [dict(row) for row in list(coarse_payload.get("candidate_ranking") or [])]
        coarse_ranked_rows = self._aggregate_trials(
            bm25_rows=chunk_rows if actual_backend == "bm25" else [],
            vector_rows=chunk_rows if actual_backend != "bm25" else [],
            limit=max(int(top_k), 1),
            query_profile=query_profile,
            stage="coarse",
        )
        enriched_candidates = self._enrich_protocol_candidates(coarse_ranked_rows)
        return {
            "query_text": query_text,
            "query_profile": query_profile,
            "backend_used": actual_backend,
            "available_backends": list(coarse_payload.get("available_backends") or self.available_backends),
            "department_tags": list(case_payload.get("department_tags") or []),
            "fallback_to_full_catalog": False,
            "candidate_ranking": [
                {
                    "nct_id": str(row.get("nct_id") or ""),
                    "title": str(row.get("title") or ""),
                    "score": float(row.get("score") or 0.0),
                    "status": str(row.get("status") or ""),
                    "enrollment_open": bool(row.get("enrollment_open")),
                    "overall_status": str(row.get("overall_status") or ""),
                    "study_type": str(row.get("study_type") or ""),
                    "phase": str(row.get("phase") or ""),
                    "primary_purpose": str(row.get("primary_purpose") or ""),
                    "conditions": list(row.get("conditions") or []),
                    "interventions": list(row.get("interventions") or []),
                }
                for row in enriched_candidates
            ],
            "coarse_candidate_ids": [
                str(row.get("nct_id") or "")
                for row in enriched_candidates
                if str(row.get("nct_id") or "").strip()
            ],
        }

    @tool(
        name="trial_coarse_retriever",
        description=(
            "Coarsely recall local trial candidates from the XML-derived trial KB. "
            "Return a minimal NCT/title bundle for protocol-stage candidate pooling."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "department_tags": "list[str]",
            },
            "top_k": "int",
            "department_tags": "list[str] | None",
            "candidate_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_coarse(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_coarse_from_structured_case(
            structured_case,
            top_k=top_k,
            department_tags=department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

    def retrieve_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        coarse_top_k: int = 30,
        chunk_top_k: int = _DEFAULT_PROTOCOL_CHUNK_TOP_K,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        normalized_department_tags = list(
            department_tags if department_tags is not None else case_payload.get("department_tags") or []
        )
        query_profile = self._build_query_profile(case_payload)
        query_text = str(query_profile.get("query_text") or "")
        resolved_backend = self.resolve_backend(backend)
        coarse_bundle = self.retrieve_coarse_from_structured_case(
            case_payload,
            top_k=coarse_top_k,
            department_tags=normalized_department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )
        coarse_candidate_ids = [
            str(item).strip()
            for item in list(coarse_bundle.get("coarse_candidate_ids") or [])
            if str(item).strip()
        ]
        if not query_text or not coarse_candidate_ids:
            return {
                "query_text": query_text,
                "query_profile": query_profile,
                "backend_used": "bm25" if resolved_backend == "bm25" or self.vector_retriever is None else "hybrid",
                "available_backends": list(self.available_backends),
                "department_tags": normalized_department_tags,
                "fallback_to_full_catalog": False,
                "coarse_candidate_ids": coarse_candidate_ids,
                "coarse_candidate_ranking": list(coarse_bundle.get("candidate_ranking") or []),
                "bm25_top5": [],
                "vector_top5": [],
                "candidate_ranking": [],
            }

        bm25_rows = self._channel_rows(
            query_text,
            top_k=max(int(chunk_top_k), 1),
            candidate_nct_ids=coarse_candidate_ids,
            backend="bm25",
        )
        vector_rows: list[dict[str, Any]] = []
        if self.vector_retriever is not None and resolved_backend != "bm25":
            vector_rows = self._channel_rows(
                query_text,
                top_k=max(int(chunk_top_k), 1),
                candidate_nct_ids=coarse_candidate_ids,
                backend="vector",
                retriever_options=self._vector_retriever_options(query_profile),
            )

        bm25_top5 = self._enrich_protocol_candidates(
            self._aggregate_trials(
                bm25_rows=bm25_rows,
                vector_rows=[],
                limit=min(max(int(top_k), 1), 5),
                query_profile=query_profile,
                stage="fine",
            )
        )
        vector_top5 = self._enrich_protocol_candidates(
            self._aggregate_trials(
                bm25_rows=[],
                vector_rows=vector_rows,
                limit=min(max(int(top_k), 1), 5),
                query_profile=query_profile,
                stage="fine",
            )
        )
        candidate_ranking = self._enrich_protocol_candidates(
            self._aggregate_trials(
                bm25_rows=bm25_rows,
                vector_rows=vector_rows,
                limit=max(int(top_k), 1),
                query_profile=query_profile,
                stage="fine",
            )
        )

        return {
            "query_text": query_text,
            "query_profile": query_profile,
            "backend_used": "bm25" if not vector_rows else "hybrid",
            "available_backends": list(self.available_backends),
            "department_tags": normalized_department_tags,
            "fallback_to_full_catalog": False,
            "coarse_candidate_ids": coarse_candidate_ids,
            "coarse_candidate_ranking": list(coarse_bundle.get("candidate_ranking") or []),
            "bm25_top5": bm25_top5,
            "vector_top5": vector_top5,
            "candidate_ranking": candidate_ranking,
        }

    @tool(
        name="trial_candidate_retriever",
        description=(
            "Run the protocol-stage two-stage trial retrieval over the XML-derived trial KB. "
            "Return coarse candidate ids, bm25 top5, vector top5, and the merged top candidate ranking."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "department_tags": "list[str]",
            },
            "top_k": "int",
            "coarse_top_k": "int",
            "chunk_top_k": "int",
            "department_tags": "list[str] | None",
            "candidate_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_candidates(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        coarse_top_k: int = 30,
        chunk_top_k: int = _DEFAULT_PROTOCOL_CHUNK_TOP_K,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_from_structured_case(
            structured_case,
            top_k=top_k,
            coarse_top_k=coarse_top_k,
            chunk_top_k=chunk_top_k,
            department_tags=department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

    def _build_query_text(self, structured_case: Mapping[str, Any]) -> str:
        return str(self._build_query_profile(structured_case).get("query_text") or "")

    def _resolve_candidate_chunk_ids(
        self,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None,
    ) -> set[str] | None:
        if candidate_nct_ids is None:
            return None
        normalized_nct_ids = {
            str(item).strip()
            for item in list(candidate_nct_ids or [])
            if str(item).strip()
        }
        candidate_chunk_ids: set[str] = set()
        for nct_id in normalized_nct_ids:
            candidate_chunk_ids.update(self.catalog.chunk_index_by_trial.get(nct_id, set()))
        return candidate_chunk_ids

    def _hydrate_chunk_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for row in list(rows or []):
            chunk_id = str(row.get("chunk_id") or row.get("document_id") or "").strip()
            if not chunk_id:
                continue
            document = self.catalog.get(chunk_id)
            if document is None:
                continue
            candidate = {
                "chunk_id": chunk_id,
                "nct_id": document.nct_id,
                "title": document.title,
                "chunk_type": document.chunk_type,
                "text": document.text,
                "source_fields": list(document.source_fields),
                "rank_weight": float(document.rank_weight),
                "score": float(row.get("score") or 0.0),
                "match_sources": sorted(set(list(row.get("match_sources") or []))),
            }
            hydrated.append(candidate)
        return hydrated

    def _aggregate_trials(
        self,
        *,
        bm25_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        limit: int,
        query_profile: Mapping[str, Any] | None = None,
        stage: str = "coarse",
    ) -> list[dict[str, Any]]:
        normalized_bm25_scores = _normalize_scores(bm25_rows)
        normalized_vector_scores = _normalize_scores(vector_rows)
        grouped_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        merged: dict[str, dict[str, Any]] = {}

        for row in list(bm25_rows or []):
            chunk_id = str(row.get("chunk_id") or row.get("document_id") or "").strip()
            if not chunk_id:
                continue
            merged.setdefault(chunk_id, {})
            merged[chunk_id]["bm25_row"] = dict(row)

        for row in list(vector_rows or []):
            chunk_id = str(row.get("chunk_id") or row.get("document_id") or "").strip()
            if not chunk_id:
                continue
            merged.setdefault(chunk_id, {})
            merged[chunk_id]["vector_row"] = dict(row)

        for chunk_id, raw_bundle in merged.items():
            document = self.catalog.get(chunk_id)
            if document is None:
                continue
            combined_score = float(normalized_bm25_scores.get(chunk_id) or 0.0) + float(
                normalized_vector_scores.get(chunk_id) or 0.0
            )
            weighted_score = combined_score * float(document.rank_weight)
            chunk_payload = {
                "chunk_id": chunk_id,
                "chunk_type": document.chunk_type,
                "score": float(weighted_score),
                "bm25_score": float((raw_bundle.get("bm25_row") or {}).get("score") or 0.0),
                "vector_score": float((raw_bundle.get("vector_row") or {}).get("score") or 0.0),
                "match_sources": sorted(
                    {
                        *list((raw_bundle.get("bm25_row") or {}).get("match_sources") or []),
                        *list((raw_bundle.get("vector_row") or {}).get("match_sources") or []),
                    }
                ),
                "source_fields": list(document.source_fields),
                "text": document.text,
            }
            grouped_chunks[document.nct_id].append(chunk_payload)

        ranked_trials: list[dict[str, Any]] = []
        for nct_id, chunk_rows in grouped_chunks.items():
            record = self.catalog.get_record(nct_id) or {"nct_id": nct_id}
            status_meta = _map_protocol_trial_status(record)
            chunk_rows.sort(
                key=lambda row: (
                    -float(row.get("score") or 0.0),
                    str(row.get("chunk_type") or ""),
                    str(row.get("chunk_id") or ""),
                )
            )
            best_chunk = chunk_rows[0]
            matched_fields = sorted(
                {
                    field_name
                    for row in chunk_rows[:3]
                    for field_name in list(row.get("source_fields") or [])
                    if str(field_name).strip()
                }
            )
            rerank_payload = _trial_level_rerank_payload(
                record=record,
                chunk_rows=chunk_rows,
                query_profile=query_profile,
                stage=stage,
            )
            ranked_trials.append(
                {
                    "nct_id": nct_id,
                    "display_title": _normalize_whitespace(
                        record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
                    ),
                    "title": _normalize_whitespace(
                        record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
                    ),
                    "score": float(rerank_payload.get("score") or 0.0),
                    "best_chunk_score": float(rerank_payload.get("best_chunk_score") or 0.0),
                    "chunk_support_score": float(rerank_payload.get("chunk_support_score") or 0.0),
                    "coverage_bonus": float(rerank_payload.get("coverage_bonus") or 0.0),
                    "query_signal_bonus": float(rerank_payload.get("query_signal_bonus") or 0.0),
                    "eligibility_bonus": float(rerank_payload.get("eligibility_bonus") or 0.0),
                    "eligibility_penalty": float(rerank_payload.get("eligibility_penalty") or 0.0),
                    "status_bonus": float(rerank_payload.get("status_bonus") or 0.0),
                    "coverage_chunk_types": list(rerank_payload.get("coverage_chunk_types") or []),
                    "matched_condition_terms": list(rerank_payload.get("matched_condition_terms") or []),
                    "matched_intervention_terms": list(rerank_payload.get("matched_intervention_terms") or []),
                    "matched_intent_terms": list(rerank_payload.get("matched_intent_terms") or []),
                    "must_not_conflicts": list(rerank_payload.get("must_not_conflicts") or []),
                    "eligibility_conflicts": list(rerank_payload.get("eligibility_conflicts") or []),
                    "eligibility_signals": list(rerank_payload.get("eligibility_signals") or []),
                    "score_breakdown": dict(rerank_payload.get("score_breakdown") or {}),
                    "matched_chunks": [
                        {
                            "chunk_id": str(row.get("chunk_id") or ""),
                            "chunk_type": str(row.get("chunk_type") or ""),
                            "score": float(row.get("score") or 0.0),
                            "match_sources": list(row.get("match_sources") or []),
                            "source_fields": list(row.get("source_fields") or []),
                        }
                        for row in chunk_rows[:3]
                    ],
                    "best_evidence_text": str(best_chunk.get("text") or ""),
                    "matched_fields": matched_fields,
                    "status": str(status_meta.get("status") or ""),
                    "enrollment_open": bool(status_meta.get("enrollment_open")),
                    "overall_status": _normalize_whitespace(record.get("overall_status")),
                    "phase": _normalize_whitespace(record.get("phase")),
                    "study_type": _normalize_whitespace(record.get("study_type")),
                    "conditions": list(record.get("conditions") or []),
                    "interventions": list(record.get("interventions") or []),
                    "source_url": _normalize_whitespace(record.get("source_url")),
                }
            )

        ranked_trials.sort(
            key=lambda row: (
                -float(row.get("score") or 0.0),
                -_status_priority(str(row.get("status") or ""), bool(row.get("enrollment_open"))),
                -len(list(row.get("matched_chunks") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("nct_id") or ""),
            )
        )
        return ranked_trials[: max(int(limit), 1)]


def create_trial_chunk_retrieval_tool(
    *,
    output_root: str | Path | None = None,
    backend: str = "hybrid",
    vector_retriever: Any | None = None,
    vector_store: str = "faiss",
    qdrant_collection_name: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_path: str | Path | None = None,
    qdrant_client: Any | None = None,
) -> TrialChunkRetrievalTool:
    resolved_root = resolve_trial_vector_output_root(output_root)
    normalized_backend = _normalize_backend_name(backend)
    normalized_vector_store = _normalize_vector_store_name(vector_store)
    qdrant_defaults = _default_qdrant_runtime_config()
    if normalized_vector_store == "auto":
        normalized_vector_store = "qdrant" if qdrant_defaults["enabled"] else "faiss"
    resolved_qdrant_collection_name = str(
        qdrant_collection_name
        or qdrant_defaults.get("collection_name")
        or DEFAULT_QDRANT_COLLECTION_NAME
    )
    resolved_qdrant_url = qdrant_url or qdrant_defaults.get("url")
    resolved_qdrant_api_key = qdrant_api_key or qdrant_defaults.get("api_key")
    resolved_qdrant_path = qdrant_path or qdrant_defaults.get("path")
    cache_key = (
        str(resolved_root),
        normalized_backend,
        normalized_vector_store,
        str(resolved_qdrant_collection_name or ""),
        str(resolved_qdrant_url or ""),
        str(resolved_qdrant_path or ""),
    )
    should_cache = vector_retriever is None and qdrant_client is None and not resolved_qdrant_api_key
    if should_cache:
        with _CACHE_LOCK:
            cached = _TRIAL_CHUNK_RETRIEVER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    catalog = TrialChunkCatalog.from_output_root(resolved_root)
    resolved_vector_retriever = vector_retriever
    if resolved_vector_retriever is None and normalized_backend in {"vector", "hybrid"}:
        if normalized_vector_store == "qdrant":
            try:
                resolved_vector_retriever = create_qdrant_trial_chunk_retriever(
                    catalog,
                    collection_name=resolved_qdrant_collection_name,
                    url=resolved_qdrant_url,
                    api_key=resolved_qdrant_api_key,
                    path=resolved_qdrant_path,
                    client=qdrant_client,
                )
            except Exception:
                try:
                    resolved_vector_retriever = MedCPTRetriever(catalog)
                except Exception:
                    resolved_vector_retriever = None
        else:
            try:
                resolved_vector_retriever = MedCPTRetriever(catalog)
            except Exception:
                resolved_vector_retriever = None

    tool_instance = TrialChunkRetrievalTool(
        catalog,
        vector_retriever=resolved_vector_retriever,
        backend="bm25" if resolved_vector_retriever is None else normalized_backend,
    )
    if should_cache:
        with _CACHE_LOCK:
            _TRIAL_CHUNK_RETRIEVER_CACHE.setdefault(cache_key, tool_instance)
    return tool_instance


__all__ = [
    "TrialChunkCatalog",
    "TrialChunkDocument",
    "TrialChunkKeywordRetriever",
    "TrialChunkRetrievalTool",
    "create_trial_chunk_retrieval_tool",
]
