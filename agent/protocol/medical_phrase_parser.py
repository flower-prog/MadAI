from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_CONCEPT_LEXICON: tuple[dict[str, Any], ...] = (
    {
        "canonical_name": "Atrial Fibrillation",
        "semantic_type": "condition",
        "aliases": ("atrial fibrillation", "afib", "af"),
        "ontology_ids": {"mesh": "D001281"},
    },
    {
        "canonical_name": "Transient Ischemic Attack",
        "semantic_type": "condition",
        "aliases": ("transient ischemic attack", "tia"),
        "ontology_ids": {"mesh": "D002546"},
    },
    {
        "canonical_name": "CHA2DS2-VASc",
        "semantic_type": "calculator",
        "aliases": ("cha2ds2-vasc", "cha2ds2 vasc", "stroke risk"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "HAS-BLED",
        "semantic_type": "calculator",
        "aliases": ("has-bled", "has bled", "bleeding risk"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "ECOG Performance Status",
        "semantic_type": "calculator",
        "aliases": ("ecog", "performance status", "eastern cooperative oncology group"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "QTc Interval",
        "semantic_type": "measurement",
        "aliases": ("qtc", "corrected qt"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "Central Nervous System Metastases",
        "semantic_type": "condition",
        "aliases": ("cns metastases", "brain metastases", "central nervous system metastases"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "Autoimmune Disease",
        "semantic_type": "condition",
        "aliases": ("autoimmune disease", "autoimmune disorder"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "Systemic Therapy",
        "semantic_type": "treatment_or_medication",
        "aliases": ("systemic therapy", "systemic corticosteroids", "systemic immunosuppressive therapy"),
        "ontology_ids": {},
    },
    {
        "canonical_name": "Warfarin",
        "semantic_type": "drug",
        "aliases": ("warfarin", "coumadin"),
        "ontology_ids": {"rxnorm": "11289"},
    },
    {
        "canonical_name": "Anticoagulation",
        "semantic_type": "treatment_or_medication",
        "aliases": ("anticoagulation", "anticoagulant", "doac", "direct oral anticoagulant"),
        "ontology_ids": {},
    },
)

_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|denies|negative for|absence of|free of)\b|无|没有|否认", re.I)
_TEMPORAL_CURRENT_PATTERN = re.compile(r"\b(?:active|current|currently|ongoing|requiring)\b|活动性|当前", re.I)
_TEMPORAL_RECENT_PATTERN = re.compile(r"\bwithin\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b|近期", re.I)
_NUMERIC_CRITERION_PATTERN = re.compile(
    r"\b(?:ecog|qtc|lvef|egfr|crcl|creatinine clearance|platelets?|hemoglobin|age)\b"
    r"[^.;,\n]{0,30}(?:<=|>=|<|>|=|\bto\b|-)\s*\d+",
    re.I,
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _append_source_text(texts: list[dict[str, str]], *, source: str, text: Any) -> None:
    normalized = _normalize_text(text)
    if not normalized:
        return
    key = (source, normalized.casefold())
    if any((item["source"], item["text"].casefold()) == key for item in texts):
        return
    texts.append({"source": source, "text": normalized})


def _case_texts(structured_case: Mapping[str, Any]) -> list[dict[str, str]]:
    texts: list[dict[str, str]] = []
    payload = dict(structured_case or {})
    if isinstance(payload.get("structured_case"), Mapping):
        payload = dict(payload["structured_case"])
    for key in ("raw_text", "raw_request", "case_summary"):
        _append_source_text(texts, source=key, text=payload.get(key))
    for key in ("problem_list", "known_facts"):
        for index, item in enumerate(list(payload.get(key) or []), start=1):
            _append_source_text(texts, source=f"{key}[{index}]", text=item)
    for key, value in _coerce_mapping(payload.get("structured_inputs")).items():
        _append_source_text(texts, source=f"structured_inputs.{key}", text=f"{key}: {value}")
    return texts


def _criterion_texts(eligibility_assessment_bundle: Mapping[str, Any] | None) -> list[dict[str, str]]:
    texts: list[dict[str, str]] = []
    for trial_index, trial in enumerate(list((eligibility_assessment_bundle or {}).get("assessed_trials") or []), start=1):
        if not isinstance(trial, Mapping):
            continue
        nct_id = _normalize_text(trial.get("nct_id")) or f"trial_{trial_index}"
        for criterion_index, criterion in enumerate(list(trial.get("criteria") or []), start=1):
            if not isinstance(criterion, Mapping):
                continue
            raw_text = _normalize_text(criterion.get("raw_text"))
            if raw_text:
                _append_source_text(
                    texts,
                    source=f"eligibility_criteria.{nct_id}[{criterion_index}]",
                    text=raw_text,
                )
    return texts


def _trial_texts(trial_retrieval_bundle: Mapping[str, Any] | None, *, limit: int) -> list[dict[str, str]]:
    texts: list[dict[str, str]] = []
    for index, candidate in enumerate(list((trial_retrieval_bundle or {}).get("candidate_ranking") or [])[: max(int(limit), 1)], start=1):
        if not isinstance(candidate, Mapping):
            continue
        nct_id = _normalize_text(candidate.get("nct_id")) or f"candidate_{index}"
        for key in ("conditions", "interventions"):
            for item in list(candidate.get(key) or []):
                _append_source_text(texts, source=f"trial_candidate.{nct_id}.{key}", text=item)
    return texts


def _attributes(text: str) -> dict[str, Any]:
    temporality = ""
    if _TEMPORAL_RECENT_PATTERN.search(text):
        temporality = "recent"
    elif _TEMPORAL_CURRENT_PATTERN.search(text):
        temporality = "current"
    return {
        "negated": bool(_NEGATION_PATTERN.search(text)),
        "temporality": temporality,
        "contains_numeric_criterion": bool(_NUMERIC_CRITERION_PATTERN.search(text)),
        "required_evidence_type": "clinical_fact",
    }


def _contains_alias(text: str, alias: str) -> bool:
    normalized_alias = str(alias or "").casefold().strip()
    if not normalized_alias:
        return False
    escaped = re.escape(normalized_alias)
    if re.search(r"[a-zA-Z0-9]", normalized_alias):
        pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
        return bool(re.search(pattern, text.casefold()))
    return normalized_alias in text.casefold()


def _required_evidence(concepts: list[dict[str, Any]], text: str) -> list[str]:
    concept_names = {str(item.get("canonical_name") or "").casefold() for item in concepts}
    evidence: list[str] = []
    if "autoimmune disease" in concept_names:
        evidence.extend(
            [
                "history of autoimmune disease",
                "current autoimmune disease activity",
                "current systemic immunosuppressive therapy",
            ]
        )
    if "central nervous system metastases" in concept_names:
        evidence.append("presence or absence of CNS or brain metastases")
    if "ecog performance status" in concept_names:
        evidence.append("ECOG performance status value")
    if "qtc interval" in concept_names or "qtc" in text.casefold():
        evidence.append("QTc interval measurement")
    if "warfarin" in concept_names or "anticoagulation" in concept_names:
        evidence.append("current anticoagulant medication list")
    return list(dict.fromkeys(evidence))


def _rule_parse_text(text: str, *, source: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    haystack = normalized.casefold()
    concepts: list[dict[str, Any]] = []
    for entry in _CONCEPT_LEXICON:
        matched_alias = ""
        for alias in entry["aliases"]:
            if _contains_alias(haystack, str(alias)):
                matched_alias = str(alias)
                break
        if not matched_alias:
            continue
        concepts.append(
            {
                "surface": matched_alias,
                "canonical_name": entry["canonical_name"],
                "semantic_type": entry["semantic_type"],
                "ontology_ids": dict(entry.get("ontology_ids") or {}),
                "aliases": list(entry.get("aliases") or []),
                "confidence": 0.82,
            }
        )

    return {
        "raw_text": normalized,
        "source": source,
        "language": "en",
        "normalized_text": normalized,
        "concepts": concepts,
        "attributes": _attributes(normalized),
        "required_patient_evidence": _required_evidence(concepts, normalized),
        "parser": "rule",
        "warnings": [],
    }


def _call_external_parser(parser: Any, text: str, *, context: dict[str, Any], top_k: int) -> dict[str, Any] | None:
    parse = getattr(parser, "parse", None)
    if not callable(parse):
        return None
    try:
        result = parse(text, context=context, top_k=top_k)
    except TypeError:
        result = parse(text)
    if isinstance(result, Mapping):
        return dict(result)
    return None


def parse_medical_phrases_for_protocol(
    *,
    structured_case: Mapping[str, Any],
    trial_retrieval_bundle: Mapping[str, Any] | None = None,
    eligibility_assessment_bundle: Mapping[str, Any] | None = None,
    parser: Any | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    texts: list[dict[str, str]] = []
    texts.extend(_case_texts(structured_case))
    texts.extend(_criterion_texts(eligibility_assessment_bundle))
    texts.extend(_trial_texts(trial_retrieval_bundle, limit=limit))

    parsed_phrases: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item in texts[: max(int(limit) * 4, 1)]:
        context = {
            "source": item["source"],
            "structured_case": dict(structured_case or {}),
        }
        parsed = _call_external_parser(parser, item["text"], context=context, top_k=max(int(limit), 1))
        if parsed is None:
            parsed = _rule_parse_text(item["text"], source=item["source"])
        else:
            parsed.setdefault("raw_text", item["text"])
            parsed.setdefault("source", item["source"])
            parsed.setdefault("concepts", [])
            parsed.setdefault("attributes", {})
            parsed.setdefault("required_patient_evidence", [])
            parsed.setdefault("parser", parser.__class__.__name__)
            parsed.setdefault("warnings", [])
        parsed_phrases.append(parsed)

    concepts: list[dict[str, Any]] = []
    seen_concepts: set[tuple[str, str]] = set()
    for phrase in parsed_phrases:
        for concept in list(phrase.get("concepts") or []):
            if not isinstance(concept, Mapping):
                continue
            canonical = _normalize_text(concept.get("canonical_name") or concept.get("name") or concept.get("surface"))
            semantic_type = _normalize_text(concept.get("semantic_type"))
            if not canonical:
                continue
            key = (canonical.casefold(), semantic_type.casefold())
            if key in seen_concepts:
                continue
            seen_concepts.add(key)
            payload = dict(concept)
            payload.setdefault("canonical_name", canonical)
            payload.setdefault("semantic_type", semantic_type)
            concepts.append(payload)

    for phrase in parsed_phrases:
        warnings.extend([_normalize_text(item) for item in list(phrase.get("warnings") or []) if _normalize_text(item)])

    return {
        "schema_version": 1,
        "status": "completed",
        "backend_used": parser.__class__.__name__ if parser is not None else "rule",
        "parsed_phrases": parsed_phrases,
        "concepts": concepts,
        "concept_count": len(concepts),
        "warnings": list(dict.fromkeys(warnings)),
    }
