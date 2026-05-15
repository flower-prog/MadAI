from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|denies|deny|negative for|absence of|free of)\b|无|没有|否认",
    re.IGNORECASE,
)
_AGE_PATTERN = re.compile(r"\b(?P<age>\d{1,3})\s*(?:M|F|male|female|year[- ]old|yo|y/o)\b", re.IGNORECASE)
_MALE_PATTERN = re.compile(r"\b(?:male|man|M)\b", re.IGNORECASE)
_FEMALE_PATTERN = re.compile(r"\b(?:female|woman|F)\b", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
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

    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _normalize_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _append_unique(items: list[str], value: Any) -> None:
    text = _normalize_text(value)
    if not text:
        return
    key = text.casefold()
    if any(item.casefold() == key for item in items):
        return
    items.append(text)


def _case_payload(structured_case: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(structured_case or {})
    if isinstance(payload.get("structured_case"), Mapping):
        payload = dict(payload["structured_case"])
    return payload


def _looks_negative(text: Any) -> bool:
    return bool(_NEGATION_PATTERN.search(_normalize_text(text)))


def _extract_negated_clauses(text: Any) -> list[str]:
    raw_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    clauses = re.split(r"(?<=[.!?。！？])\s+|\n+|;", raw_text)
    facts: list[str] = []
    for clause in clauses:
        normalized = _normalize_text(clause)
        if not normalized or not _looks_negative(normalized):
            continue
        _append_unique(facts, normalized)
    return facts


def _extract_age(payload: Mapping[str, Any], constraints: Mapping[str, Any], combined_text: str) -> int | None:
    raw_age = constraints.get("age")
    if raw_age in {None, ""}:
        structured_inputs = dict(payload.get("structured_inputs") or {})
        for key in ("age", "age_years", "patient_age"):
            raw_age = structured_inputs.get(key)
            if raw_age not in {None, ""}:
                break
    try:
        parsed = int(float(raw_age))
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None and 0 < parsed < 130:
        return parsed

    match = _AGE_PATTERN.search(combined_text)
    if not match:
        return None
    try:
        parsed = int(match.group("age"))
    except (TypeError, ValueError):
        return None
    return parsed if 0 < parsed < 130 else None


def _extract_sex(payload: Mapping[str, Any], constraints: Mapping[str, Any], combined_text: str) -> str:
    raw_sex = _normalize_text(constraints.get("sex"))
    if not raw_sex:
        structured_inputs = dict(payload.get("structured_inputs") or {})
        raw_sex = _normalize_text(structured_inputs.get("sex") or structured_inputs.get("gender"))
    lowered = raw_sex.casefold()
    if lowered in {"m", "male", "man"}:
        return "Male"
    if lowered in {"f", "female", "woman"}:
        return "Female"
    if _MALE_PATTERN.search(combined_text):
        return "Male"
    if _FEMALE_PATTERN.search(combined_text):
        return "Female"
    return ""


def _anchor_fact(anchor: Mapping[str, Any]) -> str:
    return _normalize_text(anchor.get("canonical") or anchor.get("text"))


def build_patient_evidence_card(
    structured_case: Mapping[str, Any],
    *,
    trial_search_intent: Mapping[str, Any] | None = None,
    patient_evidence_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, strict patient fact card for trial rerank/eligibility prompts.

    The card only carries facts present in upstream case state or deterministic trial
    intent anchors. Missing facts are not invented; trial-specific missing data is
    produced later by eligibility assessment.
    """

    payload = _case_payload(structured_case)
    intent = dict(trial_search_intent or payload.get("trial_search_intent") or {})
    constraints = dict(intent.get("patient_constraints") or {})
    problem_items = _coerce_text_list(payload.get("problem_list"))
    known_fact_items = _coerce_text_list(payload.get("known_facts"))
    missing_items = _coerce_text_list(payload.get("missing_information"))
    combined_text = " ".join(
        item
        for item in [
            _normalize_text(payload.get("case_summary")),
            _normalize_text(payload.get("raw_text")),
            _normalize_text(payload.get("raw_request")),
            " ".join(problem_items),
            " ".join(known_fact_items),
        ]
        if item
    )

    summary = _normalize_text(payload.get("case_summary")) or _normalize_text(payload.get("raw_text"))[:800]
    age = _extract_age(payload, constraints, combined_text)
    sex = _extract_sex(payload, constraints, combined_text)

    positive_facts: list[str] = []
    negative_facts: list[str] = []

    for anchor in list(intent.get("primary_conditions") or []):
        if isinstance(anchor, Mapping):
            _append_unique(positive_facts, _anchor_fact(anchor))
    for anchor in list(intent.get("interventions_of_interest") or []):
        if isinstance(anchor, Mapping):
            _append_unique(positive_facts, _anchor_fact(anchor))
    for key, value in dict(constraints.get("key_measurements") or {}).items():
        key_text = _normalize_text(key)
        value_text = _normalize_text(value)
        if key_text and value_text:
            _append_unique(positive_facts, f"{key_text} {value_text}")

    for item in [*problem_items, *known_fact_items]:
        if _looks_negative(item):
            _append_unique(negative_facts, item)
        else:
            _append_unique(positive_facts, item)
    for item in _coerce_text_list(constraints.get("negated_findings")):
        _append_unique(negative_facts, item)
    for item in _extract_negated_clauses(payload.get("raw_text")):
        _append_unique(negative_facts, item)

    evidence_spans = [
        dict(item)
        for item in list((patient_evidence_bundle or {}).get("evidence_spans") or [])
        if isinstance(item, Mapping)
    ]

    return {
        "schema_version": 1,
        "age": age,
        "sex": sex,
        "summary": summary,
        "positive_facts": positive_facts,
        "negative_facts": negative_facts,
        "unknown_facts": missing_items,
        "evidence_spans": evidence_spans[:20],
    }
