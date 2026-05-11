from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .types import CriterionType, EligibilityCriterion


_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]+|\(?[0-9]{1,3}[\).:]|[a-zA-Z][\).:])\s*")
_SECTION_HEADING = re.compile(r"^\s*(?:inclusion|exclusion)\s+criteria\s*:?\s*$", re.IGNORECASE)
_AGE_PATTERN = re.compile(
    r"(?:(?:age|aged|ages)\s*)?(?P<operator>>=|≤|<=|>|<|=)?\s*(?P<age>\d{1,3})\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
_ECOG_RANGE_PATTERN = re.compile(r"\bECOG\b.*?(?P<low>[0-5])\s*(?:-|to|through)\s*(?P<high>[0-5])", re.IGNORECASE)
_KARNOFSKY_PATTERN = re.compile(r"\bKarnofsky\b.*?(?P<operator>>=|≤|<=|>|<|=)?\s*(?P<value>\d{2,3})", re.IGNORECASE)
_LAB_PATTERN = re.compile(
    r"\b(?P<lab>ANC|absolute neutrophil count|platelets?|hemoglobin|bilirubin|AST|ALT|creatinine clearance|CrCl)\b"
    r".*?(?P<operator>>=|≤|<=|>|<|=)\s*(?P<value>[0-9]+(?:\.[0-9]+)?(?:\s*[a-zA-Z/%^0-9]+)?)",
    re.IGNORECASE,
)
_MALE_PATTERN = re.compile(r"\bmale\b|\bmen\b|\bman\b|男性|男", re.IGNORECASE)
_FEMALE_PATTERN = re.compile(r"\bfemale\b|\bwomen\b|\bwoman\b|女性|女", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split()).strip()


def _candidate_title(record: Mapping[str, Any]) -> str:
    for key in ("display_title", "title", "brief_title", "official_title", "name"):
        text = _normalize_text(record.get(key))
        if text:
            return text
    return _normalize_text(record.get("nct_id"))


def _split_criteria_text(text: Any) -> list[str]:
    raw_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    items: list[str] = []
    buffer: list[str] = []

    for raw_line in raw_text.split("\n"):
        line = raw_line.strip()
        if not line:
            if buffer:
                items.append(_normalize_text(" ".join(buffer)))
                buffer = []
            continue
        if _SECTION_HEADING.match(line):
            continue

        stripped = _BULLET_PREFIX.sub("", line).strip()
        starts_new = bool(_BULLET_PREFIX.match(line))
        if starts_new and buffer:
            items.append(_normalize_text(" ".join(buffer)))
            buffer = []
        buffer.append(stripped)

    if buffer:
        items.append(_normalize_text(" ".join(buffer)))

    if len(items) <= 1 and ";" in raw_text:
        semicolon_items = [_normalize_text(item) for item in raw_text.split(";")]
        if all(len(item.split()) >= 3 for item in semicolon_items if item):
            items = semicolon_items

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_text(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _extract_common_fields(raw_text: str) -> dict[str, Any]:
    text = _normalize_text(raw_text)
    lowered = text.casefold()
    fields: dict[str, Any] = {}

    age_match = _AGE_PATTERN.search(text)
    if age_match and ("age" in lowered or "years" in lowered):
        operator = age_match.group("operator") or ("<=" if "maximum age" in lowered else ">=")
        fields.update(
            {
                "condition": "age",
                "operator": "<=" if operator == "≤" else operator,
                "value": f"{age_match.group('age')} years",
                "required_evidence_type": "demographic",
            }
        )
        return fields

    has_female = bool(_FEMALE_PATTERN.search(text))
    has_male = bool(_MALE_PATTERN.search(text))
    if has_female and not has_male:
        return {
            "condition": "sex",
            "operator": "=",
            "value": "Female",
            "required_evidence_type": "demographic",
        }
    if has_male and not has_female:
        return {
            "condition": "sex",
            "operator": "=",
            "value": "Male",
            "required_evidence_type": "demographic",
        }

    ecog_match = _ECOG_RANGE_PATTERN.search(text)
    if ecog_match:
        return {
            "condition": "ECOG",
            "operator": "between",
            "value": f"{ecog_match.group('low')}-{ecog_match.group('high')}",
            "required_evidence_type": "performance_status",
        }

    karnofsky_match = _KARNOFSKY_PATTERN.search(text)
    if karnofsky_match:
        return {
            "condition": "Karnofsky",
            "operator": karnofsky_match.group("operator") or ">=",
            "value": karnofsky_match.group("value"),
            "required_evidence_type": "performance_status",
        }

    lab_match = _LAB_PATTERN.search(text)
    if lab_match:
        return {
            "condition": lab_match.group("lab").lower(),
            "operator": "<=" if lab_match.group("operator") == "≤" else lab_match.group("operator"),
            "value": lab_match.group("value").strip(),
            "required_evidence_type": "lab",
        }

    if "pregnan" in lowered:
        return {
            "condition": "pregnancy",
            "operator": "present",
            "value": "pregnancy",
            "required_evidence_type": "clinical_fact",
            "negation": any(token in lowered for token in ("not pregnant", "negative pregnancy", "non-pregnant")),
        }

    if "brain metast" in lowered or "cns metast" in lowered or "central nervous system metast" in lowered:
        return {
            "condition": "CNS metastases",
            "operator": "present",
            "value": "active CNS metastases" if "active" in lowered else "CNS metastases",
            "required_evidence_type": "clinical_fact",
        }

    if any(token in lowered for token in ("diagnosed", "diagnosis", "histologically confirmed", "confirmed")):
        return {
            "condition": "diagnosis",
            "operator": "includes",
            "value": text,
            "required_evidence_type": "diagnosis",
        }

    return fields


def _build_criterion(
    *,
    nct_id: str,
    criterion_type: CriterionType,
    index: int,
    raw_text: str,
) -> EligibilityCriterion:
    fields = _extract_common_fields(raw_text)
    return EligibilityCriterion(
        criterion_id=f"{nct_id}::{criterion_type}::{index:03d}",
        nct_id=nct_id,
        type=criterion_type,
        raw_text=_normalize_text(raw_text),
        condition=str(fields.get("condition") or ""),
        operator=str(fields.get("operator") or ""),
        value=str(fields.get("value") or ""),
        required_evidence_type=str(fields.get("required_evidence_type") or "clinical_fact"),
        negation=bool(fields.get("negation")),
    )


def parse_trial_criteria(trial_record: Mapping[str, Any]) -> list[EligibilityCriterion]:
    record = dict(trial_record or {})
    nct_id = _normalize_text(record.get("nct_id")) or "UNKNOWN"
    criteria: list[EligibilityCriterion] = []
    section_parse_status = _normalize_text(record.get("eligibility_section_parse_status"))
    if section_parse_status == "unsplit":
        return criteria

    inclusion_items = _split_criteria_text(record.get("eligibility_inclusion_text") or "")
    exclusion_items = _split_criteria_text(record.get("eligibility_exclusion_text") or "")

    if not inclusion_items and not exclusion_items and not section_parse_status:
        eligibility_text = str(record.get("eligibility_text") or record.get("eligibility") or "")
        inclusion_items = _split_criteria_text(eligibility_text)

    for index, item in enumerate(inclusion_items, start=1):
        criteria.append(_build_criterion(nct_id=nct_id, criterion_type="inclusion", index=index, raw_text=item))
    for index, item in enumerate(exclusion_items, start=1):
        criteria.append(_build_criterion(nct_id=nct_id, criterion_type="exclusion", index=index, raw_text=item))

    structured_items: list[tuple[CriterionType, str]] = []
    minimum_age = _normalize_text(record.get("minimum_age"))
    if minimum_age and minimum_age.casefold() not in {"n/a", "none", "no minimum age"}:
        structured_items.append(("inclusion", f"Minimum age {minimum_age}"))
    maximum_age = _normalize_text(record.get("maximum_age"))
    if maximum_age and maximum_age.casefold() not in {"n/a", "none", "no maximum age"}:
        structured_items.append(("inclusion", f"Maximum age {maximum_age}"))
    gender = _normalize_text(record.get("gender"))
    if gender and gender.casefold() not in {"all", "both"}:
        structured_items.append(("inclusion", f"Sex {gender}"))

    for criterion_type, raw_text in structured_items:
        existing = {item.raw_text.casefold() for item in criteria}
        if _normalize_text(raw_text).casefold() in existing:
            continue
        next_index = 1 + sum(1 for item in criteria if item.type == criterion_type)
        criteria.append(_build_criterion(nct_id=nct_id, criterion_type=criterion_type, index=next_index, raw_text=raw_text))

    return criteria
