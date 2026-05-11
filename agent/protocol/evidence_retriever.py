from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .types import EligibilityCriterion, PatientEvidenceSpan


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_AGE_PATTERN = re.compile(r"\b\d{1,3}\s*[- ]?\s*year[- ]old\b|\bage\s*\d{1,3}\b|\b\d{1,3}\s*岁", re.IGNORECASE)
_MALE_PATTERN = re.compile(r"\bmale\b|\bman\b|男性|男", re.IGNORECASE)
_FEMALE_PATTERN = re.compile(r"\bfemale\b|\bwoman\b|女性|女", re.IGNORECASE)
_ECOG_PATTERN = re.compile(r"\bECOG\b\s*(?:performance status)?\s*[:=]?\s*[0-5]", re.IGNORECASE)
_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|denies|negative for|absence of|free of)\b|无|没有|否认", re.IGNORECASE)
_CALCULATOR_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "ECOG": ("ecog", "performance status", "eastern cooperative oncology group"),
    "Karnofsky": ("karnofsky", "kps"),
    "Child-Pugh": ("child-pugh", "child pugh", "child-pugh class", "child class"),
    "MELD": ("meld", "model for end-stage liver disease"),
    "Gleason": ("gleason", "gleason score"),
    "TNM stage": ("tnm", "tumor node metastasis", "cancer stage", "stage"),
    "creatinine clearance": ("creatinine clearance", "crcl"),
    "eGFR": ("egfr", "estimated glomerular filtration rate"),
    "LVEF": ("lvef", "left ventricular ejection fraction", "ejection fraction"),
    "CHA2DS2-VASc": ("cha2ds2-vasc", "cha2ds2 vasc", "stroke risk"),
    "HAS-BLED": ("has-bled", "has bled", "bleeding risk"),
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return {}


def _coerce_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_normalize_text(value)] if _normalize_text(value) else []
    if isinstance(value, Mapping):
        return [_normalize_text(item) for item in value.values() if _normalize_text(item)]
    try:
        return [_normalize_text(item) for item in list(value) if _normalize_text(item)]
    except TypeError:
        text = _normalize_text(value)
        return [text] if text else []


def _calculator_concept(payload: Mapping[str, Any]) -> str:
    text = " ".join(
        _normalize_text(payload.get(key))
        for key in (
            "calculator",
            "name",
            "title",
            "category",
            "rationale",
            "linked_calculator",
            "required_evidence_type",
        )
    ).casefold()
    for concept, aliases in _CALCULATOR_CONCEPT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return concept
    return ""


def _format_calculator_evidence_text(payload: Mapping[str, Any], concept: str) -> str:
    value = _normalize_text(payload.get("value"))
    unit = _normalize_text(payload.get("unit"))
    status = _normalize_text(payload.get("status"))
    rationale = _normalize_text(payload.get("rationale"))
    label = concept or _normalize_text(payload.get("calculator") or payload.get("name") or payload.get("title"))
    pieces = []
    if label and value:
        pieces.append(f"{label} = {value}")
    elif label:
        pieces.append(label)
    elif value:
        pieces.append(value)
    if unit:
        pieces.append(unit)
    if status:
        pieces.append(f"status: {status}")
    if rationale:
        pieces.append(rationale)
    return _normalize_text("; ".join(pieces))


def build_calculator_evidence_spans(
    *,
    calculation_results: Iterable[Any] | None = None,
    calculator_evidence_bundle: Mapping[str, Any] | None = None,
) -> list[PatientEvidenceSpan]:
    spans: list[PatientEvidenceSpan] = []
    for index, item in enumerate(list(calculation_results or []), start=1):
        payload = _as_mapping(item)
        concept = _calculator_concept(payload)
        text = _format_calculator_evidence_text(payload, concept)
        value = _normalize_text(payload.get("value"))
        unit = _normalize_text(payload.get("unit"))
        if text:
            spans.append(
                PatientEvidenceSpan(
                    source=f"calculation_results[{index}]",
                    text=text,
                    normalized_concept=concept,
                    value=value,
                    unit=unit,
                    score=0.0,
                )
            )

    bundle = dict(calculator_evidence_bundle or {})
    for index, item in enumerate(list(bundle.get("risk_evidence_items") or []), start=1):
        payload = _as_mapping(item)
        usable_for = {str(value).strip() for value in list(payload.get("usable_for") or [])}
        if usable_for and "eligibility_evidence" not in usable_for:
            continue
        concept = _calculator_concept(payload)
        text = _format_calculator_evidence_text(payload, concept)
        value = _normalize_text(payload.get("value"))
        unit = _normalize_text(payload.get("unit"))
        if text:
            spans.append(
                PatientEvidenceSpan(
                    source=f"calculator_evidence_bundle.risk_evidence_items[{index}]",
                    text=text,
                    normalized_concept=concept,
                    value=value,
                    unit=unit,
                    score=0.0,
                )
            )
    return spans


def _split_fragments(source: str, text: str) -> list[PatientEvidenceSpan]:
    fragments: list[PatientEvidenceSpan] = []
    cursor = 0
    for part in _SENTENCE_SPLIT.split(text):
        normalized = _normalize_text(part)
        if not normalized:
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = None
            end = None
        else:
            end = start + len(part)
            cursor = end
        fragments.append(
            PatientEvidenceSpan(
                source=source,
                text=normalized,
                start=start,
                end=end,
                score=0.0,
            )
        )
    return fragments


def build_patient_evidence_index(
    structured_case: Mapping[str, Any],
    *,
    calculation_results: Iterable[Any] | None = None,
    calculator_matches: Iterable[Any] | None = None,
    calculator_evidence_bundle: Mapping[str, Any] | None = None,
) -> list[PatientEvidenceSpan]:
    case_payload = dict(structured_case or {})
    if isinstance(case_payload.get("structured_case"), Mapping):
        case_payload = dict(case_payload["structured_case"])

    spans: list[PatientEvidenceSpan] = []
    for key in ("raw_text", "raw_request", "case_summary"):
        text = _normalize_text(case_payload.get(key))
        if text:
            spans.extend(_split_fragments(key, text))
    for key in ("problem_list", "known_facts"):
        for index, text in enumerate(_coerce_text_items(case_payload.get(key)), start=1):
            spans.append(PatientEvidenceSpan(source=f"{key}[{index}]", text=text, score=0.0))

    structured_inputs = _as_mapping(case_payload.get("structured_inputs"))
    for key, value in structured_inputs.items():
        text = _normalize_text(value)
        if text:
            spans.append(PatientEvidenceSpan(source=f"structured_inputs.{key}", text=f"{key}: {text}", value=text, score=0.0))

    spans.extend(
        build_calculator_evidence_spans(
            calculation_results=calculation_results,
            calculator_evidence_bundle=calculator_evidence_bundle,
        )
    )

    for index, item in enumerate(list(calculator_matches or []), start=1):
        payload = _as_mapping(item)
        text = _normalize_text(" ".join(str(payload.get(key) or "") for key in ("title", "rationale", "value")))
        if text:
            spans.append(PatientEvidenceSpan(source=f"calculator_matches[{index}]", text=text, score=0.0))

    deduped: list[PatientEvidenceSpan] = []
    seen: set[tuple[str, str]] = set()
    for span in spans:
        key = (span.source, span.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def _tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text)
        if len(token) > 1
    }


def _score_span(criterion: EligibilityCriterion, span: PatientEvidenceSpan) -> float:
    text = span.text
    condition = criterion.condition.casefold()
    raw_text = criterion.raw_text.casefold()
    score = 0.0

    if condition == "age" and _AGE_PATTERN.search(text):
        score += 3.0
    if condition == "sex":
        desired = criterion.value.casefold()
        if desired in {"male", "female"} and (_MALE_PATTERN.search(text) or _FEMALE_PATTERN.search(text)):
            score += 3.0
    if condition == "ecog" and _ECOG_PATTERN.search(text):
        score += 3.0
    if span.normalized_concept and span.normalized_concept.casefold() == condition:
        score += 3.5
    if condition in {"cns metastases", "pregnancy"}:
        matched_condition = False
        for token in condition.split():
            if token and token in text.casefold():
                matched_condition = True
                score += 1.2
        if matched_condition and _NEGATION_PATTERN.search(text):
            score += 0.5

    criterion_tokens = _tokenize(f"{criterion.condition} {criterion.value} {criterion.raw_text}")
    span_tokens = _tokenize(text)
    if criterion_tokens and span_tokens:
        score += len(criterion_tokens.intersection(span_tokens)) / max(len(criterion_tokens), 1)
    if raw_text and raw_text in text.casefold():
        score += 2.0
    return score


def find_evidence_for_criterion(
    evidence_index: list[PatientEvidenceSpan],
    criterion: EligibilityCriterion,
    *,
    limit: int = 3,
) -> list[PatientEvidenceSpan]:
    scored: list[PatientEvidenceSpan] = []
    for span in list(evidence_index or []):
        score = _score_span(criterion, span)
        if score <= 0.0:
            continue
        scored.append(
            PatientEvidenceSpan(
                source=span.source,
                text=span.text,
                start=span.start,
                end=span.end,
                score=float(score),
                normalized_concept=span.normalized_concept,
                value=span.value,
                unit=span.unit,
                observed_time=span.observed_time,
            )
        )
    scored.sort(key=lambda item: (-item.score, item.source, item.text))
    return scored[: max(int(limit), 1)]
