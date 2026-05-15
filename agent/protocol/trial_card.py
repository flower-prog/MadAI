from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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


def _title(record: Mapping[str, Any]) -> str:
    for key in ("display_title", "title", "brief_title", "official_title", "name", "nct_id"):
        text = _normalize_text(record.get(key))
        if text:
            return text
    return ""


def _candidate_chunks(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_chunks = (
        candidate.get("matched_chunks")
        or candidate.get("supporting_chunks")
        or candidate.get("top_chunks")
        or []
    )
    chunks: list[dict[str, Any]] = []
    for item in list(raw_chunks or []):
        if not isinstance(item, Mapping):
            continue
        chunk_id = _normalize_text(item.get("chunk_id") or item.get("document_id"))
        chunk_type = _normalize_text(item.get("chunk_type"))
        text = _normalize_text(item.get("text") or item.get("summary") or item.get("eligibility"))
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "score": float(item.get("score") or 0.0),
                "text": text,
            }
        )
    return chunks


def build_trial_card(
    trial_record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
    matched_chunks: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact trial card from shallow XML-derived trial fields."""

    record = dict(trial_record or {})
    candidate_payload = dict(candidate or {})
    nct_id = _normalize_text(record.get("nct_id") or candidate_payload.get("nct_id"))
    chunks = [
        {
            "chunk_id": _normalize_text(item.get("chunk_id") or item.get("document_id")),
            "chunk_type": _normalize_text(item.get("chunk_type")),
            "score": float(item.get("score") or 0.0),
            "text": _normalize_text(item.get("text") or item.get("summary") or item.get("eligibility")),
        }
        for item in list(matched_chunks or [])
        if isinstance(item, Mapping)
    ] or _candidate_chunks(candidate_payload)

    return {
        "schema_version": 1,
        "nct_id": nct_id,
        "title": _title({**candidate_payload, **record}),
        "source_url": _normalize_text(record.get("source_url") or candidate_payload.get("source_url")),
        "overall_status": _normalize_text(record.get("overall_status") or candidate_payload.get("overall_status")),
        "status": _normalize_text(candidate_payload.get("status")),
        "enrollment_open": bool(record.get("enrollment_open") or candidate_payload.get("enrollment_open")),
        "study_type": _normalize_text(record.get("study_type") or candidate_payload.get("study_type")),
        "phase": _normalize_text(record.get("phase") or candidate_payload.get("phase")),
        "primary_purpose": _normalize_text(record.get("primary_purpose") or candidate_payload.get("primary_purpose")),
        "conditions": _coerce_text_list(record.get("conditions") or candidate_payload.get("conditions")),
        "condition_terms": _coerce_text_list(record.get("condition_terms") or candidate_payload.get("condition_terms")),
        "interventions": _coerce_text_list(record.get("interventions") or candidate_payload.get("interventions")),
        "intervention_terms": _coerce_text_list(
            record.get("intervention_terms") or candidate_payload.get("intervention_terms")
        ),
        "brief_summary": _normalize_text(record.get("brief_summary") or candidate_payload.get("brief_summary")),
        "eligibility": {
            "gender": _normalize_text(record.get("gender") or candidate_payload.get("gender")),
            "minimum_age": _normalize_text(record.get("minimum_age") or candidate_payload.get("minimum_age")),
            "maximum_age": _normalize_text(record.get("maximum_age") or candidate_payload.get("maximum_age")),
            "healthy_volunteers": _normalize_text(record.get("healthy_volunteers")),
            "inclusion_text": _normalize_text(
                record.get("eligibility_inclusion_text") or candidate_payload.get("eligibility_inclusion_text")
            ),
            "exclusion_text": _normalize_text(
                record.get("eligibility_exclusion_text") or candidate_payload.get("eligibility_exclusion_text")
            ),
            "raw_text": _normalize_text(record.get("eligibility_text") or candidate_payload.get("eligibility_text")),
            "unsplit_text": _normalize_text(record.get("eligibility_unsplit_text")),
            "parse_status": _normalize_text(record.get("eligibility_section_parse_status")),
            "parse_warnings": _coerce_text_list(record.get("eligibility_section_parse_warnings")),
        },
        "matched_chunks": chunks[:5],
        "coarse_score": float(candidate_payload.get("score") or candidate_payload.get("coarse_score") or 0.0),
    }


def build_trial_card_text(trial_card: Mapping[str, Any]) -> str:
    eligibility = dict(trial_card.get("eligibility") or {})
    chunks = [
        _normalize_text(item.get("text"))
        for item in list(trial_card.get("matched_chunks") or [])
        if isinstance(item, Mapping) and _normalize_text(item.get("text"))
    ]
    lines = [
        f"nct_id: {_normalize_text(trial_card.get('nct_id'))}",
        f"title: {_normalize_text(trial_card.get('title'))}",
        f"status: {_normalize_text(trial_card.get('overall_status') or trial_card.get('status'))}",
        f"study type: {_normalize_text(trial_card.get('study_type'))}",
        f"phase: {_normalize_text(trial_card.get('phase'))}",
        f"primary purpose: {_normalize_text(trial_card.get('primary_purpose'))}",
        "conditions: " + ", ".join(_coerce_text_list(trial_card.get("conditions"))),
        "interventions: " + ", ".join(_coerce_text_list(trial_card.get("interventions"))),
        f"summary: {_normalize_text(trial_card.get('brief_summary'))}",
        f"eligibility gender: {_normalize_text(eligibility.get('gender'))}",
        "eligibility age: "
        + " to ".join(
            item
            for item in [_normalize_text(eligibility.get("minimum_age")), _normalize_text(eligibility.get("maximum_age"))]
            if item
        ),
        f"inclusion criteria: {_normalize_text(eligibility.get('inclusion_text'))}",
        f"exclusion criteria: {_normalize_text(eligibility.get('exclusion_text'))}",
    ]
    raw_text = _normalize_text(eligibility.get("raw_text"))
    if raw_text and not (_normalize_text(eligibility.get("inclusion_text")) or _normalize_text(eligibility.get("exclusion_text"))):
        lines.append(f"eligibility criteria: {raw_text}")
    if chunks:
        lines.append("matched chunks: " + " | ".join(chunks[:3]))
    return "\n".join(line for line in lines if _normalize_text(line.split(":", 1)[-1] if ":" in line else line)).strip()
