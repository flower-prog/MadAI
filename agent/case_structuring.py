from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def coerce_string_list(value: Any, *, limit: int | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Mapping):
        values = list(value.values())
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    if limit is None:
        return normalized
    return normalized[: max(int(limit), 0)]


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    payload: dict[str, Any] = {}
    for key, item in value.items():
        if is_dataclass(item):
            payload[str(key)] = asdict(item)
        elif isinstance(item, Mapping):
            payload[str(key)] = dict(item)
        else:
            payload[str(key)] = item
    return payload


def build_structured_case_payload(
    *,
    seeded_structured_case: Mapping[str, Any] | None = None,
    source_mode: str = "baseline",
    patient_id: Any = None,
    raw_request: Any = "",
    raw_text: Any = "",
    case_summary: Any = "",
    problem_list: list[str] | None = None,
    known_facts: Any = None,
    missing_information: Any = None,
    department: Any = "",
    department_tags: list[str] | None = None,
    data_readiness: Mapping[str, Any] | None = None,
    structured_inputs: Mapping[str, Any] | None = None,
    interval_inputs: Mapping[str, Any] | None = None,
    multimodal_inputs: Mapping[str, Any] | None = None,
    reporter_feedback: list[str] | None = None,
) -> dict[str, Any]:
    """Build the normalized case object shared by calculator, protocol, and reporter.

    The graph node owns state routing; this module owns the shape of the
    structured case payload so downstream retrieval/matching code can reuse it
    without importing graph internals.
    """

    seed = dict(seeded_structured_case or {})
    merged_structured_inputs = _plain_mapping(seed.get("structured_inputs"))
    merged_structured_inputs.update(_plain_mapping(structured_inputs))

    merged_interval_inputs = _plain_mapping(seed.get("interval_inputs"))
    merged_interval_inputs.update(_plain_mapping(interval_inputs))

    merged_multimodal_inputs = _plain_mapping(seed.get("multimodal_inputs"))
    merged_multimodal_inputs.update(_plain_mapping(multimodal_inputs))

    return {
        **seed,
        "source_mode": str(source_mode or seed.get("source_mode") or "baseline"),
        "patient_id": patient_id if patient_id is not None else seed.get("patient_id"),
        "raw_request": str(seed.get("raw_request") or raw_request or ""),
        "raw_text": str(seed.get("raw_text") or raw_text or ""),
        "case_summary": str(case_summary or seed.get("case_summary") or ""),
        "problem_list": list(problem_list or coerce_string_list(seed.get("problem_list"))),
        "known_facts": coerce_string_list(known_facts if known_facts is not None else seed.get("known_facts")),
        "missing_information": coerce_string_list(
            missing_information if missing_information is not None else seed.get("missing_information")
        ),
        "department": str(department or seed.get("department") or ""),
        "department_tags": list(department_tags or coerce_string_list(seed.get("department_tags"))),
        "data_readiness": dict(data_readiness or seed.get("data_readiness") or {}),
        "structured_inputs": merged_structured_inputs,
        "interval_inputs": merged_interval_inputs,
        "multimodal_inputs": merged_multimodal_inputs,
        "reporter_feedback": list(reporter_feedback or seed.get("reporter_feedback") or []),
    }
