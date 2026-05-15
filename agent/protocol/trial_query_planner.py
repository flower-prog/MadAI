from __future__ import annotations

import re
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


def _append_unique(items: list[str], value: Any) -> None:
    text = _normalize_text(value)
    if not text:
        return
    key = text.casefold()
    if any(item.casefold() == key for item in items):
        return
    items.append(text)


def _append_anchor(
    anchors: list[dict[str, Any]],
    *,
    text: Any,
    canonical: Any = "",
    role: str,
    confidence: float,
    evidence: Any = "",
) -> None:
    surface = _normalize_text(text)
    canonical_text = _normalize_text(canonical) or surface
    if not surface and not canonical_text:
        return
    key = (canonical_text.casefold(), role.casefold())
    if any((str(item.get("canonical") or "").casefold(), str(item.get("role") or "").casefold()) == key for item in anchors):
        return
    anchors.append(
        {
            "text": surface or canonical_text,
            "canonical": canonical_text,
            "role": role,
            "confidence": float(confidence),
            "evidence": _normalize_text(evidence),
        }
    )


def _case_payload(structured_case: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(structured_case or {})
    if isinstance(payload.get("structured_case"), Mapping):
        payload = dict(payload["structured_case"])
    return payload


def _combined_case_text(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("case_summary", "raw_text", "raw_request"):
        text = _normalize_text(payload.get(key))
        if text:
            parts.append(text)
    parts.extend(_coerce_text_list(payload.get("problem_list")))
    parts.extend(_coerce_text_list(payload.get("known_facts")))
    return " ".join(parts)


def _extract_measurements(payload: Mapping[str, Any], text: str) -> dict[str, Any]:
    structured_inputs = dict(payload.get("structured_inputs") or {})
    measurements: dict[str, Any] = {}
    for key, value in structured_inputs.items():
        if value not in {None, ""}:
            measurements[str(key)] = value
    ef_match = re.search(r"\b(?:EF|LVEF|ejection fraction)\s*(?:was|is|=|:)?\s*(\d{1,3})\s*%", text, re.IGNORECASE)
    if ef_match:
        measurements.setdefault("LVEF", f"{ef_match.group(1)}%")
    return measurements


def _extract_age_gender(payload: Mapping[str, Any], text: str) -> tuple[int | None, str]:
    structured_inputs = dict(payload.get("structured_inputs") or {})
    age: int | None = None
    for key in ("age", "age_years", "patient_age"):
        value = structured_inputs.get(key)
        if value in {None, ""}:
            continue
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            continue
        if 0 < parsed < 130:
            age = parsed
            break
    if age is None:
        age_match = re.search(r"\b(\d{1,3})\s*(?:M|F|male|female|year[- ]old|yo|y/o)\b", text, re.IGNORECASE)
        if age_match:
            parsed = int(age_match.group(1))
            if 0 < parsed < 130:
                age = parsed

    gender = ""
    sex_value = _normalize_text(structured_inputs.get("sex") or structured_inputs.get("gender"))
    if sex_value.casefold() in {"m", "male", "man"}:
        gender = "Male"
    elif sex_value.casefold() in {"f", "female", "woman"}:
        gender = "Female"
    elif re.search(r"\b(?:male|man|M)\b", text):
        gender = "Male"
    elif re.search(r"\b(?:female|woman|F)\b", text):
        gender = "Female"
    return age, gender


def _looks_like_intervention(text: str) -> bool:
    lowered = text.casefold()
    return any(
        cue in lowered
        for cue in (
            "treatment",
            "therapy",
            "surgery",
            "replacement",
            "implantation",
            "tavi",
            "savr",
            "tdcs",
            "stimulation",
            "rehabilitation",
            "训练",
            "治疗",
            "手术",
            "置换",
            "刺激",
        )
    )


def _infer_domain_anchors(
    *,
    text: str,
    problem_items: list[str],
    known_fact_items: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    condition_anchors: list[dict[str, Any]] = []
    intervention_anchors: list[dict[str, Any]] = []
    query_terms: list[str] = []
    notes: list[str] = []
    lowered = text.casefold()

    if re.search(r"aortic stenosis|主动脉瓣狭窄", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="aortic stenosis",
            canonical="Aortic Valve Stenosis",
            role="disease_anchor",
            confidence=0.9,
            evidence="case mentions severe or critical aortic stenosis",
        )
        for term in ("aortic valve stenosis", "critical aortic stenosis"):
            _append_unique(query_terms, term)
    if re.search(r"bicuspid aortic valve|二叶式主动脉瓣", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="bicuspid aortic valve",
            canonical="Bicuspid Aortic Valve",
            role="disease_context",
            confidence=0.86,
            evidence="case mentions bicuspid aortic valve",
        )
        _append_unique(query_terms, "bicuspid aortic valve")
    if re.search(r"\b(?:EF|LVEF|ejection fraction)\b|left ventricular dysfunction|左室功能|射血分数", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="left ventricular dysfunction",
            canonical="Left Ventricular Dysfunction",
            role="severity_or_comorbidity",
            confidence=0.78,
            evidence="case mentions reduced left ventricular function or EF",
        )
        _append_unique(query_terms, "reduced ejection fraction")
    if re.search(r"valvular replacement|valve replacement|aortic valve replacement|瓣膜置换", text, re.IGNORECASE):
        _append_anchor(
            intervention_anchors,
            text="aortic valve replacement",
            canonical="Aortic Valve Replacement",
            role="procedure_or_treatment",
            confidence=0.9,
            evidence="case mentions preoperative workup for valve replacement",
        )
        for term in ("aortic valve replacement", "TAVI", "transcatheter aortic valve implantation", "SAVR", "surgical aortic valve replacement"):
            _append_unique(query_terms, term)

    if re.search(r"\bstroke\b|\binfarct(?:ion)?\b|\bcerebral ischem|卒中|中风|脑梗|梗死", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="stroke",
            canonical="Stroke",
            role="disease_anchor",
            confidence=0.86,
            evidence="case mentions stroke or infarction",
        )
        _append_unique(query_terms, "stroke")
    if re.search(r"\brehabilitation\b|康复", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="post-stroke rehabilitation",
            canonical="Post-Stroke Rehabilitation",
            role="disease_context",
            confidence=0.76,
            evidence="case mentions rehabilitation context",
        )
        _append_anchor(
            intervention_anchors,
            text="rehabilitation training",
            canonical="Rehabilitation Training",
            role="procedure_or_treatment",
            confidence=0.82,
            evidence="case mentions rehabilitation training or therapy",
        )
        for term in ("post-stroke rehabilitation", "stroke rehabilitation", "rehabilitation training"):
            _append_unique(query_terms, term)
    if re.search(r"\btDCS\b|transcranial direct current stimulation|经颅直流电刺激", text, re.IGNORECASE):
        _append_anchor(
            intervention_anchors,
            text="tDCS",
            canonical="Transcranial Direct Current Stimulation",
            role="procedure_or_treatment",
            confidence=0.92,
            evidence="case mentions tDCS or transcranial direct current stimulation",
        )
        for term in ("tDCS", "transcranial direct current stimulation", "non-invasive brain stimulation"):
            _append_unique(query_terms, term)
    if re.search(r"\baphasia\b|失语", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="aphasia",
            canonical="Aphasia",
            role="symptom_or_functional_deficit",
            confidence=0.78,
            evidence="case mentions aphasia",
        )
        _append_unique(query_terms, "aphasia")
    if re.search(r"\bapraxia\b|失用", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="apraxia",
            canonical="Apraxia",
            role="symptom_or_functional_deficit",
            confidence=0.78,
            evidence="case mentions apraxia",
        )
        _append_unique(query_terms, "apraxia")

    if re.search(r"nipple discharge|乳头溢液", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="nipple discharge",
            canonical="Nipple Discharge",
            role="presenting_symptom",
            confidence=0.84,
            evidence="case mentions nipple discharge",
        )
        _append_unique(query_terms, "nipple discharge")
    if re.search(r"intraductal papilloma|导管内乳头状瘤", text, re.IGNORECASE):
        _append_anchor(
            condition_anchors,
            text="intraductal papilloma",
            canonical="Intraductal Papilloma",
            role="diagnosis",
            confidence=0.9,
            evidence="case mentions intraductal papilloma",
        )
        for term in ("intraductal papilloma", "breast duct lesion"):
            _append_unique(query_terms, term)
    if re.search(r"ductoscopy|乳管镜", text, re.IGNORECASE):
        _append_anchor(
            intervention_anchors,
            text="ductoscopy",
            canonical="Ductoscopy",
            role="diagnostic_or_therapeutic_procedure",
            confidence=0.78,
            evidence="case mentions ductoscopy",
        )
        _append_unique(query_terms, "ductoscopy")
    if re.search(r"breast (?:ultrasound|MRI|imaging)|乳腺超声|乳腺MRI|钼靶|影像", text, re.IGNORECASE):
        for term in ("breast imaging", "breast ultrasound", "breast MRI"):
            _append_unique(query_terms, term)
    if re.search(r"surgery|surgical|手术", text, re.IGNORECASE):
        _append_anchor(
            intervention_anchors,
            text="surgical treatment",
            canonical="Surgical Treatment",
            role="procedure_or_treatment",
            confidence=0.72,
            evidence="case mentions surgery or surgical treatment",
        )
        _append_unique(query_terms, "surgical excision")

    for item in problem_items:
        if _looks_like_intervention(item):
            continue
        _append_anchor(
            condition_anchors,
            text=item,
            role="case_problem",
            confidence=0.62,
            evidence=item,
        )
        _append_unique(query_terms, item)

    for item in known_fact_items:
        if not _looks_like_intervention(item):
            continue
        _append_anchor(
            intervention_anchors,
            text=item,
            role="case_treatment_context",
            confidence=0.58,
            evidence=item,
        )
        _append_unique(query_terms, item)

    if not condition_anchors:
        notes.append("No high-confidence trial disease anchor was extracted; fallback terms come from the general structured case.")
    if not intervention_anchors:
        notes.append("No high-confidence trial intervention anchor was extracted; retrieval should avoid intervention hard filters.")
    return condition_anchors, intervention_anchors, query_terms, notes


def _query_variants(
    *,
    condition_anchors: list[dict[str, Any]],
    intervention_anchors: list[dict[str, Any]],
    soft_query_terms: list[str],
    case_summary: str,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(name: str, text: str, source_terms: list[str]) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append({"name": name, "text": normalized, "source_terms": list(source_terms)})

    condition_terms = [
        _normalize_text(item.get("canonical") or item.get("text"))
        for item in condition_anchors
        if _normalize_text(item.get("canonical") or item.get("text"))
    ]
    intervention_terms = [
        _normalize_text(item.get("canonical") or item.get("text"))
        for item in intervention_anchors
        if _normalize_text(item.get("canonical") or item.get("text"))
    ]
    append("case_summary", case_summary[:500], ["case_summary"])
    if condition_terms:
        append("condition", f"{' '.join(condition_terms[:5])} clinical trial", condition_terms[:5])
    if condition_terms and intervention_terms:
        append(
            "condition_intervention",
            f"{' '.join(condition_terms[:4])} {' '.join(intervention_terms[:4])} clinical trial",
            [*condition_terms[:4], *intervention_terms[:4]],
        )
    if intervention_terms:
        append("intervention", f"{' '.join(intervention_terms[:5])} interventional trial", intervention_terms[:5])
    if soft_query_terms:
        append("soft_terms", f"{' '.join(soft_query_terms[:8])} trial", soft_query_terms[:8])
    return variants[:8]


def build_trial_search_intent(
    *,
    structured_case: Mapping[str, Any],
    calculation_bundle: Mapping[str, Any] | None = None,
    calculator_matches: list[Any] | None = None,
    department_tags: list[str] | None = None,
    planner: Any | None = None,
) -> dict[str, Any]:
    """Build a protocol-entry trial-search contract from the general case structure.

    The deterministic implementation is intentionally conservative and schema-first:
    it creates the same object that an LLM-backed trial query planner agent should
    return, while keeping hard filters empty unless a future validator promotes
    a constraint explicitly.
    """

    payload = _case_payload(structured_case)
    existing = payload.get("trial_search_intent")
    if isinstance(existing, Mapping) and existing.get("schema_version"):
        return dict(existing)

    if planner is not None:
        plan = getattr(planner, "build_trial_search_intent", None) or getattr(planner, "plan", None)
        if callable(plan):
            try:
                planned = plan(
                    structured_case=dict(payload),
                    calculation_bundle=dict(calculation_bundle or {}),
                    calculator_matches=list(calculator_matches or []),
                    department_tags=list(department_tags or []),
                )
            except TypeError:
                planned = plan(dict(payload))
            if isinstance(planned, Mapping):
                planned_payload = dict(planned)
                planned_payload.setdefault("schema_version", 1)
                planned_payload.setdefault("source", "trial_query_planner_agent")
                planned_payload.setdefault("hard_filters", [])
                planned_payload.setdefault("do_not_hard_filter", ["condition_terms", "intervention_terms", "age", "gender"])
                return planned_payload

    problem_items = _coerce_text_list(payload.get("problem_list"))
    known_fact_items = _coerce_text_list(payload.get("known_facts"))
    combined_text = _combined_case_text(payload)
    case_summary = _normalize_text(payload.get("case_summary")) or combined_text[:500]
    condition_anchors, intervention_anchors, inferred_terms, notes = _infer_domain_anchors(
        text=combined_text,
        problem_items=problem_items,
        known_fact_items=known_fact_items,
    )

    soft_query_terms: list[str] = []
    for item in inferred_terms:
        _append_unique(soft_query_terms, item)
    if not soft_query_terms:
        for item in [*problem_items[:6], *known_fact_items[:4]]:
            _append_unique(soft_query_terms, item)

    age, gender = _extract_age_gender(payload, combined_text)
    patient_constraints: dict[str, Any] = {
        "age": age,
        "sex": gender,
        "key_measurements": _extract_measurements(payload, combined_text),
        "negated_findings": [
            item
            for item in known_fact_items
            if re.search(r"\b(?:no|not|without|negative for|absence of|denies)\b|无|没有|否认", item, re.IGNORECASE)
        ],
    }

    trial_intents: list[str] = []
    if intervention_anchors:
        _append_unique(trial_intents, "treatment")
    if any("procedure" in str(item.get("role") or "") for item in intervention_anchors):
        _append_unique(trial_intents, "procedure")
    if not trial_intents:
        _append_unique(trial_intents, "condition_context")

    query_variants = _query_variants(
        condition_anchors=condition_anchors,
        intervention_anchors=intervention_anchors,
        soft_query_terms=soft_query_terms,
        case_summary=case_summary,
    )

    return {
        "schema_version": 1,
        "source": "protocol_entry_trial_structurer",
        "status": "completed",
        "primary_conditions": condition_anchors,
        "interventions_of_interest": intervention_anchors,
        "trial_intents": trial_intents,
        "patient_constraints": patient_constraints,
        "soft_query_terms": soft_query_terms,
        "query_variants": query_variants,
        "soft_filters": {"study_type": ["Interventional"]} if intervention_anchors else {},
        "hard_filters": [],
        "do_not_hard_filter": ["condition_terms", "intervention_terms", "age", "gender"],
        "warnings": notes,
    }
