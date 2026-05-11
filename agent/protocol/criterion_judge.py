from __future__ import annotations

import re

from .types import CriterionAssessment, EligibilityCriterion, PatientEvidenceSpan


_AGE_VALUE_PATTERN = re.compile(r"(?P<age>\d{1,3})")
_ECOG_VALUE_PATTERN = re.compile(r"\bECOG\b\s*(?:performance status)?\s*[:=]?\s*(?P<value>[0-5])", re.IGNORECASE)
_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|denies|negative for|absence of|free of)\b|无|没有|否认", re.IGNORECASE)


def _first_number(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _comparison_met(patient_value: float, operator: str, criterion_value: float) -> bool:
    if operator == ">":
        return patient_value > criterion_value
    if operator == ">=":
        return patient_value >= criterion_value
    if operator == "<":
        return patient_value < criterion_value
    if operator == "<=":
        return patient_value <= criterion_value
    if operator == "=":
        return patient_value == criterion_value
    return patient_value >= criterion_value


def _assessment(
    criterion: EligibilityCriterion,
    *,
    label: str,
    confidence: float,
    evidence_spans: list[PatientEvidenceSpan],
    rationale: str,
    missing_data: list[str] | None = None,
) -> CriterionAssessment:
    return CriterionAssessment(
        criterion_id=criterion.criterion_id,
        nct_id=criterion.nct_id,
        type=criterion.type,
        raw_text=criterion.raw_text,
        condition=criterion.condition,
        operator=criterion.operator,
        value=criterion.value,
        time_window=criterion.time_window,
        required_evidence_type=criterion.required_evidence_type,
        negation=criterion.negation,
        label=label,  # type: ignore[arg-type]
        confidence=max(0.0, min(float(confidence), 1.0)),
        evidence_spans=list(evidence_spans),
        rationale=rationale,
        missing_data=list(missing_data or []),
    )


def judge_criterion(
    criterion: EligibilityCriterion,
    evidence_spans: list[PatientEvidenceSpan],
) -> CriterionAssessment:
    condition = criterion.condition.casefold()
    spans = list(evidence_spans or [])

    if condition == "age":
        required_age = _first_number(criterion.value)
        patient_age = None
        for span in spans:
            match = _AGE_VALUE_PATTERN.search(span.text)
            if match:
                patient_age = float(match.group("age"))
                break
        if required_age is None:
            return _assessment(
                criterion,
                label="unknown",
                confidence=0.0,
                evidence_spans=[],
                rationale="年龄标准没有被解析出可比较的数值。",
                missing_data=["age eligibility threshold"],
            )
        if patient_age is None:
            return _assessment(
                criterion,
                label="unknown",
                confidence=0.0,
                evidence_spans=[],
                rationale="病例中没有找到可用于判断年龄标准的证据。",
                missing_data=["patient age"],
            )
        met = _comparison_met(patient_age, criterion.operator or ">=", required_age)
        return _assessment(
            criterion,
            label="met" if met else "not_met",
            confidence=0.95,
            evidence_spans=spans[:1],
            rationale=(
                f"患者年龄 {int(patient_age)} 岁，符合 {criterion.raw_text}。"
                if met
                else f"患者年龄 {int(patient_age)} 岁，不符合 {criterion.raw_text}。"
            ),
        )

    if condition == "sex":
        desired = criterion.value.casefold()
        combined = " ".join(span.text for span in spans).casefold()
        if not combined:
            return _assessment(
                criterion,
                label="unknown",
                confidence=0.0,
                evidence_spans=[],
                rationale="病例中没有找到可用于判断性别标准的证据。",
                missing_data=["patient sex"],
            )
        patient_value = ""
        if re.search(r"\bmale\b|\bman\b|男性|男", combined):
            patient_value = "male"
        if re.search(r"\bfemale\b|\bwoman\b|女性|女", combined):
            patient_value = "female"
        if not patient_value:
            return _assessment(
                criterion,
                label="unknown",
                confidence=0.0,
                evidence_spans=spans[:1],
                rationale="找到的证据不足以明确患者性别。",
                missing_data=["patient sex"],
            )
        met = patient_value == desired
        return _assessment(
            criterion,
            label="met" if met else "not_met",
            confidence=0.9,
            evidence_spans=spans[:1],
            rationale="患者性别与试验要求兼容。" if met else "患者性别与试验要求不兼容。",
        )

    if condition == "ecog":
        value = None
        for span in spans:
            match = _ECOG_VALUE_PATTERN.search(span.text)
            if match:
                value = int(match.group("value"))
                break
        if value is None:
            return _assessment(
                criterion,
                label="unknown",
                confidence=0.0,
                evidence_spans=[],
                rationale="病例中没有找到 ECOG 评分。",
                missing_data=["ECOG performance status"],
            )
        bounds = [int(item) for item in re.findall(r"[0-5]", criterion.value)]
        if len(bounds) >= 2:
            met = min(bounds) <= value <= max(bounds)
        elif len(bounds) == 1:
            met = value == bounds[0]
        else:
            met = True
        return _assessment(
            criterion,
            label="met" if met else "not_met",
            confidence=0.9,
            evidence_spans=spans[:1],
            rationale=f"患者 ECOG 为 {value}，{'符合' if met else '不符合'}该标准。",
        )

    if condition in {"cns metastases", "pregnancy"}:
        if not spans:
            return _assessment(
                criterion,
                label="unknown",
                confidence=0.0,
                evidence_spans=[],
                rationale="病例中没有找到可判断该标准的证据。",
                missing_data=[criterion.condition],
            )
        has_negation = any(_NEGATION_PATTERN.search(span.text) for span in spans)
        if criterion.type == "exclusion":
            label = "not_met" if has_negation else "met"
            rationale = "病例证据提示未触发该排除标准。" if has_negation else "病例证据可能触发该排除标准。"
        else:
            label = "not_met" if has_negation else "met"
            rationale = "病例证据是否定表达，不满足该纳入标准。" if has_negation else "病例证据支持该纳入标准。"
        return _assessment(
            criterion,
            label=label,
            confidence=0.75,
            evidence_spans=spans[:2],
            rationale=rationale,
        )

    if spans:
        return _assessment(
            criterion,
            label="met",
            confidence=0.55,
            evidence_spans=spans[:2],
            rationale="找到与该标准相关的病例证据；第一版规则将其作为支持证据，需人工复核。",
        )

    return _assessment(
        criterion,
        label="unknown",
        confidence=0.0,
        evidence_spans=[],
        rationale="病例中没有找到可判断该标准的证据。",
        missing_data=[criterion.condition or criterion.raw_text[:80]],
    )
