from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import CriterionAssessment, MissingDataQuestion, TrialEligibilityAssessment


_NOT_CURRENT_STATUSES = {
    "abandoned",
    "terminated",
    "withdrawn",
    "suspended",
    "no longer available",
    "temporarily not available",
}
_EVIDENCE_SUPPORT_STATUSES = {
    "completed",
    "active, not recruiting",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def aggregate_trial_eligibility(
    trial_record: Mapping[str, Any],
    assessments: list[CriterionAssessment],
    *,
    missing_questions: list[MissingDataQuestion] | None = None,
) -> TrialEligibilityAssessment:
    record = dict(trial_record or {})
    status = _normalize_text(record.get("status")).casefold()
    overall_status = _normalize_text(record.get("overall_status"))
    overall_status_key = overall_status.casefold()
    enrollment_open = bool(record.get("enrollment_open"))

    blocking: list[str] = []
    unknown: list[str] = []
    for item in list(assessments or []):
        if item.type == "exclusion" and item.label == "met":
            blocking.append(item.criterion_id)
        if item.type == "inclusion" and item.label == "not_met":
            blocking.append(item.criterion_id)
        if item.label == "unknown":
            unknown.append(item.criterion_id)

    aggregate_status = "needs_data" if not assessments else "likely_eligible"
    if status in _NOT_CURRENT_STATUSES or overall_status_key in _NOT_CURRENT_STATUSES:
        aggregate_status = "not_current_option"
    elif overall_status_key in _EVIDENCE_SUPPORT_STATUSES or (overall_status and not enrollment_open):
        aggregate_status = "evidence_support"
    elif blocking:
        aggregate_status = "ineligible"
    elif unknown:
        aggregate_status = "needs_data"

    if aggregate_status == "not_current_option":
        reason = "试验当前状态不适合作为入组选项。"
    elif aggregate_status == "evidence_support":
        reason = "试验可能不能作为当前入组选项，但资格评估结果可作为证据支持参考。"
    elif aggregate_status == "ineligible":
        reason = "存在明确不满足的纳入标准或已触发的排除标准。"
    elif aggregate_status == "needs_data":
        reason = (
            "未能解析出可逐条判断的入排标准。"
            if not assessments
            else "未发现明确阻断标准，但仍有关键入排标准缺少病例证据。"
        )
    else:
        reason = "主要可解析标准均有支持证据，且未发现明确排除标准。"

    title = _normalize_text(
        record.get("title")
        or record.get("display_title")
        or record.get("brief_title")
        or record.get("official_title")
        or record.get("name")
    )
    return TrialEligibilityAssessment(
        nct_id=_normalize_text(record.get("nct_id")),
        title=title,
        overall_status=overall_status,
        enrollment_open=enrollment_open,
        aggregate_status=aggregate_status,  # type: ignore[arg-type]
        aggregate_reason=reason,
        criteria=list(assessments),
        blocking_criteria=blocking,
        unknown_criteria=unknown,
        missing_questions=list(missing_questions or []),
    )
