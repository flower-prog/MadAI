from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


CriterionType = Literal["inclusion", "exclusion"]
CriterionLabel = Literal["met", "not_met", "unknown", "not_applicable"]
AggregateStatus = Literal[
    "likely_eligible",
    "ineligible",
    "needs_data",
    "evidence_support",
    "not_current_option",
]


@dataclass(slots=True)
class EligibilityCriterion:
    criterion_id: str
    nct_id: str
    type: CriterionType
    raw_text: str
    condition: str = ""
    operator: str = ""
    value: str = ""
    time_window: str = ""
    required_evidence_type: str = "clinical_fact"
    negation: bool = False
    parse_method: str = "rule"


@dataclass(slots=True)
class PatientEvidenceSpan:
    source: str
    text: str
    start: int | None = None
    end: int | None = None
    score: float = 0.0
    normalized_concept: str = ""
    value: str = ""
    unit: str = ""
    observed_time: str = ""


@dataclass(slots=True)
class CriterionAssessment:
    criterion_id: str
    nct_id: str
    type: CriterionType
    raw_text: str
    condition: str = ""
    operator: str = ""
    value: str = ""
    time_window: str = ""
    required_evidence_type: str = "clinical_fact"
    negation: bool = False
    label: CriterionLabel = "unknown"
    confidence: float = 0.0
    evidence_spans: list[PatientEvidenceSpan] = field(default_factory=list)
    rationale: str = ""
    missing_data: list[str] = field(default_factory=list)
    judge_method: str = "rule"


@dataclass(slots=True)
class MissingDataQuestion:
    question_id: str
    priority: Literal["high", "medium", "low"]
    question: str
    required_data: list[str] = field(default_factory=list)
    linked_criteria: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrialEligibilityAssessment:
    nct_id: str
    title: str = ""
    overall_status: str = ""
    enrollment_open: bool = False
    aggregate_status: AggregateStatus = "needs_data"
    aggregate_reason: str = ""
    criteria: list[CriterionAssessment] = field(default_factory=list)
    blocking_criteria: list[str] = field(default_factory=list)
    unknown_criteria: list[str] = field(default_factory=list)
    missing_questions: list[MissingDataQuestion] = field(default_factory=list)


def to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    return value
