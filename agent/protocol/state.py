from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProtocolGraphState:
    request: str = ""
    structured_case_json: dict[str, Any] = field(default_factory=dict)
    calculation_results: list[Any] = field(default_factory=list)
    calculator_matches: list[Any] = field(default_factory=list)
    calculation_bundle: dict[str, Any] = field(default_factory=dict)
    department_tags: list[str] = field(default_factory=list)
    trial_retriever: Any = None
    eligibility_chat_client: Any = None
    medical_phrase_parser: Any = None
    trial_query_planner: Any = None
    medical_knowledge_retriever: Any = None
    llm_model: str | None = None
    retriever_backend: str = "hybrid"
    trial_search_intent: dict[str, Any] = field(default_factory=dict)
    patient_evidence_bundle: dict[str, Any] = field(default_factory=dict)
    trial_retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    eligibility_assessment_bundle: dict[str, Any] = field(default_factory=dict)
    calculator_evidence_bundle: dict[str, Any] = field(default_factory=dict)
    medical_phrase_bundle: dict[str, Any] = field(default_factory=dict)
    medical_knowledge_bundle: dict[str, Any] = field(default_factory=dict)
    missing_data_bundle: dict[str, Any] = field(default_factory=dict)
    protocol_decision_bundle: dict[str, Any] = field(default_factory=dict)
