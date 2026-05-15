from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_PROTOCOL_TRIAL_COARSE_TOP_K = 300
DEFAULT_PROTOCOL_TRIAL_TOP_K = 50
DEFAULT_PROTOCOL_ELIGIBILITY_LIMIT = 10


def _env_int(names: tuple[str, ...], *, default: int) -> int:
    for name in names:
        raw_value = str(os.getenv(name) or "").strip()
        if not raw_value:
            continue
        try:
            return max(int(raw_value), 1)
        except ValueError:
            continue
    return default


def _env_bool(names: tuple[str, ...], *, default: bool = False) -> bool:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        normalized = str(raw_value).strip().lower()
        if not normalized:
            continue
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


@dataclass(slots=True, frozen=True)
class ProtocolGraphConfig:
    coarse_top_k: int = DEFAULT_PROTOCOL_TRIAL_COARSE_TOP_K
    fine_top_k: int = DEFAULT_PROTOCOL_TRIAL_TOP_K
    eligibility_limit: int = DEFAULT_PROTOCOL_ELIGIBILITY_LIMIT
    enable_llm_eligibility_judge: bool = False
    skip_trial_agent: bool = False
    skip_medical_knowledge_agent: bool = False
    skip_patient_calculator_evidence_agent: bool = False

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "ProtocolGraphConfig":
        payload = dict(overrides or {})
        return cls(
            coarse_top_k=max(
                int(
                    payload.get("coarse_top_k")
                    or _env_int(
                        ("MEDAI_PROTOCOL_TRIAL_COARSE_TOP_K", "MEDAI_PROTOCOL_COARSE_TOP_K"),
                        default=DEFAULT_PROTOCOL_TRIAL_COARSE_TOP_K,
                    )
                ),
                1,
            ),
            fine_top_k=max(
                int(
                    payload.get("fine_top_k")
                    or payload.get("top_k")
                    or _env_int(
                        ("MEDAI_PROTOCOL_TRIAL_TOP_K", "MEDAI_PROTOCOL_FINE_TOP_K"),
                        default=DEFAULT_PROTOCOL_TRIAL_TOP_K,
                    )
                ),
                1,
            ),
            eligibility_limit=max(
                int(
                    payload.get("eligibility_limit")
                    or _env_int(
                        ("MEDAI_PROTOCOL_ELIGIBILITY_LIMIT", "MEDAI_PROTOCOL_TRIAL_ELIGIBILITY_LIMIT"),
                        default=DEFAULT_PROTOCOL_ELIGIBILITY_LIMIT,
                    )
                ),
                1,
            ),
            enable_llm_eligibility_judge=bool(
                payload.get("enable_llm_eligibility_judge")
                or _env_bool(
                    (
                        "MEDAI_PROTOCOL_ENABLE_LLM_ELIGIBILITY_JUDGE",
                        "MEDAI_ENABLE_LLM_ELIGIBILITY_JUDGE",
                    ),
                    default=False,
                )
            ),
            skip_trial_agent=bool(
                payload.get("skip_trial_agent")
                or _env_bool(("MEDAI_PROTOCOL_SKIP_TRIAL_AGENT", "MEDAI_SKIP_TRIAL_AGENT"), default=False)
            ),
            skip_medical_knowledge_agent=bool(
                payload.get("skip_medical_knowledge_agent")
                or _env_bool(
                    ("MEDAI_PROTOCOL_SKIP_MEDICAL_KNOWLEDGE_AGENT", "MEDAI_SKIP_MEDICAL_KNOWLEDGE_AGENT"),
                    default=False,
                )
            ),
            skip_patient_calculator_evidence_agent=bool(
                payload.get("skip_patient_calculator_evidence_agent")
                or _env_bool(
                    (
                        "MEDAI_PROTOCOL_SKIP_PATIENT_CALCULATOR_EVIDENCE_AGENT",
                        "MEDAI_PROTOCOL_SKIP_RISK_EVIDENCE_AGENT",
                        "MEDAI_SKIP_PATIENT_CALCULATOR_EVIDENCE_AGENT",
                        "MEDAI_SKIP_RISK_EVIDENCE_AGENT",
                    ),
                    default=False,
                )
            ),
        )
