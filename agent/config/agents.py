from typing import Literal

LLMType = Literal["rule_based", "basic", "reasoning", "coding"]

AGENT_LLM_MAP: dict[str, LLMType] = {
    "orchestrator": "reasoning",
    "protocol_entry": "rule_based",
    "clinical_assisstment": "reasoning",
    "calculator": "coding",
    "protocol": "reasoning",
    "reporter": "basic",
}
