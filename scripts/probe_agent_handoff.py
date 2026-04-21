from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)


from agent.tools import RiskCalcRegistry  # noqa: E402


_SELECTED_PMID_PATTERN = re.compile(r"Selected Calculator PMID:\s*(?P<calc_id>[^\n]+)")
_REQUIRED_PARAMETERS_PATTERN = re.compile(
    r"Selected Calculator Required Parameters:\s*(?P<parameter_names>[^\n]+)"
)

_BOOLEAN_PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "hypertension": ("hypertension", "htn", "high blood pressure"),
    "diabetes": ("diabetes", "dm", "diabetic"),
    "stroke_history": ("stroke history", "prior stroke", "stroke", "tia", "prior tia"),
    "heart_failure": ("heart failure", "chf", "congestive heart failure"),
    "congestive_heart_failure": ("heart failure", "chf", "congestive heart failure"),
    "smoker": ("smoker", "smoking", "tobacco use"),
    "sex_female": ("female", "woman"),
    "sex_male": ("male", "man"),
}


def _extract_selected_calc_id(prompt: str) -> str | None:
    match = _SELECTED_PMID_PATTERN.search(prompt)
    if not match:
        return None
    calc_id = str(match.group("calc_id") or "").strip()
    return calc_id or None


def _extract_parameter_names(prompt: str) -> list[str]:
    match = _REQUIRED_PARAMETERS_PATTERN.search(prompt)
    if not match:
        return []
    raw_parameter_names = str(match.group("parameter_names") or "").strip()
    if not raw_parameter_names:
        return []
    return [name.strip() for name in raw_parameter_names.split(",") if name.strip()]


def _normalize_text(text: str) -> str:
    return str(text or "").strip().lower()


def _candidate_aliases(parameter_name: str) -> tuple[str, ...]:
    normalized_name = str(parameter_name or "").strip()
    if not normalized_name:
        return ()

    aliases = list(_BOOLEAN_PARAMETER_ALIASES.get(normalized_name, ()))
    spaced_name = normalized_name.replace("_", " ").replace("-", " ").strip()
    if spaced_name:
        aliases.append(spaced_name)
    aliases.append(normalized_name.lower())
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _infer_parameter_value(parameter_name: str, prompt: str) -> Any | None:
    normalized_prompt = _normalize_text(prompt)
    for alias in _candidate_aliases(parameter_name):
        if alias.lower() in normalized_prompt:
            return True
    return None


class ProbeChatClient:
    def __init__(self, registry: RiskCalcRegistry) -> None:
        self.registry = registry
        self.last_messages: list[dict[str, str]] | None = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        del model, temperature
        self.last_messages = [dict(message) for message in list(messages or [])]
        prompt = "\n\n".join(str(message.get("content") or "") for message in self.last_messages)

        calc_id = _extract_selected_calc_id(prompt)
        if calc_id and self.registry.has(calc_id):
            parameter_names = list(self.registry.get(calc_id).parameter_names)
        else:
            parameter_names = _extract_parameter_names(prompt)

        inferred_inputs: dict[str, Any] = {}
        for parameter_name in parameter_names:
            value = _infer_parameter_value(parameter_name, prompt)
            if value is not None:
                inferred_inputs[str(parameter_name)] = value

        return json.dumps({"inputs": inferred_inputs}, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(
        "ProbeChatClient is intended to be imported by tests or local debugging helpers."
    )
