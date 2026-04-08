from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agent.config import AGENT_LLM_MAP, TEAM_MEMBERS


PROMPT_DIR = Path(__file__).resolve().parent


def get_agent_prompt_path(agent_name: str) -> Path:
    normalized = str(agent_name).strip()
    prompt_path = PROMPT_DIR / f"{normalized}.md"
    if prompt_path.exists():
        return prompt_path
    if normalized in TEAM_MEMBERS or normalized in AGENT_LLM_MAP:
        raise KeyError(f"Prompt file is missing for agent: {agent_name!r}")
    raise KeyError(f"Unknown agent prompt requested: {agent_name!r}")


def load_agent_prompt(agent_name: str) -> str:
    prompt_path = get_agent_prompt_path(agent_name)
    return prompt_path.read_text(encoding="utf-8").strip()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def render_agent_prompt(agent_name: str, *, context: dict[str, Any] | None = None) -> str:
    prompt_text = load_agent_prompt(agent_name)
    if not context:
        return prompt_text

    context_block = json.dumps(_json_ready(context), ensure_ascii=False, indent=2, sort_keys=True)
    context_block = context_block.replace("]]>", "]]]]><![CDATA[>")
    return (
        f"{prompt_text}\n\n"
        "<runtime_context>\n"
        "  <instruction>Use the following runtime context as the only factual input for this step. If fields are missing, say so explicitly instead of inventing data.</instruction>\n"
        "  <json_payload><![CDATA[\n"
        f"{context_block}\n"
        "]]></json_payload>\n"
        "</runtime_context>"
    )


def build_agent_run_spec(agent_name: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt_path = get_agent_prompt_path(agent_name)
    return {
        "agent_name": agent_name,
        "llm_type": AGENT_LLM_MAP.get(agent_name, "basic"),
        "prompt_path": str(prompt_path),
        "system_prompt": render_agent_prompt(agent_name, context=context),
    }


def build_all_agent_run_specs(
    *, context_by_agent: dict[str, dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    context_by_agent = context_by_agent or {}
    return {
        agent_name: build_agent_run_spec(agent_name, context=context_by_agent.get(agent_name))
        for agent_name in TEAM_MEMBERS
    }
