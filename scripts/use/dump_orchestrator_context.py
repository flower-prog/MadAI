from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.corpus_paths import resolve_default_corpus_paths
from agent.graph.nodes import (
    _build_default_plan,
    _build_orchestrator_query,
    _build_orchestrator_prompt_context,
    _default_tool_specs,
)
from agent.graph.types import GraphState, ensure_state
from agent.prompt import build_agent_run_spec
from agent.workflow import _build_clinical_tool_job_payload


DEFAULT_CASE_TEXT = (
    "A 78-year-old male patient presents to your clinic for a routine check-up. "
    "He has a history of hypertension and recently experienced a transient ischemic attack. "
    "He does not have diabetes or congestive heart failure. He is not currently prescribed warfarin. "
    "Based on these factors, what is the estimated stroke rate per 100 patient-years without antithrombotic therapy for this patient?"
)
DEFAULT_MODE = "question"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "上下文"


FIELD_EXPLANATIONS: dict[str, str] = {
    "request": "工作流级请求，不是病例原文。run_workflow 默认会把它设成 'Run MedAI clinical tool workflow in {mode} mode.'。",
    "messages": "对话消息历史。这里第一条 user message 就是原始病例问题文本。",
    "patient_case": "结构化病例入口；这个问题场景下默认没有额外 patient_case，因此通常是 null。",
    "clinical_tool_job": "临床工具任务配置，包含 mode/text/case_summary/structured_case/检索参数/语料路径等。",
    "reporter_feedback": "reporter 回退给 orchestrator 的阻塞反馈。首轮执行时通常为空。",
    "workflow": "固定主链路字符串：orchestrator -> clinical_assisstment -> protocol -> reporter。",
    "workflow_roles": "四个顶层节点各自的角色标签。",
    "department": "orchestrator 分类前的主科室。首轮进入时一般为空字符串，等待 orchestrator 判定。",
    "department_tags": "orchestrator 分类前的科室标签列表。首轮进入时一般为空列表。",
    "department_tag_library": "允许 orchestrator 选择的预定义科室标签库。",
    "deep_thinking_mode": "工作流深度推理开关，默认 True。",
    "tools": "在 orchestrator 被调用前，graph 节点已经挂入的工具契约列表。",
}


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


def _build_state(case_text: str, *, mode: str) -> GraphState:
    riskcalcs_path, pmid_metadata_path = resolve_default_corpus_paths(PROJECT_ROOT)
    clinical_tool_job = _build_clinical_tool_job_payload(
        case_text,
        mode=mode,
        riskcalcs_path=str(riskcalcs_path) if riskcalcs_path is not None else None,
        pmid_metadata_path=str(pmid_metadata_path) if pmid_metadata_path is not None else None,
    )
    request = f"Run MedAI clinical tool workflow in {mode} mode."
    state = ensure_state(
        {
            "request": request,
            "messages": [{"role": "user", "content": case_text}],
            "clinical_tool_job": clinical_tool_job,
        }
    )
    if not state.plan:
        state.plan = _build_default_plan()
    if not state.tools:
        state.tools = _default_tool_specs()
    return state


def _build_context_payload(case_text: str, *, mode: str) -> dict[str, Any]:
    state = _build_state(case_text, mode=mode)
    context = _build_orchestrator_prompt_context(state)
    query = _build_orchestrator_query(state)
    prompt_spec = build_agent_run_spec("orchestrator")

    context_json = _json_ready(context)
    tool_names = [str(item.get("name") or "").strip() for item in list(context_json.get("tools") or [])]
    clinical_tool_job = dict(context_json.get("clinical_tool_job") or {})
    messages = list(context_json.get("messages") or [])

    parsed = {
        "summary": {
            "mode": clinical_tool_job.get("mode"),
            "request": context_json.get("request"),
            "user_message_count": len(messages),
            "query_line_count": len(query.splitlines()),
            "tool_count": len(tool_names),
            "tool_names": tool_names,
            "department_before_orchestrator": context_json.get("department"),
            "department_tags_before_orchestrator": list(context_json.get("department_tags") or []),
            "reporter_feedback_count": len(list(context_json.get("reporter_feedback") or [])),
            "riskcalcs_path": clinical_tool_job.get("riskcalcs_path"),
            "pmid_metadata_path": clinical_tool_job.get("pmid_metadata_path"),
        },
        "observations": [
            "真正发给 orchestrator 的 user content 是纯文本 query，不是 context_json。",
            "request、content、mode、reporter_feedback 会被展开到 query 文本里。",
            "system prompt 现在只保留规则定义，不再额外内嵌 <runtime_context> JSON。",
            "病例文本同时出现在 messages[0].content 和 clinical_tool_job.text 两个位置，但实际 user content 会被整理成单段 query 文本。",
            "department 和 department_tags 在 orchestrator 分类前还是空值，分类责任留给 orchestrator。",
            "tools 依然属于图内执行契约，但不会作为 JSON 原样发给 orchestrator。",
        ],
        "field_explanations": {
            key: {
                "meaning": FIELD_EXPLANATIONS.get(key, ""),
                "value": context_json.get(key),
            }
            for key in context_json.keys()
        },
    }

    return {
        "case_text": case_text,
        "mode": mode,
        "query": query,
        "state_debug": context_json,
        "parsed": parsed,
        "prompt_spec": {
            "agent_name": prompt_spec["agent_name"],
            "llm_type": prompt_spec["llm_type"],
            "prompt_path": prompt_spec["prompt_path"],
        },
        "system_prompt": prompt_spec["system_prompt"],
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    parsed = dict(payload.get("parsed") or {})
    summary = dict(parsed.get("summary") or {})
    prompt_spec = dict(payload.get("prompt_spec") or {})

    lines = [
        "# Orchestrator 上下文解析",
        "",
        "## 病例",
        "",
        payload.get("case_text") or "",
        "",
        "## 结论摘要",
        "",
        f"- mode: `{summary.get('mode')}`",
        f"- request: `{summary.get('request')}`",
        f"- user_message_count: `{summary.get('user_message_count')}`",
        f"- query_line_count: `{summary.get('query_line_count')}`",
        f"- tool_count: `{summary.get('tool_count')}`",
        f"- department_before_orchestrator: `{summary.get('department_before_orchestrator')}`",
        f"- department_tags_before_orchestrator: `{json.dumps(summary.get('department_tags_before_orchestrator') or [], ensure_ascii=False)}`",
        f"- reporter_feedback_count: `{summary.get('reporter_feedback_count')}`",
        "",
        "## 实际 LLM 输入",
        "",
        "### user content（query）",
        "",
        "```text",
        str(payload.get("query") or ""),
        "```",
        "",
        "### system prompt",
        "",
        "```xml",
        str(payload.get("system_prompt") or ""),
        "```",
        "",
        "## 关键信息",
        "",
        "- 真正发给 LLM 的 `user.content` 是一段纯文本 query，不再是 `runtime_context JSON`。",
        "- `request` 不是病例原文，而是 workflow 入口包装语；病例正文会出现在 query 的 `content:` 段落里。",
        "- `patient_case` 在这个场景下为空，说明 orchestrator 主要依赖 question-mode 的 `clinical_tool_job` 和原始病例文本。",
        "- 科室尚未判定，所以 `department` 和 `department_tags` 在调用前为空。",
        "- tools 仍然存在于 graph 状态里，但现在只作为调试视图保留，不直接塞给 orchestrator。",
        "",
        "## Prompt 信息",
        "",
        f"- prompt_path: `{prompt_spec.get('prompt_path')}`",
        f"- llm_type: `{prompt_spec.get('llm_type')}`",
        "",
        "## 状态调试视图（不是实际发给 LLM 的内容）",
        "",
    ]

    field_explanations = dict(parsed.get("field_explanations") or {})
    for key, item in field_explanations.items():
        explanation = dict(item or {})
        lines.append(f"### {key}")
        lines.append("")
        lines.append(explanation.get("meaning") or "")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(explanation.get("value"), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    query_path = output_dir / "orchestrator_query.txt"
    debug_context_path = output_dir / "orchestrator_context_debug.json"
    parsed_path = output_dir / "orchestrator_context_parsed.json"
    prompt_path = output_dir / "orchestrator_system_prompt.txt"
    markdown_path = output_dir / "orchestrator_context_readme.md"

    query_path.write_text(str(payload.get("query") or ""), encoding="utf-8")
    debug_context_path.write_text(
        json.dumps(payload.get("state_debug"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    parsed_path.write_text(
        json.dumps(
            {
                "case_text": payload.get("case_text"),
                "mode": payload.get("mode"),
                "query": payload.get("query"),
                "prompt_spec": payload.get("prompt_spec"),
                "parsed": payload.get("parsed"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    prompt_path.write_text(str(payload.get("system_prompt") or ""), encoding="utf-8")
    markdown_path.write_text(_build_markdown(payload), encoding="utf-8")

    return {
        "query": str(query_path),
        "state_debug": str(debug_context_path),
        "parsed": str(parsed_path),
        "system_prompt": str(prompt_path),
        "readme": str(markdown_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump the real MedAI orchestrator query and system prompt for a single case.")
    parser.add_argument("--case-text", default=DEFAULT_CASE_TEXT, help="Clinical case/question text to inspect.")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=("question", "patient_note"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated context files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_text = str(args.case_text or "").strip()
    if not case_text:
        raise ValueError("`--case-text` could not be empty.")

    payload = _build_context_payload(case_text, mode=str(args.mode))
    outputs = _write_outputs(payload, Path(args.output_dir).expanduser())
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
