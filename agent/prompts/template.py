import os
import re
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState


_SKILLS_ONLY_TOOL_MODES = {"skills_test", "skills_only", "skills"}
_FULL_ACCESS_TOOL_MODES = {"full_access"}


def _resolve_tools_mode(raw_mode: str | None) -> str:
    """Normalize legacy aliases to: skills_only | full_access | full."""
    mode = str(raw_mode or "").strip().lower()
    if mode in _SKILLS_ONLY_TOOL_MODES:
        return "skills_only"
    if mode in _FULL_ACCESS_TOOL_MODES:
        return "full_access"
    return "full"


def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    # 使用反斜杠语义转义花括号
    template = template.replace("{", "{{").replace("}", "}}")
    # 将 `<<VAR>>` 替换为 `{VAR}`
    template = re.sub(r"<<([^>>]+)>>", r"{\1}", template)
    return template


def apply_prompt_template(prompt_name: str, state: AgentState) -> list:
    from src.config import TEAM_MEMBERS, SKILLS_PROMPT_TARGETS
    # Import TEAM_MEMBERS for default value
    
    # 从 state 获取 TEAM_MEMBERS，若缺失则使用默认值；并将列表转换为模板可用字符串
    team_members = state.get("TEAM_MEMBERS", TEAM_MEMBERS)
    if isinstance(team_members, list):
        team_members_str = ", ".join(team_members)
    else:
        team_members_str = str(team_members)
    
    # 为可能缺失的模板变量提供默认值
    template_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        "deep_thinking_mode": state.get("deep_thinking_mode", False),
        "search_before_planning": state.get("search_before_planning", False),
        "thread_id": state.get("thread_id", "default"),
        "TEAM_MEMBERS": team_members_str,  # 将列表转换为模板字符串
        "RAG_SCOPE_SUMMARY": state.get("RAG_SCOPE_SUMMARY", ""),
        "PLANNER_FILE_CONTEXT": state.get("PLANNER_FILE_CONTEXT", "(none)"),
    }
    # 将state中的所有字段也添加到template_vars，但不覆盖已设置的默认值
    for key, value in state.items():
        if key not in template_vars:
            template_vars[key] = value

    system_prompt = PromptTemplate.from_template(
        get_prompt_template(prompt_name)
    ).format(**template_vars)

    # Append skills system block if configured for this prompt
    targets = []
    if SKILLS_PROMPT_TARGETS:
        targets = [
            t.strip().lower()
            for t in str(SKILLS_PROMPT_TARGETS).split(",")
            if t.strip()
        ]

    prompt_key = str(prompt_name).strip().lower()
    expert_tools_mode = "full"
    coder_tools_mode = "full"
    try:
        from src.config import EXPERT_TOOLS_MODE, CODER_TOOLS_MODE

        expert_tools_mode = _resolve_tools_mode(EXPERT_TOOLS_MODE)
        coder_tools_mode = _resolve_tools_mode(CODER_TOOLS_MODE)
    except Exception:
        pass

    expert_skill_enabled = expert_tools_mode in {"skills_only", "full_access"}
    coder_skill_enabled = coder_tools_mode in {"skills_only", "full_access"}
    coder_skills_only = coder_tools_mode == "skills_only"

    include_skills_section = bool(targets and ("all" in targets or prompt_key in targets))
    if prompt_key == "expert" and expert_skill_enabled:
        include_skills_section = True
    if prompt_key == "coder" and coder_skill_enabled:
        include_skills_section = True

    if include_skills_section:
        try:
            from src.skills.runtime import build_skills_section

            read_tool = "read_data"
            if (
                prompt_key == "expert" and expert_skill_enabled
            ) or (
                prompt_key == "coder" and coder_skill_enabled
            ):
                read_tool = "read_skill"
            skills_section = build_skills_section(read_tool=read_tool)
        except Exception:
            skills_section = ""
        if skills_section:
            system_prompt = f"{system_prompt}\n\n{skills_section}"

    # In coder skills mode, override direct-tool instructions to avoid conflicts
    # with skill-only tool exposure (`read_skill` + `skill_call`).
    if prompt_key == "coder" and coder_skills_only:
        coder_skills_override = """
<coder_skills_mode_override>
  <rule priority="critical">You are running in CODER SKILLS MODE.</rule>
  <rule priority="critical">The only callable tools are `read_skill` and `skill_call`.</rule>
  <rule priority="critical">Any direct tool mention in this prompt (for example `create_sandbox`, `execute_in_sandbox`, `list_dir`, `file_info`, `read_data`) MUST be executed via `skill_call`.</rule>
  <rule priority="critical">Do NOT fail just because direct tool names are unavailable.</rule>
  <rule priority="critical">If a required tool is not whitelisted by any available skill, report `STEP_FAILED` with the missing tool/skill name.</rule>
  <rule>Directory path should use `list_dir`; `file_info` is for files only.</rule>
</coder_skills_mode_override>
""".strip()
        system_prompt = f"{system_prompt}\n\n{coder_skills_override}"
    return [{"role": "system", "content": system_prompt}] + state["messages"]
