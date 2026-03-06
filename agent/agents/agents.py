from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    browser_tool,
    crawl_tool,
    searxng_tool,
    ncbi_search_tool,
    rag_tool,
    remember_tool,
    search_memory_tool,
    load_memory_tool,
    estimate_tokens_tool,
    summarize_and_remember_tool,
    budget_guard_tool,
    create_sandbox,
    execute_in_sandbox,
    remove_sandbox,
    bioc_advisor_tool,
    BIO_DB_TOOLS,
    # 文件检查工具
    list_dir,
    file_info,
    read_data,
    peek_file_tool,
    infer_table_schema,
    check_sample_overlap,
)
from src.tools.skill_reader import read_skill
from src.tools.skill_call import skill_call

from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.config import EXPERT_TOOLS_MODE, CODER_TOOLS_MODE
from src.graph.types import State

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


def _merge_tools(*tool_groups):
    merged = []
    seen = set()
    for group in tool_groups:
        for tool_obj in group:
            key = getattr(tool_obj, "name", None) or id(tool_obj)
            if key in seen:
                continue
            seen.add(key)
            merged.append(tool_obj)
    return merged

# Create agents using configured LLM types
research_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["researcher"]),
    tools=[
        searxng_tool,
        crawl_tool,
        remember_tool,
        search_memory_tool,
        load_memory_tool,
        ncbi_search_tool,
        *BIO_DB_TOOLS,
    ],
    prompt=lambda state: apply_prompt_template("researcher", state),
    state_schema=State,
)

coder_global_tools = [
    create_sandbox,
    execute_in_sandbox,
    remove_sandbox,
    remember_tool,
    search_memory_tool,
    load_memory_tool,       # 按 id 读取记忆（压缩恢复必需）
    estimate_tokens_tool,
    summarize_and_remember_tool,
    budget_guard_tool,
    bioc_advisor_tool,
    # 数据探索工具（帮助 coder 理解数据）
    list_dir,               # 目录浏览
    file_info,              # 文件元信息
    read_data,              # 数据读取
    infer_table_schema,     # 表格结构推断（分隔符/列数/header）
    check_sample_overlap,   # matrix vs metadata 样本交集检查
]
coder_skill_tools = [read_skill, skill_call]

coder_tools_mode = _resolve_tools_mode(CODER_TOOLS_MODE)
if coder_tools_mode == "skills_only":
    # Skill mode: expose only skill reader + skill executor.
    coder_tools = [read_skill, skill_call]
elif coder_tools_mode == "full_access":
    # Full access: allow both skill tools and global tools.
    coder_tools = _merge_tools(coder_skill_tools, coder_global_tools)
else:
    # Full mode: global tools only (legacy default behavior).
    coder_tools = coder_global_tools

coder_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["coder"]),
    tools=coder_tools,
    prompt=lambda state: apply_prompt_template("coder", state),
    state_schema=State,
)

browser_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["browser"]),
    tools=[
        browser_tool
        ],
    prompt=lambda state: apply_prompt_template("browser", state),
    state_schema=State,
)

expert_global_tools = [
    rag_tool,               # 本地 RAG 知识库（支持 mode='simple' 和 mode='deep'）
    search_memory_tool,     # 搜索记忆
    load_memory_tool,       # 按 id 读取记忆（压缩恢复必需）
    # 数据探索工具（从宏观到微观）
    list_dir,               # 目录浏览：文件树、类型统计
    file_info,              # 文件元信息：大小、行数、格式
    read_data,              # 数据读取：支持任意位置、多种格式
    infer_table_schema,     # 表格结构推断
    check_sample_overlap,   # 样本对齐检查
]
expert_skill_tools = [read_skill, skill_call]

expert_tools_mode = _resolve_tools_mode(EXPERT_TOOLS_MODE)
if expert_tools_mode == "skills_only":
    # Skill mode: expose only skill reader + skill executor.
    expert_tools = [read_skill, skill_call]
elif expert_tools_mode == "full_access":
    # Full access: allow both skill tools and global tools.
    expert_tools = _merge_tools(expert_skill_tools, expert_global_tools)
else:
    # Full mode: global tools only (legacy default behavior).
    expert_tools = expert_global_tools

expert_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["expert"]),
    tools=expert_tools,
    prompt=lambda state: apply_prompt_template("expert", state),
    state_schema=State,
)
