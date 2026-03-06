import logging
import math
import json
import re
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Literal, List, Dict, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt
from langgraph.graph import END

from src.agents import research_agent, coder_agent, browser_agent, expert_agent
from src.agents.llm import (
    get_compression_llm,
    get_llm_by_type,
    estimate_context_usage_for_llm_type,
)
from src.config import TEAM_MEMBERS, EXPERT_TOOLS_MODE, SKILLS_ENABLED, SKILLS_PROMPT_TARGETS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.skills.runtime import build_skills_section
from src.skills.tool_resolver import (
    get_skill_tool_names,
    resolve_skill_entry,
)
from src.tools.search import searxng_tool
from .types import State, Router
from ..utils.json_utils import repair_json_output

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
_INTERNAL_MESSAGE_NAMES = set(TEAM_MEMBERS) | {
    "planner",
    "coordinator",
    "expert",
    "context_index",
    "context_summary",
    "plan_summary",
    "system",
}
_ROUTER_ALLOWLIST = set(TEAM_MEMBERS) | {"FINISH"}
_STEP_STATUS_VALUES = {"pending", "in_progress", "completed", "failed"}

_BASE_EXPERT_TOOL_NAMES = [
    "rag_tool",
    "ncbi_search_tool",
    "search_memory_tool",
    "load_memory_tool",
]

_SKILLS_DYNAMIC_MODES = {"skills_test", "skills_only", "skills", "skills_dynamic", "skills_graph"}


def _skills_dynamic_enabled() -> bool:
    if not SKILLS_ENABLED:
        return False
    mode = str(EXPERT_TOOLS_MODE).strip().lower()
    return mode in _SKILLS_DYNAMIC_MODES


def _strip_skills_system_block(text: str) -> str:
    if not text or "<skills_system>" not in text:
        return text
    return re.sub(r"\n?<skills_system>.*?</skills_system>\n?", "\n", text, flags=re.DOTALL).strip()


def _build_skill_selector_prompt(state: State) -> list:
    messages = apply_prompt_template("expert_skill_select", state)
    if not SKILLS_ENABLED:
        return messages

    skills_section = build_skills_section(read_tool="read_skill")
    if not skills_section:
        return messages

    if messages:
        first = messages[0]
        if isinstance(first, dict):
            content = first.get("content", "")
            if "<skills_system>" not in content:
                first["content"] = f"{content}\n\n{skills_section}"
        else:
            content = getattr(first, "content", "")
            if "<skills_system>" not in content:
                setattr(first, "content", f"{content}\n\n{skills_section}")

    return messages


def _extract_selected_skill(messages: list) -> Optional[str]:
    for msg in messages or []:
        tool_calls = None
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls") or (msg.get("additional_kwargs") or {}).get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                tool_calls = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls")

        if not tool_calls:
            continue

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args")
            if not name and isinstance(call.get("function"), dict):
                func = call.get("function", {})
                name = func.get("name")
                args = func.get("arguments")

            if name != "read_skill":
                continue

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if isinstance(args, dict):
                selected = args.get("name") or args.get("skill") or args.get("path")
                if selected:
                    return str(selected).strip()

    # Fallback: parse from last assistant text
    last_text = None
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            last_text = last.get("content")
        else:
            last_text = getattr(last, "content", None)

    if isinstance(last_text, str):
        match = re.search(r"SKILL_SELECTED\s*:\s*(.+)", last_text, flags=re.IGNORECASE)
        if match:
            selected = match.group(1).strip()
            if selected.lower() in {"none", "no", "null", "n/a"}:
                return None
            return selected

    return None


def _extract_skill_reason(messages: list) -> Optional[str]:
    last_text = None
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            last_text = last.get("content")
        else:
            last_text = getattr(last, "content", None)

    if isinstance(last_text, str):
        match = re.search(r"SKILL_REASON\s*:\s*(.+)", last_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _normalize_skill_key(skill_key: Optional[str]) -> Optional[str]:
    entry = resolve_skill_entry(skill_key)
    if entry:
        return entry.skill.name
    if isinstance(skill_key, str):
        return skill_key.strip() or None
    return None


def _build_expert_prompt_without_skills(state: State) -> list:
    messages = apply_prompt_template("expert", state)
    if messages:
        first = messages[0]
        if isinstance(first, dict):
            content = first.get("content", "")
            first["content"] = _strip_skills_system_block(content)
        else:
            content = getattr(first, "content", "")
            setattr(first, "content", _strip_skills_system_block(content))
    return messages


def _ensure_skills_section_in_messages(messages: list, read_tool: str = "read_skill") -> None:
    """Ensure the skills system block is present in the message list."""
    if not messages:
        return
    targets = []
    if SKILLS_PROMPT_TARGETS:
        targets = [
            t.strip().lower()
            for t in str(SKILLS_PROMPT_TARGETS).split(",")
            if t.strip()
        ]
    if targets and ("all" in targets or "expert" in targets):
        return
    for msg in messages:
        content = _message_content(msg)
        if "<skills_system>" in content:
            return

    skills_section = build_skills_section(read_tool=read_tool)
    if skills_section:
        messages.insert(0, {"role": "system", "content": skills_section})


def _format_skill_selection_system_message(
    selected_skill: Optional[str],
    reason: Optional[str] = None,
    tools: Optional[list[str]] = None,
) -> str:
    """Build a system message describing the preselected skill and how to use it."""
    selected = selected_skill or "none"
    lines = [
        "<skill_selection>",
        f"  <selected>{selected}</selected>",
    ]
    if reason:
        lines.append(f"  <reason>{reason}</reason>")
    if tools:
        lines.append(f"  <allowed_tools>{', '.join(tools)}</allowed_tools>")
    lines.append("  <rules>")
    if selected_skill:
        lines.append("    <rule>Use ONLY the selected skill; do not re-select.</rule>")
        lines.append("    <rule>Call `read_skill` with the selected skill before any other tool use.</rule>")
        lines.append("    <rule>Execute skill tools via `skill_call` with: skill=<selected>, tool=<tool name>, args=<dict>.</rule>")
        lines.append("    <rule>Do NOT call tools directly.</rule>")
    else:
        lines.append("    <rule>No skill selected: do NOT call `read_skill` or `skill_call` unless explicitly instructed.</rule>")
    lines.append("  </rules>")
    lines.append("</skill_selection>")
    return "\n".join(lines)


def _select_skill_for_expert(
    state: State,
    base_messages: list,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Run the skill selector prompt and return (skill, reason, allowed_tools)."""
    if not _skills_dynamic_enabled():
        return None, None, []

    temp_state = state.copy()
    temp_state["messages"] = base_messages
    messages = _build_skill_selector_prompt(temp_state)

    llm = get_llm_by_type(AGENT_LLM_MAP["expert"])
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.exception("Skill selector failed: %s", exc)
        return None, None, []

    selected_raw = _extract_selected_skill([response])
    reason = _extract_skill_reason([response])
    selected = _normalize_skill_key(selected_raw)

    if selected:
        entry = resolve_skill_entry(selected)
        if not entry:
            logger.warning("Skill selector chose unknown skill: %s", selected_raw)
            selected = None

    tools = get_skill_tool_names(selected) if selected else []
    return selected, reason, tools


def expert_skill_select_node(state: State) -> Command[Literal["expert"]]:
    """Select a skill for expert usage without adding a separate agent node."""
    if not _skills_dynamic_enabled():
        return Command(goto="expert")

    original_messages = state.get("messages", [])
    compressed_msgs, updated_index, was_compressed = _compress_context(original_messages, state)

    filtered_messages = []
    user_msg = _get_first_user_message(compressed_msgs)
    if user_msg is not None:
        filtered_messages.append(user_msg)

    for m in compressed_msgs:
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ("context_index", "context_summary"):
            if m not in filtered_messages:
                filtered_messages.append(m)

    file_context = state.get("initial_file_context")
    if file_context:
        file_context = _clamp_context_block(file_context, max_chars=1500, max_lines=100)
        filtered_messages.append(
            HumanMessage(
                content=(
                    "[AVAILABLE DATA FILES IN ./]\n"
                    f"{file_context}\n"
                    "(You can read these files directly using their filenames)"
                ),
                name="system",
            )
        )

    selected_skill, reason, tools = _select_skill_for_expert(state, filtered_messages)

    updates: dict[str, Any] = {
        "selected_skill": selected_skill,
        "selected_skill_reason": reason,
        "selected_skill_tools": tools,
    }
    if was_compressed:
        updates["context_index"] = updated_index
        updates["compression_count"] = int(state.get("compression_count") or 0) + 1

    return Command(update=updates, goto="expert")

# ---------------------------------------------------------------------------
# Context window budget constants
# ---------------------------------------------------------------------------
# Legacy fallback when model-aware token estimation fails.
_FALLBACK_CONTEXT_WINDOW_CHARS = int(os.getenv("BIOINFO_CHAT_CONTEXT_CHARS", "800000"))
_FALLBACK_COMPRESS_THRESHOLD = float(os.getenv("BIOINFO_COMPRESS_THRESHOLD", "0.80"))
_PLANNER_SEARCH_QUERY_MAX_CHARS = int(
    os.getenv("BIOINFO_PLANNER_SEARCH_QUERY_MAX_CHARS", "400")
)
_PLANNER_FILE_CONTEXT_MAX_CHARS = int(
    os.getenv("BIOINFO_PLANNER_FILE_CONTEXT_MAX_CHARS", "3000")
)
_PLANNER_FILE_CONTEXT_MAX_LINES = int(
    os.getenv("BIOINFO_PLANNER_FILE_CONTEXT_MAX_LINES", "160")
)
_CODER_FILE_CONTEXT_MAX_CHARS = int(
    os.getenv("BIOINFO_CODER_FILE_CONTEXT_MAX_CHARS", "1200")
)
_CODER_FILE_CONTEXT_MAX_LINES = int(
    os.getenv("BIOINFO_CODER_FILE_CONTEXT_MAX_LINES", "80")
)
_SMALL_TALK_PATTERNS = (
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bhow are you\b",
    r"\bthanks?\b",
    r"\bthank you\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"你好",
    r"您好",
    r"嗨",
    r"谢谢",
    r"早上好",
    r"下午好",
    r"晚上好",
)
_TASK_HINT_PATTERNS = (
    r"\banaly(sis|ze)\b",
    r"\bcompute\b",
    r"\bcalculate\b",
    r"\bplot\b",
    r"\bcsv\b",
    r"\btsv\b",
    r"\bxlsx?\b",
    r"\bfile\b",
    r"\bdataset\b",
    r"\brna\b",
    r"\bgene\b",
    r"\bprotein\b",
    r"分析",
    r"计算",
    r"数据",
    r"文件",
    r"绘图",
    r"基因",
    r"蛋白",
)
_CODER_STEP_FAILED_RETRY_MAX_ATTEMPTS = int(
    os.getenv("BIOINFO_CODER_STEP_FAILED_RETRY_MAX_ATTEMPTS", "-1")
)
_CODER_VALIDATION_MAX_ATTEMPTS = int(
    os.getenv("BIOINFO_CODER_VALIDATION_MAX_ATTEMPTS", "-1")
)
_CODER_RETRY_MIN_REMAINING_STEPS = int(
    os.getenv("BIOINFO_CODER_RETRY_MIN_REMAINING_STEPS", "4")
)
_RECOVERABLE_STEP_FAILED_HINTS = (
    "parse",
    "parser",
    "delimiter",
    "single column",
    "one column",
    "dtype",
    "object",
    "rows: 0",
    "0 rows",
    "zero rows",
    "no rows left",
    "no rows remained",
    "no analyzable",
    "zero-size",
    "empty",
    "dropna",
    "encoding",
    "coerce",
    "not numeric",
    "factor",
    "covariate",
    "mismatch",
    "bitr",
    "entrez",
    "ensembl",
    "no overlapping samples",
    "all samples have 0 counts",
    "null",
    "no gene can be mapped",
    "return null",
    "no enriched",
    "go enrichment returned null",
    "missing column",
    "column missing",
    "column not found",
    "not in columns",
    "keyerror",
    "aesev missing",
    "malformed",
    "syntax",
    "parse error",
    "unexpected symbol",
    "unexpected input",
    "quoting",
    "quote",
    "variable substitution",
    "invalid code",
    "解析",
    "分隔",
    "单列",
    "交集",
    "样本数=0",
    "全为0",
    "转换",
    "无富集结果",
    "无法富集",
    "缺少列",
    "列不存在",
    "列缺失",
    "语法",
    "引号",
    "变量替换",
    "代码错误",
    "报错",
    "无法完成",
)
_STRUCTURE_PROBE_TOOL_NAMES = ("infer_table_schema", "check_sample_overlap")
_STRUCTURE_MODEL_MARKERS = (
    "deseq",
    "edger",
    "limma",
    "differential expression",
    "deg",
    "enrichgo",
    "clusterprofiler",
    "差异表达",
    "富集",
)
_STRUCTURE_MATRIX_MARKERS = (
    "count matrix",
    "count data",
    "featurecounts",
    "gene counts",
    "counts",
    "matrix",
    "计数矩阵",
    "表达矩阵",
)
_STRUCTURE_METADATA_MARKERS = (
    "metadata",
    "meta data",
    "coldata",
    "sample",
    "condition",
    "样本",
    "分组",
)


def _build_local_rag_scope_summary(*, max_topics: int = 20) -> str:
    """Build a concise local RAG coverage summary for planner prompt injection."""
    try:
        default_docs_dir = (Path(__file__).resolve().parent.parent / "RAG" / "retrieve_docs").resolve()
    except Exception:
        default_docs_dir = (Path(os.getcwd()) / "src" / "RAG" / "retrieve_docs").resolve()

    docs_dir = Path(os.getenv("BIOINFO_RAG_DOCS_DIR", str(default_docs_dir))).resolve()
    default_topic_file = (Path(__file__).resolve().parent.parent / "rag_system" / "topic_descriptions.json").resolve()
    custom_topic_file = docs_dir / "topic_descriptions.json"
    topic_file = custom_topic_file if custom_topic_file.exists() else default_topic_file

    topic_lines: list[str] = []
    topic_count = 0

    if topic_file.exists():
        try:
            payload = json.loads(topic_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if str(key).startswith("_"):
                        continue
                    if not isinstance(value, dict):
                        continue
                    desc = str(value.get("description") or "").strip()
                    topic_count += 1
                    if len(topic_lines) < max_topics:
                        if desc:
                            topic_lines.append(f"- {key}: {desc}")
                        else:
                            topic_lines.append(f"- {key}")
        except Exception:
            topic_lines = []
            topic_count = 0

    if not topic_lines:
        return (
            f"Local RAG docs dir: {docs_dir}\n"
            "Topic summary: unavailable (topic_descriptions.json not found or unreadable).\n"
            "Planner policy: if task needs external latest/public evidence, add Researcher."
        )

    suffix = ""
    if topic_count > len(topic_lines):
        suffix = f"\n- ... and {topic_count - len(topic_lines)} more topics"

    return (
        f"Local RAG docs dir: {docs_dir}\n"
        f"Known topic summaries ({topic_count}):\n"
        + "\n".join(topic_lines)
        + suffix
    )


def _estimate_chars(text: str) -> int:
    """Return character count for context budget estimation."""
    if not text:
        return 0
    return len(text)


def _messages_total_chars(messages: list) -> int:
    """Calculate total characters across all messages."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(msg, dict):
            content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
    return total


def _format_context_index(index: list[dict] | None) -> str:
    """Format the accumulated context index as a readable string for injection.

    This block is always present in the Agent's context after compression,
    teaching the Agent how to recover full details from saved files.
    """
    if not index:
        return ""
    lines = [
        "[CONTEXT INDEX]\n"
        "Your conversation history has been compressed to save context space.\n"
        "The full original content is saved in the files listed below.\n"
        "If you need details that are NOT in the summary above, read the original:\n"
        "  → load_memory_tool(memory_id=\"<id>\")  — read a specific file\n"
        "  → search_memory_tool(query=\"...\")      — search across all saved memories\n"
        "\nSaved files:"
    ]
    for i, entry in enumerate(index, 1):
        title = entry.get("title", "untitled")
        mem_id = entry.get("id", "?")
        entry_type = entry.get("type", "memory")
        lines.append(f"  {i}. [{entry_type}] \"{title}\" → id=\"{mem_id}\"")
    return "\n".join(lines)


def _compress_context(
    messages: list,
    state: State,
    *,
    keep_recent: int = 10,
    target_llm_type: str = "basic",
) -> Tuple[list, list[dict], bool]:
    """Auto-compress context when token usage exceeds threshold.

    Strategy:
    1. Estimate total tokens of messages
    2. If < threshold (default 70%), return as-is
    3. If >= threshold:
       a. Extract essential messages (user request, plan)
       b. Summarize compressible messages via a single LLM call
       c. Save full original messages to context_store
       d. Return rebuilt context: [user_request, plan, summary_with_index, recent_N]

    Returns:
        (compressed_messages, updated_context_index, was_compressed)
    """
    existing_index: list[dict] = list(state.get("context_index") or [])
    usage_stats: dict[str, Any] | None = None
    ratio = 0.0
    threshold = _FALLBACK_COMPRESS_THRESHOLD
    try:
        llm_type_value = str(target_llm_type or "basic").strip().lower()
        if llm_type_value not in {"basic", "reasoning", "vision", "coding"}:
            llm_type_value = "basic"
        usage_stats = estimate_context_usage_for_llm_type(
            messages,
            llm_type=llm_type_value,  # type: ignore[arg-type]
        )
        ratio = float(usage_stats["utilization_ratio"])
        threshold = float(usage_stats["compress_threshold"])
        if ratio < threshold:
            return messages, existing_index, False

        logger.info(
            (
                "Context compression triggered: model=%s llm_type=%s message_tokens=%d "
                "projected_tokens=%d effective_input=%d overhead=%d ratio=%.2f threshold=%.2f"
            ),
            usage_stats.get("model_name"),
            usage_stats.get("llm_type"),
            int(usage_stats.get("message_tokens") or 0),
            int(usage_stats.get("projected_prompt_tokens") or 0),
            int(usage_stats.get("effective_input_tokens") or 0),
            int(usage_stats.get("internal_overhead_tokens") or 0),
            ratio,
            threshold,
        )
    except Exception as exc:
        total_chars = _messages_total_chars(messages)
        window = _FALLBACK_CONTEXT_WINDOW_CHARS
        ratio = total_chars / window if window > 0 else 0.0
        if ratio < _FALLBACK_COMPRESS_THRESHOLD:
            return messages, existing_index, False
        logger.warning(
            (
                "Model-aware token usage estimation failed (%s); fallback to char-threshold "
                "compression: chars=%d window=%d ratio=%.2f threshold=%.2f"
            ),
            _truncate_error_text(exc),
            total_chars,
            window,
            ratio,
            _FALLBACK_COMPRESS_THRESHOLD,
        )

    # --- Extract essential messages ---
    user_msg = _get_first_user_message(messages)
    planner_msg = next(
        (m for m in reversed(messages) if getattr(m, "name", "") == "planner"),
        None,
    )

    # Recent N messages (keep for continuity)
    recent_msgs = messages[-keep_recent:] if len(messages) > keep_recent else []

    # Determine which messages are compressible vs protected:
    #
    # PROTECTED (never sent to LLM summarizer, never lost):
    #   1. user_msg — original user request
    #   2. planner_msg — the plan
    #   3. recent_msgs — last N for continuity
    #
    # EXCLUDED from compression input (not useful to summarize, will be regenerated):
    #   4. context_index — old index block (will be replaced by updated_index)
    #   5. context_summary — old summary block (will be replaced by new summary)
    #
    # COMPRESSIBLE (sent to LLM for summarization):
    #   Everything else (agent responses, tool outputs, etc.)

    protected_ids = set()
    if user_msg is not None:
        protected_ids.add(id(user_msg))
    if planner_msg is not None:
        protected_ids.add(id(planner_msg))
    for m in recent_msgs:
        protected_ids.add(id(m))

    # Identify old context_index / context_summary messages (exclude from compression input)
    meta_msg_ids = set()
    for m in messages:
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ("context_index", "context_summary"):
            meta_msg_ids.add(id(m))

    skip_ids = protected_ids | meta_msg_ids
    compressible = [m for m in messages if id(m) not in skip_ids]

    if not compressible:
        return messages, existing_index, False

    # --- Build text to summarize ---
    compress_text_parts = []
    for m in compressible:
        name = getattr(m, "name", "")
        if isinstance(m, dict):
            name = m.get("name", name)
        content = getattr(m, "content", "")
        if isinstance(m, dict):
            content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            prefix = f"[{name}] " if name else ""
            compress_text_parts.append(f"{prefix}{content}")

    compress_text = "\n---\n".join(compress_text_parts)

    # --- Eight-section structured compression via LLM ---
    # Agent conversation history is compressed in a single pass.
    # Multi-pass compression is handled separately in crawl_tool for external content.

    # Collect all user messages verbatim (for section 6)
    user_messages_verbatim: list[str] = []
    for m in messages:
        if not _is_user_message(m):
            continue
        c = getattr(m, "content", "")
        if isinstance(m, dict):
            c = m.get("content", "")
        if isinstance(c, str) and c.strip():
            user_messages_verbatim.append(c.strip()[:500])

    user_msgs_block = "\n".join(
        f"  - {msg}" for msg in user_messages_verbatim
    ) if user_messages_verbatim else "  (none)"

    try:
        llm = get_compression_llm()
        logger.info("Compression: single-pass (content=%d chars)", len(compress_text))

        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "你是 AI Agent 的上下文压缩专家。请将以下多轮对话历史压缩为 **八段式结构化摘要**。\n"
                    "严格按照以下 8 个 section 输出，每个 section 用标题行开头，内容用分点列出。\n"
                    "如果某个 section 没有相关信息，写 '(none)'。\n"
                    "每个 section 可以写 1500-2000 字符，总字符数限制在 15000 字符以内。\n"
                    "尽量详细保留关键信息，宁多勿少。\n\n"
                    "## 八段式结构\n\n"
                    "### 1. Primary Request and Intent (主要请求和意图)\n"
                    "用户最初想要完成什么？最终目标是什么？保留原始表述。\n\n"
                    "### 2. Key Technical Concepts (关键技术概念)\n"
                    "重要的技术决策、约束条件、使用的方法/工具/包名/版本号。\n\n"
                    "### 3. Files and Code Sections (文件和代码段)\n"
                    "涉及的文件路径、数据文件、输出路径、关键代码片段（保留路径原文）。\n\n"
                    "### 4. Errors and Fixes (错误和修复)\n"
                    "遇到的错误、报错信息摘要、已采取的修复措施（避免重复踩坑）。\n\n"
                    "### 5. Problem Solving (问题解决)\n"
                    "已尝试的方案、成功/失败的方案、关键发现和结论。\n\n"
                    "### 6. All User Messages (所有用户消息)\n"
                    "保留用户的原始表达（摘要形式），确保意图不丢失。\n\n"
                    "### 7. Pending Tasks (待处理任务)\n"
                    "尚未完成的步骤、下一步计划、待验证的事项。\n\n"
                    "### 8. Current Work (当前工作)\n"
                    "最近在做什么？进行到哪一步？当前状态是什么？"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请压缩以下对话历史为八段式结构化摘要：\n\n"
                    f"{compress_text}\n\n"
                    f"---\n"
                    f"补充：以下是对话中所有用户消息的原文（供 section 6 参考）：\n"
                    f"{user_msgs_block}"
                ),
            },
        ]
        gen = llm.invoke(summary_prompt)
        summary = (getattr(gen, "content", str(gen)) or "").strip()[:15000]
    except Exception as e:
        logger.error(f"Context compression LLM call failed: {e}")
        # Fallback: simple truncation with section headers
        summary = (
            "### 1. Primary Request and Intent\n(compression failed, see context_index for original)\n\n"
            "### 8. Current Work\n"
            + compress_text[:8000]
            + "\n[Compression fallback: original truncated]"
        )

    # --- Extract memory references from compressible messages before they're lost ---
    # Tools like crawl_tool and remember_tool embed "id=xxx" in message content.
    # If we don't capture these, the ids will only survive if the LLM summary
    # happens to include them — which is unreliable.
    import re
    # Match patterns like: id=abc, id="abc", memory_id=abc, memory id=abc
    _id_pattern = re.compile(
        r'\b(?:memory_?id|id)\s*[:=]\s*"?([a-f0-9]{32})"?',
        re.IGNORECASE,
    )
    for m in compressible:
        content = getattr(m, "content", "")
        if isinstance(m, dict):
            content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for match in _id_pattern.finditer(content):
            found_id = match.group(1)
            # Avoid duplicates
            if not any(e.get("id") == found_id for e in existing_index):
                # Try to extract title from nearby text
                title_match = re.search(r'title="([^"]*)"', content)
                title = title_match.group(1) if title_match else "Saved Memory"
                existing_index.append({
                    "id": found_id,
                    "path": "",  # Path not always available from message text
                    "title": title,
                    "type": "memory",
                })
                logger.debug(f"Extracted memory reference from message: id={found_id}")

    # --- Save full original messages to context_store ---
    new_index_entry = None
    try:
        from src.memory.context_store import ContextStore
        from src.tools.memory import _default_store_root

        store = ContextStore(_default_store_root(state.get("thread_id")))
        full_text = "\n\n---\n\n".join(compress_text_parts)
        compression_count = int(state.get("compression_count") or 0) + 1
        item = store.save_text(
            content=full_text,
            title=f"Context Compression #{compression_count}",
            tags=["context", "compression", f"round-{compression_count}"],
        )
        new_index_entry = {
            "id": item.id,
            "path": item.doc_path,
            "title": f"Context Compression #{compression_count}",
            "type": "compression",
        }
        logger.info(f"Compressed context saved: id={item.id}, path={item.doc_path}")
    except Exception as e:
        logger.error(f"Failed to save compressed context: {e}")

    # --- Update index ---
    updated_index = list(existing_index)
    if new_index_entry:
        updated_index.append(new_index_entry)

    # --- Rebuild context ---
    rebuilt = []

    # 1. User request (always first)
    if user_msg is not None:
        rebuilt.append(user_msg)

    # 2. Plan (always present if exists)
    if planner_msg is not None:
        rebuilt.append(planner_msg)

    # 3. Context index (accumulated across all compressions)
    index_text = _format_context_index(updated_index)
    if index_text:
        rebuilt.append(
            SystemMessage(
                content=index_text,
                name="context_index",
            )
        )

    # 4. Eight-section structured compression summary
    rebuilt.append(
        SystemMessage(
            content=f"[COMPRESSED CONTEXT - Eight-Section Structured Summary]\n\n{summary}",
            name="context_summary",
        )
    )

    # 5. Recent messages (for continuity)
    for m in recent_msgs:
        if id(m) != id(user_msg) and id(m) != id(planner_msg):
            rebuilt.append(m)

    before_tokens = None
    after_tokens = None
    if usage_stats:
        try:
            model_name = str(usage_stats.get("model_name") or "")
            before_tokens = int(usage_stats.get("projected_prompt_tokens") or 0)
            rebuilt_usage = estimate_context_usage_for_llm_type(
                rebuilt,
                llm_type=str(usage_stats.get("llm_type") or "basic"),  # type: ignore[arg-type]
            )
            # Reuse rebuilt usage estimate (same model family) for after-compression signal.
            if model_name and str(rebuilt_usage.get("model_name") or "") != model_name:
                rebuilt_usage = usage_stats
            after_tokens = int(rebuilt_usage.get("projected_prompt_tokens") or 0)
        except Exception:
            before_tokens = None
            after_tokens = None

    if before_tokens is not None and after_tokens is not None:
        logger.info(
            "Context compressed: %d messages -> %d messages, %d prompt-tokens -> ~%d prompt-tokens",
            len(messages),
            len(rebuilt),
            before_tokens,
            after_tokens,
        )
    else:
        logger.info(
            "Context compressed: %d messages -> %d messages, %d chars -> ~%d chars",
            len(messages),
            len(rebuilt),
            _messages_total_chars(messages),
            _messages_total_chars(rebuilt),
        )

    return rebuilt, updated_index, True


def _clamp_context_block(text: str | None, max_chars: int, max_lines: int) -> str | None:
    """Clamp long context strings to reduce LLM token pressure."""
    if not text:
        return text
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    new_text = "\n".join(lines)
    if len(new_text) > max_chars:
        new_text = new_text[:max_chars]
        truncated = True
    if truncated:
        if not new_text.endswith("\n"):
            new_text += "\n"
        new_text += "[File list truncated. Use peek_file_tool to inspect specific files.]"
    return new_text


def _clamp_message_block(text: str | None, max_chars: int, max_lines: int) -> str | None:
    """Clamp long non-file messages to reduce LLM token pressure."""
    if not text:
        return text
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    new_text = "\n".join(lines)
    if len(new_text) > max_chars:
        new_text = new_text[:max_chars]
        truncated = True
    if truncated:
        if not new_text.endswith("\n"):
            new_text += "\n"
        new_text += "[Content truncated]"
    return new_text


def _last_message_after(messages: list, target_name: str, after_name: str) -> object | None:
    """Return the last message named `target_name` that appears after the last `after_name`."""
    after_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        msg_name = getattr(msg, "name", "")
        if isinstance(msg, dict):
            msg_name = msg.get("name", msg_name)
        if msg_name == after_name:
            after_idx = idx
            break

    for idx in range(len(messages) - 1, after_idx, -1):
        msg = messages[idx]
        msg_name = getattr(msg, "name", "")
        if isinstance(msg, dict):
            msg_name = msg.get("name", msg_name)
        if msg_name == target_name:
            return msg
    return None


def _collect_recent_named_messages(
    messages: list,
    target_name: str,
    *,
    after_name: str | None = None,
    max_items: int = 8,
) -> list:
    """Collect recent messages by name, preserving chronological order."""
    if not messages or max_items <= 0:
        return []

    after_idx = -1
    if after_name:
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            msg_name = getattr(msg, "name", "")
            if isinstance(msg, dict):
                msg_name = msg.get("name", msg_name)
            if msg_name == after_name:
                after_idx = idx
                break

    collected = []
    for idx in range(len(messages) - 1, after_idx, -1):
        msg = messages[idx]
        msg_name = getattr(msg, "name", "")
        if isinstance(msg, dict):
            msg_name = msg.get("name", msg_name)
        if msg_name != target_name:
            continue
        collected.append(msg)
        if len(collected) >= max_items:
            break

    collected.reverse()
    return collected


def _find_message_index(messages: list, target_name: str, start_idx: int) -> int | None:
    """Return the first index of `target_name` after `start_idx`."""
    for idx in range(start_idx + 1, len(messages)):
        msg = messages[idx]
        msg_name = getattr(msg, "name", "")
        if isinstance(msg, dict):
            msg_name = msg.get("name", msg_name)
        if msg_name == target_name:
            return idx
    return None


def _truncate_error_text(exc: Exception, *, max_chars: int = 240) -> str:
    """Return a compact single-line error description."""
    text = str(exc or "").strip() or exc.__class__.__name__
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _is_rate_limit_or_quota_error(exc: Exception) -> bool:
    """Best-effort detection for provider-side quota / 429 throttling errors."""
    lowered = f"{exc.__class__.__name__}: {exc}".lower()
    markers = (
        "ratelimit",
        "rate limit",
        "429",
        "quota",
        "insufficient_quota",
        "exceeded_current_quota",
        "resource_exhausted",
        "too many requests",
    )
    return any(marker in lowered for marker in markers)


def _is_model_connectivity_error(exc: Exception) -> bool:
    """Best-effort detection for model-provider connectivity/network errors."""
    lowered = f"{exc.__class__.__name__}: {exc}".lower()
    markers = (
        "apiconnectionerror",
        "connection error",
        "connecterror",
        "connection reset",
        "connection aborted",
        "remote protocol error",
        "unexpected eof while reading",
        "ssl",
        "timed out",
        "timeout",
        "name or service not known",
        "temporary failure in name resolution",
        "failed to establish a new connection",
    )
    return any(marker in lowered for marker in markers)


def _build_llm_system_warning(agent_name: str, exc: Exception) -> str:
    """Create a user-facing warning text for LLM invocation failures."""
    detail = _truncate_error_text(exc, max_chars=220)
    if _is_rate_limit_or_quota_error(exc):
        return (
            f"SYSTEM_WARNING: {agent_name} LLM call hit provider quota/rate-limit (429). "
            f"Please check API balance/limits and retry. Detail: {detail}"
        )
    if _is_model_connectivity_error(exc):
        return (
            f"SYSTEM_WARNING: {agent_name} LLM call failed due to model-service "
            f"connectivity/network issue (not an external database error). "
            f"Please retry. Detail: {detail}"
        )
    return f"SYSTEM_WARNING: {agent_name} LLM call failed. Detail: {detail}"


def _build_reporter_failure_hint(checklist: list | None) -> str:
    """Provide reporter with explicit failure classification to avoid wrong blame."""
    if not checklist:
        return ""

    model_failures: list[str] = []
    data_failures: list[str] = []
    for step in checklist:
        if step.get("status") != "failed":
            continue
        step_id = step.get("step_id", "?")
        step_name = str(step.get("agent_name") or "").strip() or "unknown"
        result_text = str(step.get("result") or "")
        lowered = result_text.lower()
        if "system_warning:" in lowered and "llm call" in lowered:
            model_failures.append(f"- Step {step_id} ({step_name}): model-service/LLM failure")
        elif any(token in lowered for token in ("api error", "http", "request failed", "database", "tool")):
            data_failures.append(f"- Step {step_id} ({step_name}): tool/data-source failure")

    if not model_failures and not data_failures:
        return ""

    lines = [
        "=== FAILURE CLASSIFICATION HINTS ===",
        "When writing the report:",
        "1) If failed step contains SYSTEM_WARNING about LLM call, classify as model-service issue.",
        "2) Do NOT claim external database/data-source failure unless tool output explicitly shows it.",
    ]
    if model_failures:
        lines.append("Model-service failures:")
        lines.extend(model_failures)
    if data_failures:
        lines.append("Confirmed tool/data-source failures:")
        lines.extend(data_failures)
    lines.append("=== END FAILURE CLASSIFICATION HINTS ===")
    return "\n".join(lines)


def _mark_in_progress_steps_failed(
    checklist: list | None,
    *,
    reason: str,
    agent_name: str | None = None,
) -> None:
    """Mark matching in_progress checklist steps as failed with a concise reason."""
    if not checklist:
        return

    reason_text = str(reason or "").strip()[:500]
    for step in checklist:
        if step.get("status") != "in_progress":
            continue
        if agent_name and step.get("agent_name") != agent_name:
            continue
        step["status"] = "failed"
        if reason_text:
            step["result"] = reason_text


def _next_checklist_agent(checklist: list | None) -> str:
    """Return the next safe route based on checklist state."""
    if not checklist:
        return "FINISH"

    # Prefer currently running steps first to preserve local progress.
    for step in checklist:
        if step.get("status") == "in_progress" and step.get("agent_name") in TEAM_MEMBERS:
            return str(step["agent_name"])

    # Then pick the earliest pending step.
    for step in checklist:
        if step.get("status") == "pending" and step.get("agent_name") in TEAM_MEMBERS:
            return str(step["agent_name"])

    return "FINISH"


def _sanitize_supervisor_route(raw_next: object, checklist: list | None) -> str:
    """Validate supervisor route and fallback to checklist-driven safe route."""
    normalized = str(raw_next or "").strip()
    safe_next = _next_checklist_agent(checklist)
    if not normalized:
        return safe_next

    upper = normalized.upper()
    if upper == "FINISH":
        if safe_next != "FINISH":
            logger.info(
                "Supervisor requested FINISH but checklist not complete; fallback to %s",
                safe_next,
            )
            return safe_next
        return "FINISH"

    lowered = normalized.lower()
    if lowered in TEAM_MEMBERS:
        if safe_next != "FINISH" and lowered != safe_next:
            logger.info(
                "Supervisor requested route %s but checklist-safe next is %s; overriding",
                lowered,
                safe_next,
            )
            return safe_next
        return lowered

    logger.warning("Supervisor returned unsupported route %r, fallback to safe next step", raw_next)
    return safe_next


def _ensure_routed_step_in_progress(checklist: list | None, routed_agent: str) -> None:
    """Ensure the routed agent has an active in_progress step when checklist is used.

    This prevents repeated no-op loops where an agent is routed repeatedly while all
    its remaining steps stay pending.
    """
    if not checklist or routed_agent not in TEAM_MEMBERS:
        return

    has_in_progress = any(
        step.get("status") == "in_progress" and step.get("agent_name") == routed_agent
        for step in checklist
    )
    if has_in_progress:
        return

    first_pending = next((step for step in checklist if step.get("status") == "pending"), None)
    if first_pending is None:
        return
    if first_pending.get("agent_name") != routed_agent:
        return

    first_pending["status"] = "in_progress"
    logger.info(
        "Checklist auto-advance: Step %s -> in_progress for routed agent %s",
        first_pending.get("step_id"),
        routed_agent,
    )


def _is_valid_step_transition(old_status: str, new_status: str) -> bool:
    """Return True if checklist status transition is allowed."""
    if old_status == new_status:
        return False

    if old_status in {"completed", "failed"}:
        return False

    if old_status == "pending":
        return new_status in {"in_progress", "failed"}

    if old_status == "in_progress":
        return new_status in {"completed", "failed"}

    return False


def _apply_guarded_step_updates(
    checklist: list,
    step_updates: list,
    *,
    next_agent: str,
) -> None:
    """Apply LLM step_updates with strict transition and routing guards."""
    if not checklist or not step_updates:
        return

    step_by_id = {int(step["step_id"]): step for step in checklist}
    first_pending_id = None
    for step in checklist:
        if step.get("status") == "pending":
            first_pending_id = int(step["step_id"])
            break

    in_progress_applied = False
    for raw_update in step_updates:
        if not isinstance(raw_update, dict):
            continue
        step_id_raw = raw_update.get("step_id")
        new_status_raw = raw_update.get("status")
        result_text = raw_update.get("result")

        try:
            step_id = int(step_id_raw)
        except (TypeError, ValueError):
            continue

        new_status = str(new_status_raw or "").strip().lower()
        if new_status not in _STEP_STATUS_VALUES:
            logger.warning("Skip invalid step status update: step_id=%s status=%r", step_id, new_status_raw)
            continue

        step = step_by_id.get(step_id)
        if step is None:
            logger.warning("Skip step update for unknown step_id=%s", step_id)
            continue

        old_status = str(step.get("status") or "")
        if not _is_valid_step_transition(old_status, new_status):
            logger.info(
                "Skip step transition %s -> %s for step %s (guarded)",
                old_status,
                new_status,
                step_id,
            )
            continue

        if new_status == "in_progress":
            # Cautious mode: only the earliest pending step can start,
            # and it should match the routed worker.
            if in_progress_applied:
                logger.info("Skip extra in_progress update for step %s", step_id)
                continue
            if first_pending_id is not None and step_id != first_pending_id:
                logger.info("Skip out-of-order in_progress update for step %s", step_id)
                continue
            if next_agent in TEAM_MEMBERS and step.get("agent_name") != next_agent:
                logger.info(
                    "Skip in_progress update for step %s: agent mismatch (step=%s, next=%s)",
                    step_id,
                    step.get("agent_name"),
                    next_agent,
                )
                continue
            in_progress_applied = True

        step["status"] = new_status
        if result_text:
            step["result"] = str(result_text).strip()[:500]
        logger.info("  Step %s: %s -> %s", step_id, old_status, new_status)


def _get_first_user_message(messages: list) -> object | None:
    """Return the first user/human message from a mixed message list.

    The workflow service may prepend system/env notes, so we cannot assume
    messages[0] is the user's request.
    """
    if not messages:
        return None

    for msg in messages:
        if _is_user_message(msg):
            return msg

    # Fallback: preserve previous behavior if no user message is found
    return messages[0]


def _get_latest_user_message(messages: list) -> object | None:
    """Return the latest user/human message from a mixed message list."""
    if not messages:
        return None

    for msg in reversed(messages):
        if _is_user_message(msg):
            return msg

    return _get_first_user_message(messages)


def _is_user_message(message: object) -> bool:
    """Return True if message is a user/human input rather than agent intermediate output."""
    if isinstance(message, dict):
        role = (message.get("role") or message.get("type") or "").lower()
        if role not in {"user", "human"}:
            return False
        name = str(message.get("name") or "").strip().lower()
        return name not in _INTERNAL_MESSAGE_NAMES

    msg_type = (getattr(message, "type", "") or "").lower()
    if msg_type not in {"human", "user"}:
        return False
    name = str(getattr(message, "name", "") or "").strip().lower()
    return name not in _INTERNAL_MESSAGE_NAMES


def _message_content(message: object | None) -> str:
    """Extract message content from either dict or LangChain message object."""
    if message is None:
        return ""
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _looks_like_internal_agent_payload(text: str) -> bool:
    """Detect intermediate agent payload that should never be used as search query."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("response from "):
        return True
    markers = (
        "<response>",
        "</response>",
        "*please execute the next step.*",
        "\"agent_name\"",
        "\"step_id\"",
        "\"description\"",
    )
    marker_hits = sum(1 for marker in markers if marker in lowered)
    return marker_hits >= 2


def _is_small_talk_request(text: str) -> bool:
    """Conservative small-talk detector for the coordinator gateway."""
    normalized = (text or "").strip()
    if not normalized:
        return False

    lowered = normalized.lower()
    if len(lowered) > 120:
        return False

    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _TASK_HINT_PATTERNS):
        return False

    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _SMALL_TALK_PATTERNS):
        return True

    return lowered in {"ok", "okay", "cool", "great", "nice", "thx", "thanks", "thank you"}


def _small_talk_reply(user_text: str) -> str:
    """Generate a short gateway reply for chit-chat turns."""
    if re.search(r"[\u4e00-\u9fff]", user_text or ""):
        return "你好，我在。请告诉我你要完成的生信或数据分析任务。"
    return "Hi, I'm here. Share your bioinformatics or analysis task and I'll plan it."


def _sanitize_planner_search_query(raw_text: str, *, max_chars: int) -> str:
    """Normalize planner pre-search query into concise intent text."""
    if not raw_text:
        return ""

    text = str(raw_text)

    # Prefer keeping response payload body if wrapped by internal tags.
    inner = re.search(r"(?is)<response>\s*(.*?)\s*</response>", text)
    if inner:
        text = inner.group(1)

    text = re.sub(r"(?im)^\s*response from [^:\n]+:\s*", "", text)
    text = re.sub(r"(?i)\*?\s*please execute the next step\.?\s*\*?", " ", text)
    text = re.sub(r"(?is)```.*?```", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links
    text = re.sub(r"https?://\S+", " ", text)  # urls
    text = re.sub(r"(?is)<[^>]+>", " ", text)  # html/xml tags
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text).strip()

    if not text:
        return ""

    # Keep whole sentences when possible before hard truncation.
    if len(text) > max_chars:
        sentences = [
            seg.strip() for seg in re.split(r"(?<=[。！？.!?;；])\s+", text) if seg.strip()
        ]
        picked: list[str] = []
        total = 0
        for seg in sentences:
            extra = len(seg) + (1 if picked else 0)
            if total + extra > max_chars:
                break
            picked.append(seg)
            total += extra

        if picked:
            text = " ".join(picked)
        if len(text) > max_chars:
            text = text[:max_chars]

    return text.strip(" \t\r\n,;；。")


def _build_planner_search_query(state: State) -> str:
    """Build a safe pre-planning search query and avoid upstream payload pollution."""
    messages = list(state.get("messages") or [])
    if not messages:
        return ""

    latest_user = _get_latest_user_message(messages)
    first_user = _get_first_user_message(messages)
    last_msg = messages[-1]

    candidates: list[tuple[str, object | None]] = [("latest_user", latest_user)]
    if first_user is not None and first_user is not latest_user:
        candidates.append(("first_user", first_user))
    candidates.append(("last_message", last_msg))

    for source, candidate in candidates:
        raw = _message_content(candidate).strip()
        if not raw:
            continue

        # Never let internal agent payload be used as the last-resort query source.
        if source == "last_message" and _looks_like_internal_agent_payload(raw):
            logger.info("Skip planner pre-search: last message is internal agent payload")
            continue

        cleaned = _sanitize_planner_search_query(
            raw,
            max_chars=_PLANNER_SEARCH_QUERY_MAX_CHARS,
        )
        if not cleaned:
            continue
        if _looks_like_internal_agent_payload(cleaned):
            logger.info(
                "Skip planner pre-search candidate from %s: still looks like internal payload",
                source,
            )
            continue
        logger.debug(
            "Planner pre-search query prepared from %s (chars=%d)",
            source,
            len(cleaned),
        )
        return cleaned

    logger.info("Planner pre-search skipped: no valid sanitized query candidate")
    return ""


def _extract_step_failed_entries(output: str) -> list[tuple[int, str]]:
    """Parse STEP_FAILED markers from agent output."""
    if not output:
        return []
    entries: list[tuple[int, str]] = []
    matches = re.findall(r"STEP_FAILED:\s*(\d+),\s*(.+?)(?:\n|$)", output, flags=re.IGNORECASE)
    for step_id_str, reason in matches:
        try:
            entries.append((int(step_id_str), str(reason or "").strip()))
        except ValueError:
            continue
    return entries


def _is_recoverable_step_failed_reason(reason: str) -> bool:
    lowered = str(reason or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in _RECOVERABLE_STEP_FAILED_HINTS)


def _build_step_failed_retry_hint(entries: list[tuple[int, str]]) -> str:
    reasons = "; ".join(
        f"step {step_id}: {reason[:180]}" for step_id, reason in entries
    )[:600]
    return (
        "Detected recoverable STEP_FAILED. Retry the same step now with concrete fixes.\n"
        f"Previous failure detail: {reasons}\n"
        "Required retry checklist:\n"
        "1) Re-inspect files first with list_dir/file_info/infer_table_schema/read_data before rewriting the script.\n"
        "2) For matrix + metadata tasks, run check_sample_overlap before fitting models.\n"
        "3) If parsing failed, try alternate delimiter/encoding and print parsed shape/column names.\n"
        "4) Before claiming any column is missing, print the actual loaded dataframe columns (e.g., list(df.columns) or colnames(df)) "
        "from the exact file used in analysis.\n"
        "5) If preprocessing drops all rows, do NOT stop immediately: print per-column NA rates, avoid global dropna, "
        "use targeted imputation/encoding or reduced covariate set, then retry.\n"
        "6) Re-run analysis after the fix and report either STEP_COMPLETED or a final STEP_FAILED with concrete evidence."
    )


def _remaining_steps_budget(state: State) -> int | None:
    """Return remaining graph step budget when available."""
    try:
        remaining = int(state.get("remaining_steps"))  # type: ignore[arg-type]
    except Exception:
        return None
    if remaining <= 0:
        return None
    return remaining


def _can_retry_with_budget(
    *,
    state: State,
    attempts: int,
    explicit_cap: int = -1,
) -> bool:
    """Retry until LangGraph budget is near exhaustion, unless explicit cap is set."""
    if explicit_cap > 0 and attempts > explicit_cap:
        return False
    remaining = _remaining_steps_budget(state)
    if remaining is not None and remaining <= _CODER_RETRY_MIN_REMAINING_STEPS:
        return False
    return True


def _resolve_agent_remaining_steps(state: State, default: int = 100) -> int:
    """Use graph-provided remaining_steps when available; otherwise fallback."""
    remaining = _remaining_steps_budget(state)
    if remaining is not None:
        return remaining
    return int(default)


def _update_checklist_from_output(
    checklist: list,
    output: str,
    *,
    evidence_text: str = "",
) -> list:
    """解析 coder/agent 输出，更新清单状态（证据优先，避免误判）。

    优先级：
    1. 显式标记：STEP_COMPLETED / STEP_FAILED
    2. 产物证据：SAVED_SCRIPT_PATH / SAVED_FILE / SAVED_PLOT_PATH / FINAL_ANSWER
    3. 明确错误：Traceback / Exception / exit code / failed
    4. 无证据不自动完成，保持 in_progress
    """
    if not checklist:
        return checklist
    if not output and not evidence_text:
        return checklist
    
    # 匹配 STEP_FAILED: 2, 原因（失败优先于完成）
    failed_matches = _extract_step_failed_entries(output)
    has_explicit_failed = False
    failed_ids: set[int] = set()
    for step_id, reason in failed_matches:
        failed_ids.add(step_id)
        for step in checklist:
            if step["step_id"] == step_id:
                step["status"] = "failed"
                step["result"] = reason.strip()
                logger.debug(f"Checklist: Step {step_id} marked as failed: {reason}")
                has_explicit_failed = True

    # 匹配 STEP_COMPLETED: [1, 2, 3]（同一步骤若同时失败，则保持失败）
    completed_match = re.search(r'STEP_COMPLETED:\s*\[([^\]]+)\]', output)
    has_explicit_completed = False
    if completed_match:
        try:
            completed_ids = [int(x.strip()) for x in completed_match.group(1).split(',')]
            if failed_ids:
                completed_ids = [sid for sid in completed_ids if sid not in failed_ids]
            for step in checklist:
                if step["step_id"] in completed_ids:
                    step["status"] = "completed"
                    logger.debug(f"Checklist: Step {step['step_id']} marked as completed")
            has_explicit_completed = bool(completed_ids)
        except ValueError:
            logger.warning(f"Failed to parse STEP_COMPLETED ids: {completed_match.group(1)}")

    # 如果有显式状态，直接返回，不做推断覆盖
    if has_explicit_completed or has_explicit_failed:
        return checklist

    # 显式状态缺失时，改为“证据驱动”判定，不再默认成功
    combined_text = f"{output or ''}\n{evidence_text or ''}"
    artifact_markers = ("SAVED_SCRIPT_PATH:", "SAVED_FILE:", "SAVED_PLOT_PATH:")
    has_artifacts = any(marker in combined_text for marker in artifact_markers)
    has_final_answer = "FINAL_ANSWER:" in combined_text

    error_indicators = [
        "Error:",
        "Exception:",
        "Traceback",
        "failed",
        "FAILED",
        "Command failed with exit code",
        "Failed to execute",
        "Agent error:",
    ]
    has_error = any(indicator in combined_text for indicator in error_indicators)

    if not has_explicit_completed and not has_explicit_failed:
        target_step = next(
            (
                step
                for step in checklist
                if step.get("status") == "in_progress" and step.get("agent_name") == "coder"
            ),
            None,
        )
        if target_step is not None:
            if has_artifacts or has_final_answer:
                target_step["status"] = "completed"
                logger.info(
                    "Checklist: Evidence-based completion for step %s",
                    target_step["step_id"],
                )
            elif has_error:
                target_step["status"] = "failed"
                target_step["result"] = "执行过程中检测到明确错误"
                logger.info(
                    "Checklist: Evidence-based failure for step %s",
                    target_step["step_id"],
                )
            else:
                logger.info(
                    "Checklist: Step %s remains in_progress (missing completion evidence)",
                    target_step["step_id"],
                )
    
    return checklist


def _format_checklist_for_supervisor(checklist: list) -> str:
    """格式化 checklist 用于 supervisor LLM 的输入。
    
    输出格式：
    Step 1 [pending] coder: 读取数据
      描述: 读取counts.csv文件
      注意: 检查列名格式
    Step 2 [in_progress] coder: 差异分析
      ...
    """
    if not checklist:
        return "（无任务清单）"
    
    status_map = {
        "pending": "待执行",
        "in_progress": "执行中",
        "completed": "已完成",
        "failed": "失败"
    }
    
    lines = []
    for step in checklist:
        status_label = status_map.get(step['status'], step['status'])
        lines.append(f"Step {step['step_id']} [{status_label}] {step['agent_name']}: {step['title']}")
        if step.get('description'):
            lines.append(f"  描述: {step['description']}")
        if step.get('note'):
            lines.append(f"  注意: {step['note']}")
        if step.get('result'):
            lines.append(f"  结果: {step['result']}")
    
    return "\n".join(lines)


def _message_contains_marker(messages: list, marker: str) -> bool:
    """Check whether any message content contains a marker substring."""
    if not messages or not marker:
        return False
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and marker in content:
            return True
    return False


def _has_tool_execution_evidence(messages: list) -> bool:
    """Return True when message stream includes concrete tool execution traces."""
    if not messages:
        return False
    evidence_tools = {
        "execute_in_sandbox",
        "python_repl_tool",
        "create_sandbox",
        "remove_sandbox",
        "list_dir",
        "file_info",
        "read_data",
        "infer_table_schema",
        "check_sample_overlap",
    }
    for msg in messages:
        name = str(getattr(msg, "name", "") or "").lower()
        content = str(getattr(msg, "content", "") or "")
        content_lower = content.lower()
        if name in evidence_tools:
            return True
        if "command executed successfully" in content_lower:
            return True
        if "command executed failed" in content_lower:
            return True
        if "sandbox created" in content_lower:
            return True
        if "successfully removed sandbox" in content_lower:
            return True
    return False


def _has_command_execution_evidence(messages: list) -> bool:
    """Return True when real code/command execution happened (not just file inspection)."""
    if not messages:
        return False
    for msg in messages:
        name = str(getattr(msg, "name", "") or "").lower()
        content = str(getattr(msg, "content", "") or "").lower()
        if name in {"execute_in_sandbox", "python_repl_tool"}:
            return True
        if "command executed successfully" in content:
            return True
        if "command executed failed" in content:
            return True
    return False


def _iter_message_tool_names(messages: list) -> set[str]:
    """Extract tool names referenced by message.name and tool_calls payloads."""
    names: set[str] = set()
    if not messages:
        return names
    for msg in messages:
        if isinstance(msg, dict):
            name = str(msg.get("name", "") or "").strip().lower()
            if name:
                names.add(name)
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_name = str(call.get("name", "") or "").strip().lower()
                    if call_name:
                        names.add(call_name)
            continue

        name = str(getattr(msg, "name", "") or "").strip().lower()
        if name:
            names.add(name)

        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_name = str(call.get("name", "") or "").strip().lower()
                if call_name:
                    names.add(call_name)

        additional = getattr(msg, "additional_kwargs", None)
        if isinstance(additional, dict):
            extra_calls = additional.get("tool_calls")
            if isinstance(extra_calls, list):
                for call in extra_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if isinstance(fn, dict):
                        call_name = str(fn.get("name", "") or "").strip().lower()
                        if call_name:
                            names.add(call_name)
    return names


def _missing_structure_probe_tools(messages: list) -> list[str]:
    seen = _iter_message_tool_names(messages)
    missing = [name for name in _STRUCTURE_PROBE_TOOL_NAMES if name not in seen]
    return missing


def _should_require_structure_probe(
    *,
    messages: list,
    checklist: list | None,
    file_context: str | None,
) -> bool:
    """Decide whether this coder task should enforce matrix+metadata structure probing."""
    text_chunks: list[str] = []

    if isinstance(file_context, str) and file_context.strip():
        text_chunks.append(file_context)

    if checklist:
        for step in checklist:
            if step.get("agent_name") != "coder":
                continue
            for key in ("title", "description", "note"):
                value = step.get(key)
                if isinstance(value, str) and value.strip():
                    text_chunks.append(value)

    if messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    text_chunks.append(content)
                    break

    if not text_chunks:
        return False

    text = "\n".join(text_chunks).lower()
    has_model_marker = any(token in text for token in _STRUCTURE_MODEL_MARKERS)
    has_matrix_marker = any(token in text for token in _STRUCTURE_MATRIX_MARKERS)
    has_metadata_marker = any(token in text for token in _STRUCTURE_METADATA_MARKERS)
    return bool(has_model_marker and has_matrix_marker and has_metadata_marker)


def _sanitize_llm_text(text: str | None) -> str:
    """Remove tool-like traces from LLM text before showing to users."""
    if not text:
        return ""

    cleaned = str(text)
    # Remove explicit XML-like tool blocks.
    cleaned = re.sub(r"(?is)<tool\b[^>]*>.*?</tool>", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*</?tool\b[^>]*>\s*$", "", cleaned)

    # Remove common pseudo tool-invocation lines leaked by some models.
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        if stripped.lower().startswith("code_execution "):
            continue
        if stripped.lower().startswith("code_execution{"):
            continue
        lines.append(raw_line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _normalize_reporter_images_to_markdown(text: str) -> str:
    """Convert HTML <img> tags in reporter output to Markdown image syntax."""
    if not text:
        return ""

    def _extract_attr(tag: str, attr: str) -> str:
        quoted = re.search(
            rf"""\b{attr}\s*=\s*(?:"([^"]*)"|'([^']*)')""",
            tag,
            flags=re.IGNORECASE,
        )
        if quoted:
            return (quoted.group(1) or quoted.group(2) or "").strip()
        unquoted = re.search(
            rf"""\b{attr}\s*=\s*([^\s>]+)""",
            tag,
            flags=re.IGNORECASE,
        )
        if unquoted:
            return (unquoted.group(1) or "").strip().strip('"').strip("'")
        return ""

    def _replace_img(match: re.Match[str]) -> str:
        tag = match.group(0)
        src = _extract_attr(tag, "src")
        if not src:
            return ""
        alt = _extract_attr(tag, "alt") or "image"
        return f"![{alt}]({src})"

    normalized = re.sub(r"(?is)<img\b[^>]*>", _replace_img, text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def _should_require_saved_script(checklist: list | None) -> bool:
    """Return True when current coder step explicitly requires script persistence."""
    if not checklist:
        return False

    target_steps = [
        step
        for step in checklist
        if step.get("agent_name") == "coder"
        and step.get("status") in {"in_progress", "pending"}
    ]
    if not target_steps:
        target_steps = [step for step in checklist if step.get("agent_name") == "coder"]

    if not target_steps:
        return False

    combined_text = " ".join(
        str(step.get("title") or "")
        + " "
        + str(step.get("description") or "")
        + " "
        + str(step.get("note") or "")
        for step in target_steps
    ).lower()

    required_keywords = (
        "saved_script_path",
        "generated_scripts",
        "/app/workspace/generated_scripts",
        "save script",
        "script file",
        "脚本落盘",
    )
    return any(keyword in combined_text for keyword in required_keywords)


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_planner_plan(raw_text: str) -> dict[str, Any] | None:
    """Best-effort parse planner output into a JSON object."""
    cleaned = _strip_json_fence(raw_text)
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)

    if cleaned:
        repaired = repair_json_output(cleaned)
        if repaired and repaired not in candidates:
            candidates.append(repaired)

    embedded = _extract_first_json_object(cleaned)
    if embedded:
        embedded = embedded.strip()
        if embedded and embedded not in candidates:
            candidates.append(embedded)
        repaired_embedded = repair_json_output(embedded)
        if repaired_embedded and repaired_embedded not in candidates:
            candidates.append(repaired_embedded)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_plan_steps(plan: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize planner steps and drop unsupported entries."""
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list):
        return []

    normalized: list[dict[str, str]] = []
    for idx, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        agent_name = str(raw_step.get("agent_name", "")).strip().lower()
        if agent_name not in TEAM_MEMBERS:
            logger.warning(
                "Planner step %s has unsupported agent_name=%r, skipping",
                idx,
                raw_step.get("agent_name"),
            )
            continue

        title = str(raw_step.get("title") or f"Step {idx}").strip()
        description = str(raw_step.get("description") or "").strip()
        note = str(raw_step.get("note") or "").strip()
        normalized.append(
            {
                "agent_name": agent_name,
                "title": title or f"Step {idx}",
                "description": description,
                "note": note,
            }
        )
    return normalized


def _build_fallback_plan(state: State) -> dict[str, Any]:
    """Build a minimal valid plan when planner JSON is invalid."""
    user_msg = _get_first_user_message(state.get("messages", []))
    user_text = ""
    if isinstance(user_msg, dict):
        user_text = str(user_msg.get("content") or "")
    elif user_msg is not None:
        user_text = str(getattr(user_msg, "content", "") or "")

    lower_text = user_text.lower()
    has_path_or_data = any(
        token in lower_text
        for token in [
            "csv",
            "xlsx",
            "tsv",
            "json",
            "file",
            "path",
            "目录",
            "路径",
            "文件",
            "/",
            "\\",
        ]
    )
    needs_compute = any(
        token in lower_text
        for token in [
            "mean",
            "average",
            "median",
            "sum",
            "计算",
            "统计",
            "分析",
            "quant",
            "count",
        ]
    )
    first_agent = "coder" if (has_path_or_data or needs_compute) else "researcher"
    use_zh = bool(re.search(r"[\u4e00-\u9fff]", user_text))

    if use_zh:
        thought = "Planner 输出非结构化内容，已自动回退为最小可执行计划。"
        title = "自动回退执行计划"
        if first_agent == "coder":
            steps = [
                {
                    "agent_name": "coder",
                    "title": "执行数据读取与计算",
                    "description": "读取用户指定的数据或上下文，先确认数据结构，再完成用户请求的计算/分析。",
                    "note": "必须输出可核验的中间结果和 FINAL_ANSWER。",
                },
                {
                    "agent_name": "reporter",
                    "title": "生成最终答复",
                    "description": "基于上一步结果生成最终答复，保持与用户语言一致。",
                    "note": "包含 FINAL_ANSWER。",
                },
            ]
        else:
            steps = [
                {
                    "agent_name": "researcher",
                    "title": "检索关键信息",
                    "description": "围绕用户问题检索并整理关键信息，提供结构化结论。",
                    "note": "优先权威来源并保留核心证据。",
                },
                {
                    "agent_name": "reporter",
                    "title": "生成最终答复",
                    "description": "整合检索结果并生成最终答复。",
                    "note": "包含 FINAL_ANSWER。",
                },
            ]
    else:
        thought = "Planner returned non-JSON output, so a minimal fallback plan is used."
        title = "Fallback Execution Plan"
        if first_agent == "coder":
            steps = [
                {
                    "agent_name": "coder",
                    "title": "Load data and execute analysis",
                    "description": "Read the user-provided data/context, inspect structure first, then complete the requested computation/analysis.",
                    "note": "Return verifiable intermediate results and FINAL_ANSWER.",
                },
                {
                    "agent_name": "reporter",
                    "title": "Produce final response",
                    "description": "Generate the final response from prior step outputs in the user's language.",
                    "note": "Must include FINAL_ANSWER.",
                },
            ]
        else:
            steps = [
                {
                    "agent_name": "researcher",
                    "title": "Collect key evidence",
                    "description": "Search and summarize key facts needed to answer the user request.",
                    "note": "Prefer authoritative sources and include essential evidence.",
                },
                {
                    "agent_name": "reporter",
                    "title": "Produce final response",
                    "description": "Synthesize findings into the final response.",
                    "note": "Must include FINAL_ANSWER.",
                },
            ]

    return {
        "thought": thought,
        "title": title,
        "steps": steps,
    }


def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the researcher agent that performs research tasks."""
    logger.info("Research agent starting task")
    checklist = state.get("plan_checklist", [])

    # --- Auto context compression ---
    original_messages = state.get("messages", [])
    compressed_msgs, updated_index, was_compressed = _compress_context(
        original_messages,
        state,
        target_llm_type=AGENT_LLM_MAP["researcher"],
    )
    temp_state = state.copy()
    temp_state["messages"] = compressed_msgs
    temp_state["remaining_steps"] = _resolve_agent_remaining_steps(state)

    try:
        result = research_agent.invoke(temp_state)
        logger.info("Research agent completed task")
        logger.debug(f"Research agent response: {result['messages'][-1].content}")
        
        # 更新 checklist：仅推进一个 in_progress researcher 步骤，避免误批量完成
        research_text = str(result["messages"][-1].content or "")
        has_research_error = any(
            marker in research_text
            for marker in ("Traceback", "Exception", "Error:", "Research failed")
        )
        if checklist and research_text.strip() and not has_research_error:
            target_step = next(
                (
                    step
                    for step in checklist
                    if step.get("status") == "in_progress" and step.get("agent_name") == "researcher"
                ),
                None,
            )
            if target_step is not None:
                target_step["status"] = "completed"
                logger.info("Checklist: Marked researcher step %s as completed", target_step["step_id"])
        
        updates = {
            "messages": [
                AIMessage(
                    content=RESPONSE_FORMAT.format(
                        "researcher", result["messages"][-1].content
                    ),
                    name="researcher",
                )
            ],
        }
        if checklist:
            updates["plan_checklist"] = checklist
        if was_compressed:
            updates["context_index"] = updated_index
            updates["compression_count"] = int(state.get("compression_count") or 0) + 1
        
        return Command(update=updates, goto="supervisor")
    except Exception as exc:
        logger.exception("Research agent failed to complete task")
        warning_text = _build_llm_system_warning("researcher", exc)
        error_text = _truncate_error_text(exc, max_chars=200)
        if "SYSTEM_WARNING:" in warning_text and "LLM call" in warning_text:
            error_text = f"{error_text} (FailureCategory=model_service_unavailable)"
        
        # 将 in_progress 的 researcher 步骤标记为 failed
        if checklist:
            target_step = next(
                (
                    step
                    for step in checklist
                    if step.get("status") == "in_progress" and step.get("agent_name") == "researcher"
                ),
                None,
            )
            if target_step is not None:
                target_step["status"] = "failed"
                target_step["result"] = warning_text[:500]
                logger.info("Checklist: Marked researcher step %s as failed", target_step["step_id"])
        
        updates = {
            "messages": [
                AIMessage(
                    content=RESPONSE_FORMAT.format("researcher", f"{warning_text}\nResearch failed: {error_text}"),
                    name="researcher",
                )
            ],
        }
        if checklist:
            updates["plan_checklist"] = checklist
        if was_compressed:
            updates["context_index"] = updated_index
            updates["compression_count"] = int(state.get("compression_count") or 0) + 1
        
        return Command(update=updates, goto="supervisor")


def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info("Code agent starting task")
    
    # --- Auto context compression (Graph-level, before any filtering) ---
    original_messages = state.get("messages", [])
    compressed_msgs, updated_index, was_compressed = _compress_context(
        original_messages,
        state,
        target_llm_type=AGENT_LLM_MAP["coder"],
    )

    # --- Context Optimization: further filter for Coder ---
    filtered_messages = []
    
    # === 检查是否有清单模式 ===
    checklist = state.get("plan_checklist", [])
    has_checklist = bool(checklist)
    
    # 1. Always keep the first *user* message
    user_msg = _get_first_user_message(compressed_msgs)
    if user_msg is not None:
        filtered_messages.append(user_msg)
    
    # 2. 【关键改动】如果有 checklist，就不需要 planner 的原始输出了
    # 清单已经包含了所有必要的步骤信息
    planner_msg = None
    if not has_checklist:
        # 只有在没有清单时才保留 planner 消息
        planner_msg = next((m for m in reversed(compressed_msgs) if getattr(m, "name", "") == "planner"), None)
        if planner_msg:
            filtered_messages.append(planner_msg)

    # 3. Keep context_index and context_summary (from compression)
    for m in compressed_msgs:
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ("context_index", "context_summary"):
            if m not in filtered_messages:
                filtered_messages.append(m)

    # 4. Include last Researcher/Browser outputs (truncated) for factual context
    for name in ("researcher", "browser"):
        msg = _last_message_after(compressed_msgs, name, "planner")
        if msg:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                content = _clamp_message_block(content, max_chars=1500, max_lines=80) or content
                filtered_messages.append(AIMessage(content=content, name=name))
            else:
                filtered_messages.append(msg)
        
    # 5. Keep recent Coder/Validation feedback for error recovery
    recent_context = []
    for m in reversed(compressed_msgs):
        if planner_msg and m == planner_msg: 
            break # Stop at planner
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ["coder", "tool"]:
            recent_context.insert(0, m)
    
    if len(recent_context) > 200:
        recent_context = recent_context[-200:]
        
    filtered_messages.extend(recent_context)
    
    # 6. Inject file context + system reminder
    file_context = state.get("initial_file_context")
    file_context_msg = ""
    if file_context:
        file_context = _clamp_context_block(
            file_context,
            max_chars=_CODER_FILE_CONTEXT_MAX_CHARS,
            max_lines=_CODER_FILE_CONTEXT_MAX_LINES,
        )
        file_context_msg = (
            f"\n\n[AVAILABLE DATA FILES IN ./]\n{file_context}\n"
            "(You can read these files directly using their filenames)"
        )

    # === 构建清单视图传递给 coder ===
    checklist_view = ""
    if has_checklist:
        # 获取所有 coder 的步骤（包含 pending 和 in_progress 的）
        coder_steps = [s for s in checklist if s["agent_name"] == "coder"]
        if coder_steps:
            checklist_lines = []
            for s in coder_steps:
                if s["status"] == "completed":
                    status_mark = "[✔]"
                elif s["status"] == "in_progress":
                    status_mark = "[>]"  # 当前需要执行的步骤
                elif s["status"] == "failed":
                    status_mark = "[!]"
                else:
                    status_mark = "[ ]"
                checklist_lines.append(
                    f"{status_mark} 步骤{s['step_id']}: {s['title']}\n"
                    f"    描述: {s['description']}\n"
                    f"    注意: {s['note']}"
                )
            checklist_view = "\n".join(checklist_lines)

    # === 添加系统提醒 ===
    validation_hint = state.get("coder_validation_hint")
    if has_checklist and checklist_view:
        # 清单模式：只使用清单，不需要 planner 消息
        checklist_reminder = f"""=== 你的任务清单 ===
{checklist_view}

【执行规则】
1. 请执行所有标记为 [>] 的步骤（当前需要执行的步骤）
2. 严格按照每个步骤的"描述"和"注意"执行
3. 完成后必须报告：STEP_COMPLETED: [步骤编号列表]
4. 如果某步骤失败：STEP_FAILED: 步骤编号, 原因

示例输出：
STEP_COMPLETED: [1, 2, 3]
或
STEP_COMPLETED: [1, 2]
STEP_FAILED: 3, DESeq2 安装失败
==========================={file_context_msg}"""
        if validation_hint:
            checklist_reminder += f"\n\n【上次失败后的必做修复提示】\n{validation_hint}"
        filtered_messages.append(
            SystemMessage(
                content=checklist_reminder,
                name="system",
            )
        )
    elif planner_msg:
        # 非清单模式：使用 planner 消息 + validation hint
        system_reminder = (
            f"SYSTEM REMINDER: You MUST follow the instructions in the 'planner' message above exactly. "
            f"Do NOT deviate from the recommended methods (e.g. if it says use fold-change, do NOT use DESeq2). "
            f"Check constraints like sample size (n=1) before writing code.{file_context_msg}"
        )
        if validation_hint:
            system_reminder += f"\n\nVALIDATION NOTICE: {validation_hint}"
        filtered_messages.append(SystemMessage(content=system_reminder, name="system"))
    elif file_context:
        filtered_messages.append(
            SystemMessage(
                content=f"SYSTEM REMINDER: {file_context_msg}",
                name="system",
            )
        )
    
    temp_state = state.copy()
    temp_state["messages"] = filtered_messages
    
    # 【关键修复】确保 InjectedState 所需的 Key 存在
    # 如果 State 中尚未生成这些字段，必须给它们赋予默认值 (None)，
    # 否则 ToolNode 在尝试注入参数时会报 KeyError。
    temp_state["sandbox_session_id"] = state.get("sandbox_session_id", None)
    temp_state["sandbox_available"] = state.get("sandbox_available", None) # Tool 会自行处理 None
    temp_state["sandbox_failed"] = state.get("sandbox_failed", False)
    temp_state["remaining_steps"] = _resolve_agent_remaining_steps(state)
    temp_state["current_step_label"] = state.get("current_step_label", None)  # 工具注入需要
    try:
        # Use temp_state instead of state
        result = coder_agent.invoke(temp_state)
    except Exception as exc:
        logger.exception("Code agent failed to complete task")
        error_text = str(exc).strip()
        if len(error_text) > 1000:
            error_text = error_text[:997] + "..."
        fallback_message = (
            "Code agent encountered an internal error while deciding the next step.\n\n"
            f"Error detail: {error_text}\n\n"
            "Please inspect LLM credentials/network settings and retry once resolved."
        )
        
        # 【关键修复】将 in_progress 的 coder 步骤标记为 failed
        updated_checklist = checklist
        if checklist:
            for step in checklist:
                if step["status"] == "in_progress" and step["agent_name"] == "coder":
                    step["status"] = "failed"
                    step["result"] = f"Agent error: {error_text[:200]}"
                    logger.info(f"Checklist: Marked step {step['step_id']} as failed due to agent error")
            updated_checklist = checklist
        
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=RESPONSE_FORMAT.format("coder", fallback_message),
                        name="coder",
                    )
                ],
                "plan_checklist": updated_checklist,  # 更新 checklist 状态
            },
            goto="supervisor",
        )
    logger.info("Code agent completed task")
    logger.debug(f"Code agent response: {result['messages'][-1].content}")

    result_messages = result.get("messages", [])
    coder_output = result["messages"][-1].content if result.get("messages") else ""
    step_failed_entries = _extract_step_failed_entries(str(coder_output or ""))
    if checklist and step_failed_entries:
        in_progress_ids = {
            int(step["step_id"])
            for step in checklist
            if step.get("status") == "in_progress" and step.get("agent_name") == "coder"
        }
        relevant_entries = [
            (step_id, reason)
            for step_id, reason in step_failed_entries
            if not in_progress_ids or step_id in in_progress_ids
        ]
        recoverable_entries = [
            (step_id, reason)
            for step_id, reason in relevant_entries
            if _is_recoverable_step_failed_reason(reason)
        ]
        if recoverable_entries:
            attempts = int(state.get("coder_step_failed_retry_attempts") or 0) + 1
            if _can_retry_with_budget(
                state=state,
                attempts=attempts,
                explicit_cap=_CODER_STEP_FAILED_RETRY_MAX_ATTEMPTS,
            ):
                hint = _build_step_failed_retry_hint(recoverable_entries)
                retry_notice = (
                    "Recoverable STEP_FAILED detected. Retrying coder step with explicit repair instructions."
                )
                logger.warning(
                    "Coder recoverable STEP_FAILED detected; scheduling retry attempt=%s (cap=%s)",
                    attempts,
                    _CODER_STEP_FAILED_RETRY_MAX_ATTEMPTS,
                )
                return Command(
                    update={
                        "messages": [
                            AIMessage(
                                content=RESPONSE_FORMAT.format("coder", retry_notice),
                                name="coder",
                            )
                        ],
                        "coder_validation_failed": True,
                        "coder_validation_reason": "recoverable_step_failed",
                        "coder_validation_attempts": 0,
                        "coder_validation_hint": hint,
                        "coder_step_failed_retry_attempts": attempts,
                    },
                    goto="supervisor",
                )

    has_saved_script = _message_contains_marker(result_messages, "SAVED_SCRIPT_PATH:")
    has_saved_file = _message_contains_marker(result_messages, "SAVED_FILE:")
    has_saved_plot = _message_contains_marker(result_messages, "SAVED_PLOT_PATH:")
    has_final_answer = _message_contains_marker(result_messages, "FINAL_ANSWER:")
    has_step_completed = _message_contains_marker(result_messages, "STEP_COMPLETED:")
    has_tool_execution = _has_tool_execution_evidence(result_messages)
    has_execution_evidence = (
        has_saved_script
        or has_saved_file
        or has_saved_plot
        or has_tool_execution
    )
    require_saved_script = _should_require_saved_script(checklist)

    if require_saved_script and not has_saved_script:
        attempts = int(state.get("coder_validation_attempts") or 0) + 1
        hint = (
            "Current coder step explicitly requires script persistence. "
            "Please save script to /app/workspace/generated_scripts/<case_id>/... and print "
            "`SAVED_SCRIPT_PATH: ...`."
        )
        updates = {
            "coder_validation_failed": True,
            "coder_validation_reason": "missing_saved_script_path",
            "coder_validation_attempts": attempts,
            "coder_validation_hint": hint,
        }
        if _can_retry_with_budget(
            state=state,
            attempts=attempts,
            explicit_cap=_CODER_VALIDATION_MAX_ATTEMPTS,
        ):
            logger.warning(
                "Code agent validation failed (attempt=%s, cap=%s): script is required but SAVED_SCRIPT_PATH is missing",
                attempts,
                _CODER_VALIDATION_MAX_ATTEMPTS,
            )
            return Command(update=updates, goto="supervisor")

        failure_message = (
            "Code agent validation failed: this task requires script persistence, "
            "but SAVED_SCRIPT_PATH was still missing after retries."
        )
        updates["messages"] = [
            AIMessage(
                content=RESPONSE_FORMAT.format("coder", failure_message),
                name="coder",
            )
        ]
        return Command(update=updates, goto="supervisor")

    if not has_execution_evidence:
        attempts = int(state.get("coder_validation_attempts") or 0) + 1
        hint = (
            "No concrete tool execution evidence detected. "
            "You must run real commands in sandbox (e.g., execute_in_sandbox) and include execution output. "
            "FINAL_ANSWER or STEP_COMPLETED alone is not enough."
        )
        updates = {
            "coder_validation_failed": True,
            "coder_validation_reason": "missing_execution_evidence",
            "coder_validation_attempts": attempts,
            "coder_validation_hint": hint,
        }
        if _can_retry_with_budget(
            state=state,
            attempts=attempts,
            explicit_cap=_CODER_VALIDATION_MAX_ATTEMPTS,
        ):
            logger.warning(
                "Code agent validation failed (attempt=%s, cap=%s): no execution evidence detected",
                attempts,
                _CODER_VALIDATION_MAX_ATTEMPTS,
            )
            return Command(update=updates, goto="supervisor")

        failure_message = (
            "Code agent validation failed: no execution evidence was produced after retry."
        )
        updates["messages"] = [
            AIMessage(
                content=RESPONSE_FORMAT.format("coder", failure_message),
                name="coder",
            )
        ]
        return Command(update=updates, goto="supervisor")

    require_structure_probe = _should_require_structure_probe(
        messages=state.get("messages", []),
        checklist=checklist,
        file_context=file_context,
    )
    if require_structure_probe:
        probe_messages = []
        historical_messages = state.get("messages", [])
        if isinstance(historical_messages, list):
            probe_messages.extend(historical_messages)
        probe_messages.extend(result_messages)
        missing_probe_tools = _missing_structure_probe_tools(probe_messages)
        if missing_probe_tools:
            attempts = int(state.get("coder_validation_attempts") or 0) + 1
            missing_str = ", ".join(missing_probe_tools)
            hint = (
                "Matrix + metadata modeling task detected. Before DESeq2/edgeR/limma fitting, "
                "you must call both structure tools and use their results to choose parser and align samples.\n"
                f"Missing required tool calls: {missing_str}\n"
                "Minimum required sequence:\n"
                "1) infer_table_schema(<count_matrix_file>)\n"
                "2) check_sample_overlap(<count_matrix_file>, <metadata_file>, metadata_sample_col='sample')\n"
                "Then revise script by inferred delimiter/header and sample overlap result, rerun analysis, "
                "and report STEP_COMPLETED or STEP_FAILED with concrete command output."
            )
            updates = {
                "coder_validation_failed": True,
                "coder_validation_reason": "missing_structure_probe_tools",
                "coder_validation_attempts": attempts,
                "coder_validation_hint": hint,
            }
            if _can_retry_with_budget(
                state=state,
                attempts=attempts,
                explicit_cap=_CODER_VALIDATION_MAX_ATTEMPTS,
            ):
                logger.warning(
                    "Code agent validation failed (attempt=%s, cap=%s): missing structure-probe tools (%s)",
                    attempts,
                    _CODER_VALIDATION_MAX_ATTEMPTS,
                    missing_str,
                )
                return Command(update=updates, goto="supervisor")

            failure_message = (
                "Code agent validation failed: required structure-probe tools were still missing "
                "after retries for this matrix+metadata modeling task."
            )
            updates["messages"] = [
                AIMessage(
                    content=RESPONSE_FORMAT.format("coder", failure_message),
                    name="coder",
                )
            ]
            return Command(update=updates, goto="supervisor")

        if not _has_command_execution_evidence(result_messages):
            attempts = int(state.get("coder_validation_attempts") or 0) + 1
            hint = (
                "Matrix + metadata modeling task requires actual model execution after structure checks.\n"
                "You already inspected schema/overlap, now you must run real commands via execute_in_sandbox "
                "(or python_repl_tool) to execute DESeq2/edgeR/limma workflow and provide concrete outputs.\n"
                "Do not stop at 'pending execution'."
            )
            updates = {
                "coder_validation_failed": True,
                "coder_validation_reason": "missing_model_execution",
                "coder_validation_attempts": attempts,
                "coder_validation_hint": hint,
            }
            if _can_retry_with_budget(
                state=state,
                attempts=attempts,
                explicit_cap=_CODER_VALIDATION_MAX_ATTEMPTS,
            ):
                logger.warning(
                    "Code agent validation failed (attempt=%s, cap=%s): structure tools used but no command execution evidence",
                    attempts,
                    _CODER_VALIDATION_MAX_ATTEMPTS,
                )
                return Command(update=updates, goto="supervisor")

            failure_message = (
                "Code agent validation failed: no model execution evidence after retries "
                "for this matrix+metadata modeling task."
            )
            updates["messages"] = [
                AIMessage(
                    content=RESPONSE_FORMAT.format("coder", failure_message),
                    name="coder",
                )
            ]
            return Command(update=updates, goto="supervisor")

    # 检查是否有 python_repl_tool 的返回结果（包含图片）
    image_url = None
    try:
        for msg in result_messages:
            if hasattr(msg, "name") and msg.name == "python_repl_tool":
                if (
                    isinstance(msg.content, str)
                    and msg.content.startswith("{")
                    and msg.content.endswith("}")
                ):
                    payload = json.loads(msg.content)

                    # 只有当payload中确实包含image_path时才提取
                    if "image_path" in payload and payload["image_path"]:
                        image_url = payload["image_path"]
                        print("提取到 image_url:", image_url)
                        logger.info(f"Plot image detected: {image_url}")
                        break
                    else:
                        logger.info("No image_path found in python_repl_tool result")
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.warning(f"Failed to parse python_repl_tool result: {e}")
        print(f"解析 JSON 失败: {e}")

    # 构造返回给前端的消息
    messages_update = []
    print("\n构造 messages_update")

    coder_text = ""
    if result.get("messages"):
        coder_text = str(result["messages"][-1].content or "")
    formatted_content = RESPONSE_FORMAT.format("coder", _sanitize_llm_text(coder_text))
    # print("格式化后的 content:", formatted_content)
    messages_update.append(
        AIMessage(
            content=formatted_content,
            name="coder",
        )
    )

    # 1. 透传工具产生的关键完成标记，便于 supervisor 正确终止流程
    try:
        artifact_lines: list[str] = []
        for msg in result_messages:
            content_str = getattr(msg, "content", "")
            if not isinstance(content_str, str):
                continue
            for line in content_str.splitlines():
                stripped = line.strip()
                if (
                    stripped.startswith("SAVED_SCRIPT_PATH:")
                    or stripped.startswith("SAVED_PLOT_PATH:")
                    or stripped.startswith("SAVED_FILE:")
                    or "Successfully removed sandbox" in stripped
                ):
                    artifact_lines.append(stripped)
        # 去重但保持顺序
        seen = set()
        deduped = [x for x in artifact_lines if not (x in seen or seen.add(x))]
        for mark in deduped:
            messages_update.append(AIMessage(content=mark, name="coder"))
    except Exception as _:
        # 不阻断主流程
        pass

    # 2. 如果有图片，添加图片显示消息
    if image_url:
        print("检测到 image_url，添加图片消息...")
        # 使用相对路径，利用Next.js代理
        # image_url 格式: "/static/figures/plot_xxx.png"
        image_message = f"![Generated Plot]({image_url})"
        messages_update.append(
            AIMessage(
                content=image_message,
                name="coder",
            )
        )

    updates = {
        "messages": messages_update,
        "coder_validation_failed": False,
        "coder_validation_reason": None,
        "coder_validation_attempts": 0,
        "coder_validation_hint": None,
        "coder_step_failed_retry_attempts": 0,
    }

    # Propagate compression state
    if was_compressed:
        updates["context_index"] = updated_index
        updates["compression_count"] = int(state.get("compression_count") or 0) + 1

    # 3. 透传 sandbox 状态回全局 State，确保 InjectedState 能读到最新值
    for key in ("sandbox_session_id", "sandbox_available", "sandbox_failed"):
        if key in result:
            updates[key] = result[key]
    
    # === 解析 coder 输出，更新清单状态 ===
    if checklist:
        coder_output = result["messages"][-1].content if result.get("messages") else ""
        evidence_blob = "\n".join(
            str(getattr(m, "content", "") or "")
            for m in result_messages
        )
        updated_checklist = _update_checklist_from_output(
            checklist,
            coder_output,
            evidence_text=evidence_blob,
        )
        updates["plan_checklist"] = updated_checklist
        logger.info(f"Updated checklist after coder execution")
    
    return Command(update=updates, goto="supervisor")


def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    checklist = state.get("plan_checklist", [])
    
    # --- Auto context compression ---
    original_messages = state.get("messages", [])
    compressed_msgs, updated_index, was_compressed = _compress_context(
        original_messages,
        state,
        target_llm_type=AGENT_LLM_MAP["browser"],
    )
    temp_state = state.copy()
    temp_state["messages"] = compressed_msgs

    try:
        result = browser_agent.invoke(temp_state)
        logger.info("Browser agent completed task")
        logger.debug(f"Browser agent response: {result['messages'][-1].content}")
        
        # 更新 checklist：仅推进一个 in_progress browser 步骤，避免误批量完成
        browser_text = str(result["messages"][-1].content or "")
        has_browser_error = any(
            marker in browser_text
            for marker in ("Traceback", "Exception", "Error:", "Browser failed")
        )
        if checklist and browser_text.strip() and not has_browser_error:
            target_step = next(
                (
                    step
                    for step in checklist
                    if step.get("status") == "in_progress" and step.get("agent_name") == "browser"
                ),
                None,
            )
            if target_step is not None:
                target_step["status"] = "completed"
                logger.info("Checklist: Marked browser step %s as completed", target_step["step_id"])
        
        updates = {
            "messages": [
                AIMessage(
                    content=RESPONSE_FORMAT.format(
                        "browser", result["messages"][-1].content
                    ),
                    name="browser",
                )
            ],
        }
        if checklist:
            updates["plan_checklist"] = checklist
        if was_compressed:
            updates["context_index"] = updated_index
            updates["compression_count"] = int(state.get("compression_count") or 0) + 1
        
        return Command(update=updates, goto="supervisor")
    except Exception as exc:
        logger.exception("Browser agent failed to complete task")
        error_text = str(exc).strip()[:200]
        
        # 将 in_progress 的 browser 步骤标记为 failed
        if checklist:
            target_step = next(
                (
                    step
                    for step in checklist
                    if step.get("status") == "in_progress" and step.get("agent_name") == "browser"
                ),
                None,
            )
            if target_step is not None:
                target_step["status"] = "failed"
                target_step["result"] = f"Agent error: {error_text}"
                logger.info("Checklist: Marked browser step %s as failed", target_step["step_id"])
        
        updates = {
            "messages": [
                AIMessage(
                    content=RESPONSE_FORMAT.format("browser", f"Browser failed: {error_text}"),
                    name="browser",
                )
            ],
        }
        if checklist:
            updates["plan_checklist"] = checklist
        if was_compressed:
            updates["context_index"] = updated_index
            updates["compression_count"] = int(state.get("compression_count") or 0) + 1
        
        return Command(update=updates, goto="supervisor")


def supervisor_node(
    state: State,
) -> Command[Literal["expert", "researcher", "coder", "browser", "reporter", "__end__"]]:
    """Supervisor node that decides which agent should act next and maintains checklist state."""
    logger.info("Supervisor evaluating next action")
    
    checklist = state.get("plan_checklist")
    
    # === 记录当前 checklist 状态 ===
    if checklist:
        status_symbols = {
            "completed": "✔",
            "failed": "!",
            "in_progress": "<",
            "pending": "○"
        }
        logger.info("=== Current checklist state ===")
        for s in checklist:
            symbol = status_symbols.get(s['status'], '?')
            logger.info(f"  [{symbol}] Step {s['step_id']}: {s['agent_name']} - {s['title']}")
            logger.info(f"      描述: {s.get('description', 'N/A')}")
            logger.info(f"      注意: {s.get('note', 'N/A')}")
        logger.info("=== End checklist state ===")
    
    # === LLM 决策逻辑 ===
    original_messages = state.get("messages", [])

    # --- Auto context compression ---
    compressed_msgs, updated_index, was_compressed = _compress_context(
        original_messages,
        state,
        target_llm_type=AGENT_LLM_MAP["supervisor"],
    )

    # --- Enforce plan order (if available) ---
    plan = state.get("full_plan")
    if not checklist and isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            planner_idx = -1
            for idx in range(len(compressed_msgs) - 1, -1, -1):
                msg = compressed_msgs[idx]
                msg_name = getattr(msg, "name", "")
                if isinstance(msg, dict):
                    msg_name = msg.get("name", msg_name)
                if msg_name == "planner":
                    planner_idx = idx
                    break
            if planner_idx != -1:
                current_idx = planner_idx
                for step in steps:
                    agent_name = step.get("agent_name") if isinstance(step, dict) else None
                    if not agent_name or agent_name not in TEAM_MEMBERS:
                        continue
                    msg_idx = _find_message_index(compressed_msgs, agent_name, current_idx)
                    if msg_idx is None:
                        logger.info(f"Supervisor enforcing plan order: next={agent_name}")
                        sup_updates = {"next": agent_name}
                        # 关键修复：plan_order_enforcement 提前 return 时，
                        # 必须把目标 agent 对应的 pending 步骤标记为 in_progress，
                        # 否则 coder 收到的清单全是 [ ] 没有 [>]，导致无事可做返回空内容
                        if checklist:
                            for s in checklist:
                                if s["status"] == "pending" and s["agent_name"] == agent_name:
                                    s["status"] = "in_progress"
                                    logger.info(f"  Plan enforcement: Step {s['step_id']} -> in_progress")
                            sup_updates["plan_checklist"] = checklist
                        if was_compressed:
                            sup_updates["context_index"] = updated_index
                            sup_updates["compression_count"] = int(state.get("compression_count") or 0) + 1
                        return Command(goto=agent_name, update=sup_updates)
                    current_idx = msg_idx

    # --- Further filter for Supervisor: User + Plan + context_index/summary + Recent ---
    filtered_messages = []
    
    user_msg = _get_first_user_message(compressed_msgs)
    if user_msg is not None:
        filtered_messages.append(user_msg)
    
    # 2. Plan (仅在无 checklist 时包含原始 planner 消息)
    if not checklist:
        planner_msg = next((m for m in reversed(original_messages) if getattr(m, "name", "") == "planner"), None)
        if planner_msg:
            filtered_messages.append(planner_msg)
        
    # 3. Keep context_index and context_summary
    for m in compressed_msgs:
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ("context_index", "context_summary"):
            if m not in filtered_messages:
                filtered_messages.append(m)

    # 4. Recent context to track immediate state
    recent_count = 100
    recent_msgs = compressed_msgs[-recent_count:] if len(compressed_msgs) > recent_count else compressed_msgs
    
    for msg in recent_msgs:
        if msg != user_msg and msg not in filtered_messages:
            filtered_messages.append(msg)
    
    # 4. 如果有 checklist，添加格式化的 checklist 信息给 LLM
    if checklist:
        checklist_text = _format_checklist_for_supervisor(checklist)
        filtered_messages.append(
            SystemMessage(
                content=f"""=== 当前任务清单状态 ===
{checklist_text}

请根据上述清单状态和最近的 agent 执行结果：
1. 判断哪些步骤需要更新状态（completed/failed/in_progress）
2. 决定下一个应该执行的 agent

注意：
- 如果 agent 的输出包含 "STEP_COMPLETED" 或 "FINAL_ANSWER"，标记为 completed
- 如果 agent 的输出包含错误或 "STEP_FAILED"，标记为 failed
- 将下一个要执行的步骤标记为 in_progress
- 所有非-reporter 步骤完成后，路由到 reporter
- 所有步骤完成后，返回 FINISH
===========================""",
                name="system",
            )
        )
            
    # Use filtered messages for the prompt
    temp_state = state.copy()
    temp_state["messages"] = filtered_messages
    
    messages = apply_prompt_template("supervisor", temp_state)
    
    # 调用 LLM 获取结构化输出
    try:
        response = (
            get_llm_by_type(AGENT_LLM_MAP["supervisor"])
            .with_structured_output(Router)
            .invoke(messages)
        )
    except Exception as exc:
        logger.exception("Supervisor LLM invocation failed")
        warning_text = _build_llm_system_warning("supervisor", exc)
        _mark_in_progress_steps_failed(
            checklist,
            reason=warning_text,
            agent_name=None,
        )

        final_text = (
            f"{warning_text}\n"
            "FINAL_ANSWER: Workflow stopped due to orchestrator model failure. "
            "Please resolve model quota/connectivity and retry."
        )
        updates = {
            "next": "FINISH",
            "messages": [
                AIMessage(
                    content=RESPONSE_FORMAT.format("reporter", final_text),
                    name="reporter",
                )
            ],
        }
        if checklist:
            updates["plan_checklist"] = checklist
        if was_compressed:
            updates["context_index"] = updated_index
            updates["compression_count"] = int(state.get("compression_count") or 0) + 1
        return Command(goto="__end__", update=updates)

    if response is None:
        logger.warning("Supervisor returned None response, defaulting to FINISH")
        goto = _next_checklist_agent(checklist)
        step_updates = []
    else:
        goto = _sanitize_supervisor_route(response.get("next", "FINISH"), checklist)
        raw_updates = response.get("step_updates", []) if isinstance(response, dict) else []
        step_updates = raw_updates if isinstance(raw_updates, list) else []

    if goto not in _ROUTER_ALLOWLIST:
        logger.warning("Supervisor produced non-allowlisted next=%r, fallback route applied", goto)
        goto = _next_checklist_agent(checklist)

    logger.info(f"Supervisor LLM response (guarded): next={goto}, updates={step_updates}")
    
    # === 应用 step_updates 更新 checklist ===
    if checklist and step_updates:
        logger.info("Applying %d step updates from LLM (guarded)", len(step_updates))
        _apply_guarded_step_updates(checklist, step_updates, next_agent=goto)
        # Re-sanitize with updated checklist state to keep routing and checklist aligned.
        goto = _sanitize_supervisor_route(goto, checklist)

    # Ensure routed worker actually has an in_progress step to execute.
    if checklist:
        _ensure_routed_step_in_progress(checklist, goto)

    # === 处理路由 ===
    if goto == "FINISH":
        goto = "__end__"
        logger.info("Workflow completed")
    elif goto == "reporter":
        logger.info("Routing to reporter")
    else:
        logger.info(f"Supervisor delegating to: {goto}")

    # 返回更新后的 checklist、压缩状态和路由
    update_dict = {"next": goto if goto != "__end__" else "FINISH"}
    if checklist:
        update_dict["plan_checklist"] = checklist
    if was_compressed:
        update_dict["context_index"] = updated_index
        update_dict["compression_count"] = int(state.get("compression_count") or 0) + 1
    
    return Command(goto=goto, update=update_dict)


def planner_node(state: State) -> Command[Literal["human_feedback", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    planner_state = state.copy()
    planner_state["RAG_SCOPE_SUMMARY"] = _build_local_rag_scope_summary()
    planner_file_context = state.get("initial_file_context")
    if not planner_file_context:
        manifest_path = state.get("capsule_manifest_path")
        if manifest_path:
            try:
                planner_file_context = Path(str(manifest_path)).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception:
                planner_file_context = None

    if planner_file_context:
        planner_file_context = (
            _clamp_context_block(
                planner_file_context,
                max_chars=_PLANNER_FILE_CONTEXT_MAX_CHARS,
                max_lines=_PLANNER_FILE_CONTEXT_MAX_LINES,
            )
            or planner_file_context
        )
        planner_state["PLANNER_FILE_CONTEXT"] = planner_file_context
    else:
        planner_state["PLANNER_FILE_CONTEXT"] = "(none)"
    messages = apply_prompt_template("planner", planner_state)
    # whether to enable deep thinking mode
    llm = get_llm_by_type("basic")
    if state.get("deep_thinking_mode"):
        llm = get_llm_by_type("reasoning")
    if state.get("search_before_planning"):
        search_query = _build_planner_search_query(state)
        normalized_results: list[dict[str, str]] = []
        if search_query:
            try:
                searched_content = searxng_tool.invoke({"query": search_query})
            except Exception:
                logger.exception("Planner pre-search failed; continue without pre-search context")
                searched_content = []

            if isinstance(searched_content, str):
                try:
                    searched_content = json.loads(searched_content)
                except Exception:
                    searched_content = []
            if isinstance(searched_content, dict):
                if isinstance(searched_content.get("results"), list):
                    searched_content = searched_content["results"]
                elif isinstance(searched_content.get("data"), list):
                    searched_content = searched_content["data"]
                else:
                    searched_content = []
            if isinstance(searched_content, list):
                for elem in searched_content:
                    if isinstance(elem, dict):
                        title = elem.get("title") or elem.get("name") or ""
                        content = (
                            elem.get("content")
                            or elem.get("snippet")
                            or elem.get("summary")
                            or ""
                        )
                        if title or content:
                            normalized_results.append(
                                {"title": title, "content": content}
                            )
                    elif isinstance(elem, str):
                        text = elem.strip()
                        if text:
                            normalized_results.append({"title": "", "content": text})

        if normalized_results:
            messages = deepcopy(messages)
            messages[
                -1
            ].content += f"\n\n# Relative Search Results\n\n{json.dumps(normalized_results, ensure_ascii=False)}"
    full_response = ""
    try:
        stream = llm.stream(messages)
        for chunk in stream:
            full_response += chunk.content
    except Exception as exc:
        logger.exception("Planner LLM invocation failed")
        warning_text = _build_llm_system_warning("planner", exc)
        emergency_text = (
            f"{warning_text}\n"
            "FINAL_ANSWER: Unable to generate a valid plan because model service is unavailable."
        )
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=RESPONSE_FORMAT.format("reporter", emergency_text),
                        name="reporter",
                    )
                ],
            },
            goto="__end__",
        )
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: \n {full_response}")

    parsed_plan = _parse_planner_plan(full_response)
    if parsed_plan is None:
        logger.warning("Planner response is not a valid JSON; applying fallback plan")
        parsed_plan = _build_fallback_plan(state)

    normalized_steps = _normalize_plan_steps(parsed_plan)
    if not normalized_steps:
        logger.warning("Planner plan has no valid steps; rebuilding fallback plan")
        parsed_plan = _build_fallback_plan(state)
        normalized_steps = _normalize_plan_steps(parsed_plan)

    if not normalized_steps:
        logger.warning("Fallback plan still empty; forcing reporter-only step")
        normalized_steps = [
            {
                "agent_name": "reporter",
                "title": "Generate final answer",
                "description": "Provide the best possible direct answer in user's language.",
                "note": "Must include FINAL_ANSWER.",
            }
        ]

    normalized_plan = {
        "thought": str(parsed_plan.get("thought") or ""),
        "title": str(parsed_plan.get("title") or "") or "Execution Plan",
        "steps": normalized_steps,
    }
    normalized_plan_json = json.dumps(normalized_plan, ensure_ascii=False)

    checklist = []
    for idx, step in enumerate(normalized_steps):
        checklist.append(
            {
                "step_id": idx + 1,
                "agent_name": step.get("agent_name"),
                "title": step.get("title"),
                "description": step.get("description"),
                "note": step.get("note", ""),
                "status": "pending",
                "result": None,
            }
        )
    logger.info(f"Built plan checklist with {len(checklist)} steps")

    return Command(
        update={
            "messages": [AIMessage(content=normalized_plan_json, name="planner")],
            "full_plan": normalized_plan_json,
            "plan_checklist": checklist,  # 新增：带状态的步骤清单
        },
        goto="human_feedback",
    )


def human_feedback_node(
    state,
) -> Command[Literal["planner", "supervisor", "__end__"]]:
    full_plan = state.get("full_plan", "")
    # check if the plan is auto accepted
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    if not auto_accepted_plan:
        logger.info("human feedback is starting")
        feedback = interrupt("Please Review the Plan.")

        # if the feedback is not accepted, return the planner node
        if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                },
                goto="planner",
            )
        elif feedback and str(feedback).upper().startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            raise TypeError(f"Interrupt value of {feedback} is not supported.")

    # if the plan is accepted, run the following node
    plan_iterations = int(state.get("plan_iterations", 0) or 0)
    goto = "supervisor"

    parsed_plan: dict[str, Any] | None = None
    if isinstance(full_plan, dict):
        parsed_plan = full_plan
    elif isinstance(full_plan, str):
        parsed_plan = _parse_planner_plan(full_plan)
        if parsed_plan is None:
            repaired = repair_json_output(full_plan)
            parsed_plan = _parse_planner_plan(repaired)

    if parsed_plan is None:
        logger.warning("Planner response is not a valid JSON in human_feedback; using fallback plan")
        parsed_plan = _build_fallback_plan(state)

    normalized_steps = _normalize_plan_steps(parsed_plan)
    if not normalized_steps:
        logger.warning("Plan in human_feedback has no valid steps; rebuilding fallback plan")
        parsed_plan = _build_fallback_plan(state)
        normalized_steps = _normalize_plan_steps(parsed_plan)

    normalized_plan = {
        "thought": str(parsed_plan.get("thought") or ""),
        "title": str(parsed_plan.get("title") or "") or "Execution Plan",
        "steps": normalized_steps,
    }
    plan_iterations += 1

    updates: dict[str, Any] = {
        "full_plan": normalized_plan,
        "plan_iterations": plan_iterations,
    }

    if not state.get("plan_checklist"):
        checklist = []
        for idx, step in enumerate(normalized_steps):
            checklist.append(
                {
                    "step_id": idx + 1,
                    "agent_name": step.get("agent_name"),
                    "title": step.get("title"),
                    "description": step.get("description"),
                    "note": step.get("note", ""),
                    "status": "pending",
                    "result": None,
                }
            )
        updates["plan_checklist"] = checklist

    return Command(
        update=updates,
        goto=goto,
    )


def coordinator_node(
    state: State,
) -> Command[Literal["planner", "__end__"]]:
    """Deterministic gateway: small-talk ends here, otherwise route to planner."""
    logger.info("Coordinator gateway evaluating request type")
    latest_user = _get_latest_user_message(state.get("messages", []))
    user_text = _message_content(latest_user)

    if _is_small_talk_request(user_text):
        reply = _small_talk_reply(user_text)
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=reply, name="coordinator")]},
        )

    return Command(goto="planner", update={})


def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    
    # --- Auto context compression ---
    original_messages = state.get("messages", [])
    compressed_msgs, updated_index, was_compressed = _compress_context(
        original_messages,
        state,
        target_llm_type=AGENT_LLM_MAP["reporter"],
    )

    # --- Further filter for Reporter: User + Plan + context_index/summary + Coder result ---
    filtered_messages = []
    
    # === 检查是否有清单模式 ===
    checklist = state.get("plan_checklist", [])
    has_checklist = bool(checklist)
    
    # 1. User Input
    user_msg = _get_first_user_message(compressed_msgs)
    if user_msg is not None:
        filtered_messages.append(user_msg)
        
    # 2. 【关键改动】如果有 checklist，使用清单摘要代替 planner 消息
    if has_checklist:
        # 构建清单完成摘要
        checklist_summary_lines = ["=== 任务执行摘要 ==="]
        for step in checklist:
            status_text = {
                "completed": "✔ 已完成",
                "failed": "✗ 失败",
                "in_progress": "→ 进行中",
                "pending": "○ 待执行"
            }.get(step["status"], step["status"])
            checklist_summary_lines.append(
                f"{status_text} | 步骤{step['step_id']}: {step['title']}"
            )
            if step.get("result"):
                checklist_summary_lines.append(f"    结果: {step['result']}")
        checklist_summary = "\n".join(checklist_summary_lines)
        filtered_messages.append(SystemMessage(content=checklist_summary, name="plan_summary"))
    else:
        # 没有清单时使用 planner 消息
        planner_msg = next((m for m in reversed(compressed_msgs) if getattr(m, "name", "") == "planner"), None)
        if planner_msg:
            filtered_messages.append(planner_msg)

    # 3. Keep context_index and context_summary
    for m in compressed_msgs:
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ("context_index", "context_summary"):
            if m not in filtered_messages:
                filtered_messages.append(m)

    # 4. Researcher/Expert outputs (latest after planner)
    for name in ("researcher", "expert"):
        msg = _last_message_after(compressed_msgs, name, "planner")
        if msg is None and name == "expert":
            msg = next(
                (m for m in reversed(compressed_msgs) if getattr(m, "name", "") == "expert"),
                None,
            )
        if msg:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                content = _clamp_message_block(content, max_chars=2000, max_lines=120) or content
                filtered_messages.append(AIMessage(content=content, name=name))
            else:
                filtered_messages.append(msg)

    failure_hint = _build_reporter_failure_hint(checklist)
    if failure_hint:
        filtered_messages.append(SystemMessage(content=failure_hint, name="failure_hint"))

    # Include multiple recent coder outputs so FINAL_ANSWER is not lost when the
    # latest coder message is only an artifact marker or image placeholder.
    coder_msgs = _collect_recent_named_messages(
        compressed_msgs,
        "coder",
        after_name="planner",
        max_items=8,
    )
    if not coder_msgs:
        coder_msgs = _collect_recent_named_messages(
            compressed_msgs,
            "coder",
            max_items=8,
        )

    if coder_msgs:
        coder_blocks: list[str] = []
        for msg in coder_msgs:
            content = getattr(msg, "content", "")
            if isinstance(msg, dict):
                content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            coder_blocks.append(content.strip())
        if coder_blocks:
            merged_coder_context = "\n\n---\n\n".join(coder_blocks)
            merged_coder_context = (
                _clamp_message_block(merged_coder_context, max_chars=4500, max_lines=260)
                or merged_coder_context
            )
            filtered_messages.append(AIMessage(content=merged_coder_context, name="coder"))
    
    temp_state = state.copy()
    temp_state["messages"] = filtered_messages

    messages = apply_prompt_template("reporter", temp_state)
    try:
        response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(messages)
        logger.debug(f"Current state messages: {state['messages']}")
        logger.debug(f"reporter response: {response}")
        reporter_text = _sanitize_llm_text(str(getattr(response, "content", "") or ""))
        reporter_text = _normalize_reporter_images_to_markdown(reporter_text)
    except Exception as exc:
        logger.exception("Reporter LLM invocation failed")
        warning_text = _build_llm_system_warning("reporter", exc)
        reporter_text = (
            f"{warning_text}\n"
            "FINAL_ANSWER: Unable to generate final natural-language report automatically."
        )

    updates = {
        "messages": [
            AIMessage(
                content=RESPONSE_FORMAT.format("reporter", reporter_text),
                name="reporter",
            )
        ],
    }
    if checklist:
        target_step = next(
            (
                step
                for step in checklist
                if step.get("status") == "in_progress" and step.get("agent_name") == "reporter"
            ),
            None,
        )
        if target_step is not None:
            target_step["status"] = "completed"
            logger.info(
                "Checklist: Marked reporter step %s as completed",
                target_step.get("step_id"),
            )
        updates["plan_checklist"] = checklist

    if was_compressed:
        updates["context_index"] = updated_index
        updates["compression_count"] = int(state.get("compression_count") or 0) + 1

    return Command(update=updates, goto="supervisor")


def expert_node(state: State) -> Command[Literal["planner"]]:
    """Expert node that retrieve the solution with RAG"""
    logger.info("expert retrieve the solution")
    
    # --- Auto context compression ---
    original_messages = state.get("messages", [])
    compressed_msgs, updated_index, was_compressed = _compress_context(
        original_messages,
        state,
        target_llm_type=AGENT_LLM_MAP["expert"],
    )

    # Expert only needs user input + context_index/summary + file context
    filtered_messages = []
    
    user_msg = _get_first_user_message(compressed_msgs)
    if user_msg is not None:
        filtered_messages.append(user_msg)

    # Keep context_index and context_summary
    for m in compressed_msgs:
        msg_name = getattr(m, "name", "")
        if isinstance(m, dict):
            msg_name = m.get("name", msg_name)
        if msg_name in ("context_index", "context_summary"):
            if m not in filtered_messages:
                filtered_messages.append(m)
    
    file_context = state.get("initial_file_context")
    if file_context:
        file_context = _clamp_context_block(file_context, max_chars=1500, max_lines=100)
        filtered_messages.append(
            SystemMessage(
                content=f"[AVAILABLE DATA FILES IN ./]\n{file_context}\n(You can read these files directly using their filenames)",
                name="system",
            )
        )
    
    selected_skill: Optional[str] = None
    selected_skill_reason: Optional[str] = None
    selected_skill_tools: list[str] = []

    if _skills_dynamic_enabled():
        selected_skill, selected_skill_reason, selected_skill_tools = _select_skill_for_expert(
            state, filtered_messages
        )
        _ensure_skills_section_in_messages(filtered_messages, read_tool="read_skill")
        selection_msg = _format_skill_selection_system_message(
            selected_skill, selected_skill_reason, selected_skill_tools
        )
        if selection_msg:
            filtered_messages.insert(0, {"role": "system", "content": selection_msg})

    temp_state = state.copy()
    temp_state["messages"] = filtered_messages
    temp_state["remaining_steps"] = _resolve_agent_remaining_steps(state)
    if _skills_dynamic_enabled():
        temp_state["selected_skill"] = selected_skill
        temp_state["selected_skill_reason"] = selected_skill_reason
        temp_state["selected_skill_tools"] = selected_skill_tools
    try:
        result = expert_agent.invoke(temp_state)
        logger.info("expert agent completed task")
        logger.debug(f"Expert agent response: {result['messages'][-1].content}")
        expert_text = result["messages"][-1].content
    except Exception as exc:
        logger.exception("Expert agent failed to complete task")
        expert_text = _build_llm_system_warning("expert", exc)

    updates: dict[str, Any] = {
        "messages": [
            AIMessage(
                content=RESPONSE_FORMAT.format("expert", expert_text),
                name="expert",
            )
        ]
    }
    if _skills_dynamic_enabled():
        updates["selected_skill"] = selected_skill
        updates["selected_skill_reason"] = selected_skill_reason
        updates["selected_skill_tools"] = selected_skill_tools
    if was_compressed:
        updates["context_index"] = updated_index
        updates["compression_count"] = int(state.get("compression_count") or 0) + 1

    return Command(update=updates, goto="planner")
