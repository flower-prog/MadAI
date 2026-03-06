import operator
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from src.config import TEAM_MEMBERS

# Define routing options
OPTIONS = TEAM_MEMBERS + ["FINISH"]


class StepUpdate(TypedDict, total=False):
    """单个步骤的状态更新"""
    step_id: int
    status: Literal["pending", "in_progress", "completed", "failed"]
    result: str | None  # 可选的结果/原因说明


class Router(TypedDict, total=False):
    """Supervisor 路由决策的结构化输出"""
    next: Literal["expert", "researcher", "coder", "browser", "reporter", "FINISH"]
    step_updates: list[StepUpdate]  # 需要更新的步骤状态列表（可选）


class PlanStep(TypedDict):
    """单个计划步骤"""
    step_id: int                # 步骤编号（从1开始）
    agent_name: str             # "expert" | "researcher" | "coder" | "browser" | "reporter"
    title: str                  # 步骤标题
    description: str            # 详细描述
    note: str                   # 约束/注意事项
    status: Literal["pending", "in_progress", "completed", "failed"]
    result: str | None          # 执行结果摘要
    
class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Constants
    TEAM_MEMBERS: list[str]

    # Runtime Variables
    next: str
    full_plan: str
    thread_id: str | None
    output_session_dir: str | None
    deep_thinking_mode: bool
    search_before_planning: bool
    auto_accepted_plan: bool
    plan_iterations: int = 0
    case_dir: str | None
    local_capsule_dir: str | None
    capsule_manifest_path: str | None
    files_listed: bool
    sandbox_available: bool
    sandbox_failed: bool
    initial_file_context: str | None # [Mod] Store file structure context
    sandbox_session_id: str | None # [Mod] Store sandbox session ID
    plan_checklist: list[PlanStep] | None  # 带状态的步骤清单（清单模式）
    current_step_label: str | None  # 当前步骤标签（用于日志/调试）
    coder_validation_failed: bool
    coder_validation_reason: str | None
    coder_validation_attempts: int
    coder_validation_hint: str | None
    coder_step_failed_retry_attempts: int
    remaining_steps: Annotated[int, operator.add]  # 剩余可执行步骤数（用于工具调用限制）
    selected_skill: str | None  # Selected skill name for dynamic tool injection
    selected_skill_reason: str | None  # Optional reason from selector
    selected_skill_tools: list[str] | None  # Tool names authorized by selected skill

    # Context compression: accumulated index of saved files (persists across compressions)
    context_index: list[dict] | None  # [{"id": ..., "path": ..., "title": ..., "type": ...}]
    compression_count: int | None  # Number of times context has been compressed (None = 0)
