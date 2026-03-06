import logging
from pathlib import Path
from src.config import TEAM_MEMBERS
from src.graph import build_graph_with_memory
from src.tools import python_repl_tool
from src.utils.capsule_utils import snapshot_capsule_contents
from src.utils.runtime import detect_sandbox_availability, format_env_note
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)
    # Also set the root logger handlers to DEBUG level
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
# graph = build_graph()
graph = build_graph_with_memory()


def _load_text_excerpt(path: str | None, *, max_chars: int = 8000, max_lines: int = 240) -> str | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return None
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("[Manifest truncated by lines]")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[Manifest truncated by chars]"
    text = text.strip()
    return text or None


def run_agent_workflow(
    user_input: str,
    debug: bool = False,
    deep_thinking_mode: bool = True,
    search_before_planning: bool = False,
    pass_through_expert: bool = False,
    auto_accepted_plan: bool = False,
    config: dict[str, Any] | None = None,
    *,
    case_dir: str | None = None,
    local_capsule_dir: str | None = None,
    sandbox_available: bool | None = None,
):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        deep_thinking_mode: If True, enables reasoning model
        search_before_planning: If True, search on web before planning
        pass_through_expert: Deprecated; retained for API compatibility.
        auto_accepted_plan: If True, planner goto supervisor
    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input}")
    if sandbox_available is None:
        sandbox_available = detect_sandbox_availability()
    if case_dir and not local_capsule_dir:
        local_capsule_dir = case_dir
    manifest_path = snapshot_capsule_contents(case_dir, local_capsule_dir)
    initial_file_context = _load_text_excerpt(manifest_path)
    decorated_messages = []
    for note in format_env_note(sandbox_available, manifest_path):
        decorated_messages.append({"role": "system", "content": note})
    decorated_messages.append({"role": "user", "content": user_input})
    initial_state = {
        # Constants
        "TEAM_MEMBERS": TEAM_MEMBERS,
        # Runtime Variables
        "messages": decorated_messages,
        "deep_thinking_mode": deep_thinking_mode,
        "search_before_planning": search_before_planning,
        "auto_accepted_plan": auto_accepted_plan,
        "case_dir": case_dir,
        "local_capsule_dir": local_capsule_dir,
        "capsule_manifest_path": manifest_path,
        "files_listed": bool(manifest_path),
        "initial_file_context": initial_file_context,
        "sandbox_available": sandbox_available,
        "sandbox_failed": False,
        # 清单/日志辅助字段：保证 InjectedState 注入时 key 一定存在
        "current_step_label": None,
        "remaining_steps": 100,  # 剩余可执行步骤数（用于工具调用限制）
    }

    # 确保配置包含必要的 thread_id
    if config is None:
        import uuid

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    elif "configurable" not in config or "thread_id" not in config["configurable"]:
        import uuid

        if "configurable" not in config:
            config["configurable"] = {}
        config["configurable"]["thread_id"] = str(uuid.uuid4())
    thread_id = str(config.get("configurable", {}).get("thread_id"))
    initial_state["thread_id"] = thread_id
    initial_state["output_session_dir"] = f"sessions/{thread_id}"

    result = graph.invoke(
        input=initial_state,
        config=config,
    )

    # Assuming 'result' contains the Python code to execute
    python_code = result.get("python_code")  # Extract the code from the result
    logger.info(f"Extracted Python code: {python_code}")
    if python_code:
        logger.info("Executing generated Python code...")
        python_result = python_repl_tool(python_code)  # Execute the generated code
        logger.info("Code executed successfully.")
        logger.debug(f"Python execution result: {python_result}")

    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
