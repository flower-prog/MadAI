from typing import Literal

# Define available LLM types
LLMType = Literal["basic", "reasoning", "vision", "coding"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",  # 协调默认使用basic llm
    "planner": "reasoning",  # 计划默认使用reasoning llm
    "supervisor": "basic",  # 决策使用basic llm
    "researcher": "basic",  # 简单搜索任务使用basic llm
    "coder": "coding",  # 编程任务使用 coding llm (Azure Codex)
    # 浏览器 agent 主要负责触发 browser_tool；真正的视觉/页面操作由工具内部模型完成。
    # 这里保持 basic，避免在未配置 VL_* 时导入即失败。
    "browser": "basic",
    "reporter": "basic",  # 编写报告使用basic llm
    "expert": "basic",  # 检索任务流程使用basic llm
}
