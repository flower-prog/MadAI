from __future__ import annotations

from typing import Any, Callable

from .snapshots import FileSnapshotStore
from .nodes import (
    clinical_assisstment_node,
    orchestrator_node,
    protocol_node,
    reporter_node,
)
from .types import AgentName, GraphState, ensure_state, state_to_dict


class SimpleAgentGraph:
    """MedAI 的轻量路由图。

    当前流程遵循固定主链路：
    `orchestrator -> clinical_assisstment -> protocol -> reporter`

    其中 calculator 仍然作为 `clinical_assisstment` 节点下的子执行器。
    `reporter` 节点负责判断本轮结果是否通过，并在需要时回退到
    `orchestrator`，最多执行三轮总迭代。
    """

    def __init__(self) -> None:
        """注册这个轻量图中可用的 MedAI 路由节点。"""
        self.node_map: dict[AgentName, Callable[[GraphState], GraphState]] = {
            "orchestrator": orchestrator_node,
            "clinical_assisstment": clinical_assisstment_node,
            "protocol": protocol_node,
            "reporter": reporter_node,
        }

    def compile(self, checkpointer: Any | None = None) -> "SimpleAgentGraph":
        """兼容 LangGraph 的 compile 接口，但当前直接返回图本身。"""
        return self

    def invoke(self, input: GraphState | dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """按路由顺序执行节点，直到结束、失败或触发防死循环保护。"""
        state = ensure_state(input)
        snapshot_store = _build_snapshot_store(config)
        if snapshot_store is not None:
            snapshot_store.record(state, phase="input")
        current_agent: AgentName | str = cast_agent_name(state.next_agent or "orchestrator")
        max_steps = max(
            14,
            (state.max_reporter_attempts * len(self.node_map)) + 2,
        )

        step_count = 0
        while current_agent != "FINISH":
            step_count += 1
            if step_count > max_steps:
                state.status = "failed"
                state.errors.append(
                    f"Graph exceeded the maximum routed steps ({max_steps}) and was stopped to avoid a loop."
                )
                state.next_agent = "FINISH"
                break

            node = self.node_map.get(cast_agent_name(current_agent))
            if node is None:
                state.status = "failed"
                state.errors.append(f"Unknown graph node requested: {current_agent}")
                state.next_agent = "FINISH"
                break

            state.next_agent = None
            state = node(state)
            if snapshot_store is not None:
                snapshot_store.record(state, phase="node", agent_name=str(current_agent))
            if state.next_agent is None:
                if state.status in {"completed", "failed"}:
                    break
                state.status = "failed"
                state.errors.append(f"Node {current_agent} did not set `next_agent`.")
                state.next_agent = "FINISH"
                break

            current_agent = state.next_agent

        if snapshot_store is not None:
            snapshot_store.record(state, phase="final")
        return state_to_dict(state)


def cast_agent_name(value: AgentName | str) -> AgentName | str:
    """规范化 agent 名称，同时保留未知值供后续报错使用。"""
    return str(value).strip() or "orchestrator"


def _build_snapshot_store(config: dict[str, Any] | None) -> FileSnapshotStore | None:
    workflow_config = dict(config or {})
    configurable = dict(workflow_config.get("configurable") or {})
    snapshot_dir = str(
        configurable.get("snapshot_dir")
        or workflow_config.get("snapshot_dir")
        or ""
    ).strip()
    if not snapshot_dir:
        return None

    thread_id = str(configurable.get("thread_id") or "").strip() or "default-thread"
    return FileSnapshotStore(snapshot_dir, thread_id=thread_id)


def _build_base_graph() -> SimpleAgentGraph:
    """创建公开构建函数共用的基础图实例。"""
    return SimpleAgentGraph()


def build_graph() -> SimpleAgentGraph:
    """构建默认的 MedAI 工作流图。"""
    return _build_base_graph()


def build_graph_with_memory() -> SimpleAgentGraph:
    """保留向后兼容的别名接口。

    当前通过 `configurable.snapshot_dir` 提供基于文件的快照能力，
    但图实例本身仍然是同一个轻量实现。
    """
    return _build_base_graph()
