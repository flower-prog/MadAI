from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from agent.prompt import build_agent_run_spec
from agent.tools import (
    RiskCalcComputationRetrievalTool,
    RiskCalcExecutionTool,
    RiskCalcRetrievalTool,
    StructuredRetrievalTool,
    TrialChunkRetrievalTool,
    append_agent_trace,
    build_agent_trace,
    build_case_summary,
    build_patient_note_queries,
    build_default_chat_client,
    build_tool_call,
    create_trial_chunk_retrieval_tool,
    export_tool_specs,
    generate_risk_hints,
    maybe_load_json,
    summarize_text,
)

from .types import (
    TEAM_MEMBERS,
    AgentName,
    CalculationArtifact,
    CalculationTask,
    CalculatorMatch,
    CounterfactualScenario,
    GraphState,
    IntervalValue,
    PlanStep,
    ProtocolRecommendation,
    RetrievalQuery,
    SafetyIssue,
    ToolSpec,
    TreatmentRecommendation,
)


DEPARTMENT_TAG_LIBRARY: tuple[str, ...] = (
    "内科",
    "外科",
    "妇产科",
    "儿科",
    "五官科",
    "肿瘤科",
    "皮肤性病科",
    "传染科",
    "精神心理科",
    "麻醉医学科",
    "医学影像科",
)

_QUESTION_CHOICE_SECTION_PATTERN = re.compile(r"(?im)^\s*choices\s*:\s*$")
_QUESTION_OPTION_LINE_PATTERN = re.compile(r"^\s*[A-Z]\.\s+\S+")
_QUESTION_INSTRUCTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*please compute the answer using the appropriate calculator", re.IGNORECASE),
    re.compile(r"^\s*end your final answer with the format", re.IGNORECASE),
    re.compile(r"^\s*answer\s*:\s*<letter>", re.IGNORECASE),
)
_PROTOCOL_TRIAL_COARSE_TOP_K = 30
_PROTOCOL_TRIAL_TOP_K = 10


def _mark_step(state: GraphState, agent_name: AgentName, *, status: str, result: str | None = None) -> None:
    """更新指定 agent 对应的计划步骤状态，不影响其他步骤。"""
    for step in state.plan:
        if step.agent_name != agent_name:
            continue
        step.status = status
        if result is not None:
            step.result = result
        return


def _record_agent_prompt(
    state: GraphState,
    agent_name: str,
    *,
    context: dict[str, Any] | None = None,
    query: str | None = None,
    use_context_in_system_prompt: bool = True,
) -> None:
    """把某个 agent 的 prompt 规格写入最终输出，便于追踪。"""
    prompt_bundle = dict(state.final_output.get("agent_prompts") or {})
    prompt_bundle[agent_name] = build_agent_run_spec(
        agent_name,
        context=context if use_context_in_system_prompt else None,
    )
    if query is not None:
        prompt_bundle[agent_name]["query"] = query
    state.final_output["agent_prompts"] = prompt_bundle


def _build_prompt_specs() -> dict[str, dict[str, Any]]:
    """为核心工作流中的每个 agent 预构建默认 prompt 规格。"""
    return {agent_name: build_agent_run_spec(agent_name) for agent_name in TEAM_MEMBERS}


def _normalize_department_tag(value: Any) -> str:
    tag = str(value or "").strip()
    return tag if tag in DEPARTMENT_TAG_LIBRARY else ""


def _normalize_department_selection(department: Any, department_tags: Any) -> tuple[str, list[str]]:
    normalized_department = _normalize_department_tag(department)
    normalized_tags: list[str] = []
    for item in list(department_tags or []):
        tag = _normalize_department_tag(item)
        if tag and tag not in normalized_tags:
            normalized_tags.append(tag)

    if not normalized_department and normalized_tags:
        normalized_department = normalized_tags[0]
    if normalized_department and normalized_department not in normalized_tags:
        normalized_tags.insert(0, normalized_department)
    if not normalized_tags and normalized_department:
        normalized_tags = [normalized_department]
    return normalized_department, normalized_tags


def _ensure_execution_trace(state: GraphState) -> None:
    """确保 `final_output` 中存在统一的执行轨迹骨架。"""
    trace = dict(state.final_output.get("execution_trace") or {})
    request_text = _source_text(state)
    mode = state.clinical_tool_job.mode if state.clinical_tool_job is not None else "baseline"
    trace.setdefault("workflow", list(TEAM_MEMBERS))
    trace.setdefault("mode", mode)
    trace.setdefault("request", summarize_text(request_text))
    trace.setdefault("agents", [])
    trace.setdefault("tool_calls", [])
    state.final_output["execution_trace"] = trace


def _default_tool_specs() -> list[ToolSpec]:
    """定义 MedAI 图默认暴露的工具契约。"""
    decorated_tool_specs = {
        spec.name: spec
        for spec in export_tool_specs(
            StructuredRetrievalTool,
            TrialChunkRetrievalTool,
            RiskCalcRetrievalTool,
            RiskCalcComputationRetrievalTool,
            RiskCalcExecutionTool,
        )
    }
    return [
        ToolSpec(
            name="case_json_builder",
            description="Normalize the incoming case into a stable JSON payload for downstream agents.",
            required=True,
            input_schema={"request": "str", "patient_case": "dict|None", "clinical_tool_job": "dict|None"},
        ),
        ToolSpec(
            name="progressive_query_builder",
            description="Build retrieval-ready case summary and staged queries from the normalized case.",
            required=True,
            input_schema={"problem_list": "list[str]", "case_summary": "str"},
        ),
        ToolSpec(
            name="calculation_subagent",
            description="Run the calculator child-agent workflow under clinical_assisstment.",
            required=False,
            input_schema={"clinical_tool_job": "dict", "calculation_tasks": "list[dict]"},
        ),
        decorated_tool_specs["structured_bm25_retriever"],
        decorated_tool_specs["structured_vector_retriever"],
        decorated_tool_specs["trial_coarse_retriever"],
        decorated_tool_specs["trial_candidate_retriever"],
        decorated_tool_specs["riskcalc_coarse_retriever"],
        decorated_tool_specs["riskcalc_computation_retriever"],
        decorated_tool_specs["riskcalc_executor"],
        ToolSpec(
            name="similar_case_estimator",
            description="Estimate a single missing parameter from similar cases when calculation is nearly ready.",
            required=False,
            input_schema={"calculator": "str", "missing_inputs": "list[str]", "case_summary": "str"},
        ),
        ToolSpec(
            name="treatment_matcher",
            description="Map risk outputs to treatment or clinical trial strategies.",
            required=True,
            input_schema={"structured_case": "dict", "calculation_results": "list[dict]"},
        ),
        ToolSpec(
            name="report_compiler",
            description="Assemble the physician-facing report from all upstream artifacts.",
            required=True,
            input_schema={"structured_case": "dict", "calculation_bundle": "dict", "treatment_bundle": "dict"},
        ),
        ToolSpec(
            name="report_reviewer",
            description="Judge whether the current iteration is acceptable or should be rerun.",
            required=True,
            input_schema={"report_payload": "dict", "attempt": "int", "max_attempts": "int"},
        ),
    ]


def _build_default_plan() -> list[PlanStep]:
    """构建默认的四步 MedAI 工作流计划。"""
    return [
        PlanStep(
            step_id=1,
            agent_name="orchestrator",
            title="Initialize MedAI core workflow",
            description="Create the workflow contract and publish the role map for the four-agent graph.",
            note="This node coordinates only; it does not make medical decisions.",
        ),
        PlanStep(
            step_id=2,
            agent_name="clinical_assisstment",
            title="Clinical Assisstment: intake plus child calculator dispatch",
            description=(
                "Normalize the case into JSON, prepare retrieval context, "
                "and dispatch calculation tasks to the child calculator."
            ),
            note="This node combines agent1 intake work with agent2 coordination.",
        ),
        PlanStep(
            step_id=3,
            agent_name="protocol",
            title="Agent3: Treatment and clinical-trial decision",
            description="Map calculation outputs to treatment paths, similar-case fallbacks, or direct advice.",
            note="This node performs treatment reasoning but does not compile the final physician-facing report.",
        ),
        PlanStep(
            step_id=4,
            agent_name="reporter",
            title="Agent4: Report and iteration review",
            description="汇总报告，判断当前结果是否成立；若不通过，则回退到 orchestrator，最多执行三轮总迭代。",
            note="该节点负责最终的通过/重试判定，并控制回退到 orchestrator。",
        ),
    ]


def _source_text(state: GraphState) -> str:
    """选取当前最适合用于打标签、检索和摘要的源文本。"""
    if state.clinical_tool_job is not None and str(state.clinical_tool_job.text).strip():
        return str(state.clinical_tool_job.text).strip()

    if state.patient_case is not None and state.patient_case.structured_inputs:
        parts = []
        for key, value in state.patient_case.structured_inputs.items():
            if value in {None, "", False}:
                continue
            parts.append(f"{key}: {value}")
        if parts:
            return ". ".join(parts)

    return str(state.request or "").strip()


def _build_orchestrator_prompt_context(state: GraphState) -> dict[str, Any]:
    return {
        "request": state.request,
        "messages": list(state.messages),
        "patient_case": state.patient_case,
        "clinical_tool_job": state.clinical_tool_job,
        "reporter_feedback": list(state.reporter_feedback),
        "department": state.department,
        "department_tags": list(state.department_tags),
        "department_tag_library": list(DEPARTMENT_TAG_LIBRARY),
        "deep_thinking_mode": state.deep_thinking_mode,
        "tools": [asdict(tool) for tool in state.tools],
    }


def _build_orchestrator_query(state: GraphState) -> str:
    request_text = str(state.request or "").strip() or "(empty)"
    content_text = ""
    if state.clinical_tool_job is not None and str(state.clinical_tool_job.text).strip():
        content_text = str(state.clinical_tool_job.text).strip()
    if not content_text:
        for message in list(state.messages or []):
            if str(message.get("role") or "").strip().lower() != "user":
                continue
            candidate = str(message.get("content") or "").strip()
            if candidate:
                content_text = candidate
                break
    if not content_text and state.patient_case is not None and state.patient_case.structured_inputs:
        parts = []
        for key, value in state.patient_case.structured_inputs.items():
            if value in {None, "", False}:
                continue
            parts.append(f"{key}: {value}")
        if parts:
            content_text = ". ".join(parts)
    if not content_text:
        content_text = request_text
    mode = state.clinical_tool_job.mode if state.clinical_tool_job is not None else "baseline"
    feedback_items = [
        str(item).strip()
        for item in list(state.reporter_feedback or [])
        if str(item).strip()
    ]

    lines = [
        f"mode: {mode}",
        f"iteration_attempt: {state.reporter_attempts + 1}",
        f"max_iterations: {state.max_reporter_attempts}",
        f"request: {request_text}",
        "",
        "content:",
        content_text,
        "",
        "reporter_feedback:",
    ]
    if feedback_items:
        lines.extend(f"- {item}" for item in feedback_items)
    else:
        lines.append("- none")
    return "\n".join(lines).strip()


def _resolve_orchestrator_chat_client(state: GraphState):
    client = state.tool_registry.get("orchestrator_chat_client")
    if client is not None:
        return client
    model = state.clinical_tool_job.llm_model if state.clinical_tool_job is not None else None
    return build_default_chat_client(model=model)


def _populate_orchestrator_department(state: GraphState) -> None:
    existing_department, existing_tags = _normalize_department_selection(
        state.orchestrator_result.get("department") or state.department,
        state.orchestrator_result.get("department_tags") or state.department_tags,
    )
    if existing_department and existing_tags:
        state.department = existing_department
        state.department_tags = existing_tags
        return

    prompt_spec = build_agent_run_spec("orchestrator")
    query = _build_orchestrator_query(state)
    model = state.clinical_tool_job.llm_model if state.clinical_tool_job is not None else None
    chat_client = _resolve_orchestrator_chat_client(state)
    for attempt in range(3):
        try:
            answer = chat_client.complete(
                [
                    {"role": "system", "content": prompt_spec["system_prompt"]},
                    {"role": "user", "content": query},
                ],
                model=model,
                temperature=0.0,
            )
        except Exception as exc:
            state.errors.append(f"Orchestrator department classification failed on attempt {attempt + 1}/3: {exc}")
            continue

        payload = maybe_load_json(answer)
        if not isinstance(payload, dict):
            state.errors.append(
                f"Orchestrator did not return a JSON object on attempt {attempt + 1}/3."
            )
            continue

        state.orchestrator_result = {
            **dict(state.orchestrator_result or {}),
            **payload,
        }
        if isinstance(payload.get("structured_case"), dict):
            state.structured_case_json = dict(payload["structured_case"])

        resolved_department, resolved_tags = _normalize_department_selection(
            state.orchestrator_result.get("department") or state.department,
            state.orchestrator_result.get("department_tags") or state.department_tags,
        )
        if resolved_department:
            state.department = resolved_department
            state.department_tags = resolved_tags
            return

        state.errors.append(
            f"Orchestrator returned payload without a valid department on attempt {attempt + 1}/3."
        )

    resolved_department, resolved_tags = _normalize_department_selection("内科", ["内科"])

    state.department = resolved_department
    state.department_tags = resolved_tags


def _sanitize_question_retrieval_text(text: Any) -> str:
    """为选择题场景清理检索文本，去掉选项和答题指令噪音。"""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    match = _QUESTION_CHOICE_SECTION_PATTERN.search(normalized)
    candidate_text = normalized[: match.start()].strip() if match else normalized
    filtered_lines: list[str] = []
    for raw_line in candidate_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _QUESTION_OPTION_LINE_PATTERN.match(line):
            continue
        if any(pattern.search(line) for pattern in _QUESTION_INSTRUCTION_PATTERNS):
            continue
        filtered_lines.append(line)

    cleaned = re.sub(r"\s+", " ", " ".join(filtered_lines)).strip()
    if cleaned:
        return cleaned

    return re.sub(r"\s+", " ", candidate_text).strip()


def _retrieval_source_text(state: GraphState) -> str:
    """选取真正送入检索的文本；问题模式下优先使用清洗后的 clinical stem。"""
    source_text = _source_text(state)
    job = state.clinical_tool_job
    if job is None or job.mode != "question":
        return source_text
    return _sanitize_question_retrieval_text(source_text)


def _dedupe_strings(values: list[str]) -> list[str]:
    """在保持原顺序的前提下，按大小写不敏感方式去重字符串。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        normalized = str(raw or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _coerce_string_list(value: Any, *, limit: int | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, dict):
        values = list(value.values())
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    normalized = _dedupe_strings([str(item or "").strip() for item in values])
    if limit is None:
        return normalized
    return normalized[: max(int(limit), 0)]


def _seed_structured_case(state: GraphState) -> dict[str, Any]:
    seeded: dict[str, Any] = {}
    if isinstance(state.structured_case_json, dict):
        seeded.update(dict(state.structured_case_json))
    job = state.clinical_tool_job
    if job is not None and isinstance(job.structured_case, dict):
        seeded.update(dict(job.structured_case))
    return seeded


def _derive_problem_list(state: GraphState) -> list[str]:
    """从结构化输入中提炼问题列表，必要时回退到自由文本。"""
    seeded_problem_list = _coerce_string_list(_seed_structured_case(state).get("problem_list"), limit=8)
    if seeded_problem_list:
        return seeded_problem_list

    patient_case = state.patient_case
    if patient_case is not None:
        structured_problems = []
        for key, value in patient_case.structured_inputs.items():
            if value in {None, "", False}:
                continue
            structured_problems.append(f"{str(key).replace('_', ' ')}: {value}")
        if structured_problems:
            return structured_problems[:8]

    text = _retrieval_source_text(state)
    if not text:
        return []

    for separator in ["\n", ".", ";"]:
        if separator not in text:
            continue
        chunks = [chunk.strip(" -") for chunk in text.split(separator) if chunk.strip(" -")]
        if chunks:
            return chunks[:8]
    return [text[:240]]


def _resolve_case_summary(state: GraphState, problem_list: list[str]) -> str:
    job = state.clinical_tool_job
    if job is not None and str(job.case_summary or "").strip():
        return str(job.case_summary).strip()

    seeded_case_summary = str(_seed_structured_case(state).get("case_summary") or "").strip()
    if seeded_case_summary:
        return seeded_case_summary

    return build_case_summary(problem_list=problem_list)


def _compute_data_readiness(state: GraphState, problem_list: list[str]) -> dict[str, Any]:
    """估计当前病例对下游计算器执行来说准备程度如何。"""
    patient_case = state.patient_case
    structured_count = len(patient_case.structured_inputs) if patient_case is not None else 0
    interval_count = len(patient_case.interval_inputs) if patient_case is not None else 0
    multimodal_count = len(patient_case.multimodal_inputs) if patient_case is not None else 0
    problem_count = len(problem_list)
    risk_hint_count = len(state.clinical_tool_job.risk_hints) if state.clinical_tool_job is not None else 0

    if interval_count >= 1 or structured_count >= 4:
        status = "ready"
    elif structured_count >= 2 or problem_count >= 2 or risk_hint_count >= 2:
        status = "partial"
    else:
        status = "insufficient"

    return {
        "status": status,
        "structured_field_count": structured_count,
        "interval_field_count": interval_count,
        "multimodal_field_count": multimodal_count,
        "problem_count": problem_count,
        "risk_hint_count": risk_hint_count,
    }


def _build_query_set(
    *,
    state: GraphState,
    problem_list: list[str],
    case_summary: str,
) -> list[RetrievalQuery]:
    """为病历驱动、问题驱动或基线模式构建检索查询集合。"""
    if state.clinical_tool_job is not None:
        job = state.clinical_tool_job
        if job.retrieval_queries:
            return _dedupe_queries(
                [query for query in job.retrieval_queries if str(query.text or "").strip()]
            )
        if job.mode == "patient_note":
            return build_patient_note_queries(
                case_summary=case_summary,
                risk_hints=job.risk_hints,
                problem_list=problem_list,
                risk_count=job.risk_count,
            )

        question_text = _retrieval_source_text(state) or job.text.strip() or state.request
        return _dedupe_queries(
            [
                RetrievalQuery(
                    stage="question_anchor",
                    text=question_text,
                    intent="clinical_question",
                    rationale="Keep the original question wording for the first retrieval pass.",
                    priority=1,
                ),
                RetrievalQuery(
                    stage="case_summary_dense",
                    text=case_summary or question_text,
                    intent="case_summary_dense",
                    rationale="Keep a compact normalized summary beside the original question for retrieval.",
                    priority=2,
                ),
                RetrievalQuery(
                    stage="problem_anchor",
                    text=problem_list[0] if problem_list else question_text,
                    intent="problem_anchor",
                    rationale="Carry forward the primary problem statement as a retrieval anchor.",
                    priority=3,
                ),
            ]
        )

    manual_queries = [
        RetrievalQuery(
            stage="case_summary",
            text=case_summary or (problem_list[0] if problem_list else state.request),
            intent="case_summary_dense",
            rationale="Use a compact case summary as the main retrieval anchor.",
            priority=1,
        )
    ]
    if problem_list:
        manual_queries.append(
            RetrievalQuery(
                stage="problem_anchor",
                text=problem_list[0],
                intent="problem_anchor",
                rationale="Carry forward the main patient problem for calculator retrieval.",
                priority=2,
            )
        )
    manual_queries.append(
        RetrievalQuery(
            stage="direct_request",
            text=state.request,
            intent="clinical_question",
            rationale="Keep the original request wording available for downstream retrieval.",
            priority=3,
        )
    )
    return _dedupe_queries(manual_queries)


def _dedupe_queries(queries: list[RetrievalQuery]) -> list[RetrievalQuery]:
    """按规范化后的查询文本去重，只保留首次出现的查询。"""
    seen: set[str] = set()
    deduped: list[RetrievalQuery] = []
    for query in queries:
        key = query.text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def _build_structured_case(
    state: GraphState,
    *,
    problem_list: list[str],
    case_summary: str,
    data_readiness: dict[str, Any],
) -> dict[str, Any]:
    """组装供下游 agent 共用的标准化病例 JSON。"""
    patient_case = state.patient_case
    seeded_structured_case = _seed_structured_case(state)

    structured_inputs = (
        dict(seeded_structured_case.get("structured_inputs") or {})
        if isinstance(seeded_structured_case.get("structured_inputs"), dict)
        else {}
    )
    if patient_case is not None:
        structured_inputs.update(dict(patient_case.structured_inputs))

    interval_inputs = (
        {
            str(name): dict(value)
            for name, value in dict(seeded_structured_case.get("interval_inputs") or {}).items()
        }
        if isinstance(seeded_structured_case.get("interval_inputs"), dict)
        else {}
    )
    if patient_case is not None:
        interval_inputs.update(
            {
                name: asdict(value)
                for name, value in patient_case.interval_inputs.items()
            }
        )

    multimodal_inputs = (
        dict(seeded_structured_case.get("multimodal_inputs") or {})
        if isinstance(seeded_structured_case.get("multimodal_inputs"), dict)
        else {}
    )
    if patient_case is not None:
        multimodal_inputs.update(dict(patient_case.multimodal_inputs))

    return {
        **seeded_structured_case,
        "source_mode": state.clinical_tool_job.mode if state.clinical_tool_job is not None else "baseline",
        "patient_id": patient_case.patient_id if patient_case is not None else None,
        "raw_request": str(seeded_structured_case.get("raw_request") or state.request),
        "raw_text": str(seeded_structured_case.get("raw_text") or _retrieval_source_text(state)),
        "case_summary": case_summary,
        "problem_list": list(problem_list),
        "known_facts": _coerce_string_list(seeded_structured_case.get("known_facts")),
        "missing_information": _coerce_string_list(seeded_structured_case.get("missing_information")),
        "department": state.department,
        "department_tags": list(state.department_tags),
        "data_readiness": data_readiness,
        "structured_inputs": structured_inputs,
        "interval_inputs": interval_inputs,
        "multimodal_inputs": multimodal_inputs,
        "reporter_feedback": list(state.reporter_feedback),
    }


def _issue(severity: str, message: str, source: str, *, blocking: bool = False) -> SafetyIssue:
    """便捷创建类型化 `SafetyIssue` 对象。"""
    return SafetyIssue(severity=severity, message=message, source=source, blocking=blocking)


def _resolve_clinical_tool_runner(state: GraphState):
    """从注册表解析子级 clinical tool 执行器，必要时构建默认实例。"""
    runner = state.tool_registry.get("clinical_tool_agent")
    if runner is not None:
        return runner

    from agent.clinical_tool_agent import ClinicalToolAgent

    if state.clinical_tool_job is None:
        raise ValueError("clinical_tool_job is required to resolve the clinical tool runner.")
    return ClinicalToolAgent.from_job(state.clinical_tool_job)


def _resolve_clinical_tool_selector(state: GraphState):
    """解析支持 `plan_selection(job)` 的 calculator 选择器；缺失时允许回退。"""
    runner = state.tool_registry.get("clinical_tool_agent")
    if runner is not None:
        if hasattr(runner, "plan_selection"):
            return runner
        nested_agent = getattr(runner, "agent", None)
        if nested_agent is not None and hasattr(nested_agent, "plan_selection"):
            return nested_agent
        return None

    from agent.clinical_tool_agent import ClinicalToolAgent

    if state.clinical_tool_job is None:
        raise ValueError("clinical_tool_job is required to resolve the clinical tool selector.")
    return ClinicalToolAgent.from_job(state.clinical_tool_job)


def _plan_clinical_tool_selection(state: GraphState) -> dict[str, Any] | None:
    """尝试在父节点完成 PMID 预选；若当前 runner 不支持则返回 `None`。"""
    selector = _resolve_clinical_tool_selector(state)
    if selector is None:
        return None

    plan_selection = getattr(selector, "plan_selection", None)
    if not callable(plan_selection):
        return None

    job = state.clinical_tool_job
    if job is None:
        raise ValueError("clinical_tool_job is missing from graph state.")
    selection_bundle = plan_selection(job)
    if not isinstance(selection_bundle, dict):
        raise TypeError("clinical_tool_agent.plan_selection(job) must return a dict.")
    return selection_bundle


def _apply_clinical_tool_selection_to_job(
    job: ClinicalToolJob,
    selection_bundle: dict[str, Any],
) -> None:
    """把父节点的 calculator 选择结果写回 `clinical_tool_job`，供子 agent 直接执行。"""
    selected_tool = dict(selection_bundle.get("selected_tool") or {})
    selected_pmid = str(selected_tool.get("pmid") or "").strip() or None
    dispatch_query_text = str(selection_bundle.get("dispatch_query_text") or "").strip()
    dispatch_query_source = str(selection_bundle.get("dispatch_query_source") or "").strip()
    selection_mode = str(selection_bundle.get("selection_mode") or "").strip()
    if job.forced_tool_pmid:
        selection_mode = "oracle_forced"
    elif selection_mode != "oracle_forced":
        selection_mode = "parent_selected"

    job.selected_tool_pmid = selected_pmid
    job.selected_tool = dict(selected_tool)
    job.dispatch_query_text = dispatch_query_text
    job.selection_context = {
        "risk_hints": list(selection_bundle.get("risk_hints") or job.risk_hints),
        "retrieval_batches": [
            dict(item)
            for item in list(selection_bundle.get("retrieval_batches") or [])
            if isinstance(item, dict)
        ],
        "retrieved_tools": [
            dict(item)
            for item in list(selection_bundle.get("retrieved_tools") or [])
            if isinstance(item, dict)
        ],
        "candidate_ranking": [
            dict(item)
            for item in list(selection_bundle.get("candidate_ranking") or [])
            if isinstance(item, dict)
        ],
        "selection_candidates": [
            dict(item)
            for item in list(selection_bundle.get("selection_candidates") or [])
            if isinstance(item, dict)
        ],
        "bm25_raw_top5": [
            dict(item)
            for item in list(selection_bundle.get("bm25_raw_top5") or [])
            if isinstance(item, dict)
        ],
        "vector_raw_top5": [
            dict(item)
            for item in list(selection_bundle.get("vector_raw_top5") or [])
            if isinstance(item, dict)
        ],
        "recommended_pmids": [
            str(item).strip()
            for item in list(selection_bundle.get("recommended_pmids") or [])
            if str(item).strip()
        ],
        "selected_tool": dict(selected_tool),
        "selected_tool_pmid": selected_pmid,
        "selection_mode": selection_mode,
        "dispatch_query_text": dispatch_query_text,
        "dispatch_query_source": dispatch_query_source,
    }


def _run_clinical_tool_job(state: GraphState) -> dict[str, object]:
    """按统一调用约定执行子级 clinical tool 任务。"""
    runner = _resolve_clinical_tool_runner(state)
    job = state.clinical_tool_job
    if job is None:
        raise ValueError("clinical_tool_job is missing from graph state.")
    if hasattr(runner, "run"):
        return runner.run(job)
    if callable(runner):
        return runner(job)
    raise TypeError("clinical_tool_agent registry entry must be callable or expose a .run(job) method.")


def _float_from_value(value: float | IntervalValue | None) -> float | None:
    """把标量或区间值折算成一个中点浮点数。"""
    if value is None:
        return None
    if isinstance(value, IntervalValue):
        return (float(value.lower) + float(value.upper)) / 2.0
    return float(value)


def _compute_interval_summary(state: GraphState) -> dict[str, Any]:
    """为基线计算构建占位性质的区间汇总结果。"""
    patient_case = state.patient_case
    if patient_case is None or not patient_case.interval_inputs:
        return {
            "factor_count": 0,
            "feature_bounds": {},
            "aggregate_risk_interval": None,
            "comment": "No interval inputs were available for the baseline scaffold.",
        }

    feature_bounds: dict[str, dict[str, float | str | None]] = {}
    lower_sum = 0.0
    upper_sum = 0.0
    for feature_name, interval in patient_case.interval_inputs.items():
        lower = float(interval.lower)
        upper = float(interval.upper)
        if lower > upper:
            lower, upper = upper, lower
        lower_sum += lower
        upper_sum += upper
        feature_bounds[feature_name] = {
            "lower": lower,
            "upper": upper,
            "unit": interval.unit,
            "source": interval.source,
        }

    factor_count = len(feature_bounds)
    aggregate_interval = {
        "lower": lower_sum / factor_count,
        "upper": upper_sum / factor_count,
    }
    midpoint = (aggregate_interval["lower"] + aggregate_interval["upper"]) / 2.0
    if midpoint >= 0.75:
        band = "high"
    elif midpoint >= 0.4:
        band = "moderate"
    else:
        band = "low"
    aggregate_interval["band"] = band
    return {
        "factor_count": factor_count,
        "feature_bounds": feature_bounds,
        "aggregate_risk_interval": aggregate_interval,
        "comment": "This remains a placeholder interval aggregator until disease-specific formulas are wired in.",
    }


def _compute_counterfactuals(state: GraphState, baseline_interval: dict[str, Any] | None) -> list[dict[str, Any]]:
    """基于基线区间汇总，估算简化版反事实变化结果。"""
    patient_case = state.patient_case
    if patient_case is None or not patient_case.counterfactual_scenarios:
        return []

    baseline_mid = None
    if baseline_interval:
        lower = baseline_interval.get("lower")
        upper = baseline_interval.get("upper")
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            baseline_mid = (float(lower) + float(upper)) / 2.0

    results: list[dict[str, Any]] = []
    for scenario in patient_case.counterfactual_scenarios:
        if not isinstance(scenario, CounterfactualScenario):
            continue
        current_mid = _float_from_value(scenario.current_value)
        target_mid = _float_from_value(scenario.target_value)
        estimated_delta = None
        if current_mid is not None and target_mid is not None:
            estimated_delta = max((current_mid - target_mid) * 0.1, 0.0)
        estimated_risk_after = None
        if baseline_mid is not None and estimated_delta is not None:
            estimated_risk_after = max(baseline_mid - estimated_delta, 0.0)

        results.append(
            {
                "feature_name": scenario.feature_name,
                "direction": scenario.direction,
                "current_midpoint": current_mid,
                "target_midpoint": target_mid,
                "estimated_risk_reduction": estimated_delta,
                "estimated_risk_after": estimated_risk_after,
                "note": scenario.note or "Counterfactual output still uses a placeholder monotonic rule.",
            }
        )
    return results


def _summarize_calculator_matches(result: dict[str, Any], default_category: str) -> list[CalculatorMatch]:
    """把子 agent 的输出归一化为计算器匹配摘要。"""
    category = default_category
    mode = str(result.get("mode") or "")
    matches: list[CalculatorMatch] = []
    selected_pmid = str(dict(result.get("selected_tool") or {}).get("pmid") or "").strip()

    if mode == "question":
        selected_tool = dict(result.get("selected_tool") or {})
        execution = dict(result.get("execution") or {})
        selected_decision = next(
            (
                dict(item)
                for item in list(result.get("selection_decisions") or [])
                if str(dict(item).get("pmid") or "").strip() == str(selected_tool.get("pmid") or "").strip()
            ),
            {},
        )
        missing_inputs = [
            str(item)
            for item in list(
                selected_decision.get("missing_inputs")
                or selected_tool.get("missing_inputs")
                or execution.get("missing_inputs")
                or []
            )
            if str(item).strip()
        ]
        execution_status = str(
            execution.get("status")
            or selected_decision.get("execution_status")
            or ""
        ).strip().lower()
        if execution_status == "completed" and missing_inputs:
            execution_status = "partial"
        if str(selected_tool.get("pmid") or "").strip():
            applicability = "partial" if execution_status == "partial" else "selected"
            matches.append(
                CalculatorMatch(
                    pmid=str(selected_tool.get("pmid") or ""),
                    title=str(selected_tool.get("title") or ""),
                    category=category,
                    rationale=str(
                        selected_tool.get("reason")
                        or execution.get("final_text")
                        or "Selected calculator for the clinical question."
                    ),
                    applicability=applicability,
                    missing_inputs=missing_inputs,
                    execution_status=execution_status,
                    value=execution.get("result"),
                )
            )
        return matches

    candidate_ranking = list(result.get("candidate_ranking") or result.get("retrieved_tools") or [])
    decisions_by_pmid = {
        str(item.get("pmid") or ""): dict(item)
        for item in list(result.get("selection_decisions") or [])
        if str(item.get("pmid") or "").strip()
    }
    executions_by_pmid = {
        str(item.get("pmid") or ""): dict(item)
        for item in list(result.get("executions") or [])
        if str(item.get("pmid") or "").strip()
    }

    ordered_pmids: list[str] = []
    for candidate in candidate_ranking:
        pmid = str(dict(candidate).get("pmid") or "").strip()
        if pmid and pmid not in ordered_pmids:
            ordered_pmids.append(pmid)
    for pmid in executions_by_pmid:
        if pmid not in ordered_pmids:
            ordered_pmids.append(pmid)

    for pmid in ordered_pmids[:10]:
        candidate = next((dict(item) for item in candidate_ranking if str(dict(item).get("pmid") or "") == pmid), {})
        decision = decisions_by_pmid.get(pmid, {})
        execution = executions_by_pmid.get(pmid, {})
        gate_status = str(decision.get("gate_status") or "").strip().lower()
        decision_execution_status = str(decision.get("execution_status") or "").strip().lower()
        execution_status = str(execution.get("status") or decision_execution_status or "").strip().lower()
        missing_inputs = [
            str(item)
            for item in list(
                decision.get("missing_inputs")
                or execution.get("missing_inputs")
                or []
            )
            if str(item).strip()
        ]
        if execution_status == "completed" and missing_inputs:
            execution_status = "partial"
        value = execution.get("result")

        if execution and execution_status == "partial":
            applicability = "partial"
            rationale = str(
                execution.get("final_text")
                or decision.get("rationale")
                or "Calculator produced only a provisional result because required parameters are still missing."
            )
        elif execution:
            applicability = "selected" if pmid == selected_pmid else "computed"
            rationale = str(
                execution.get("final_text")
                or decision.get("rationale")
                or "Calculator executed by the child calculation agent."
            )
        elif gate_status == "failed_missing_inputs" or decision_execution_status == "missing_inputs":
            applicability = "data_missing"
            rationale = str(
                decision.get("rationale")
                or "Calculator could not run because required parameters are missing."
            )
        elif gate_status == "failed_execution" or decision_execution_status in {"failed", "error"}:
            applicability = "execution_failed"
            rationale = str(
                decision.get("rationale")
                or "Calculator execution failed before a usable score was produced."
            )
        elif gate_status == "failed_unregistered" or decision_execution_status == "unregistered":
            applicability = "unavailable"
            rationale = str(
                decision.get("rationale")
                or "Calculator is not available for structured execution in the current registry."
            )
        elif decision.get("patient_eligible") == "no":
            applicability = "ineligible"
            rationale = str(decision.get("rationale") or "Candidate rejected before score computation.")
        elif decision:
            applicability = "eligible_pending"
            rationale = str(
                decision.get("rationale")
                or "Candidate is partially aligned but still needs additional parameter handling."
            )
        else:
            applicability = "candidate"
            rationale = "Candidate surfaced by second-stage retrieval but was not executed."

        matches.append(
            CalculatorMatch(
                pmid=pmid,
                title=str(candidate.get("title") or decision.get("title") or execution.get("title") or ""),
                category=category,
                rationale=rationale,
                applicability=applicability,
                missing_inputs=missing_inputs,
                execution_status=execution_status,
                value=value,
            )
        )

    return matches


def _build_calculation_tasks_from_matches(
    matches: list[CalculatorMatch],
    default_category: str,
) -> tuple[list[CalculationTask], list[CalculationArtifact]]:
    """根据计算器匹配结果生成执行任务和占位结果。"""
    tasks: list[CalculationTask] = []
    results: list[CalculationArtifact] = []

    if not matches:
        task = CalculationTask(
            task_id="calc-1",
            label="no_viable_calculator_match",
            category=default_category,
            decision="skip",
            missing_inputs=["calculator_match_missing"],
            rationale="No calculator could be matched to the current case after intake and diagnosis.",
        )
        result = CalculationArtifact(
            name="no_viable_calculation",
            category=default_category,
            status="skipped",
            source="calculation_coordinator",
            rationale=task.rationale,
            linked_task_id=task.task_id,
            missing_inputs=list(task.missing_inputs),
        )
        return [task], [result]

    for index, match in enumerate(matches, start=1):
        normalized_execution_status = str(match.execution_status or "").strip().lower()
        if normalized_execution_status == "completed":
            decision = "direct"
            status = "completed"
            rationale = "Calculator completed successfully with the child calculation agent."
            source = "calculation_subagent"
        elif normalized_execution_status == "partial" or match.applicability == "partial":
            decision = "direct"
            status = "partial"
            rationale = (
                "Calculator produced only a provisional result because one or more required parameters "
                "were missing from the case."
            )
            source = "calculation_subagent"
        elif normalized_execution_status == "missing_inputs" or match.applicability == "data_missing":
            decision = "skip"
            status = "skipped"
            rationale = (
                "This calculator is not executable because the current case is missing required parameters."
            )
            source = "calculation_coordinator"
        elif match.applicability == "execution_failed" or normalized_execution_status in {"failed", "error"}:
            decision = "skip"
            status = "failed"
            rationale = match.rationale or "Calculator execution failed before a usable score was produced."
            source = "calculation_subagent"
        elif match.applicability in {"eligible_pending", "selected", "candidate"}:
            decision = "estimate_from_similar_case"
            status = "estimated"
            rationale = (
                "The case is close enough for calculation coordination, so the graph flags it for "
                "single-gap similar-case estimation in downstream implementations."
            )
            source = "similar_case_estimator_pending"
        else:
            decision = "skip"
            status = "skipped"
            rationale = "This calculator is not applicable to the current patient cohort."
            source = "calculation_coordinator"

        task = CalculationTask(
            task_id=f"calc-{index}",
            label=match.title or f"calculator_{index}",
            category=match.category,
            decision=decision,
            missing_inputs=list(match.missing_inputs),
            rationale=rationale,
        )
        artifact = CalculationArtifact(
            name=match.title or task.label,
            category=match.category,
            status=status,
            value=match.value,
            source=source,
            rationale=rationale,
            linked_task_id=task.task_id,
            linked_calculator=match.pmid,
            missing_inputs=list(match.missing_inputs),
        )
        tasks.append(task)
        results.append(artifact)
    return tasks, results


def _build_baseline_calculation_bundle(
    state: GraphState,
    default_category: str,
) -> tuple[list[CalculatorMatch], list[CalculationTask], list[CalculationArtifact], dict[str, Any]]:
    """在没有子级工具任务时，构建回退用的基线计算包。"""
    interval_summary = _compute_interval_summary(state)
    aggregate_interval = dict(interval_summary).get("aggregate_risk_interval") or {}
    counterfactuals = _compute_counterfactuals(state, aggregate_interval)
    category = default_category

    readiness_status = str(state.structured_case_json.get("data_readiness", {}).get("status") or "")
    if aggregate_interval:
        decision = "direct"
        status = "completed"
        rationale = "Baseline interval scaffold produced a usable risk interval."
        source = "baseline_interval_scaffold"
    elif readiness_status == "partial":
        decision = "estimate_from_similar_case"
        status = "estimated"
        rationale = "Baseline case is only partially ready; similar-case estimation is the next step."
        source = "similar_case_estimator_pending"
    else:
        decision = "skip"
        status = "skipped"
        rationale = "Baseline case does not expose enough structured inputs for calculation."
        source = "calculation_coordinator"

    match = CalculatorMatch(
        pmid="baseline-interval-scaffold",
        title="Baseline interval scaffold",
        category=category,
        rationale=rationale,
        applicability="selected" if decision != "skip" else "data_missing",
        missing_inputs=[] if decision == "direct" else ["structured_parameter_gap"],
        execution_status="completed" if status == "completed" else "",
    )
    task = CalculationTask(
        task_id="calc-1",
        label="baseline_interval_scaffold",
        category=category,
        decision=decision,
        missing_inputs=[] if decision == "direct" else ["structured_parameter_gap"],
        rationale=rationale,
    )
    artifact = CalculationArtifact(
        name="baseline_interval_scaffold",
        category=category,
        status=status,
        value=aggregate_interval or None,
        source=source,
        rationale=rationale,
        linked_task_id=task.task_id,
        linked_calculator=match.pmid,
        missing_inputs=list(task.missing_inputs),
    )
    bundle = {
        "interval_summary": interval_summary,
        "counterfactuals": counterfactuals,
    }
    return [match], [task], [artifact], bundle


def _resolve_trial_retriever(state: GraphState):
    """解析 protocol 使用的 trial retriever；缺省时构建 XML chunk KB 版本。"""
    trial_retriever = state.tool_registry.get("trial_retriever")
    if trial_retriever is not None:
        return trial_retriever

    backend = state.clinical_tool_job.retriever_backend if state.clinical_tool_job is not None else "hybrid"
    return create_trial_chunk_retrieval_tool(
        backend=backend,
        vector_store="auto",
    )


def _retrieve_trial_candidates(state: GraphState) -> dict[str, Any]:
    """运行 protocol 阶段的本地 trial 两阶段检索。"""
    structured_case = dict(state.structured_case_json or {})
    if not structured_case:
        return {
            "query_text": "",
            "backend_used": "bm25",
            "available_backends": ["bm25"],
            "department_tags": list(state.department_tags),
            "fallback_to_full_catalog": False,
            "coarse_candidate_ids": [],
            "bm25_top5": [],
            "vector_top5": [],
            "candidate_ranking": [],
        }

    backend = state.clinical_tool_job.retriever_backend if state.clinical_tool_job is not None else "hybrid"
    trial_retriever = _resolve_trial_retriever(state)

    if hasattr(trial_retriever, "retrieve_from_structured_case"):
        return trial_retriever.retrieve_from_structured_case(
            structured_case,
            top_k=_PROTOCOL_TRIAL_TOP_K,
            coarse_top_k=_PROTOCOL_TRIAL_COARSE_TOP_K,
            department_tags=list(state.department_tags),
            backend=backend,
        )

    candidate_tool = getattr(trial_retriever, "retrieve_candidates", None)
    if callable(candidate_tool):
        return candidate_tool(
            structured_case=structured_case,
            top_k=_PROTOCOL_TRIAL_TOP_K,
            coarse_top_k=_PROTOCOL_TRIAL_COARSE_TOP_K,
            department_tags=list(state.department_tags),
            backend=backend,
        )

    raise TypeError(
        "trial_retriever registry entry must expose retrieve_from_structured_case(...) or retrieve_candidates(...)."
    )


def _top_non_abandoned_trial_ids(trial_bundle: dict[str, Any], *, limit: int = 3) -> list[str]:
    selected_ids: list[str] = []
    for candidate in list(trial_bundle.get("candidate_ranking") or []):
        if str(candidate.get("status") or "").strip() == "abandoned":
            continue
        nct_id = str(candidate.get("nct_id") or "").strip()
        if not nct_id or nct_id in selected_ids:
            continue
        selected_ids.append(nct_id)
        if len(selected_ids) >= max(int(limit), 1):
            break
    return selected_ids


def _trial_review_actions(trial_bundle: dict[str, Any], *, limit: int = 3) -> list[str]:
    actions: list[str] = []
    for candidate in list(trial_bundle.get("candidate_ranking") or [])[: max(int(limit), 1)]:
        for action in list(candidate.get("actions") or []):
            text = str(action).strip()
            if text and text not in actions:
                actions.append(text)
    return actions


def _normalize_trial_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = " ".join(str(item or "").split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _trial_candidate_text(candidate: dict[str, Any]) -> str:
    text_parts = [
        str(candidate.get("title") or ""),
        str(candidate.get("brief_summary") or ""),
        str(candidate.get("best_evidence_text") or ""),
        " ; ".join(_normalize_trial_text_list(candidate.get("conditions"))),
        " ; ".join(_normalize_trial_text_list(candidate.get("interventions"))),
        str(candidate.get("primary_purpose") or ""),
    ]
    return " ".join(part for part in text_parts if str(part).strip())


def _candidate_matches_term(candidate: dict[str, Any], term: str) -> bool:
    normalized_term = str(term or "").strip().casefold()
    if not normalized_term:
        return False
    candidate_text = _trial_candidate_text(candidate).casefold()
    return normalized_term in candidate_text


def _coerce_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_protocol_trial_candidate(trial_bundle: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in list(trial_bundle.get("candidate_ranking") or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}
    for candidate in candidates:
        if str(candidate.get("status") or "").strip() != "abandoned":
            return candidate
    return candidates[0]


def _assess_protocol_trial_candidate(
    candidate: dict[str, Any],
    *,
    query_profile: dict[str, Any],
) -> dict[str, Any]:
    matching_signals: list[str] = []
    conflicts: list[str] = []
    missing_information: list[str] = []

    for term in list(query_profile.get("trial_condition_terms") or []):
        if _candidate_matches_term(candidate, term):
            matching_signals.append(f"Matched condition focus: {term}.")
    for term in list(query_profile.get("trial_intervention_terms") or []):
        if _candidate_matches_term(candidate, term):
            matching_signals.append(f"Matched intervention focus: {term}.")
    for term in list(query_profile.get("trial_intent_terms") or []):
        if _candidate_matches_term(candidate, term):
            matching_signals.append(f"Matched trial intent: {term}.")
        elif term.casefold() == "stroke prevention" and str(candidate.get("primary_purpose") or "").casefold() == "prevention":
            matching_signals.append("Primary purpose aligns with stroke prevention intent.")

    for term in list(query_profile.get("patient_negative_terms") or []):
        if term in {"diabetes", "congestive heart failure"} and _candidate_matches_term(candidate, term):
            conflicts.append(
                f"Trial focus appears to include {term}, which the case explicitly says is absent."
            )

    age_years = _coerce_optional_float(query_profile.get("age_years"))
    age_floor = _coerce_optional_float(candidate.get("age_floor_years"))
    age_ceiling = _coerce_optional_float(candidate.get("age_ceiling_years"))
    if age_years is not None:
        if age_floor is not None and age_years < age_floor:
            conflicts.append(f"Case age {int(age_years)} is below the recorded minimum age {int(age_floor)}.")
        elif age_floor is None:
            missing_information.append("Minimum age is not structured on the trial record.")
        if age_ceiling is not None and age_years > age_ceiling:
            conflicts.append(f"Case age {int(age_years)} is above the recorded maximum age {int(age_ceiling)}.")
        elif age_ceiling is None:
            missing_information.append("Maximum age is not structured on the trial record.")

    patient_gender = str(query_profile.get("gender") or "").strip()
    trial_gender = str(candidate.get("gender") or "").strip()
    if patient_gender:
        if trial_gender and trial_gender not in {"All", patient_gender}:
            conflicts.append(f"Trial gender restriction is {trial_gender}, while the case is {patient_gender}.")
        elif not trial_gender:
            missing_information.append("Trial gender eligibility is not structured on the record.")

    if not str(candidate.get("best_evidence_text") or "").strip():
        missing_information.append("No matched eligibility/evidence chunk was surfaced for manual review.")

    next_checks = _normalize_trial_text_list(candidate.get("actions"))
    if not next_checks:
        next_checks = ["Review the trial inclusion and exclusion criteria against the case manually."]
    if conflicts:
        next_checks.append("Verify whether the apparent mismatch is a true exclusion or just noisy retrieval overlap.")

    fit = "possible_match"
    if str(candidate.get("status") or "").strip() == "abandoned":
        fit = "not_current_option"
    elif conflicts:
        fit = "needs_manual_review"
    elif matching_signals:
        fit = "likely_match"

    return {
        "fit": fit,
        "matching_signals": _normalize_trial_text_list(matching_signals),
        "conflicts": _normalize_trial_text_list(conflicts),
        "missing_information": _normalize_trial_text_list(missing_information),
        "next_checks": _normalize_trial_text_list(next_checks),
    }


def _build_protocol_trial_selection(
    trial_bundle: dict[str, Any],
    *,
    structured_case: dict[str, Any],
) -> dict[str, Any]:
    del structured_case
    selected_trial = _select_protocol_trial_candidate(trial_bundle)
    if not selected_trial:
        return {
            "selected_trial": None,
            "selection_reason": "No trial candidate survived retrieval for the current case.",
            "eligibility_assessment": {
                "fit": "not_available",
                "matching_signals": [],
                "conflicts": [],
                "missing_information": ["No ranked trial candidates were returned."],
                "next_checks": ["Adjust the protocol query profile or broaden the trial corpus before retrying."],
            },
            "trial_status_assessment": {},
            "evidence": [],
            "alternatives": [],
        }

    query_profile = dict(trial_bundle.get("query_profile") or {})
    eligibility_assessment = _assess_protocol_trial_candidate(
        selected_trial,
        query_profile=query_profile,
    )
    status_assessment = {
        "status": str(selected_trial.get("status") or ""),
        "overall_status": str(selected_trial.get("overall_status") or ""),
        "enrollment_open": bool(selected_trial.get("enrollment_open")),
        "status_reason": str(selected_trial.get("status_reason") or ""),
    }

    evidence: list[str] = []
    for item in list(eligibility_assessment.get("matching_signals") or []):
        if item not in evidence:
            evidence.append(str(item))
    best_evidence_text = str(selected_trial.get("best_evidence_text") or "").strip()
    if best_evidence_text:
        evidence.append(best_evidence_text[:240])
    brief_summary = str(selected_trial.get("brief_summary") or "").strip()
    if brief_summary:
        evidence.append(brief_summary[:240])

    focus_terms = _normalize_trial_text_list(
        list(query_profile.get("trial_condition_terms") or [])
        + list(query_profile.get("trial_intent_terms") or [])
    )
    selection_reason = (
        f"Selected {selected_trial.get('title') or selected_trial.get('nct_id') or 'the top-ranked trial'} "
        f"because it remained the highest-ranked non-abandoned candidate and matched the protocol focus terms "
        f"{', '.join(focus_terms[:3]) or 'from the structured case'}."
    )
    if status_assessment["overall_status"]:
        selection_reason += f" Current study status: {status_assessment['overall_status']}."

    alternatives: list[dict[str, Any]] = []
    for candidate in list(trial_bundle.get("candidate_ranking") or []):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("nct_id") or "").strip() == str(selected_trial.get("nct_id") or "").strip():
            continue
        alternatives.append(
            {
                "nct_id": str(candidate.get("nct_id") or ""),
                "title": str(candidate.get("title") or ""),
                "status": str(candidate.get("status") or ""),
                "overall_status": str(candidate.get("overall_status") or ""),
                "why_not_selected": (
                    "Lower final ranking or weaker fit than the selected trial."
                    if str(candidate.get("status") or "").strip() != "abandoned"
                    else "Not selected because the trial is not a current option."
                ),
            }
        )
        if len(alternatives) >= 3:
            break

    return {
        "selected_trial": dict(selected_trial),
        "selection_reason": selection_reason,
        "eligibility_assessment": eligibility_assessment,
        "trial_status_assessment": status_assessment,
        "evidence": evidence[:5],
        "alternatives": alternatives,
    }


def _augment_treatment_recommendations_with_trials(
    recommendations: list[TreatmentRecommendation],
    trial_bundle: dict[str, Any],
    *,
    has_completed_results: bool,
) -> list[TreatmentRecommendation]:
    """把 protocol trial retrieval 的结果并入治疗建议。"""
    top_trial_ids = _top_non_abandoned_trial_ids(trial_bundle, limit=3)
    if has_completed_results and top_trial_ids:
        for recommendation in recommendations:
            recommendation.linked_trials = list(top_trial_ids)

    if has_completed_results or not list(trial_bundle.get("candidate_ranking") or []):
        return recommendations

    candidate_ranking = list(trial_bundle.get("candidate_ranking") or [])
    top_candidate = dict(candidate_ranking[0] or {}) if candidate_ranking else {}
    trial_review_actions = _trial_review_actions(trial_bundle, limit=3)
    if "Review trial eligibility and enrollment details before surfacing any specific study." not in trial_review_actions:
        trial_review_actions.append(
            "Review trial eligibility and enrollment details before surfacing any specific study."
        )
    recommendations.insert(
        0,
        TreatmentRecommendation(
            name="trial candidate review",
            strategy="trial_candidate_review",
            source="trial_retrieval",
            status="manual_review",
            rationale=(
                "Local trial retrieval surfaced candidate studies for the current case, but there is no usable "
                "calculator-backed risk output strong enough to promote a direct trial match. "
                f"Top candidate: {top_candidate.get('title') or top_candidate.get('name') or top_candidate.get('nct_id') or 'unknown trial'}."
            ),
            linked_trials=list(top_trial_ids),
            actions=trial_review_actions,
        ),
    )
    return recommendations


def _build_treatment_recommendations(
    state: GraphState,
    *,
    trial_bundle: dict[str, Any] | None = None,
) -> list[TreatmentRecommendation]:
    """把计算结果翻译成治疗层面的建议列表。"""
    completed_results = [item for item in state.calculation_results if item.status == "completed"]
    partial_results = [item for item in state.calculation_results if item.status == "partial"]
    estimated_results = [item for item in state.calculation_results if item.status == "estimated"]
    effective_trial_bundle = dict(trial_bundle or {})

    if completed_results:
        recommendations: list[TreatmentRecommendation] = []
        for artifact in completed_results[:3]:
            recommendations.append(
                TreatmentRecommendation(
                    name=f"{artifact.name} guided treatment",
                    strategy="risk_informed_treatment",
                    source="protocol_reasoning",
                    status="manual_review",
                    rationale=(
                        "A usable risk result is available, so protocol can now derive a treatment direction. "
                        "Any final trial or regimen match still needs explicit downstream evidence."
                    ),
                    linked_calculators=[artifact.linked_calculator] if artifact.linked_calculator else [],
                    actions=[
                        "Map the risk output to treatment thresholds or protocol branches.",
                        "Keep any regimen or trial recommendation evidence-linked and explicitly reviewable.",
                    ],
                )
            )
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            has_completed_results=True,
        )

    if partial_results:
        recommendations = [
            TreatmentRecommendation(
                name="partial calculator result requires parameter completion",
                strategy="similar_case_fallback",
                source="partial_parameter_gap",
                status="similar_case_fallback",
                rationale=(
                    "A calculator produced only a provisional result because key parameters are still missing, "
                    "so treatment and trial routing should remain provisional until those inputs are completed."
                ),
                linked_calculators=[
                    artifact.linked_calculator for artifact in partial_results if artifact.linked_calculator
                ],
                actions=[
                    "Collect the listed missing calculator inputs before treating the score as final.",
                    "Use the current calculator text only as a provisional lower-bound or partial interpretation.",
                ],
            )
        ]
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            has_completed_results=False,
        )

    if estimated_results:
        recommendations = [
            TreatmentRecommendation(
                name="similar-case treatment fallback",
                strategy="similar_case_fallback",
                source="estimated_parameter_gap",
                status="similar_case_fallback",
                rationale=(
                    "Calculation is close but not fully executable, so treatment should be chosen only after "
                    "validating the estimated parameter against similar cases."
                ),
                linked_calculators=[
                    artifact.linked_calculator for artifact in estimated_results if artifact.linked_calculator
                ],
                actions=[
                    "Review the estimated parameter against one or more similar cases.",
                    "Keep the treatment recommendation provisional until the missing value is validated.",
                ],
            )
        ]
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            has_completed_results=False,
        )

    if state.calculator_matches:
        recommendations = [
            TreatmentRecommendation(
                name="similar-case assisted recommendation",
                strategy="similar_case_fallback",
                source="calculator_candidates_without_execution",
                status="similar_case_fallback",
                rationale=(
                    "Candidate calculators were found, but no risk value was strong enough to anchor treatment. "
                    "A clinician should review similar cases and trial options directly."
                ),
                linked_calculators=[match.pmid for match in state.calculator_matches[:3] if match.pmid],
                actions=[
                    "Use retrieved candidate calculators and their eligibility notes as a screening aid.",
                    "Search similar cases or evidence sources before escalating treatment.",
                ],
            )
        ]
        return _augment_treatment_recommendations_with_trials(
            recommendations,
            effective_trial_bundle,
            has_completed_results=False,
        )

    recommendations = [
        TreatmentRecommendation(
            name="direct treatment advice",
            strategy="direct_advice",
            source="no_calculation_signal",
            status="advice_only",
            rationale=(
                "Neither executable calculators nor valid fallback signals are available, so the graph can only "
                "return a clinician-reviewed recommendation path."
            ),
            actions=[
                "Collect more structured parameters before rerunning MedAI.",
                "If treatment must proceed now, provide direct conservative treatment advice and mark it as low confidence.",
            ],
        )
    ]
    return _augment_treatment_recommendations_with_trials(
        recommendations,
        effective_trial_bundle,
        has_completed_results=False,
    )


def _to_protocol_recommendations(
    recommendations: list[TreatmentRecommendation],
) -> list[ProtocolRecommendation]:
    """把治疗建议转换成 protocol 阶段使用的摘要对象。"""
    protocol_recommendations: list[ProtocolRecommendation] = []
    for recommendation in recommendations:
        if recommendation.status in {"matched", "trial_matched"}:
            status = "matched"
        elif recommendation.status == "abandoned":
            status = "insufficient_data"
        else:
            status = "needs_revision"
        protocol_recommendations.append(
            ProtocolRecommendation(
                name=recommendation.name,
                category=recommendation.strategy,
                status=status,
                rationale=recommendation.rationale,
                linked_calculators=list(recommendation.linked_calculators),
                linked_trials=list(recommendation.linked_trials),
                corrections=list(recommendation.actions),
            )
        )
    return protocol_recommendations


def _reset_iteration_outputs(state: GraphState) -> None:
    """在重新进入评估链路前清空本轮迭代产生的输出。"""
    state.errors = []
    state.safety_issues = []
    state.calculator_matches = []
    state.calculation_tasks = []
    state.calculation_results = []
    state.treatment_recommendations = []
    state.protocol_recommendations = []
    state.assessment_bundle = {}
    state.calculation_bundle = {}
    state.trial_retrieval_bundle = {}
    state.treatment_bundle = {}
    state.reporter_result = {}
    state.review_report = {}
    state.clinical_answer = []


def _build_report_review(state: GraphState, report_payload: dict[str, Any], *, attempt: int) -> dict[str, Any]:
    """评估当前这一轮输出是否足够完整、可以直接保留。"""
    calculation_bundle = dict(state.calculation_bundle or {})
    calculation_error = str(calculation_bundle.get("error") or "").strip()
    has_failed_calculation = any(item.status == "failed" for item in state.calculation_results)
    has_recommendations = bool(state.treatment_recommendations)

    checks: list[dict[str, Any]] = [
        {
            "name": "structured_case_present",
            "passed": bool(state.structured_case_json),
            "detail": "clinical_assisstment must produce a structured case payload.",
        },
        {
            "name": "protocol_recommendations_present",
            "passed": has_recommendations,
            "detail": "protocol must return at least one treatment recommendation.",
        },
        {
            "name": "no_calculation_runtime_error",
            "passed": not calculation_error and not has_failed_calculation and not state.errors,
            "detail": "The current iteration still contains calculator/runtime errors.",
        },
    ]

    if state.clinical_tool_job is not None:
        checks.append(
            {
                "name": "calculation_artifacts_present",
                "passed": bool(state.calculation_results),
                "detail": "A clinical_tool_job was supplied, but no calculation artifacts were produced.",
            }
        )

    issues: list[SafetyIssue] = []
    for check in checks:
        if check["passed"]:
            continue
        blocking = check["name"] != "calculation_artifacts_present"
        severity = "critical" if blocking else "warning"
        issues.append(_issue(severity, str(check["detail"]), str(check["name"]), blocking=blocking))

    blocking_issues = [issue for issue in issues if issue.blocking]
    passed = not blocking_issues

    return {
        "attempt": attempt,
        "max_attempts": state.max_reporter_attempts,
        "passed": passed,
        "checks": checks,
        "issues": [asdict(issue) for issue in issues],
        "report_outcome": report_payload.get("outcome"),
    }


def orchestrator_node(state: GraphState) -> GraphState:  # 定义 orchestrator 节点；输入是共享状态，输出还是更新后的共享状态
    """初始化工作流元数据，并把流程路由到 clinical_assisstment。"""  # 这行 docstring 已经点明职责：初始化 + 路由

    state.status = "running"  # 把整个图的全局状态标记为“运行中”

    state.final_output.setdefault("agent_prompts", _build_prompt_specs())  # 确保 final_output 里有 agent_prompts；如果没有，就为四个 agent 预先生成默认 prompt 规格
    _ensure_execution_trace(state)  # 确保 final_output 里有 execution_trace 骨架，后续每个节点都能往里面追加轨迹

    if not state.plan:  # 如果当前还没有工作计划
        state.plan = _build_default_plan()  # 就创建默认的四步工作流计划：orchestrator / clinical_assisstment / protocol / reporter

    if not state.tools:  # 如果当前还没有工具契约列表
        state.tools = _default_tool_specs()  # 就挂上默认工具说明，供后续节点/子 agent 使用

    _populate_orchestrator_department(state)

    raw_department_tags = list(state.orchestrator_result.get("department_tags") or [])
    if not raw_department_tags and state.department:
        raw_department_tags = [state.department]
    elif not raw_department_tags and state.department_tags:
        raw_department_tags = list(state.department_tags)

    state.department_tags = [str(item).strip() for item in raw_department_tags if str(item).strip()]
    if not state.department and state.department_tags:
        state.department = state.department_tags[0]
    elif state.department and not state.department_tags:
        state.department_tags = [state.department]

    state.orchestrator_result = {
        **dict(state.orchestrator_result or {}),
        "mode": state.clinical_tool_job.mode if state.clinical_tool_job is not None else "baseline",
        "department": state.department,
        "department_tags": list(state.department_tags),
        "department_tag_library": list(DEPARTMENT_TAG_LIBRARY),
        "notes": [
            "The graph now centers on orchestrate -> assess(+child calculator) -> protocol -> report/review.",
            "The reporter node is responsible for iteration control and can trigger up to three total passes.",
        ],
    }

    state.final_output["department"] = state.department
    state.final_output["department_tags"] = list(state.department_tags)
    state.final_output["orchestrator_result"] = dict(state.orchestrator_result)

    _record_agent_prompt(
        state,
        "orchestrator",
        query=_build_orchestrator_query(state),
        use_context_in_system_prompt=False,
    )

    _mark_step(state, "orchestrator", status="completed", result="Core workflow contract created.")  # 把 plan 里 orchestrator 这一步标记为已完成，并写入结果说明

    append_agent_trace(  # 把 orchestrator 的执行记录追加到 final_output.execution_trace.agents 里
        state.final_output,  # 最终输出容器
        build_agent_trace(  # 先构造一条标准化的 agent trace
            "orchestrator",  # 当前 trace 属于 orchestrator
            status="completed",  # 本节点执行状态：completed
            summary="Core workflow contract created.",  # 人类可读的一句话摘要
            output_payload=state.orchestrator_result,  # 这一步的输出内容直接放在主状态 orchestrator_result 里
            tool_calls=[],  # 这个节点本身没有调用任何工具，所以是空列表
        ),
    )

    state.next_agent = "clinical_assisstment"  # 指定下一跳节点为 clinical_assisstment，把控制权交出去
    return state  # 返回更新后的共享状态，供 graph 继续执行

def clinical_assisstment_node(state: GraphState) -> GraphState:
    """完成 intake 归一化、检索准备和计算器协调。"""
    _reset_iteration_outputs(state)
    _mark_step(state, "clinical_assisstment", status="in_progress")

    risk_hint_tool_call = None
    if state.clinical_tool_job is not None and state.clinical_tool_job.mode == "patient_note" and not state.clinical_tool_job.risk_hints:
        try:
            chat_client = build_default_chat_client(model=state.clinical_tool_job.llm_model)
            risk_hints, raw_response = generate_risk_hints(
                clinical_text=state.clinical_tool_job.text,
                risk_count=state.clinical_tool_job.risk_count,
                chat_client=chat_client,
                model=state.clinical_tool_job.llm_model,
                temperature=state.clinical_tool_job.temperature,
            )
            risk_hints = list(risk_hints)
            fallback_used = False
        except Exception as exc:
            risk_hints = [item[:120] for item in _derive_problem_list(state)[: max(state.clinical_tool_job.risk_count, 1)]]
            raw_response = f"fallback_problem_list:{exc}"
            fallback_used = True

        state.clinical_tool_job.risk_hints = list(risk_hints)
        risk_hint_tool_call = build_tool_call(
            "risk_hint_generator",
            input_payload={
                "clinical_text": summarize_text(state.clinical_tool_job.text),
                "risk_count": state.clinical_tool_job.risk_count,
            },
            output_payload={
                "risk_hints": list(risk_hints),
                "raw_response": raw_response,
                "fallback_used": fallback_used,
            },
            metadata={"fallback_used": fallback_used},
        )

    problem_list = _derive_problem_list(state)
    data_readiness = _compute_data_readiness(state, problem_list)
    default_category = "clinical_calculation"
    request_text = _source_text(state)
    case_summary = _resolve_case_summary(state, problem_list)
    retrieval_queries = _build_query_set(
        state=state,
        problem_list=problem_list,
        case_summary=case_summary,
    )
    structured_case = _build_structured_case(
        state,
        problem_list=problem_list,
        case_summary=case_summary,
        data_readiness=data_readiness,
    )

    state.problem_list = list(problem_list)
    state.structured_case_json = structured_case
    state.retrieval_queries = list(retrieval_queries)

    if state.clinical_tool_job is not None:
        state.clinical_tool_job.case_summary = case_summary
        state.clinical_tool_job.structured_case = dict(structured_case)
        state.clinical_tool_job.retrieval_queries = list(retrieval_queries)
        state.clinical_tool_job.selected_tool_pmid = None
        state.clinical_tool_job.selected_tool = {}
        state.clinical_tool_job.dispatch_query_text = ""
        state.clinical_tool_job.selection_context = {}
        try:
            selection_bundle = _plan_clinical_tool_selection(state)
        except Exception as exc:
            state.errors.append(f"clinical_tool parent selection failed; falling back to child selection: {exc}")
        else:
            if selection_bundle is not None:
                _apply_clinical_tool_selection_to_job(
                    state.clinical_tool_job,
                    selection_bundle,
                )

    delegated_tool_calls: list[dict[str, Any]] = []
    if state.clinical_tool_job is not None:
        try:
            child_result = _run_clinical_tool_job(state)
            matches = _summarize_calculator_matches(child_result, default_category)
            tasks, results = _build_calculation_tasks_from_matches(matches, default_category)
            raw_trace = child_result.get("trace")
            delegated_trace = raw_trace if isinstance(raw_trace, dict) else {}
            for item in list(delegated_trace.get("tool_calls") or []):
                nested_call = dict(item)
                nested_call.setdefault("agent_name", delegated_trace.get("agent_name") or "calculator")
                delegated_tool_calls.append(nested_call)
            calculation_bundle = {
                "mode": "clinical_tool_agent",
                "calculation_tasks": [asdict(task) for task in tasks],
                "calculation_results": [asdict(result) for result in results],
                "calculator_matches": [asdict(match) for match in matches],
                "child_result": child_result,
            }
            state.final_output["clinical_tool_agent"] = child_result
        except Exception as exc:
            error_message = str(exc)
            matches = []
            tasks = [
                CalculationTask(
                    task_id="calc-1",
                    label="clinical_tool_agent_failed",
                    category=default_category,
                    decision="skip",
                    missing_inputs=["child_executor_failed"],
                    rationale=error_message,
                )
            ]
            results = [
                CalculationArtifact(
                    name="clinical_tool_agent_failed",
                    category=default_category,
                    status="failed",
                    source="calculation_subagent",
                    rationale=error_message,
                    linked_task_id="calc-1",
                    missing_inputs=["child_executor_failed"],
                )
            ]
            calculation_bundle = {
                "mode": "clinical_tool_agent",
                "error": error_message,
                "calculation_tasks": [asdict(task) for task in tasks],
                "calculation_results": [asdict(result) for result in results],
                "calculator_matches": [],
            }
            state.errors.append(error_message)
    else:
        matches, tasks, results, baseline_bundle = _build_baseline_calculation_bundle(
            state,
            "interval_risk_model",
        )
        calculation_bundle = {
            "mode": "baseline",
            "calculation_tasks": [asdict(task) for task in tasks],
            "calculation_results": [asdict(result) for result in results],
            "calculator_matches": [asdict(match) for match in matches],
            **baseline_bundle,
        }

    state.calculator_matches = list(matches)
    state.calculation_tasks = list(tasks)
    state.calculation_results = list(results)
    state.calculation_bundle = calculation_bundle

    bundle = {
        "case_summary": case_summary,
        "structured_case": structured_case,
        "problem_list": list(problem_list),
        "department": state.department,
        "department_tags": list(state.department_tags),
        "progressive_queries": [asdict(query) for query in retrieval_queries],
        "reporter_feedback": list(state.reporter_feedback),
        "calculation_bundle": calculation_bundle,
    }
    state.assessment_bundle = bundle
    state.final_output["structured_case"] = structured_case
    state.final_output["assessment_bundle"] = bundle
    state.final_output["calculation_bundle"] = calculation_bundle

    _record_agent_prompt(
        state,
        "clinical_assisstment",
        context={
            "request": state.request,
            "clinical_text": _source_text(state),
            "problem_list": list(problem_list),
            "department": state.department,
            "department_tags": list(state.department_tags),
            "structured_case": structured_case,
            "retrieval_queries": [asdict(query) for query in retrieval_queries],
            "reporter_feedback": list(state.reporter_feedback),
            "iteration_attempt": state.reporter_attempts + 1,
        },
    )
    _record_agent_prompt(
        state,
        "calculator",
        context={
            "request": state.request,
            "department": state.department,
            "structured_case": structured_case,
            "retrieval_queries": [asdict(query) for query in retrieval_queries],
            "reporter_feedback": list(state.reporter_feedback),
        },
    )
    _mark_step(
        state,
        "clinical_assisstment",
        status="completed",
        result="Clinical assisstment completed with child calculator coordination.",
    )
    append_agent_trace(
        state.final_output,
        build_agent_trace(
            "clinical_assisstment",
            status="completed",
            summary="Clinical assisstment completed with child calculator coordination.",
            output_payload=bundle,
            tool_calls=[
                build_tool_call(
                    "case_json_builder",
                    input_payload={
                        "request": state.request,
                        "patient_case_present": state.patient_case is not None,
                        "clinical_tool_job_present": state.clinical_tool_job is not None,
                    },
                    output_payload=structured_case,
                ),
                build_tool_call(
                    "progressive_query_builder",
                    input_payload={"case_summary": case_summary},
                    output_payload={"queries": [asdict(query) for query in retrieval_queries]},
                ),
                build_tool_call(
                    "calculation_subagent",
                    input_payload={
                        "request": summarize_text(request_text),
                        "default_category": default_category,
                        "task_count": len(tasks),
                    },
                    output_payload={
                        "result_count": len(results),
                        "calculator_match_count": len(matches),
                    },
                ),
            ]
            + ([risk_hint_tool_call] if risk_hint_tool_call is not None else [])
            + delegated_tool_calls,
        ),
    )
    state.next_agent = "protocol"
    return state


def protocol_node(state: GraphState) -> GraphState:
    """把计算结果整理成治疗与试验推荐包。"""
    _mark_step(state, "protocol", status="in_progress")

    trial_retrieval_bundle = _retrieve_trial_candidates(state)
    trial_selection = _build_protocol_trial_selection(
        trial_retrieval_bundle,
        structured_case=dict(state.structured_case_json or {}),
    )
    recommendations = _build_treatment_recommendations(
        state,
        trial_bundle=trial_retrieval_bundle,
    )
    protocol_recommendations = _to_protocol_recommendations(recommendations)
    trial_candidates = [
        dict(item)
        for item in list(trial_retrieval_bundle.get("candidate_ranking") or [])
        if isinstance(item, dict)
    ]
    treatment_bundle = {
        "recommendations": [asdict(item) for item in recommendations],
        "trial_candidates": trial_candidates,
        "trial_candidate_ids": [
            str(item.get("nct_id") or "").strip()
            for item in trial_candidates
            if str(item.get("nct_id") or "").strip()
        ],
        "trial_selection": trial_selection,
        "note": "This node now owns treatment and clinical-trial judgment in the core MedAI workflow.",
    }

    state.treatment_recommendations = recommendations
    state.protocol_recommendations = protocol_recommendations
    state.trial_retrieval_bundle = trial_retrieval_bundle
    state.treatment_bundle = treatment_bundle
    state.final_output["trial_retrieval_bundle"] = trial_retrieval_bundle
    state.final_output["treatment_bundle"] = treatment_bundle

    _record_agent_prompt(
        state,
        "protocol",
        context={
            "request": state.request,
            "structured_case": dict(state.structured_case_json),
            "calculation_results": [asdict(item) for item in state.calculation_results],
            "calculator_matches": [asdict(item) for item in state.calculator_matches],
            "trial_retrieval_bundle": dict(trial_retrieval_bundle),
        },
    )
    _mark_step(state, "protocol", status="completed", result="Treatment decision bundle generated.")
    append_agent_trace(
        state.final_output,
        build_agent_trace(
            "protocol",
            status="completed",
            summary="Treatment decision bundle generated.",
            output_payload=treatment_bundle,
            tool_calls=[
                build_tool_call(
                    "trial_candidate_retriever",
                    input_payload={
                        "structured_case": dict(state.structured_case_json),
                        "department_tags": list(state.department_tags),
                        "top_k": _PROTOCOL_TRIAL_TOP_K,
                        "coarse_top_k": _PROTOCOL_TRIAL_COARSE_TOP_K,
                    },
                    output_payload=trial_retrieval_bundle,
                ),
                build_tool_call(
                    "treatment_matcher",
                    input_payload={
                        "structured_case": dict(state.structured_case_json),
                        "calculation_result_count": len(state.calculation_results),
                        "trial_candidate_count": len(trial_candidates),
                    },
                    output_payload=treatment_bundle,
                )
            ],
        ),
    )
    state.next_agent = "reporter"
    return state


def reporter_node(state: GraphState) -> GraphState:
    """汇总报告、判断通过或重试，并控制迭代路由。"""
    state.reporter_attempts += 1
    attempt = state.reporter_attempts
    _mark_step(state, "reporter", status="in_progress", result=f"Reporter review attempt {attempt}.")

    treatment_bundle = dict(state.treatment_bundle or {})
    calculation_bundle = dict(state.calculation_bundle or {})

    calculation_highlights = []
    for artifact in state.calculation_results:
        calculation_highlights.append(
            {
                "name": artifact.name,
                "status": artifact.status,
                "source": artifact.source,
                "linked_calculator": artifact.linked_calculator,
            }
        )

    recommendation_summary = [
        {
            "name": recommendation.name,
            "status": recommendation.status,
            "strategy": recommendation.strategy,
            "linked_trials": list(recommendation.linked_trials),
            "actions": list(recommendation.actions),
        }
        for recommendation in state.treatment_recommendations
    ]
    trial_candidate_summary = [
        {
            "nct_id": str(candidate.get("nct_id") or "").strip(),
            "title": str(candidate.get("title") or candidate.get("name") or "").strip(),
            "status": str(candidate.get("status") or "").strip(),
            "enrollment_open": bool(candidate.get("enrollment_open")),
        }
        for candidate in list(treatment_bundle.get("trial_candidates") or [])
        if isinstance(candidate, dict)
    ]

    report_payload = {
        "iteration_attempt": attempt,
        "max_iterations": state.max_reporter_attempts,
        "outcome": "review_pending",
        "summary": "Reporter is evaluating the current iteration.",
        "structured_case": dict(state.structured_case_json),
        "problem_list": list(state.problem_list),
        "calculation_bundle": calculation_bundle,
        "treatment_bundle": treatment_bundle,
        "calculation_highlights": calculation_highlights,
        "recommendation_summary": recommendation_summary,
        "trial_candidate_summary": trial_candidate_summary,
        "incoming_feedback": list(state.reporter_feedback),
        "errors": list(state.errors),
    }
    review_report = _build_report_review(state, report_payload, attempt=attempt)
    report_payload["review_report"] = review_report

    _record_agent_prompt(
        state,
        "reporter",
        context={
            "request": state.request,
            "structured_case": dict(state.structured_case_json),
            "calculation_bundle": calculation_bundle,
            "treatment_bundle": treatment_bundle,
            "attempt": attempt,
            "max_attempts": state.max_reporter_attempts,
            "errors": list(state.errors),
        },
    )

    state.reporter_result = report_payload
    state.review_report = review_report
    state.clinical_answer = recommendation_summary
    state.final_output["review_report"] = review_report
    state.final_output["clinical_report"] = report_payload
    state.final_output["clinical_answer"] = recommendation_summary

    review_tool_status = (
        "completed"
        if review_report["passed"]
        else ("retry" if attempt < state.max_reporter_attempts else "failed")
    )
    reporter_tool_calls = [
        build_tool_call(
            "report_compiler",
            input_payload={
                "attempt": attempt,
                "recommendation_count": len(recommendation_summary),
                "error_count": len(state.errors),
            },
            output_payload=report_payload,
        ),
        build_tool_call(
            "report_reviewer",
            input_payload={"attempt": attempt, "max_attempts": state.max_reporter_attempts},
            output_payload=review_report,
            status=review_tool_status,
        ),
    ]

    if review_report["passed"]:
        state.review_passed = True
        state.reporter_feedback = []
        report_payload["outcome"] = "completed"
        report_payload["summary"] = "Reporter accepted the current iteration."
        state.status = "completed"
        _mark_step(state, "reporter", status="completed", result=report_payload["summary"])
        append_agent_trace(
            state.final_output,
            build_agent_trace(
                "reporter",
                status="completed",
                summary=report_payload["summary"],
                output_payload=report_payload,
                tool_calls=reporter_tool_calls,
            ),
        )
        state.next_agent = "FINISH"
        return state

    feedback = [issue["message"] for issue in list(review_report.get("issues") or []) if issue.get("blocking")]
    state.review_passed = False
    state.reporter_feedback = feedback

    if attempt < state.max_reporter_attempts:
        report_payload["outcome"] = "iteration_requested"
        report_payload["summary"] = (
            f"Reporter 判定需要重跑，流程已回退到 orchestrator（{attempt}/{state.max_reporter_attempts}）。"
        )
        state.status = "running"
        _mark_step(state, "reporter", status="in_progress", result=report_payload["summary"])
        append_agent_trace(
            state.final_output,
            build_agent_trace(
                "reporter",
                status="retry",
                summary=report_payload["summary"],
                output_payload=report_payload,
                tool_calls=reporter_tool_calls,
            ),
        )
        state.next_agent = "orchestrator"
        return state

    state.abandoned = True
    report_payload["outcome"] = "failed_after_max_iterations"
    report_payload["summary"] = (
        f"Reporter rejected the result after {state.max_reporter_attempts} total iterations."
    )
    state.status = "failed"
    _mark_step(state, "reporter", status="failed", result=report_payload["summary"])
    append_agent_trace(
        state.final_output,
        build_agent_trace(
            "reporter",
            status="failed",
            summary=report_payload["summary"],
            output_payload=report_payload,
            tool_calls=reporter_tool_calls,
        ),
    )
    state.next_agent = "FINISH"
    return state
