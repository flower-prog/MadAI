from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field, is_dataclass
import os
from typing import Any, Literal, cast


AgentName = Literal[
    "orchestrator",
    "clinical_assisstment",
    "protocol",
    "reporter",
]
StepStatus = Literal["pending", "in_progress", "completed", "failed"]
GraphStatus = Literal["pending", "running", "completed", "failed"]
ClinicalToolMode = Literal["patient_note", "question"]
RetrieverBackend = Literal["keyword", "vector", "hybrid"]
ProtocolStatus = Literal["matched", "needs_revision", "insufficient_data"]
SafetySeverity = Literal["info", "warning", "critical"]
CalculationDecision = Literal["direct", "estimate_from_similar_case", "skip"]
CalculationResultStatus = Literal["completed", "partial", "estimated", "skipped", "failed"]
TreatmentRecommendationStatus = Literal[
    "matched",
    "fallback",
    "manual_review",
    "abandoned",
    "trial_matched",
    "similar_case_fallback",
    "advice_only",
]

TEAM_MEMBERS: tuple[AgentName, ...] = (
    "orchestrator",
    "clinical_assisstment",
    "protocol",
    "reporter",
)
OPTIONS: tuple[str, ...] = TEAM_MEMBERS + ("FINISH",)

_AGENT_NAME_ALIASES: dict[str, AgentName] = {
    "planner": "orchestrator",
    "clinical_assessment": "clinical_assisstment",
}

_LEGACY_RETRIEVER_BACKEND_MAP: dict[str, RetrieverBackend] = {
    "auto": "hybrid",
    "medcpt": "vector",
}
_DEFAULT_RETRIEVAL_TOP_K = 30


@dataclass(slots=True)
class IntervalValue:
    lower: float
    upper: float
    unit: str | None = None
    source: str | None = None


@dataclass(slots=True)
class CounterfactualScenario:
    feature_name: str
    current_value: float | IntervalValue | None = None
    target_value: float | IntervalValue | None = None
    direction: Literal["increase", "decrease", "unknown"] = "unknown"
    note: str = ""


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    required: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalQuery:
    stage: str
    text: str
    intent: str = ""
    rationale: str = ""
    priority: int = 1


@dataclass(slots=True)
class CalculatorMatch:
    pmid: str
    title: str = ""
    category: str = "risk_score"
    rationale: str = ""
    applicability: str = "unknown"
    missing_inputs: list[str] = field(default_factory=list)
    execution_status: str | None = None
    value: Any = None


@dataclass(slots=True)
class ProtocolRecommendation:
    name: str
    category: str = "care_pathway"
    status: ProtocolStatus = "insufficient_data"
    rationale: str = ""
    linked_calculators: list[str] = field(default_factory=list)
    linked_trials: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SafetyIssue:
    severity: SafetySeverity = "info"
    message: str = ""
    source: str = ""
    blocking: bool = False


@dataclass(slots=True)
class PlanStep:
    step_id: int
    agent_name: AgentName
    title: str
    description: str
    status: StepStatus = "pending"
    note: str = ""
    result: str | None = None


@dataclass(slots=True)
class PatientCase:
    patient_id: str | None = None
    structured_inputs: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    interval_inputs: dict[str, IntervalValue] = field(default_factory=dict)
    multimodal_inputs: dict[str, Any] = field(default_factory=dict)
    counterfactual_scenarios: list[CounterfactualScenario] = field(default_factory=list)


@dataclass(slots=True)
class ClinicalToolJob:
    mode: ClinicalToolMode = "patient_note"
    text: str = ""
    case_summary: str = ""
    structured_case: dict[str, Any] = field(default_factory=dict)
    risk_hints: list[str] = field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)
    selected_tool_pmid: str | None = None
    selected_tool: dict[str, Any] = field(default_factory=dict)
    dispatch_query_text: str = ""
    selection_context: dict[str, Any] = field(default_factory=dict)
    top_k: int = 30
    risk_count: int = 5
    max_selected_tools: int | None = 5
    retriever_backend: RetrieverBackend = "hybrid"
    riskcalcs_path: str | None = None
    pmid_metadata_path: str | None = None
    llm_model: str | None = None
    forced_tool_pmid: str | None = None
    temperature: float = 0.0
    max_rounds: int = 12


@dataclass(slots=True)
class CalculationTask:
    task_id: str
    label: str
    category: str
    decision: CalculationDecision = "skip"
    required_inputs: list[str] = field(default_factory=list)
    available_inputs: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    rationale: str = ""
    executor: str = "calculation_subagent"


@dataclass(slots=True)
class CalculationArtifact:
    name: str
    category: str
    status: CalculationResultStatus = "skipped"
    value: Any = None
    unit: str | None = None
    source: str = ""
    rationale: str = ""
    linked_task_id: str | None = None
    linked_calculator: str | None = None
    missing_inputs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TreatmentRecommendation:
    name: str
    strategy: str
    source: str
    status: TreatmentRecommendationStatus = "manual_review"
    rationale: str = ""
    linked_calculators: list[str] = field(default_factory=list)
    linked_trials: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphState:
    """图执行期间共享的完整运行状态。"""

    # ---- 入口输入 ----
    request: str  # 当前用户请求的标准化文本；整个工作流的主入口
    patient_case: PatientCase | None = None  # 结构化病例输入，供评估/计算/推荐节点复用
    clinical_tool_job: ClinicalToolJob | None = None  # 临床工具模式下的任务配置和检索参数

    # ---- 规划与可用资源 ----
    plan: list[PlanStep] = field(default_factory=list)  # 当前工作流的步骤计划
    tools: list[ToolSpec] = field(default_factory=list)  # 可暴露给节点或子执行器的工具描述
    tool_registry: dict[str, Any] = field(default_factory=dict)  # 运行时工具注册表，通常包含不可序列化对象
    messages: list[dict[str, str]] = field(default_factory=list)  # 对话消息历史；兼容消息驱动节点

    # ---- 临床推理中间结果 ----
    problem_list: list[str] = field(default_factory=list)  # 从病例/问题中抽取出的核心问题列表
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)  # 检索阶段生成的查询列表
    calculator_matches: list[CalculatorMatch] = field(default_factory=list)  # 候选风险计算器或评分工具匹配结果
    protocol_recommendations: list[ProtocolRecommendation] = field(default_factory=list)  # 协议/路径推荐结果
    safety_issues: list[SafetyIssue] = field(default_factory=list)  # 安全提醒、阻断项或证据不足提示
    final_output: dict[str, Any] = field(default_factory=dict)  # 最终返回给调用方的结构化输出
    orchestrator_result: dict[str, Any] = field(default_factory=dict)  # orchestrator 的结构化结果，直接放在主状态里
    assessment_bundle: dict[str, Any] = field(default_factory=dict)  # clinical_assisstment 的结构化产物
    calculation_bundle: dict[str, Any] = field(default_factory=dict)  # calculator 阶段的汇总结果
    trial_retrieval_bundle: dict[str, Any] = field(default_factory=dict)  # protocol 阶段的 trial 检索包
    treatment_bundle: dict[str, Any] = field(default_factory=dict)  # protocol 阶段的治疗决策包
    reporter_result: dict[str, Any] = field(default_factory=dict)  # reporter 生成的完整报告结果
    review_report: dict[str, Any] = field(default_factory=dict)  # reporter 的通过/重跑判定结果
    clinical_answer: list[dict[str, Any]] = field(default_factory=list)  # 面向外层调用方的简化答案列表

    # ---- 运行开关与路由控制 ----
    deep_thinking_mode: bool = True  # 是否启用更深入的规划/推理模式
    search_before_planning: bool = False  # 是否在规划前先做检索
    pass_through_expert: bool = False  # 是否走 expert 直通路径，跳过部分默认流程
    auto_accepted_plan: bool = False  # 是否自动接受计划而不等待额外确认
    next_agent: AgentName | Literal["FINISH"] | None = None  # 下一跳节点；FINISH 表示流程结束
    status: GraphStatus = "pending"  # 图整体执行状态
    errors: list[str] = field(default_factory=list)  # 运行过程中累计的错误信息

    # ---- 结构化上下文与报告回路 ----
    department: str = ""  # orchestrator 生成的主科室
    department_tags: list[str] = field(default_factory=list)  # 病例所属科室标签，值来自预定义标签库
    structured_case_json: dict[str, Any] = field(default_factory=dict)  # 面向前端/存档的结构化病例 JSON
    reporter_feedback: list[str] = field(default_factory=list)  # reporter 节点给出的审阅反馈
    reporter_attempts: int = 0  # reporter 已回路审阅的次数
    max_reporter_attempts: int = 3  # reporter 最多允许的回路次数
    review_passed: bool = False  # 当前结果是否已通过 reporter 审阅
    abandoned: bool = False  # 是否已放弃继续推进本轮工作流

    # ---- 计算与治疗建议产物 ----
    calculation_tasks: list[CalculationTask] = field(default_factory=list)  # 待执行或已规划的计算任务
    calculation_results: list[CalculationArtifact] = field(default_factory=list)  # 计算任务产出的结果制品
    treatment_recommendations: list[TreatmentRecommendation] = field(default_factory=list)  # 最终治疗/处置建议


def _resolve_retrieval_top_k(raw_value: Any) -> int:
    if raw_value is not None and raw_value != "":
        return max(int(raw_value), 1)

    for env_name in ("MEDAI_RETRIEVAL_TOP_K", "MEDAI_TOP_K"):
        value = str(os.getenv(env_name) or "").strip()
        if not value:
            continue
        return max(int(value), 1)

    return _DEFAULT_RETRIEVAL_TOP_K


def _coerce_agent_name(raw: Any, *, default: AgentName = "orchestrator") -> AgentName:
    """把原始 agent 标识规范化成受支持的 `AgentName`。

    这个函数主要给路由结果、历史状态恢复、或者 LLM 输出做兜底清洗。
    它会先去掉空白，再处理兼容别名，最后确认结果是否真的在
    `TEAM_MEMBERS` 里；如果不合法，就回退到 `default`。

    参数:
        raw: 原始输入，可能是 `None`、带空格的字符串、旧别名，或者其他可转字符串的值。
        default: 当 `raw` 为空或不合法时返回的默认节点名。

    返回:
        一个保证合法的 `AgentName`。

    示例:
        >>> _coerce_agent_name(" planner ")
        'orchestrator'
        >>> _coerce_agent_name("clinical_assessment")
        'clinical_assisstment'
        >>> _coerce_agent_name("unknown", default="reporter")
        'reporter'
    """
    normalized = str(raw or "").strip()
    if not normalized:
        return default
    normalized = _AGENT_NAME_ALIASES.get(normalized, cast(AgentName | str, normalized))
    if normalized in TEAM_MEMBERS:
        return cast(AgentName, normalized)
    return default


def _coerce_next_agent(raw: Any) -> AgentName | Literal["FINISH"] | None:
    """规范化下一跳节点名称，同时保留终止标记和空值。

    与 `_coerce_agent_name` 的区别是，这里还要支持图执行中的特殊值：
    `None` 表示暂时没有下一跳，`"FINISH"` 表示流程应当结束。

    参数:
        raw: 来自路由器、规划器或恢复状态的原始下一跳值。

    返回:
        合法的 `AgentName`、字符串 `"FINISH"`，或者 `None`。

    示例:
        >>> _coerce_next_agent(None) is None
        True
        >>> _coerce_next_agent(" FINISH ")
        'FINISH'
        >>> _coerce_next_agent("planner")
        'orchestrator'
    """
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        return None
    if normalized == "FINISH":
        return "FINISH"
    return _coerce_agent_name(normalized)


def _coerce_retriever_backend(raw: Any, *, default: RetrieverBackend = "hybrid") -> RetrieverBackend:
    """把检索后端名称映射到当前支持的标准枚举值。

    这个函数除了处理空值，还兼容旧版本配置里的别名，例如
    `"auto"` 会映射为 `"hybrid"`，`"medcpt"` 会映射为 `"vector"`。

    参数:
        raw: 原始后端名称，可能来自配置文件、旧状态或模型输出。
        default: 当输入为空或无法识别时使用的默认后端。

    返回:
        当前系统支持的 `RetrieverBackend`。

    示例:
        >>> _coerce_retriever_backend("auto")
        'hybrid'
        >>> _coerce_retriever_backend("medcpt")
        'vector'
        >>> _coerce_retriever_backend("something-else")
        'hybrid'
    """
    normalized = str(raw or default).strip().lower()
    if not normalized:
        return default
    normalized = _LEGACY_RETRIEVER_BACKEND_MAP.get(normalized, cast(RetrieverBackend | str, normalized))
    if normalized in {"keyword", "vector", "hybrid"}:
        return cast(RetrieverBackend, normalized)
    return default


def _coerce_interval_value(raw: Any) -> IntervalValue:
    """把标量、字典或现成对象转换成统一的 `IntervalValue`。

    这样后续逻辑就不需要同时处理 `3.4`、`{"min": 3, "max": 5}`、
    和 `IntervalValue(...)` 这几种输入格式了。

    参数:
        raw: 原始区间值，可以是 `IntervalValue`、数字，或者包含上下界的字典。

    返回:
        规范化后的 `IntervalValue` 实例。

    异常:
        TypeError: 当输入既不是数字、字典，也不是 `IntervalValue` 时抛出。

    示例:
        >>> _coerce_interval_value(5)
        IntervalValue(lower=5.0, upper=5.0, unit=None, source=None)
        >>> _coerce_interval_value({"min": 1.2, "max": 2.4, "unit": "mmol/L"})
        IntervalValue(lower=1.2, upper=2.4, unit='mmol/L', source=None)
        >>> _coerce_interval_value(IntervalValue(lower=3.0, upper=4.0, unit="mg/dL"))
        IntervalValue(lower=3.0, upper=4.0, unit='mg/dL', source=None)
    """
    if isinstance(raw, IntervalValue):
        return raw
    if isinstance(raw, dict):
        lower = float(raw.get("lower", raw.get("min", 0.0)))
        upper = float(raw.get("upper", raw.get("max", lower)))
        return IntervalValue(
            lower=lower,
            upper=upper,
            unit=raw.get("unit"),
            source=raw.get("source"),
        )
    if isinstance(raw, (int, float)):
        value = float(raw)
        return IntervalValue(lower=value, upper=value)
    raise TypeError(f"Unsupported interval value: {raw!r}")


def _coerce_counterfactual(raw: Any) -> CounterfactualScenario:
    """把原始反事实场景载荷转换成 `CounterfactualScenario`。

    反事实数据里常见的 `current_value` 和 `target_value` 既可能是单值，
    也可能是区间字典，这里会在需要时递归调用 `_coerce_interval_value`
    进行规范化。

    参数:
        raw: 原始反事实场景，通常是字典，也可以已经是 `CounterfactualScenario`。

    返回:
        规范化后的 `CounterfactualScenario`。

    异常:
        TypeError: 当输入不是字典且不是目标对象时抛出。

    示例:
        >>> _coerce_counterfactual({"feature_name": "age", "current_value": 60, "target_value": 50})
        CounterfactualScenario(feature_name='age', current_value=60, target_value=50, direction='unknown', note='')
        >>> scenario = _coerce_counterfactual({
        ...     "feature": "creatinine",
        ...     "target_value": {"lower": 0.9, "upper": 1.1, "unit": "mg/dL"},
        ...     "direction": "decrease",
        ... })
        >>> scenario.target_value
        IntervalValue(lower=0.9, upper=1.1, unit='mg/dL', source=None)
    """
    if isinstance(raw, CounterfactualScenario):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported counterfactual scenario: {raw!r}")
    current_value = raw.get("current_value")
    target_value = raw.get("target_value")
    return CounterfactualScenario(
        feature_name=str(raw.get("feature_name") or raw.get("feature") or "unknown_feature"),
        current_value=_coerce_interval_value(current_value) if isinstance(current_value, dict) else current_value,
        target_value=_coerce_interval_value(target_value) if isinstance(target_value, dict) else target_value,
        direction=str(raw.get("direction") or "unknown"),
        note=str(raw.get("note") or ""),
    )


def _coerce_patient_case(raw: Any) -> PatientCase | None:
    """把原始病例载荷转换成结构化的 `PatientCase`。

    这个函数会一并处理嵌套字段，例如 `interval_inputs` 会被转成
    `IntervalValue`，`counterfactual_scenarios` 会被转成
    `CounterfactualScenario` 列表。

    参数:
        raw: 原始病例数据，允许为 `None`、字典，或已经构造好的 `PatientCase`。

    返回:
        `PatientCase` 实例；如果输入是 `None`，则返回 `None`。

    异常:
        TypeError: 当输入既不是 `None`、字典，也不是 `PatientCase` 时抛出。

    示例:
        >>> case = _coerce_patient_case({
        ...     "patient_id": "P-001",
        ...     "structured_inputs": {"sex": "male"},
        ...     "interval_inputs": {"sbp": {"lower": 120, "upper": 130, "unit": "mmHg"}},
        ... })
        >>> case.patient_id
        'P-001'
        >>> case.interval_inputs["sbp"]
        IntervalValue(lower=120.0, upper=130.0, unit='mmHg', source=None)
        >>> _coerce_patient_case(None) is None
        True
    """
    if raw is None:
        return None
    if isinstance(raw, PatientCase):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported patient_case payload: {raw!r}")

    interval_inputs = {
        str(name): _coerce_interval_value(value)
        for name, value in dict(raw.get("interval_inputs") or {}).items()
    }
    counterfactuals = [_coerce_counterfactual(item) for item in list(raw.get("counterfactual_scenarios") or [])]
    return PatientCase(
        patient_id=raw.get("patient_id"),
        structured_inputs=dict(raw.get("structured_inputs") or {}),
        interval_inputs=interval_inputs,
        multimodal_inputs=dict(raw.get("multimodal_inputs") or {}),
        counterfactual_scenarios=counterfactuals,
    )


def _coerce_clinical_tool_job(raw: Any) -> ClinicalToolJob | None:
    """把临床工具任务输入转换成 `ClinicalToolJob`。

    除了类型清洗，这里还兼容若干旧字段名，例如：
    `note`/`question` 会回填到 `text`，`summary` 会回填到 `case_summary`，
    `progressive_queries` 会回填到 `retrieval_queries`。

    参数:
        raw: 原始任务载荷，允许为 `None`、字典，或已经构造好的 `ClinicalToolJob`。

    返回:
        `ClinicalToolJob` 实例；如果输入是 `None`，则返回 `None`。

    异常:
        TypeError: 当输入既不是 `None`、字典，也不是 `ClinicalToolJob` 时抛出。

    示例:
        >>> job = _coerce_clinical_tool_job({
        ...     "question": "Should this patient receive anticoagulation?",
        ...     "summary": "AF with elevated stroke risk",
        ...     "retriever_backend": "auto",
        ...     "progressive_queries": [{"stage": "initial", "query": "CHA2DS2-VASc"}],
        ... })
        >>> job.text
        'Should this patient receive anticoagulation?'
        >>> job.case_summary
        'AF with elevated stroke risk'
        >>> job.retriever_backend
        'hybrid'
    """
    if raw is None:
        return None
    if isinstance(raw, ClinicalToolJob):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported clinical_tool_job payload: {raw!r}")

    return ClinicalToolJob(
        mode=str(raw.get("mode") or "patient_note"),
        text=str(raw.get("text") or raw.get("note") or raw.get("question") or ""),
        case_summary=str(raw.get("case_summary") or raw.get("summary") or ""),
        structured_case=dict(raw.get("structured_case") or {}),
        risk_hints=[str(item) for item in list(raw.get("risk_hints") or []) if str(item).strip()],
        retrieval_queries=[
            _coerce_retrieval_query(item)
            for item in list(raw.get("retrieval_queries") or raw.get("progressive_queries") or [])
        ],
        selected_tool_pmid=(str(raw.get("selected_tool_pmid") or "").strip() or None),
        selected_tool=dict(raw.get("selected_tool") or {}),
        dispatch_query_text=str(raw.get("dispatch_query_text") or ""),
        selection_context=dict(raw.get("selection_context") or {}),
        top_k=_resolve_retrieval_top_k(raw.get("top_k")),
        risk_count=int(raw.get("risk_count") or 5),
        max_selected_tools=(
            None if raw.get("max_selected_tools") is None else int(raw.get("max_selected_tools"))
        ),
        retriever_backend=_coerce_retriever_backend(raw.get("retriever_backend")),
        riskcalcs_path=raw.get("riskcalcs_path"),
        pmid_metadata_path=raw.get("pmid_metadata_path"),
        llm_model=raw.get("llm_model"),
        forced_tool_pmid=(str(raw.get("forced_tool_pmid") or "").strip() or None),
        temperature=float(raw.get("temperature") or 0.0),
        max_rounds=int(raw.get("max_rounds") or 12),
    )


def _coerce_tool(raw: Any) -> ToolSpec:
    """把原始工具描述转换成 `ToolSpec`。

    适用于把 JSON、LLM 输出或缓存中的工具定义恢复成统一对象，
    并给缺失字段补上安全默认值。

    参数:
        raw: 原始工具定义，可以是字典或已经构造好的 `ToolSpec`。

    返回:
        规范化后的 `ToolSpec`。

    异常:
        TypeError: 当输入不是字典且不是 `ToolSpec` 时抛出。

    示例:
        >>> _coerce_tool({"name": "search", "description": "Search medical literature"})
        ToolSpec(name='search', description='Search medical literature', required=False, input_schema={})
        >>> _coerce_tool({"description": "missing name"})
        ToolSpec(name='unnamed_tool', description='missing name', required=False, input_schema={})
    """
    if isinstance(raw, ToolSpec):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported tool spec: {raw!r}")
    return ToolSpec(
        name=str(raw.get("name") or "unnamed_tool"),
        description=str(raw.get("description") or ""),
        required=bool(raw.get("required", False)),
        input_schema=dict(raw.get("input_schema") or {}),
    )


def _coerce_retrieval_query(raw: Any) -> RetrievalQuery:
    """把检索查询载荷转换成 `RetrievalQuery`。

    当上游输出字段名不一致时，这里会把 `query` 也视为 `text`，
    从而兼容不同版本的提示词或历史存档格式。

    参数:
        raw: 原始检索查询对象，通常是字典。

    返回:
        规范化后的 `RetrievalQuery`。

    异常:
        TypeError: 当输入不是字典且不是 `RetrievalQuery` 时抛出。

    示例:
        >>> _coerce_retrieval_query({"stage": "initial", "query": "sepsis lactate threshold"})
        RetrievalQuery(stage='initial', text='sepsis lactate threshold', intent='', rationale='', priority=1)
        >>> _coerce_retrieval_query({"stage": "follow_up", "text": "vasopressor timing", "priority": 2})
        RetrievalQuery(stage='follow_up', text='vasopressor timing', intent='', rationale='', priority=2)
    """
    if isinstance(raw, RetrievalQuery):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported retrieval query: {raw!r}")
    return RetrievalQuery(
        stage=str(raw.get("stage") or "unspecified"),
        text=str(raw.get("text") or raw.get("query") or ""),
        intent=str(raw.get("intent") or ""),
        rationale=str(raw.get("rationale") or ""),
        priority=int(raw.get("priority") or 1),
    )


def _coerce_calculator_match(raw: Any) -> CalculatorMatch:
    """把计算器匹配结果转换成 `CalculatorMatch`。

    这个函数适合清洗来自检索模块或决策模块的候选计算器信息，
    让后续执行器始终拿到相同的数据结构。

    参数:
        raw: 原始计算器匹配结果，通常是字典。

    返回:
        `CalculatorMatch` 实例。

    异常:
        TypeError: 当输入不是字典且不是 `CalculatorMatch` 时抛出。

    示例:
        >>> match = _coerce_calculator_match({
        ...     "pmid": "12345678",
        ...     "title": "CHA2DS2-VASc Score",
        ...     "missing_inputs": ["age", "stroke_history"],
        ... })
        >>> match.pmid
        '12345678'
        >>> match.missing_inputs
        ['age', 'stroke_history']
    """
    if isinstance(raw, CalculatorMatch):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported calculator match: {raw!r}")
    return CalculatorMatch(
        pmid=str(raw.get("pmid") or ""),
        title=str(raw.get("title") or ""),
        category=str(raw.get("category") or "risk_score"),
        rationale=str(raw.get("rationale") or ""),
        applicability=str(raw.get("applicability") or "unknown"),
        missing_inputs=[str(item) for item in list(raw.get("missing_inputs") or []) if str(item).strip()],
        execution_status=raw.get("execution_status"),
        value=raw.get("value"),
    )


def _coerce_protocol_recommendation(raw: Any) -> ProtocolRecommendation:
    """把 protocol 阶段的原始输出转换成 `ProtocolRecommendation`。

    这一步会统一协议/路径推荐的名称、状态、理由以及关联计算器列表，
    便于 reporter 或后续审查模块直接消费。

    参数:
        raw: 原始推荐对象，通常是字典。

    返回:
        `ProtocolRecommendation` 实例。

    异常:
        TypeError: 当输入不是字典且不是 `ProtocolRecommendation` 时抛出。

    示例:
        >>> rec = _coerce_protocol_recommendation({
        ...     "name": "NSTEMI early invasive strategy",
        ...     "status": "matched",
        ...     "linked_calculators": ["GRACE"],
        ... })
        >>> rec.status
        'matched'
        >>> rec.linked_calculators
        ['GRACE']
    """
    if isinstance(raw, ProtocolRecommendation):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported protocol recommendation: {raw!r}")
    return ProtocolRecommendation(
        name=str(raw.get("name") or "protocol_review"),
        category=str(raw.get("category") or "care_pathway"),
        status=str(raw.get("status") or "insufficient_data"),
        rationale=str(raw.get("rationale") or ""),
        linked_calculators=[str(item) for item in list(raw.get("linked_calculators") or []) if str(item).strip()],
        linked_trials=[str(item) for item in list(raw.get("linked_trials") or []) if str(item).strip()],
        corrections=[str(item) for item in list(raw.get("corrections") or []) if str(item).strip()],
    )


def _coerce_safety_issue(raw: Any) -> SafetyIssue:
    """把审查阶段输出转换成 `SafetyIssue`。

    适用于把 reviewer、validator 或 reporter 回传的安全提醒，
    统一成可排序、可展示、可聚合的结构。

    参数:
        raw: 原始安全问题对象，通常是字典。

    返回:
        `SafetyIssue` 实例。

    异常:
        TypeError: 当输入不是字典且不是 `SafetyIssue` 时抛出。

    示例:
        >>> issue = _coerce_safety_issue({
        ...     "severity": "critical",
        ...     "message": "Potential contraindication with active bleeding",
        ...     "blocking": True,
        ... })
        >>> issue.severity
        'critical'
        >>> issue.blocking
        True
    """
    if isinstance(raw, SafetyIssue):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported safety issue: {raw!r}")
    return SafetyIssue(
        severity=str(raw.get("severity") or "info"),
        message=str(raw.get("message") or ""),
        source=str(raw.get("source") or ""),
        blocking=bool(raw.get("blocking", False)),
    )


def _coerce_calculation_task(raw: Any) -> CalculationTask:
    """把原始计算任务转换成 `CalculationTask`。

    它会清洗任务编号、决策方式、输入缺口和执行器信息，方便把
    规划阶段输出直接交给 calculation 子代理或执行模块。

    参数:
        raw: 原始计算任务对象，通常是字典。

    返回:
        `CalculationTask` 实例。

    异常:
        TypeError: 当输入不是字典且不是 `CalculationTask` 时抛出。

    示例:
        >>> task = _coerce_calculation_task({
        ...     "task_id": "calc-1",
        ...     "label": "Calculate SOFA score",
        ...     "decision": "direct",
        ...     "required_inputs": ["platelets", "bilirubin"],
        ... })
        >>> task.task_id
        'calc-1'
        >>> task.decision
        'direct'
    """
    if isinstance(raw, CalculationTask):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported calculation task: {raw!r}")
    return CalculationTask(
        task_id=str(raw.get("task_id") or ""),
        label=str(raw.get("label") or "calculation_task"),
        category=str(raw.get("category") or "risk_score"),
        decision=str(raw.get("decision") or "skip"),
        required_inputs=[str(item) for item in list(raw.get("required_inputs") or []) if str(item).strip()],
        available_inputs=[str(item) for item in list(raw.get("available_inputs") or []) if str(item).strip()],
        missing_inputs=[str(item) for item in list(raw.get("missing_inputs") or []) if str(item).strip()],
        rationale=str(raw.get("rationale") or ""),
        executor=str(raw.get("executor") or "calculation_subagent"),
    )


def _coerce_calculation_artifact(raw: Any) -> CalculationArtifact:
    """把原始计算结果载荷转换成 `CalculationArtifact`。

    用于把执行结果、估算结果或跳过原因统一包装起来，方便后续
    reporter 汇总和状态序列化。

    参数:
        raw: 原始计算结果对象，通常是字典。

    返回:
        `CalculationArtifact` 实例。

    异常:
        TypeError: 当输入不是字典且不是 `CalculationArtifact` 时抛出。

    示例:
        >>> artifact = _coerce_calculation_artifact({
        ...     "name": "SOFA",
        ...     "status": "completed",
        ...     "value": 7,
        ...     "unit": "points",
        ... })
        >>> artifact.name
        'SOFA'
        >>> artifact.value
        7
    """
    if isinstance(raw, CalculationArtifact):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported calculation artifact: {raw!r}")
    return CalculationArtifact(
        name=str(raw.get("name") or "calculation_result"),
        category=str(raw.get("category") or "risk_score"),
        status=str(raw.get("status") or "skipped"),
        value=raw.get("value"),
        unit=raw.get("unit"),
        source=str(raw.get("source") or ""),
        rationale=str(raw.get("rationale") or ""),
        linked_task_id=raw.get("linked_task_id"),
        linked_calculator=raw.get("linked_calculator"),
        missing_inputs=[str(item) for item in list(raw.get("missing_inputs") or []) if str(item).strip()],
    )


def _coerce_treatment_recommendation(raw: Any) -> TreatmentRecommendation:
    """把原始治疗建议载荷转换成 `TreatmentRecommendation`。

    适用于将 protocol、trial matching 或最终建议模块的输出统一整理成
    同一结构，方便前端展示和后续审阅。

    参数:
        raw: 原始治疗建议对象，通常是字典。

    返回:
        `TreatmentRecommendation` 实例。

    异常:
        TypeError: 当输入不是字典且不是 `TreatmentRecommendation` 时抛出。

    示例:
        >>> rec = _coerce_treatment_recommendation({
        ...     "name": "Start therapeutic anticoagulation",
        ...     "strategy": "guideline_based",
        ...     "source": "ESC 2024",
        ...     "actions": ["review bleeding risk", "confirm renal function"],
        ... })
        >>> rec.source
        'ESC 2024'
        >>> rec.actions
        ['review bleeding risk', 'confirm renal function']
    """
    if isinstance(raw, TreatmentRecommendation):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported treatment recommendation: {raw!r}")
    return TreatmentRecommendation(
        name=str(raw.get("name") or "treatment_review"),
        strategy=str(raw.get("strategy") or "manual_review"),
        source=str(raw.get("source") or ""),
        status=str(raw.get("status") or "manual_review"),
        rationale=str(raw.get("rationale") or ""),
        linked_calculators=[str(item) for item in list(raw.get("linked_calculators") or []) if str(item).strip()],
        linked_trials=[str(item) for item in list(raw.get("linked_trials") or []) if str(item).strip()],
        actions=[str(item) for item in list(raw.get("actions") or []) if str(item).strip()],
    )


def ensure_state(data: GraphState | dict[str, Any]) -> GraphState:
    """把图输入规范化成完整、可运行的 `GraphState`。

    这是整个文件里最核心的入口之一。它允许上游直接传入已经构造好的
    `GraphState`，也允许传入一个松散的字典；函数会负责：

    1. 校验 `request` / `user_input`
    2. 递归恢复病例、任务、工具、计划和推荐对象
    3. 给缺失字段补齐默认值
    4. 把松散 JSON 风格数据转换成稳定的 dataclass 状态

    参数:
        data: `GraphState` 实例，或包含图状态信息的字典。

    返回:
        一个字段完整、类型稳定的 `GraphState`。

    异常:
        TypeError: 当输入既不是 `GraphState` 也不是字典时抛出。
        ValueError: 当输入里既没有 `request` 也没有 `user_input` 时抛出。

    示例:
        >>> state = ensure_state({"request": "Summarize the case"})
        >>> state.request
        'Summarize the case'
        >>> state.status
        'pending'
        >>> state.plan
        []
        >>> state = ensure_state({
        ...     "user_input": "Plan next step",
        ...     "plan": [{"agent": "planner", "title": "Draft plan", "description": "Create a safe plan"}],
        ... })
        >>> state.plan[0].agent_name
        'orchestrator'
        >>> state.plan[0].title
        'Draft plan'
    """
    if isinstance(data, GraphState):
        return data
    if not isinstance(data, dict):
        raise TypeError("Graph input must be a GraphState or dict")

    request = str(data.get("request") or data.get("user_input") or "").strip()
    if not request:
        raise ValueError("Graph input requires `request` or `user_input`.")

    plan: list[PlanStep] = []
    for item in list(data.get("plan") or []):
        if isinstance(item, PlanStep):
            plan.append(item)
            continue
        if not isinstance(item, dict):
            continue
        plan.append(
            PlanStep(
                step_id=int(item.get("step_id") or (len(plan) + 1)),
                agent_name=_coerce_agent_name(item.get("agent_name") or item.get("agent")),
                title=str(item.get("title") or f"Step {len(plan) + 1}"),
                description=str(item.get("description") or ""),
                status=str(item.get("status") or "pending"),
                note=str(item.get("note") or ""),
                result=item.get("result"),
            )
        )

    return GraphState(
        request=request,
        patient_case=_coerce_patient_case(data.get("patient_case")),
        clinical_tool_job=_coerce_clinical_tool_job(data.get("clinical_tool_job")),
        plan=plan,
        tools=[_coerce_tool(item) for item in list(data.get("tools") or [])],
        tool_registry=dict(data.get("tool_registry") or {}),
        messages=list(data.get("messages") or []),
        problem_list=[str(item) for item in list(data.get("problem_list") or []) if str(item).strip()],
        retrieval_queries=[_coerce_retrieval_query(item) for item in list(data.get("retrieval_queries") or [])],
        calculator_matches=[_coerce_calculator_match(item) for item in list(data.get("calculator_matches") or [])],
        protocol_recommendations=[
            _coerce_protocol_recommendation(item) for item in list(data.get("protocol_recommendations") or [])
        ],
        safety_issues=[_coerce_safety_issue(item) for item in list(data.get("safety_issues") or [])],
        final_output=dict(data.get("final_output") or {}),
        orchestrator_result=dict(data.get("orchestrator_result") or {}),
        assessment_bundle=dict(data.get("assessment_bundle") or {}),
        calculation_bundle=dict(data.get("calculation_bundle") or {}),
        trial_retrieval_bundle=dict(data.get("trial_retrieval_bundle") or {}),
        treatment_bundle=dict(data.get("treatment_bundle") or {}),
        reporter_result=dict(data.get("reporter_result") or {}),
        review_report=dict(data.get("review_report") or {}),
        clinical_answer=[
            dict(item) if isinstance(item, dict) else {"value": item}
            for item in list(data.get("clinical_answer") or [])
        ],
        deep_thinking_mode=bool(data.get("deep_thinking_mode", True)),
        search_before_planning=bool(data.get("search_before_planning", False)),
        pass_through_expert=bool(data.get("pass_through_expert", False)),
        auto_accepted_plan=bool(data.get("auto_accepted_plan", False)),
        next_agent=_coerce_next_agent(data.get("next_agent")),
        status=str(data.get("status") or "pending"),
        errors=[str(item) for item in list(data.get("errors") or []) if str(item).strip()],
        department=str(data.get("department") or "").strip(),
        department_tags=[str(item) for item in list(data.get("department_tags") or []) if str(item).strip()],
        structured_case_json=dict(data.get("structured_case_json") or {}),
        reporter_feedback=[
            str(item)
            for item in list(data.get("reporter_feedback") or [])
            if str(item).strip()
        ],
        reporter_attempts=int(data.get("reporter_attempts") or 0),
        max_reporter_attempts=int(data.get("max_reporter_attempts") or 3),
        review_passed=bool(data.get("review_passed", False)),
        abandoned=bool(data.get("abandoned", False)),
        calculation_tasks=[_coerce_calculation_task(item) for item in list(data.get("calculation_tasks") or [])],
        calculation_results=[
            _coerce_calculation_artifact(item) for item in list(data.get("calculation_results") or [])
        ],
        treatment_recommendations=[
            _coerce_treatment_recommendation(item) for item in list(data.get("treatment_recommendations") or [])
        ],
    )


def _state_value_to_dict(value: Any) -> Any:
    """递归地把状态值转换成适合 JSON 化的普通 Python 对象。

    它会处理 dataclass、字典、列表、元组、集合和可调用对象，
    从而让 `GraphState` 内部那些不方便直接序列化的值，变成更稳定、
    更容易落盘或输出给前端的结构。

    参数:
        value: 任意状态值，可能是 dataclass、容器、集合或普通标量。

    返回:
        由字典、列表、标量等组成的可序列化 Python 对象。

    示例:
        >>> _state_value_to_dict(IntervalValue(lower=2.0, upper=3.5, unit="mg/dL"))
        {'lower': 2.0, 'upper': 3.5, 'unit': 'mg/dL', 'source': None}
        >>> _state_value_to_dict({"callback": print})
        {'callback': {'callable': 'print'}}
        >>> _state_value_to_dict({"tags": {"b", "a"}})
        {'tags': ['a', 'b']}
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _state_value_to_dict(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _state_value_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_value_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [_state_value_to_dict(item) for item in value]
    if isinstance(value, set):
        return [_state_value_to_dict(item) for item in sorted(value, key=repr)]
    if callable(value):
        return {"callable": getattr(value, "__name__", value.__class__.__name__)}
    return value


def state_to_dict(state: GraphState) -> dict[str, Any]:
    """把 `GraphState` 序列化成可输出的字典表示。

    与直接递归转换不同，这个函数还会特殊处理 `tool_registry`：
    运行时注册表里往往存的是不可 JSON 序列化的工具对象，因此这里会只保留
    `registered_tools` 名称列表，避免把函数对象或复杂实例直接写入结果。

    参数:
        state: 需要序列化的 `GraphState` 实例。

    返回:
        一个适合保存、调试或向外部接口返回的字典。

    异常:
        TypeError: 当传入的不是 dataclass 实例时抛出。

    示例:
        >>> state = ensure_state({"request": "Review plan"})
        >>> payload = state_to_dict(state)
        >>> payload["request"]
        'Review plan'
        >>> payload["tool_registry"]
        {'registered_tools': []}
    """
    if not is_dataclass(state):
        raise TypeError("state_to_dict expects a GraphState dataclass instance")
    payload = _state_value_to_dict(state)
    payload["tool_registry"] = {"registered_tools": sorted(list(state.tool_registry.keys()))}
    return payload


State = GraphState
