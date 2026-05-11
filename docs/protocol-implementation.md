# MedAI Protocol 实现说明

本文档说明 `/home/yuanzy/MadAI` 当前版本中 `protocol` 节点的真实实现方式，重点覆盖它在工作流中的位置、输入输出、trial 检索解析、候选选择、治疗建议生成、trace 输出和调试入口。

注意：本文描述的是当前代码行为，不是未来计划。未来把 protocol 升级成 criteria-level eligibility agent 的设计，见 `docs/plans/2026-05-06-protocol-eligibility-agent-implementation-plan.md`。

## 1. 总体定位

当前 MedAI 的主工作流是一个轻量 graph：

```text
orchestrator -> clinical_assisstment -> protocol -> reporter
```

入口实现：

- `agent/graph/builder.py`
  - `SimpleAgentGraph`
  - `build_graph()`
  - `build_graph_with_memory()`

`protocol` 不是独立服务，也不是单纯依赖 prompt 自由生成的 LLM agent。它是 graph 中的一个 Python 节点，主函数是：

- `agent/graph/nodes.py`
  - `protocol_node(state: GraphState) -> GraphState`

它的核心职责是：

1. 接收上游 `clinical_assisstment` 已经整理好的病例结构化信息。
2. 调用本地 trial retriever 检索候选临床试验。
3. 从候选 trial 中选择一个主要候选，并做粗粒度适配评估。
4. 结合 calculator 结果生成治疗建议或 fallback 建议。
5. 写入 `trial_retrieval_bundle`、`treatment_bundle` 和 `protocol_recommendations`。
6. 把执行过程记录到 `final_output.execution_trace`。

对应 prompt 契约在：

- `agent/prompt/protocol.md`

该 prompt 描述了 protocol 的角色和输出要求，但当前实际判断逻辑主要由 `agent/graph/nodes.py` 和 `agent/tools/trial_vector_retrieval_tools.py` 中的规则代码完成。

## 2. Protocol 的输入

`protocol` 不直接解析原始病历文本，也不重新运行 calculator。它读取的是 `GraphState` 中已经存在的中间产物。

主要输入字段：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `state.request` | 外层调用输入 | 用于记录 prompt context 和 trace |
| `state.structured_case_json` | `clinical_assisstment_node` | trial query profile 和治疗判断的病例基础 |
| `state.calculation_results` | calculator 子执行器或 baseline calculator bundle | 决定是否能生成 risk-informed treatment |
| `state.calculator_matches` | calculator 检索/匹配结果 | calculator 未执行时的 fallback 依据 |
| `state.department_tags` | orchestrator / clinical_assisstment | trial 检索时传入 retriever |
| `state.clinical_tool_job.retriever_backend` | 外层任务配置 | 选择 `bm25`、`vector` 或 `hybrid` 检索后端 |
| `state.tool_registry["trial_retriever"]` | 可选运行时注入 | 覆盖默认 trial retriever |

相关结构定义在：

- `agent/graph/types.py`
  - `GraphState`
  - `ClinicalToolJob`
  - `CalculationArtifact`
  - `TreatmentRecommendation`
  - `ProtocolRecommendation`

`protocol` 当前只依赖 `structured_case`、`calculation_results`、`calculator_matches` 这条新链路，不再依赖旧的四类标签或 `patient_strata / calculator_categories / protocol_targets` 链路。

## 3. Protocol 的输出

`protocol_node` 会写入以下字段：

| 输出字段 | 位置 | 说明 |
| --- | --- | --- |
| `state.trial_retrieval_bundle` | graph state | trial 检索完整结果 |
| `state.treatment_bundle` | graph state | protocol 阶段的治疗/试验决策包 |
| `state.treatment_recommendations` | graph state | 详细治疗建议对象列表 |
| `state.protocol_recommendations` | graph state | reporter 更容易消费的 protocol 摘要对象列表 |
| `state.final_output["trial_retrieval_bundle"]` | final output | 对外暴露 trial 检索结果 |
| `state.final_output["treatment_bundle"]` | final output | 对外暴露治疗决策包 |
| `state.final_output.execution_trace` | final output | 记录 protocol 的工具调用和输出摘要 |

`treatment_bundle` 的当前结构：

```json
{
  "recommendations": [],
  "trial_candidates": [],
  "trial_candidate_ids": [],
  "trial_selection": {},
  "note": "This node now owns treatment and clinical-trial judgment in the core MedAI workflow."
}
```

`trial_retrieval_bundle` 的典型结构：

```json
{
  "query_text": "...",
  "query_profile": {},
  "backend_used": "hybrid",
  "available_backends": ["bm25", "vector", "hybrid"],
  "department_tags": [],
  "fallback_to_full_catalog": false,
  "coarse_candidate_ids": [],
  "coarse_candidate_ranking": [],
  "bm25_top5": [],
  "vector_top5": [],
  "candidate_ranking": []
}
```

`protocol_recommendations` 是由 `treatment_recommendations` 压缩得到的摘要对象，结构定义为：

```python
@dataclass(slots=True)
class ProtocolRecommendation:
    name: str
    category: str = "care_pathway"
    status: ProtocolStatus = "insufficient_data"
    rationale: str = ""
    linked_calculators: list[str] = field(default_factory=list)
    linked_trials: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
```

## 4. Protocol 主流程

主函数在 `agent/graph/nodes.py`：

```python
def protocol_node(state: GraphState) -> GraphState:
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
    ...
```

可以把它理解为四步：

```text
structured_case
  -> trial retrieval
  -> trial selection / coarse eligibility assessment
  -> treatment recommendation
  -> protocol summary + trace
```

### 4.1 先检索 trial

入口函数：

- `agent/graph/nodes.py`
  - `_retrieve_trial_candidates(state)`

逻辑：

1. 从 `state.structured_case_json` 取病例结构。
2. 如果没有 structured case，直接返回空 trial bundle。
3. 读取检索后端：
   - 如果 `state.clinical_tool_job` 存在，用 `state.clinical_tool_job.retriever_backend`。
   - 否则默认 `hybrid`。
4. 调 `_resolve_trial_retriever(state)` 获取 trial retriever。
5. 优先调用 retriever 的 `retrieve_from_structured_case(...)`。
6. 如果 retriever 没有这个方法，则尝试调用 `retrieve_candidates(...)`。
7. 如果两者都没有，抛出 `TypeError`。

默认 retriever 创建逻辑：

```python
def _resolve_trial_retriever(state: GraphState):
    trial_retriever = state.tool_registry.get("trial_retriever")
    if trial_retriever is not None:
        return trial_retriever

    backend = state.clinical_tool_job.retriever_backend if state.clinical_tool_job is not None else "hybrid"
    return create_trial_chunk_retrieval_tool(
        backend=backend,
        vector_store="auto",
    )
```

也就是说：

- 可以通过 `state.tool_registry["trial_retriever"]` 注入测试替身或自定义 retriever。
- 默认走 XML-derived trial chunk KB。
- 默认 vector store 是 `auto`。
- 默认后端一般是 `hybrid`。

### 4.2 再构建 trial selection

入口函数：

- `agent/graph/nodes.py`
  - `_build_protocol_trial_selection(trial_bundle, structured_case=...)`

它会：

1. 从 `trial_bundle["candidate_ranking"]` 中选择一个 trial。
2. 对该 trial 做粗粒度 eligibility assessment。
3. 整理 trial status assessment。
4. 收集 evidence。
5. 整理最多 3 个 alternatives。

如果没有候选，返回：

```json
{
  "selected_trial": null,
  "selection_reason": "No trial candidate survived retrieval for the current case.",
  "eligibility_assessment": {
    "fit": "not_available",
    "matching_signals": [],
    "conflicts": [],
    "missing_information": ["No ranked trial candidates were returned."],
    "next_checks": ["Adjust the protocol query profile or broaden the trial corpus before retrying."]
  },
  "trial_status_assessment": {},
  "evidence": [],
  "alternatives": []
}
```

### 4.3 再生成 treatment recommendations

入口函数：

- `agent/graph/nodes.py`
  - `_build_treatment_recommendations(state, trial_bundle=...)`

它不是简单把 top trial 当作治疗方案，而是先看 calculator 是否产生了可用结果。

分支顺序：

1. `completed_results`
2. `partial_results`
3. `estimated_results`
4. `calculator_matches`
5. 无任何 calculator 信号

如果有 completed calculator result，生成 `risk_informed_treatment`。

如果没有 completed result，但有 trial candidates，`_augment_treatment_recommendations_with_trials(...)` 会插入一条 `trial candidate review`，提醒需要人工复核 trial eligibility 和 enrollment details。

### 4.4 最后写入 state 和 trace

`protocol_node` 会把结果写入：

```python
state.treatment_recommendations = recommendations
state.protocol_recommendations = protocol_recommendations
state.trial_retrieval_bundle = trial_retrieval_bundle
state.treatment_bundle = treatment_bundle
state.final_output["trial_retrieval_bundle"] = trial_retrieval_bundle
state.final_output["treatment_bundle"] = treatment_bundle
```

并追加 agent trace：

- `trial_candidate_retriever`
- `treatment_matcher`

这些 trace 对调试很重要，因为 reporter 或外层调用方可以看到 protocol 当时用了什么 structured case、召回了哪些 trial、最后生成了多少 recommendation。

## 5. Trial query profile 的解析

核心函数：

- `agent/tools/trial_vector_retrieval_tools.py`
  - `build_protocol_trial_query_profile(...)`

它负责把 `structured_case` 解析成 trial 检索用的 query profile。

输入可以包含：

```python
structured_case: Mapping[str, Any] | None = None
raw_text: Any = ""
case_summary: Any = None
problem_list: Any = None
known_facts: Any = None
structured_inputs: Any = None
```

如果传入的 payload 里面还有一层 `structured_case`，函数会自动拆开：

```python
if isinstance(case_payload.get("structured_case"), Mapping):
    case_payload = dict(case_payload["structured_case"])
```

### 5.1 文本片段来源

函数会从 case payload 中提取多个文本片段，合并为 trial 检索语境：

- 原始文本
- 病例摘要
- problem list
- known facts
- structured inputs 中的可读内容

随后对这些文本进行规则匹配，得到：

- `patient_positive_terms`
- `patient_negative_terms`
- `referenced_intervention_terms`
- `referenced_intent_terms`
- `trial_condition_terms`

这些术语来自 `_TRIAL_TERM_DEFINITIONS` 中定义的模式匹配规则。

### 5.2 条件词、干预词、意图词

query profile 中最重要的三类 trial 搜索字段是：

| 字段 | 含义 |
| --- | --- |
| `trial_condition_terms` | 用于匹配 trial condition 的疾病/临床问题 |
| `trial_intervention_terms` | 用于匹配 intervention 的治疗/干预方向 |
| `trial_intent_terms` | 用于匹配 trial purpose 或治疗目的 |

举例：

- 病例中出现 stroke risk 和 anticoagulation 相关表达时，代码可能推断 `atrial fibrillation` 作为 trial-search anchor。
- 如果出现 `warfarin`、`anticoagulation`、`antithrombotic therapy`，会加入 intervention terms。
- 如果出现 prevention 语义，可能加入 `stroke prevention` 或 `secondary prevention`。

这类额外推断记录在 `derivation_notes` 中。

### 5.3 fallback terms

如果没有识别到 domain-specific condition term，代码不会强行把 `problem_list` 直接当作结构化 condition filter，而是把它们作为 coarse recall 的 fallback focus terms。

这点很重要：它避免把长 problem phrase 错误塞进 structured filter，导致召回过窄。

### 5.4 年龄和性别

函数会解析：

- `age_years`
- `gender`

然后生成 demographic terms：

- `"{age_years} year old"`
- `"older adult"`，当年龄大于等于 75
- gender lower-case

这些信息既会进入 `query_text`，也会进入 `payload_filters`。

### 5.5 query_text 的组成

最终 `query_text` 不是单行 keyword，而是多行结构：

```text
<structured query text>
trial focus hints: ...
patient profile: ...
screening constraints: ...
```

其中：

- `trial focus hints` 来自 condition / intent / intervention / demographic / fallback terms。
- `patient profile` 来自正向病例特征。
- `screening constraints` 来自明确否定项，例如 `no diabetes`。

### 5.6 payload_filters

`build_protocol_trial_query_profile` 还会生成结构化 filter：

```json
{
  "must": [],
  "should": [],
  "must_not": [],
  "age_years": 78,
  "gender": "Male"
}
```

常见 should filter：

```json
{
  "field": "condition_terms",
  "values": ["atrial fibrillation"]
}
```

```json
{
  "field": "intervention_terms",
  "values": ["anticoagulation"]
}
```

如果识别到 stroke prevention / secondary prevention，会加入：

```json
{
  "field": "primary_purpose",
  "values": ["Prevention"]
}
```

所有 query 都会倾向 Interventional study：

```json
{
  "field": "study_type",
  "values": ["Interventional"]
}
```

目前 `must_not` 主要处理：

- `diabetes`
- `congestive heart failure`

如果病例明确没有这些病，且 trial condition 中包含它们，会被视为冲突信号。

## 6. Trial 检索工具

默认 trial 检索工具在：

- `agent/tools/trial_vector_retrieval_tools.py`
  - `TrialChunkRetrievalTool`
  - `create_trial_chunk_retrieval_tool(...)`

底层 trial chunk catalog 在：

- `agent/retrieval/trial_chunks.py`
  - `TrialChunkCatalog`
  - `TrialChunkDocument`

### 6.1 Trial chunk 数据

`TrialChunkDocument` 表示一个 trial 的一个可检索片段。关键字段：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | chunk 唯一 ID |
| `nct_id` | ClinicalTrials.gov NCT ID |
| `title` | trial title + chunk type |
| `chunk_type` | overview / eligibility / intervention 等 chunk 类型 |
| `text` | 原始片段文本 |
| `embedding_text` | 用于 embedding 检索的文本 |
| `rank_weight` | chunk 排名权重 |
| `trial_title` | trial 标题 |
| `record_payload` | trial 级记录 |

catalog 默认数据目录解析顺序：

1. `outputs/trial_vector_kb_full`
2. `outputs/trial_vector_kb_part1_caseprobe`
3. `outputs/trial_vector_kb`
4. `data/trial_vector_kb`

如果都不存在，会返回第一个默认路径作为预期输出目录。

### 6.2 检索后端

`TrialChunkRetrievalTool` 内部创建 structured retriever：

```python
self._retriever = create_structured_retriever(
    catalog,
    bm25_retriever=self.keyword_retriever,
    vector_retriever=vector_retriever,
    query_builder=build_protocol_trial_query_text,
    default_backend=backend,
    id_field="chunk_id",
)
```

支持的常见 backend：

- `bm25`
- `vector`
- `hybrid`

名称归一化规则：

| 输入 | 实际 |
| --- | --- |
| `keyword` | `bm25` |
| `auto` | `hybrid` |
| `medcpt` | `vector` |

如果 vector 检索失败或不可用，部分路径会自动 fallback 到 BM25。

### 6.3 两阶段检索

当前 protocol 使用：

```python
TrialChunkRetrievalTool.retrieve_from_structured_case(...)
```

主要步骤：

1. 构造 `query_profile`。
2. 构造 `query_text`。
3. 调 `retrieve_coarse_from_structured_case(...)` 做 coarse recall。
4. 得到 `coarse_candidate_ids`。
5. 在 coarse candidates 范围内跑 BM25 chunk retrieval。
6. 如果 vector retriever 存在且 backend 不是 BM25，再跑 vector chunk retrieval。
7. 分别聚合成 `bm25_top5` 和 `vector_top5`。
8. 合并 BM25 与 vector chunk rows，聚合成最终 `candidate_ranking`。

返回结构：

```json
{
  "query_text": "...",
  "query_profile": {},
  "backend_used": "hybrid",
  "available_backends": [],
  "department_tags": [],
  "fallback_to_full_catalog": false,
  "coarse_candidate_ids": [],
  "coarse_candidate_ranking": [],
  "bm25_top5": [],
  "vector_top5": [],
  "candidate_ranking": []
}
```

### 6.4 chunk 到 trial 的聚合

检索底层返回的是 chunk 级结果，但 protocol 需要 trial 级候选。因此工具会把多个 chunk 聚合回同一个 NCT trial。

聚合后每个 candidate 会包含：

- `nct_id`
- `title`
- `score`
- `matched_chunks`
- `best_evidence_text`
- `status`
- `enrollment_open`
- `status_reason`
- `overall_status`
- `study_type`
- `phase`
- `primary_purpose`
- `conditions`
- `interventions`
- `gender`
- `age_floor_years`
- `age_ceiling_years`
- `actions`

其中 `best_evidence_text` 是后续 trial selection evidence 的重要来源。

### 6.5 trial 状态映射

函数：

- `agent/tools/trial_vector_retrieval_tools.py`
  - `_map_protocol_trial_status(record)`

它把 ClinicalTrials.gov 的 `overall_status` 映射为 MedAI protocol 内部状态：

| 原始 trial 状态类型 | protocol 状态 | 含义 |
| --- | --- | --- |
| open / preparing open | `trial_matched` | 可以作为直接 trial candidate，但仍需人工检查 inclusion/exclusion |
| completed / evidence support 类 | `trial_matched` | 不开放入组，但可作为 protocol/evidence support |
| withdrawn / terminated / suspended 等 | `abandoned` | 不应作为当前治疗试验推荐 |
| unknown | `manual_review` | 状态不清楚，必须人工确认 |
| 其他未清晰映射状态 | `manual_review` | 保守处理 |

这个状态会影响候选排序、trial selection 和 recommendation actions。

## 7. Trial candidate 选择和适配评估

### 7.1 选择规则

函数：

- `agent/graph/nodes.py`
  - `_select_protocol_trial_candidate(trial_bundle)`

规则：

1. 从 `trial_bundle["candidate_ranking"]` 读取候选。
2. 优先返回第一个非 `abandoned` candidate。
3. 如果所有候选都是 `abandoned`，返回第一个候选。
4. 如果没有候选，返回 `{}`。

这意味着当前 selection 是保守但简单的 top-ranked non-abandoned 规则，不是完整 eligibility 判定。

### 7.2 粗粒度 eligibility assessment

函数：

- `agent/graph/nodes.py`
  - `_assess_protocol_trial_candidate(candidate, query_profile=...)`

它会检查：

| 检查项 | 逻辑 |
| --- | --- |
| condition match | query profile 中的 `trial_condition_terms` 是否出现在 candidate 文本中 |
| intervention match | `trial_intervention_terms` 是否出现在 candidate 文本中 |
| intent match | `trial_intent_terms` 是否出现在 candidate 文本中 |
| negative conflicts | 当前主要检查 diabetes / congestive heart failure |
| age floor | 病例年龄是否低于 trial minimum age |
| age ceiling | 病例年龄是否高于 trial maximum age |
| gender | trial gender 是否是 All 或等于病例性别 |
| evidence chunk | 是否有 `best_evidence_text` |

输出：

```json
{
  "fit": "likely_match",
  "matching_signals": [],
  "conflicts": [],
  "missing_information": [],
  "next_checks": []
}
```

`fit` 取值：

| fit | 条件 |
| --- | --- |
| `likely_match` | 有匹配信号且无明显冲突 |
| `possible_match` | 没有强匹配信号，也没有明显冲突 |
| `needs_manual_review` | 存在年龄、性别或 negative term 冲突 |
| `not_current_option` | candidate 状态是 `abandoned` |

### 7.3 trial_selection 输出

函数：

- `agent/graph/nodes.py`
  - `_build_protocol_trial_selection(...)`

输出字段：

```json
{
  "selected_trial": {},
  "selection_reason": "...",
  "eligibility_assessment": {},
  "trial_status_assessment": {},
  "evidence": [],
  "alternatives": []
}
```

`evidence` 来源：

1. `eligibility_assessment.matching_signals`
2. selected trial 的 `best_evidence_text`
3. selected trial 的 `brief_summary`

`alternatives` 最多保留 3 个非 selected candidates，并解释为什么没有选中。

## 8. 治疗建议生成逻辑

函数：

- `agent/graph/nodes.py`
  - `_build_treatment_recommendations(state, trial_bundle=...)`

它会先检查 calculation result 状态：

```python
completed_results = [item for item in state.calculation_results if item.status == "completed"]
partial_results = [item for item in state.calculation_results if item.status == "partial"]
estimated_results = [item for item in state.calculation_results if item.status == "estimated"]
```

### 8.1 有 completed calculation result

如果有 completed result，生成：

```json
{
  "name": "<artifact.name> guided treatment",
  "strategy": "risk_informed_treatment",
  "source": "protocol_reasoning",
  "status": "manual_review",
  "rationale": "...",
  "linked_calculators": ["..."],
  "linked_trials": ["NCT...", "NCT...", "NCT..."],
  "actions": [
    "Map the risk output to treatment thresholds or protocol branches.",
    "Keep any regimen or trial recommendation evidence-linked and explicitly reviewable."
  ]
}
```

如果 trial bundle 中有非 abandoned trial，会把最多 3 个 NCT ID 挂到 `linked_trials`。

注意：即使有 completed risk result，当前状态仍是 `manual_review`，因为代码没有真正完成具体 regimen/protocol branch 的医学确认。

### 8.2 有 partial calculation result

生成：

```json
{
  "name": "partial calculator result requires parameter completion",
  "strategy": "similar_case_fallback",
  "source": "partial_parameter_gap",
  "status": "similar_case_fallback"
}
```

含义：calculator 只给出 provisional result，治疗/试验路由必须等缺失参数补齐。

### 8.3 有 estimated calculation result

生成：

```json
{
  "name": "similar-case treatment fallback",
  "strategy": "similar_case_fallback",
  "source": "estimated_parameter_gap",
  "status": "similar_case_fallback"
}
```

含义：计算接近可执行，但仍依赖估计参数，需要通过相似病例或人工验证。

### 8.4 只有 calculator_matches

生成：

```json
{
  "name": "similar-case assisted recommendation",
  "strategy": "similar_case_fallback",
  "source": "calculator_candidates_without_execution",
  "status": "similar_case_fallback"
}
```

含义：找到了候选 calculator，但没有可用风险值支撑治疗决策。

### 8.5 没有任何 calculator 信号

生成：

```json
{
  "name": "direct treatment advice",
  "strategy": "direct_advice",
  "source": "no_calculation_signal",
  "status": "advice_only"
}
```

含义：系统只能给出保守、低置信度、需要医生复核的建议路径。

### 8.6 trial candidate review 插入逻辑

函数：

- `agent/graph/nodes.py`
  - `_augment_treatment_recommendations_with_trials(...)`

如果没有 completed calculation result，但 trial 检索返回了 candidates，它会在 recommendations 前面插入：

```json
{
  "name": "trial candidate review",
  "strategy": "trial_candidate_review",
  "source": "trial_retrieval",
  "status": "manual_review",
  "linked_trials": ["NCT...", "NCT...", "NCT..."]
}
```

这表示：

- 本地 trial 检索发现了可能相关的研究。
- 但 calculator 结果不足以支撑直接 trial match。
- 必须人工 review trial eligibility 和 enrollment details。

## 9. TreatmentRecommendation 到 ProtocolRecommendation 的转换

函数：

- `agent/graph/nodes.py`
  - `_to_protocol_recommendations(recommendations)`

转换规则：

| TreatmentRecommendation.status | ProtocolRecommendation.status |
| --- | --- |
| `matched` | `matched` |
| `trial_matched` | `matched` |
| `abandoned` | `insufficient_data` |
| 其他 | `needs_revision` |

因此，当前大多数 recommendation 会变成 `needs_revision`，因为它们的状态通常是：

- `manual_review`
- `similar_case_fallback`
- `advice_only`

这是当前实现的保守设计：没有明确证据链和 eligibility 判断时，不把推荐标记成 matched。

## 10. Reporter 如何检查 protocol

`reporter` 会检查 protocol 是否至少给出一条 treatment recommendation。

相关函数：

- `agent/graph/nodes.py`
  - `_build_report_review(...)`

检查项之一：

```python
{
    "name": "protocol_recommendations_present",
    "passed": has_recommendations,
    "detail": "protocol must return at least one treatment recommendation.",
}
```

这里的 `has_recommendations` 实际检查的是：

```python
has_recommendations = bool(state.treatment_recommendations)
```

如果 protocol 没有生成建议，reporter 会认为本轮失败，并可能把 feedback 返回 orchestrator 触发重跑，最多三轮。

## 11. 调试入口

### 11.1 跑单条病例

推荐先用：

```bash
cd /home/yuanzy/MadAI
uv run python scripts/try_single_case_workflow.py --show-json
```

或者直接传病例：

```bash
uv run python scripts/try_single_case_workflow.py \
  --case-text "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA; which stroke risk calculator should be used and what would it compute?" \
  --mode question \
  --show-json
```

重点看输出 JSON 中：

- `structured_case`
- `calculation_bundle`
- `trial_retrieval_bundle`
- `treatment_bundle`
- `execution_trace`

### 11.2 只看 protocol prompt context

`protocol_node` 会调用 `_record_agent_prompt(...)`，把 protocol context 记录到 final output。重点检查：

```json
{
  "request": "...",
  "structured_case": {},
  "calculation_results": [],
  "calculator_matches": [],
  "trial_retrieval_bundle": {}
}
```

如果 protocol 输出异常，先看这里是否有足够输入。

### 11.3 检查 trial query profile

可以直接在 Python 中调用：

```python
from agent.tools.trial_vector_retrieval_tools import build_protocol_trial_query_profile

profile = build_protocol_trial_query_profile({
    "raw_text": "78-year-old male with atrial fibrillation and prior TIA, not currently on warfarin.",
    "case_summary": "Older male with atrial fibrillation and prior TIA.",
    "problem_list": ["stroke prevention in atrial fibrillation"],
    "known_facts": ["hypertension", "prior TIA"],
})

print(profile["query_text"])
print(profile["payload_filters"])
```

重点看：

- `trial_condition_terms`
- `trial_intervention_terms`
- `trial_intent_terms`
- `patient_negative_terms`
- `payload_filters`
- `derivation_notes`

### 11.4 检查 trial retriever 是否有 vector 后端

如果 `backend_used` 总是 `bm25`，通常是 vector retriever 不可用或初始化失败。

重点检查：

- `trial_retrieval_bundle["backend_used"]`
- `trial_retrieval_bundle["available_backends"]`
- Qdrant/FAISS 配置
- `outputs/trial_vector_kb_full` 是否存在
- `trial_record.jsonl`
- `trial_chunk.jsonl`
- `manifest.json`

### 11.5 注入自定义 trial retriever 做单元测试

`protocol` 支持通过 `state.tool_registry["trial_retriever"]` 注入 retriever。测试里可以构造 fake retriever：

```python
class FakeTrialRetriever:
    def retrieve_from_structured_case(self, structured_case, **kwargs):
        return {
            "query_text": "fake query",
            "query_profile": {
                "trial_condition_terms": ["atrial fibrillation"],
                "trial_intervention_terms": ["anticoagulation"],
                "trial_intent_terms": ["stroke prevention"],
                "patient_negative_terms": [],
            },
            "backend_used": "bm25",
            "available_backends": ["bm25"],
            "candidate_ranking": [
                {
                    "nct_id": "NCT00000001",
                    "title": "Fake AF stroke prevention trial",
                    "status": "trial_matched",
                    "overall_status": "Recruiting",
                    "enrollment_open": True,
                    "brief_summary": "Fake summary.",
                    "best_evidence_text": "Includes atrial fibrillation patients.",
                    "conditions": ["Atrial Fibrillation"],
                    "interventions": ["Anticoagulation"],
                    "primary_purpose": "Prevention",
                    "actions": ["Review detailed inclusion and exclusion criteria."],
                }
            ],
        }
```

这样可以绕开真实 trial KB，直接测试 `protocol_node` 的 selection 和 recommendation 分支。

## 12. 当前实现边界

当前 `protocol` 已经能完成 trial-level candidate matching，但还没有做到真正的 criterion-level eligibility assessment。

已有能力：

1. 基于 structured case 构造 trial query profile。
2. 使用 BM25/vector/hybrid 检索本地 XML-derived trial chunk KB。
3. 聚合 chunk 到 trial-level candidates。
4. 映射 trial lifecycle status。
5. 选择 top non-abandoned trial。
6. 做粗粒度年龄、性别、condition、intervention、intent 和 negative term 检查。
7. 基于 calculator 结果生成保守 treatment recommendations。
8. 输出 trace，方便审查和调试。

缺失能力：

1. 没有把 inclusion/exclusion criteria 拆成原子 criterion。
2. 没有逐条 criterion 判断 `met / not_met / unknown / not_applicable`。
3. 没有从 patient note 中检索每条 criterion 对应 evidence span。
4. 没有 deterministic trial eligibility aggregator。
5. 没有根据 unknown criteria 生成最小缺失问题列表。
6. 没有真正把具体 regimen/protocol branch 确认为 `matched`。

因此，当前 protocol 的输出应被理解为：

- trial retrieval support
- treatment decision scaffold
- manual review recommendation
- similar-case fallback
- conservative advice path

而不是最终、确定的临床试验入排判断。

## 13. 常见问题

### 13.1 为什么有 trial candidate，但 recommendation 还是 manual_review？

因为当前代码只做 trial-level 粗匹配，没有逐条 inclusion/exclusion criteria 判断。即使 trial 状态是 `trial_matched`，也必须人工 review eligibility 和 enrollment details。

### 13.2 为什么 protocol_recommendations 经常是 needs_revision？

`_to_protocol_recommendations` 只有在 treatment recommendation status 是 `matched` 或 `trial_matched` 时才给 `matched`。当前大多数 recommendation 是 `manual_review`、`similar_case_fallback` 或 `advice_only`，所以会转换成 `needs_revision`。

### 13.3 为什么没有 structured_case 时 trial 检索为空？

`_retrieve_trial_candidates` 明确依赖 `state.structured_case_json`。没有 structured case 时，它返回空 `candidate_ranking`，不会直接拿原始 request 检索 trial。

### 13.4 为什么 vector 检索没有参与？

可能原因：

1. retriever backend 被设置为 `bm25`。
2. vector retriever 初始化失败。
3. Qdrant/FAISS 数据不存在。
4. vector 查询运行时报错后 fallback 到 BM25。

先检查 `trial_retrieval_bundle["backend_used"]` 和 `available_backends`。

### 13.5 `数据/治疗方案` 和 XML chunk KB 是什么关系？

当前 protocol 默认走 XML-derived trial chunk KB。`数据/治疗方案` 下的 department payloads 和 XML 文件仍然是项目中的治疗方案/试验数据资源，但 protocol 当前默认 retriever 是 `create_trial_chunk_retrieval_tool(...)`，它优先解析 `outputs/trial_vector_kb_full` 等 trial vector KB 输出目录。

旧版或备用检索逻辑可参考：

- `agent/tools/trial_retrieval_tools.py`

当前主路径参考：

- `agent/tools/trial_vector_retrieval_tools.py`
- `agent/retrieval/trial_chunks.py`

## 14. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `agent/graph/builder.py` | graph 主链路和节点注册 |
| `agent/graph/nodes.py` | `protocol_node`、trial selection、treatment recommendation |
| `agent/graph/types.py` | `GraphState`、`TreatmentRecommendation`、`ProtocolRecommendation` |
| `agent/prompt/protocol.md` | protocol 行为契约 |
| `agent/tools/trial_vector_retrieval_tools.py` | protocol trial query profile、BM25/vector/hybrid 检索、候选 enrichment |
| `agent/retrieval/trial_chunks.py` | trial chunk catalog 和 trial chunk document |
| `agent/retrieval/qdrant/trial_chunk.py` | Qdrant trial chunk 检索运行时 |
| `agent/retrieval/qdrant/trial_chunk_sync.py` | trial chunk 同步到 Qdrant |
| `agent/trial_vector_kb.py` | XML trial KB 构建逻辑 |
| `tests/test_trial_vector_retrieval_tools.py` | trial vector retrieval 单元测试 |
| `tests/test_trial_retrieval_tools.py` | protocol node 与 trial retrieval 集成测试 |
| `tests/test_orchestrator_state.py` | graph state 与 protocol/reporter 状态测试 |

## 15. 当前 protocol 的一句话总结

当前 MedAI 的 `protocol` 是一个保守的规则驱动 graph 节点：它接收上游 `structured_case` 和 calculator 结果，把病例解析成 trial query profile，用 BM25/vector/hybrid 两阶段检索本地 XML-derived trial chunk KB，选择 top non-abandoned trial 做粗粒度适配评估，然后根据 calculator 结果生成 treatment recommendation、trial candidate review、similar-case fallback 或 direct advice，并把所有结果结构化写入 `trial_retrieval_bundle`、`treatment_bundle` 和 `protocol_recommendations`。
