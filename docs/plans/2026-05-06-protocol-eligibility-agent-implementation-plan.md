# Protocol 试验资格评估 Agent 实现计划

## 目标

将当前 MedAI 的 `protocol` 节点，从“试验级候选匹配”升级为“基于证据、逐条标准判断”的临床试验资格评估。

当前 `protocol` 阶段已经可以基于由 ClinicalTrials.gov XML 构建的本地知识库进行试验检索，并返回 `trial_retrieval_bundle`、`trial_selection` 和 `treatment_bundle`。下一步是在试验检索和治疗推荐之间插入一条试验资格评估流水线：

```text
clinical_assisstment
  -> calculator
  -> protocol:
       试验候选检索
       方案入排标准解析器
       患者证据检索器
       单条标准判断 Agent
       确定性的试验资格聚合器
       缺失数据问题生成器
       治疗推荐
  -> reporter
```

本计划保持 `protocol` 作为试验和方案推理的责任边界。`clinical_assisstment` 继续负责病例标准化和计算器调度；`reporter` 只负责检查并编译最终面向医生的报告。

## 当前基线

当前相关实现：

- `agent/graph/nodes.py`
  - `_retrieve_trial_candidates(...)`
  - `_build_protocol_trial_selection(...)`
  - `_build_treatment_recommendations(...)`
  - `protocol_node(...)`
- `agent/tools/trial_vector_retrieval_tools.py`
  - `build_protocol_trial_query_profile(...)`
  - `TrialChunkRetrievalTool.retrieve_from_structured_case(...)`
  - 分块到试验级别的聚合与重排
- `agent/trial_vector_kb.py`
  - 将 XML 解析为 `trial_record`
  - 基于概览、描述、纳入标准、排除标准、结局、研究臂和干预措施构建分块

当前行为：

1. 从 `structured_case` 构建试验检索查询画像。
2. 使用 BM25、向量检索或混合检索召回试验分块。
3. 将分块聚合回试验级别的 `candidate_ranking`。
4. 选择排名靠前且非废弃状态的试验候选。
5. 执行粗粒度的试验级适配评估。
6. 主要基于计算器结果可用性生成治疗推荐。

当前缺失的能力：

1. 没有将纳入和排除标准解析为原子标准。
2. 没有从患者事实中检索证据片段。
3. 没有针对每条标准输出 `met`、`not_met`、`unknown`、`not_applicable` 等标签。
4. 没有从逐条标准标签确定性聚合到试验级资格状态。
5. 没有生成最小必要的缺失数据追问。

## 总体实现框架

这部分不要实现成一个“大 prompt”或一个“大函数”。它应实现为一条可测试的 protocol eligibility pipeline。每一层只解决一个问题，并把结果交给下一层。

主流程：

```text
trial_retrieval_bundle
  -> 选择 top N trial candidates
  -> 获取每个 trial 的完整 trial_record
  -> 解析 trial 入排标准
  -> 构建患者证据索引
  -> 为每条标准检索患者证据
  -> 判断每条标准 label
  -> 聚合 trial 级 aggregate_status
  -> 生成 missing_questions
  -> 写入 eligibility_assessment_bundle
  -> 更新 trial_selection 和 treatment_bundle
```

第一版的实现原则：

1. **先结构化，再智能化**：第一版先用规则和模板跑通结构，不依赖 LLM。
2. **能判断才判断**：没有明确证据时输出 `unknown`，不要猜测。
3. **证据优先**：除 `unknown` 外，每个关键判断都应尽量带 `evidence_spans`。
4. **聚合确定性**：trial 级 `aggregate_status` 由固定规则产生，不由 LLM 自由总结。
5. **向后兼容**：保留现有 `trial_retrieval_bundle`、`trial_selection`、`recommendations`、`trial_candidates` 字段。
6. **先小后大**：默认只评估排名前 3 个 trial，先覆盖年龄、性别、诊断、ECOG、妊娠、CNS 转移和常见实验室标准。

推荐模块边界：

```text
agent/protocol/
  types.py                  protocol eligibility 数据类
  criteria_parser.py        trial 原文入排标准 -> 原子标准
  evidence_retriever.py     患者病例事实 -> evidence spans
  criterion_judge.py        单条标准 + 证据 -> met/not_met/unknown
  eligibility_aggregator.py 多条标准 -> trial 级 aggregate_status
  missing_data.py           unknown 标准 -> 缺失数据问题
  pipeline.py               串联以上模块，产出 eligibility_assessment_bundle
```

每个模块的责任边界：

| 模块 | 输入 | 输出 | 不负责 |
| --- | --- | --- | --- |
| `criteria_parser.py` | `trial_record` | `list[EligibilityCriterion]` | 不判断患者是否符合 |
| `evidence_retriever.py` | `structured_case`、计算器结果、单条标准 | `list[PatientEvidenceSpan]` | 不下资格结论 |
| `criterion_judge.py` | 单条标准和证据 | `CriterionAssessment` | 不做 trial 级总评 |
| `eligibility_aggregator.py` | trial 状态和多条标准判断 | `TrialEligibilityAssessment` | 不重新解析证据 |
| `missing_data.py` | `unknown` 标准 | `list[MissingDataQuestion]` | 不生成治疗建议 |
| `pipeline.py` | graph state 和 trial bundle | `eligibility_assessment_bundle` | 不改变检索逻辑 |

第一版只需要做到“保守可用”：

```text
能拆出来的标准 -> 拆
能找到的证据 -> 引用
能确定的判断 -> met/not_met
不能确定的判断 -> unknown
unknown 里重要的标准 -> 生成 missing_questions
```

不在第一版解决：

1. 复杂 AND/OR 组合标准的完整逻辑推理。
2. 复杂治疗线数、耐药、复发时间窗的医学推断。
3. 全面的实验室单位换算。
4. 全面的基因组变异归一化。
5. LLM 直接生成最终资格 JSON。

对齐检查点：

1. **和 trial 原文对齐**：每条 `criteria.raw_text` 必须来自 trial 原始入排标准或 trial 明确结构化字段。
2. **和患者证据对齐**：非 `unknown` 判断应尽量引用病例证据片段。
3. **和 label 语义对齐**：`inclusion.met` 是好事；`exclusion.not_met` 是好事；`exclusion.met` 是阻断风险。
4. **和 aggregate 规则对齐**：总评只由聚合器根据逐条标签和试验状态产生。
5. **和输出契约对齐**：`final_output` 和 `treatment_bundle` 都必须暴露 `eligibility_assessment_bundle`。

建议按以下顺序进入枝叶实现：

1. **数据模型枝**：先定义 `EligibilityCriterion`、`PatientEvidenceSpan`、`CriterionAssessment`、`TrialEligibilityAssessment`、`MissingDataQuestion`。这一层只解决字段统一问题。
2. **标准解析枝**：实现 `criteria_parser.py`，先能稳定拆 inclusion/exclusion，并识别年龄、性别、ECOG、常见 lab、CNS 转移、妊娠等高频标准。
3. **患者证据枝**：实现 `evidence_retriever.py`，先把 `raw_text`、`case_summary`、`problem_list`、`known_facts`、`structured_inputs` 和计算器结果转成可检索证据片段。
4. **单条判断枝**：实现 `criterion_judge.py`，只根据一条标准和证据输出 `met`、`not_met`、`unknown`、`not_applicable`。
5. **聚合枝**：实现 `eligibility_aggregator.py`，把逐条判断汇总为 trial 级 `aggregate_status`、`blocking_criteria` 和 `unknown_criteria`。
6. **缺失问题枝**：实现 `missing_data.py`，把重要 `unknown` 标准转成医生能补充的数据问题。
7. **流水线枝**：实现 `pipeline.py`，把 top N trial、完整 trial_record、标准解析、证据检索、单条判断、聚合和缺失问题串起来。
8. **graph 接入枝**：把 pipeline 接入 `protocol_node`，写入 `final_output["eligibility_assessment_bundle"]` 和 `treatment_bundle["eligibility_assessment_bundle"]`。
9. **选择与推荐枝**：让 `_build_protocol_trial_selection(...)` 和 `_build_treatment_recommendations(...)` 感知资格评估结果。

第一批提交只做 1 到 8。第 9 步可以第二批提交做，因为它会改变试验选择策略，影响面更大。

## Protocol 内部多 Agent 图结构设计

`protocol` 节点后续不应继续膨胀成一个大函数。更合理的形态是：外层 workflow 仍然只有一个 `protocol_node`，但 `protocol_node` 内部运行一个 protocol subgraph。

这个 subgraph 参考临床试验匹配、医学 RAG 和多 agent 临床决策系统的常见分工：先由检索节点抓取候选临床试验和相关医学知识，再由解析/判断节点做结构化抽取和证据比对，最后由确定性聚合节点给出可审计结论。

### 相关设计调研

当前公开设计里有几类模式值得借鉴，但不能照搬。

1. **TrialGPT 式三段链路**  
   TrialGPT 将患者-试验匹配拆成 retrieval、criterion-level matching、trial-level ranking 三个模块。这个设计和本项目最接近：先从大规模 trial 集合过滤候选，再对患者和 trial criteria 做逐条匹配，最后聚合为 trial 级排序。  
   对本项目的启发：`trial_coarse_retrieval_agent`、`trial_fine_retrieval_agent`、`criterion_judgment_agent`、`trial_eligibility_aggregation_agent` 应分开，不应让一个 agent 同时完成检索、判断和排序。

2. **MedAgents 式多学科会诊**  
   MedAgents 使用多学科专家角色进行多轮分析、总结、讨论和最终决策。它适合复杂医学推理，但成本和不确定性较高。  
   对本项目的启发：复杂 criterion 或治疗推荐可以引入“专家角色”，例如 oncology、cardiology、genomics、trial-methodology；但基础检索、section 切分和聚合不应交给会诊式 LLM 自由讨论。

3. **MDAgents 式自适应协作**  
   MDAgents 根据任务复杂度决定使用 solo 还是 group collaboration，并强调 moderator review 和外部医学知识。  
   对本项目的启发：protocol subgraph 应先做复杂度分流。简单年龄、性别、ECOG、明确否定事实走规则；复杂治疗线数、基因组、模糊时间窗才进入 LLM agent。`medical_knowledge_retrieval_agent` 应作为复杂判断的外部知识输入，而不是可选装饰。

4. **PRISM / OncoLLM 式真实 EHR 匹配**  
   PRISM 强调真实世界 EHR 很长、非结构化且复杂，trial matching 需要直接处理 inclusion/exclusion free text，并提供标准级解释。  
   对本项目的启发：患者证据检索不能只看 `case_summary`，必须索引 `raw_text`、结构化输入、known facts、calculator results 和后续 EHR 文档；每条 criterion 的判断要保留 evidence spans。

5. **AgentClinic 式顺序决策和工具使用**  
   AgentClinic 将临床任务视为不完整信息下的顺序决策，强调工具调用、检查请求和多模态信息获取。  
   对本项目的启发：`missing_data.py` 和 `missing_questions` 不是附属字段，而是 protocol subgraph 的正式输出。当证据不足时，系统应该生成下一步需要补充的数据，而不是猜 eligibility。

6. **MedAgentBoard 式审慎使用多 agent**  
   MedAgentBoard 的结论提醒：多 agent 在某些临床工作流能提升完整性，但并不总是优于单 LLM 或传统模型，且有额外复杂度。  
   对本项目的启发：本方案采用“工具/规则节点为骨架，LLM agent 只处理复杂自然语言”的混合架构。每个 agent 必须有明确输入、输出和测试，不因为名字是 multi-agent 就把所有步骤都做成 LLM 调用。

参考来源：

- MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning  
  https://arxiv.org/abs/2311.10537
- MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making  
  https://arxiv.org/abs/2404.15155
- TrialGPT: Matching Patients to Clinical Trials with Large Language Models  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10418514/
- PRISM: Patient Records Interpretation for Semantic clinical trial Matching system using large language models  
  https://www.nature.com/articles/s41746-024-01274-7
- AgentClinic: a multimodal benchmark for tool-using clinical AI agents  
  https://agentclinic.github.io/
- MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks  
  https://arxiv.org/abs/2505.12371

核心原则：

1. **多节点不等于全 LLM**：检索、过滤、聚合、状态判断优先用规则和工具；LLM 只用于复杂标准解析和单条标准判断的辅助。
2. **trial 和医学知识双检索**：protocol 不只需要抓 clinical trials，还需要抓相关医学知识区，用于解释标准、补足医学背景和生成可审计依据。
3. **逐层收敛**：先粗召回 trial，再精召回 trial chunks，再选择少量 trial 做资格评估。
4. **单条标准判断**：LLM 如参与，必须一次只处理一条 criterion，并输出严格 JSON。
5. **确定性聚合**：trial 级 `aggregate_status` 由规则聚合器生成，不由 LLM 自由总结。
6. **外层状态保持简洁**：protocol subgraph 使用自己的内部 state，最后只把稳定 bundle 写回外层 `GraphState`。

### 子图主流程

建议的 protocol subgraph：

```text
protocol_node
  -> run_protocol_subgraph

protocol_subgraph:
  1. case_protocol_profile_agent
  2. trial_coarse_retrieval_agent
  3. trial_fine_retrieval_agent
  4. medical_knowledge_retrieval_agent
  5. eligibility_candidate_selector_agent
  6. trial_record_resolver_agent
  7. eligibility_section_parser_agent
  8. criteria_extraction_agent
  9. patient_evidence_retrieval_agent
 10. criterion_judgment_agent
 11. trial_eligibility_aggregation_agent
 12. protocol_trial_selection_agent
 13. treatment_recommendation_agent
```

可以先实现为普通 Python 节点和工具函数；后续再把其中适合 LLM 的节点升级为真正 agent。

### 节点职责

| 节点 | 类型 | 职责 | 输出 |
| --- | --- | --- | --- |
| `case_protocol_profile_agent` | 规则为主，可 LLM 辅助 | 从 `structured_case`、计算器结果和科室标签生成 trial 检索画像 | `query_profile` |
| `trial_coarse_retrieval_agent` | 工具节点 | 从大规模 trial KB 粗召回候选 trial IDs | `coarse_candidate_ids`、`coarse_candidate_ranking` |
| `trial_fine_retrieval_agent` | 工具节点 | 在粗召回候选内做 BM25/vector chunk 精召回和 trial-level rerank | `candidate_ranking`、`bm25_top5`、`vector_top5` |
| `medical_knowledge_retrieval_agent` | 工具节点，可 RAG | 检索疾病、治疗、指南、药物、生物标志物等医学知识 | `medical_knowledge_bundle` |
| `eligibility_candidate_selector_agent` | 规则节点 | 从 `candidate_ranking` 中选择进入逐条资格评估的 top trial | `assessed_trial_candidates` |
| `trial_record_resolver_agent` | 工具节点 | 根据 NCT ID 获取完整 `trial_record` | `trial_records` |
| `eligibility_section_parser_agent` | 规则节点 | 稳定定位 inclusion / exclusion section | `eligibility_sections` |
| `criteria_extraction_agent` | 规则优先，LLM 辅助 | 将入排标准原文解析成原子 criteria 和结构化参数 | `criteria_by_trial` |
| `patient_evidence_retrieval_agent` | 工具节点 | 从病例、结构化输入、计算器结果中检索患者证据片段 | `patient_evidence_index`、`evidence_by_criterion` |
| `criterion_judgment_agent` | 规则优先，LLM 辅助 | 对单条 criterion 判断 `met`、`not_met`、`unknown`、`not_applicable` | `criterion_assessments_by_trial` |
| `trial_eligibility_aggregation_agent` | 确定性规则节点 | 聚合逐条标准为 trial 级资格状态 | `eligibility_assessment_bundle` |
| `protocol_trial_selection_agent` | 规则节点，可轻量 LLM 辅助 | 根据检索、资格状态和 trial 状态选择推荐 trial | `trial_selection` |
| `treatment_recommendation_agent` | 规则 + 结构化生成 | 生成治疗/试验推荐，引用 calculator、trial 和医学知识证据 | `treatment_bundle` |

### Trial 和医学知识双通道

protocol 后续应同时维护两条检索通道：

```text
trial channel:
  structured_case
    -> trial query profile
    -> trial coarse retrieval
    -> trial fine retrieval
    -> trial candidate ranking
    -> trial eligibility assessment

medical knowledge channel:
  structured_case + candidate trials + calculator results
    -> disease / treatment / biomarker / guideline query profile
    -> medical knowledge retrieval
    -> medical_knowledge_bundle
    -> support criteria interpretation and treatment recommendation
```

`medical_knowledge_bundle` 建议结构：

```json
{
  "schema_version": 1,
  "queries": [],
  "retrieved_items": [
    {
      "source": "guideline | pubmed | local_kb | calculator | drug_label",
      "title": "",
      "text": "",
      "url": "",
      "score": 0.0,
      "linked_concepts": [],
      "used_for": ["criteria_interpretation", "treatment_recommendation"]
    }
  ],
  "knowledge_gaps": []
}
```

该 bundle 不直接决定患者是否符合 trial，但可以用于：

1. 解释复杂医学术语，例如 ECOG、CNS metastases、prior line therapy。
2. 辅助 `criteria_extraction_agent` 理解复杂标准。
3. 辅助 `treatment_recommendation_agent` 生成有来源的建议。
4. 支持 reporter 输出更完整的医学背景。

### 内部状态

建议新增 protocol 内部 state，避免把外层 `GraphState` 塞得过满：

```python
@dataclass(slots=True)
class ProtocolGraphState:
    request: str = ""
    structured_case: dict[str, Any] = field(default_factory=dict)
    calculation_results: list[Any] = field(default_factory=list)
    calculator_matches: list[Any] = field(default_factory=list)
    department_tags: list[str] = field(default_factory=list)

    query_profile: dict[str, Any] = field(default_factory=dict)
    trial_retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    medical_knowledge_bundle: dict[str, Any] = field(default_factory=dict)

    coarse_candidate_ids: list[str] = field(default_factory=list)
    fine_candidates: list[dict[str, Any]] = field(default_factory=list)
    assessed_trial_candidates: list[dict[str, Any]] = field(default_factory=list)

    trial_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    eligibility_sections: dict[str, list[Any]] = field(default_factory=dict)
    criteria_by_trial: dict[str, list[EligibilityCriterion]] = field(default_factory=dict)

    patient_evidence_index: list[PatientEvidenceSpan] = field(default_factory=list)
    criterion_assessments_by_trial: dict[str, list[CriterionAssessment]] = field(default_factory=dict)

    eligibility_assessment_bundle: dict[str, Any] = field(default_factory=dict)
    trial_selection: dict[str, Any] = field(default_factory=dict)
    treatment_bundle: dict[str, Any] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
```

外层 `protocol_node` 只负责调用子图并写回稳定结果：

```python
def protocol_node(state: GraphState) -> GraphState:
    protocol_result = run_protocol_subgraph(
        request=state.request,
        structured_case=dict(state.structured_case_json or {}),
        calculation_results=list(state.calculation_results or []),
        calculator_matches=list(state.calculator_matches or []),
        department_tags=list(state.department_tags),
        trial_retriever=_resolve_trial_retriever(state),
        config=ProtocolGraphConfig.from_env(),
    )

    state.trial_retrieval_bundle = protocol_result.trial_retrieval_bundle
    state.treatment_bundle = protocol_result.treatment_bundle
    state.final_output["trial_retrieval_bundle"] = protocol_result.trial_retrieval_bundle
    state.final_output["eligibility_assessment_bundle"] = protocol_result.eligibility_assessment_bundle
    state.final_output["medical_knowledge_bundle"] = protocol_result.medical_knowledge_bundle
    state.final_output["treatment_bundle"] = protocol_result.treatment_bundle
    return state
```

### 建议目录

第一版可以保持文件少一些：

```text
agent/protocol/
  graph.py
  state.py
  config.py
  nodes.py
  types.py
  criteria_parser.py
  evidence_retriever.py
  criterion_judge.py
  eligibility_aggregator.py
  missing_data.py
  pipeline.py
```

节点变多后再拆为：

```text
agent/protocol/nodes/
  query_profile.py
  retrieval.py
  knowledge_retrieval.py
  candidate_selector.py
  record_resolver.py
  section_parser.py
  criteria_extraction.py
  evidence_retrieval.py
  criterion_judgment.py
  aggregation.py
  trial_selection.py
  recommendations.py
```

### 分阶段落地

第一阶段：结构迁移，不改变行为。

1. 新增 `ProtocolGraphState` 和 `ProtocolGraphConfig`。
2. 新增 `run_protocol_subgraph(...)`。
3. 内部仍复用现有 `_retrieve_trial_candidates(...)`、`assess_trial_eligibility_candidates(...)`、`_build_protocol_trial_selection(...)` 和 `_build_treatment_recommendations(...)`。
4. 保证现有测试不变。

第二阶段：补 trial 候选选择器。

1. 新增 `eligibility_candidate_selector_agent`。
2. 从 `fine_top_k` 候选中选择 `eligibility_assessment_limit` 个 trial。
3. 优先开放招募 trial，降级 completed / active-not-recruiting，排除 abandoned / withdrawn / suspended。
4. 使用 `score`、`coverage_chunk_types`、`matched_condition_terms`、`eligibility_signals`、`must_not_conflicts` 辅助排序。

第三阶段：补医学知识通道。

1. 新增 `medical_knowledge_retrieval_agent`。
2. 先复用本地 structured retrieval / PubMed / guideline KB 的统一接口。
3. 输出 `medical_knowledge_bundle`。
4. 在 treatment recommendation 和 reporter 中引用该 bundle。

第四阶段：引入 LLM 辅助 agent。

1. `criteria_extraction_agent` 对规则无法解析的复杂 criterion 调 LLM。
2. `criterion_judgment_agent` 对证据和标准复杂、规则无法处理的单条 criterion 调 LLM。
3. 所有 LLM 输出必须严格 JSON schema 校验。
4. 任何证据不足的情况仍然输出 `unknown`。

## 当前优化方案

大规模 trial 检索、inclusion/exclusion 稳定定位和资格比对链路是当前需要直接解决的问题，单独维护在：

- `docs/plans/2026-05-07-trial-retrieval-eligibility-optimization-plan.md`

## 目标输出契约

在 `treatment_bundle` 和 `final_output` 中新增顶层资格评估包：

```json
{
  "eligibility_assessment_bundle": {
    "schema_version": 1,
    "assessed_trial_count": 0,
    "assessed_trials": [
      {
        "nct_id": "NCT...",
        "title": "...",
        "overall_status": "Recruiting",
        "enrollment_open": true,
        "aggregate_status": "likely_eligible",
        "aggregate_reason": "...",
        "criteria": [
          {
            "criterion_id": "NCT...::inclusion::001",
            "type": "inclusion",
            "raw_text": "Age >= 18 years",
            "condition": "age",
            "operator": ">=",
            "value": "18 years",
            "time_window": "",
            "required_evidence_type": "demographic",
            "negation": false,
            "label": "met",
            "confidence": 0.92,
            "evidence_spans": [
              {
                "source": "raw_text",
                "text": "62-year-old male",
                "start": 0,
                "end": 16,
                "score": 1.0
              }
            ],
            "rationale": "患者年龄高于方案要求的最低年龄。",
            "missing_data": []
          }
        ],
        "blocking_criteria": [],
        "unknown_criteria": [],
        "missing_questions": []
      }
    ]
  }
}
```

`treatment_bundle` 应包含：

```json
{
  "recommendations": [],
  "trial_candidates": [],
  "trial_candidate_ids": [],
  "trial_selection": {},
  "eligibility_assessment_bundle": {}
}
```

`final_output` 应包含：

```json
{
  "trial_retrieval_bundle": {},
  "eligibility_assessment_bundle": {},
  "treatment_bundle": {}
}
```

## 模块 1：方案标准解析器

### 职责

将每个入选试验中的 `eligibility_inclusion_text` 和 `eligibility_exclusion_text` 解析为原子资格标准。

### 输入

```python
trial_record: dict[str, Any]
```

必需字段：

- `nct_id`
- `display_title`
- `eligibility_inclusion_text`
- `eligibility_exclusion_text`
- `eligibility_text`
- `gender`
- `minimum_age`
- `maximum_age`

### 输出

```python
list[EligibilityCriterion]
```

建议的数据类：

```python
@dataclass(slots=True)
class EligibilityCriterion:
    criterion_id: str
    nct_id: str
    type: Literal["inclusion", "exclusion"]
    raw_text: str
    condition: str = ""
    operator: str = ""
    value: str = ""
    time_window: str = ""
    required_evidence_type: str = "clinical_fact"
    negation: bool = False
    parse_method: str = "rule"
```

### 解析策略

第一阶段应保持确定性和简洁性：

1. 按项目符号、编号、换行符拆分标准；仅在安全场景下按分号拆分。
2. 移除 `Inclusion Criteria:`、`Exclusion Criteria:` 等小节标题。
3. 规范化空白字符，同时保留原始文本语义。
4. 分配稳定 ID：

```text
{nct_id}::inclusion::001
{nct_id}::exclusion::001
```

5. 为常见模式添加轻量字段抽取：
   - 年龄：`>= 18 years`、`18 Years and older`
   - 性别：`male`、`female`、`all`
   - ECOG / Karnofsky
   - 实验室阈值：ANC、platelet、hemoglobin、bilirubin、AST、ALT、creatinine clearance
   - 疾病诊断
   - 既往治疗
   - 妊娠
   - CNS 转移
   - 生物标志物和基因组标志物

第二阶段可仅对规则无法解析的标准加入 LLM 辅助解析。

### 文件位置

新增：

- `agent/protocol/__init__.py`
- `agent/protocol/criteria_parser.py`

测试：

- `tests/test_protocol_criteria_parser.py`

## 模块 2：患者证据检索器

### 职责

从病例事实构建患者证据索引，并为每条解析后的标准检索证据片段。

该模块不应只搜索 `case_summary`，而应索引所有可用患者证据：

- `structured_case.raw_text`
- `structured_case.case_summary`
- `structured_case.problem_list`
- `structured_case.known_facts`
- `structured_case.structured_inputs`
- `calculation_results`
- `calculator_matches`
- 后续扩展：实验室检查、用药、诊断、病理、基因组、影像

### 输出

```python
list[PatientEvidenceSpan]
```

建议的数据类：

```python
@dataclass(slots=True)
class PatientEvidenceSpan:
    source: str
    text: str
    start: int | None = None
    end: int | None = None
    score: float = 0.0
    normalized_concept: str = ""
    value: str = ""
    unit: str = ""
    observed_time: str = ""
```

### 检索策略

第一阶段：

1. 从每个患者信息来源创建证据文档。
2. 将长文本切分为句子级或分句级证据片段。
3. 使用 BM25 将标准文本和证据片段进行匹配。
4. 增加基于规则的直接抽取：
   - 从结构化输入和原始文本抽取年龄、性别
   - 从已知事实抽取被否定的疾病
   - 抽取计算器结果名称、数值和关联计算器

第二阶段：

1. 如有需要，对证据片段增加向量检索。
2. 增加医学同义词归一化。
3. 增加实验室单位归一化。

### 文件位置

新增：

- `agent/protocol/evidence_retriever.py`

测试：

- `tests/test_protocol_evidence_retriever.py`

## 模块 3：单条标准判断 Agent 与确定性聚合器

### 职责

根据检索到的患者证据判断每条原子标准，并将逐条标准决策聚合为试验级资格状态。

启用 LLM 时，LLM 应一次只判断一条标准，并且必须返回受约束的 JSON 对象。聚合器必须保持确定性。

### 标准标签

允许的标签：

- `met`
- `not_met`
- `unknown`
- `not_applicable`

### 单条标准判断输出

```python
@dataclass(slots=True)
class CriterionAssessment:
    criterion_id: str
    nct_id: str
    type: Literal["inclusion", "exclusion"]
    label: Literal["met", "not_met", "unknown", "not_applicable"]
    confidence: float
    evidence_spans: list[PatientEvidenceSpan]
    rationale: str
    missing_data: list[str]
    judge_method: str = "rule"
```

### 规则基线

在使用 LLM 之前，先实现确定性基线：

1. 年龄标准：
   - 满足纳入标准的最低年龄要求 -> `met`
   - 年龄低于最低要求 -> `not_met`
   - 年龄缺失 -> `unknown`
2. 性别标准：
   - 性别兼容 -> `met`
   - 性别不兼容 -> `not_met`
   - 性别缺失 -> `unknown`
3. 明确的诊断或干预阳性证据：
   - 找到匹配证据 -> 对纳入标准标记为 `met`
   - 找到匹配证据 -> 对排除标准标记为 `met`，后续会阻断该试验
4. 明确否定的疾病：
   - 对应疾病的排除标准 -> `not_met`
   - 要求该疾病存在的纳入标准 -> `not_met`
5. 无证据 -> `unknown`

### LLM 判断契约

LLM 判断器应接收：

- 单条标准
- 排名前列的证据片段
- 结构化患者事实
- 与该标准相关的计算器结果

必须返回：

```json
{
  "label": "met | not_met | unknown | not_applicable",
  "confidence": 0.0,
  "evidence_span_ids": [],
  "rationale": "",
  "missing_data": []
}
```

该模块不得生成自由文本形式的最终推荐。

### 确定性试验级聚合

建议的聚合状态：

- `likely_eligible`
- `ineligible`
- `needs_data`
- `evidence_support`
- `not_current_option`

聚合规则：

1. 如果试验生命周期状态为 abandoned、withdrawn 或 suspended：
   - `aggregate_status = "not_current_option"`
2. 如果试验已完成或处于 active-not-recruiting：
   - 保留资格评估结果，但将当前用途解释为 `evidence_support`
3. 如果任一排除标准为 `met`：
   - `aggregate_status = "ineligible"`
4. 如果任一纳入标准为 `not_met`：
   - `aggregate_status = "ineligible"`
5. 如果任一纳入标准为 `unknown`，或任一高风险排除标准为 `unknown`：
   - `aggregate_status = "needs_data"`
6. 如果所有纳入标准均为 `met` 或 `not_applicable`，且没有排除标准为 `met`：
   - `aggregate_status = "likely_eligible"`

### 文件位置

新增：

- `agent/protocol/criterion_judge.py`
- `agent/protocol/eligibility_aggregator.py`

测试：

- `tests/test_protocol_criterion_judge.py`
- `tests/test_protocol_eligibility_aggregator.py`

## 模块 4：缺失数据问题生成器

### 职责

将 `unknown` 标准转换为最小集合的、临床上有用的追问问题。

### 输入

```python
assessments: list[CriterionAssessment]
```

### 输出

```python
list[MissingDataQuestion]
```

建议的数据类：

```python
@dataclass(slots=True)
class MissingDataQuestion:
    question_id: str
    priority: Literal["high", "medium", "low"]
    question: str
    required_data: list[str]
    linked_criteria: list[str]
```

### 生成规则

先使用模板生成：

- ANC / platelet / hemoglobin / bilirubin / AST / ALT / creatinine clearance：
  - “请提供最近一次 {lab list} 数值和采样日期，优先提供方案要求时间窗内的结果。”
- ECOG：
  - “请确认患者的 ECOG 体能状态评分。”
- CNS 转移：
  - “请确认患者是否存在未经治疗或不稳定的 CNS 转移。”
- 妊娠：
  - “如临床适用，请确认妊娠状态。”
- 生物标志物：
  - “请确认 {marker} 状态及检测方法。”
- 既往治疗：
  - “请确认既往治疗线数和治疗日期。”

按所需数据类型去重，并合并关联标准。

### 文件位置

新增：

- `agent/protocol/missing_data.py`

测试：

- `tests/test_protocol_missing_data.py`

## Protocol 节点集成

### 新辅助函数

在 `agent/graph/nodes.py` 中新增辅助函数：

```python
def _assess_trial_eligibility_candidates(
    state: GraphState,
    *,
    trial_bundle: dict[str, Any],
    limit: int = 3,
) -> dict[str, Any]:
    ...
```

该辅助函数应：

1. 选择排名靠前的 `limit` 个非废弃试验候选。
2. 从试验目录或检索器中获取每个完整 `trial_record`。
3. 解析资格标准。
4. 检索患者证据片段。
5. 判断每条标准。
6. 聚合试验级资格。
7. 生成缺失数据问题。
8. 返回 `eligibility_assessment_bundle`。

### Protocol 节点修改

当前结构：

```python
trial_retrieval_bundle = _retrieve_trial_candidates(state)
trial_selection = _build_protocol_trial_selection(...)
recommendations = _build_treatment_recommendations(...)
treatment_bundle = {...}
```

目标结构：

```python
trial_retrieval_bundle = _retrieve_trial_candidates(state)
eligibility_assessment_bundle = _assess_trial_eligibility_candidates(
    state,
    trial_bundle=trial_retrieval_bundle,
    limit=3,
)
trial_selection = _build_protocol_trial_selection(
    trial_retrieval_bundle,
    structured_case=dict(state.structured_case_json or {}),
    eligibility_assessment_bundle=eligibility_assessment_bundle,
)
recommendations = _build_treatment_recommendations(
    state,
    trial_bundle=trial_retrieval_bundle,
    eligibility_assessment_bundle=eligibility_assessment_bundle,
)
treatment_bundle = {
    "recommendations": [asdict(item) for item in recommendations],
    "trial_candidates": trial_candidates,
    "trial_candidate_ids": trial_candidate_ids,
    "trial_selection": trial_selection,
    "eligibility_assessment_bundle": eligibility_assessment_bundle,
    "note": "This node owns treatment and clinical-trial protocol judgment.",
}
state.final_output["eligibility_assessment_bundle"] = eligibility_assessment_bundle
```

### 向后兼容

保留现有字段：

- `trial_retrieval_bundle`
- `treatment_bundle["trial_candidates"]`
- `treatment_bundle["trial_candidate_ids"]`
- `treatment_bundle["trial_selection"]`
- `treatment_bundle["recommendations"]`

只新增字段，不移除既有字段：

- `treatment_bundle["eligibility_assessment_bundle"]`
- `final_output["eligibility_assessment_bundle"]`

更新必要的预期载荷后，现有测试应继续通过。

## 试验记录访问

资格评估流水线需要完整试验记录，而不仅是候选片段。

推荐做法：

1. 给 `TrialChunkRetrievalTool` 增加方法：

```python
def get_trial_record(self, nct_id: str) -> dict[str, Any] | None:
    return self.catalog.get_record(nct_id)
```

2. 在 `_assess_trial_eligibility_candidates(...)` 中复用 `_resolve_trial_retriever(state)` 返回的同一个检索器。
3. 如果注入的检索器没有暴露 `get_trial_record`，则退回到仅使用候选字段，并在 bundle 中标记缺失的试验字段。

## 数据模型位置

方案 A：将 protocol 相关数据类放在 `agent/protocol/types.py`。

方案 B：将它们加入 `agent/graph/types.py`。

建议使用方案 A。这些对象属于 protocol 阶段内部模型，在 schema 稳定之前不应膨胀中央图状态。写入 `state.final_output` 前转换为字典。

新增：

- `agent/protocol/types.py`

数据类：

- `EligibilityCriterion`
- `PatientEvidenceSpan`
- `CriterionAssessment`
- `TrialEligibilityAssessment`
- `MissingDataQuestion`

## 测试

### 单元测试

新增：

- `tests/test_protocol_criteria_parser.py`
- `tests/test_protocol_evidence_retriever.py`
- `tests/test_protocol_criterion_judge.py`
- `tests/test_protocol_eligibility_aggregator.py`
- `tests/test_protocol_missing_data.py`

### 集成测试

新增或扩展：

- `tests/test_trial_retrieval_tools.py`
- `tests/test_orchestrator_state.py`

最低集成场景：

1. 年龄和性别兼容，且没有排除证据：
   - 预期聚合状态：根据未知标准数量，结果为 `needs_data` 或 `likely_eligible`。
2. 存在明确排除证据：
   - 预期聚合状态：`ineligible`。
3. 已关闭或已完成试验：
   - 预期聚合解释：`evidence_support`。
4. 实验室标准未知：
   - 预期缺失问题提到所需实验室检查。
5. 现有 `trial_selection` 字段仍然可用。

## 评估计划

### 内部冒烟评估

使用合成病例和小型 fixture 试验知识库：

- 10 条试验记录
- 20 个患者病例
- 人工标注的逐条标准决策

指标：

- 标准标签准确率
- `unknown` 召回率
- 证据片段命中率
- 试验级聚合准确率
- 缺失问题有用性

### TREC Clinical Trials 评估

使用 TREC topics 和 qrels 评估试验检索指标：

- Recall@30
- NDCG@10
- MRR@10

然后人工标注一小部分资格评估样本：

- 20 个 topic
- 每个 topic 取排名前 5 的试验
- 为高信号标准标注逐条标准标签

### 研究指标

报告：

- 引入资格重排前后的试验检索质量
- 逐条标准标签准确率
- 减少的人工复核项目数量
- 缺失数据问题精确率
- 每个试验和每条标准的延迟

## 开发阶段

### 阶段 1：确定性骨架

目标：在不调用 LLM 的前提下实现四个核心模块。

任务：

1. 增加 protocol 数据类。
2. 增加基于规则的标准解析器。
3. 增加患者证据片段索引。
4. 增加基于规则的标准判断器，覆盖年龄、性别、简单疾病和否定匹配。
5. 增加确定性聚合器。
6. 增加缺失问题模板。
7. 将 bundle 接入 `protocol_node`。
8. 增加单元测试和小型集成 fixture。

预期输出：

- `eligibility_assessment_bundle` 存在。
- 排名前列的试验候选包含已解析标准和聚合状态。
- 为未知标准生成缺失问题。

### 阶段 2：使用严格 JSON 的 LLM 判断器

目标：增加可选的 LLM 标准判断，同时保留确定性回退。

任务：

1. 增加 `agent/prompt/protocol_criterion_judge.md`。
2. 增加严格 JSON schema 校验。
3. 一次只判断一条标准。
4. 如果 JSON 解析或校验失败，回退到规则判断器。
5. 为每条标准决策增加追踪信息。

预期输出：

- 更好地处理复杂自由文本标准。
- 判断器不会输出不受控的自由文本推荐。

### 阶段 3：感知资格状态的试验选择

目标：使用逐条标准结果来选择试验。

任务：

1. 修改 `_build_protocol_trial_selection(...)`，使其接收 `eligibility_assessment_bundle`。
2. 优先选择 `likely_eligible`，再考虑 `needs_data`。
3. 排除 `ineligible` 和 `not_current_option`，除非没有更好的选项。
4. 区分开放入组和证据支持用途。
5. 更新治疗推荐，使其提到缺失数据和资格阻断因素。

预期输出：

- `trial_selection` 不再只是简单选择排名最高的非废弃候选。
- 选择理由引用资格聚合状态和关键标准。

### 阶段 4：生物标志物与计算器集成

目标：让系统能支撑生物信息学和肿瘤学工作流，并具备可发表的研究价值。

任务：

1. 增加 EGFR、ALK、KRAS、BRAF、ERBB2/HER2、MSI、TMB、PD-L1 的生物标志物解析。
2. 归一化变异表达以及 AND/OR 逻辑。
3. 将计算器结果送入证据检索。
4. 允许标准引用已计算风险或评分阈值。
5. 增加肿瘤学专项评估病例。

预期输出：

- Protocol 匹配支持基因组信息和计算器驱动的资格判断。
- 研究叙事强于通用试验 RAG。

## 风险与控制

### 风险：标准解析噪声较高

控制：

- 保留 `raw_text`。
- 使用稳定 ID。
- 保持解析字段可选。
- 当结构化解析不完整时，允许标准判断器直接使用原始文本。

### 风险：LLM 夸大资格匹配

控制：

- LLM 一次只判断一条标准。
- 试验级状态由确定性聚合器负责。
- 证据不足时优先输出 `unknown`。
- 每个非 `unknown` 标签都必须引用证据片段。

### 风险：试验状态被误用

控制：

- 开放入组试验可作为入组候选展示。
- 已完成或 active-not-recruiting 试验可作为证据支持。
- abandoned、withdrawn、suspended 试验不应作为当前选项。

### 风险：延迟过高

控制：

- 默认只评估排名前 3 的试验。
- 在早期冒烟模式中限制每个试验的标准数量。
- 按 `nct_id` 和 XML hash 缓存已解析标准。
- 仅对规则无法解决的标准运行 LLM 判断器。

## 建议的第一个提交

第一个提交应避免调用 LLM，聚焦于可测试的确定性流水线：

1. `agent/protocol/types.py`
2. `agent/protocol/criteria_parser.py`
3. `agent/protocol/evidence_retriever.py`
4. `agent/protocol/criterion_judge.py`
5. `agent/protocol/eligibility_aggregator.py`
6. `agent/protocol/missing_data.py`
7. `agent/protocol/pipeline.py`
8. `tests/test_protocol_*.py`

然后在第二个提交中将流水线接入 `protocol_node`。

## 完成定义

第一版实现满足以下条件时视为完成：

1. 运行工作流时产生 `final_output["eligibility_assessment_bundle"]`。
2. 每个被评估试验都有已解析的纳入和排除标准。
3. 每条标准都有标签、证据字段和缺失数据字段。
4. 试验级状态由确定性规则产生。
5. 对未知的高价值标准生成缺失问题。
6. 现有 protocol 试验检索测试仍然通过。
7. 至少有一个集成测试验证 `treatment_bundle` 包含新的资格评估 bundle，并且没有破坏既有字段。
