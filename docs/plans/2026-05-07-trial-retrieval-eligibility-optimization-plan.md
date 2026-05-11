# Trial 检索与资格比对优化方案

## 目标

当前要直接解决的问题是：在大规模 ClinicalTrials.gov trial 数据上，稳定完成“粗召回、精召回、inclusion/exclusion 定位、抽参比对”的在线链路。

本方案聚焦三个问题：

1. 大规模 trial 数据下，如何先粗召回、再精召回，避免全库逐条资格比对。
2. 如何稳定定位 ClinicalTrials.gov XML 中的 inclusion 和 exclusion。
3. 如何只在 top candidates 上做入排标准抽参和患者证据比对。

在实现顺序上，建议先搭建 `protocol` 内部子多 agent / subgraph 框架，再把 trial 检索、资格比对、医学知识检索和计算器风险证据作为子模块接入。否则后续逻辑会继续散落在 `protocol_node` 和若干 helper 函数里，难以扩展到大规模 trial 和医学知识联合推理。

不在本方案内解决：

1. 完整治疗推荐策略。
2. 报告生成。
3. 复杂标准的 LLM 全量自动判断。
4. 生物标志物和复杂治疗线数的完整医学推理。

## 当前基线

当前 `protocol` trial 检索已经具备两阶段结构：

```text
retrieve_from_structured_case(...)
  -> build query_profile
  -> retrieve_coarse_from_structured_case(...)
  -> coarse_candidate_ids
  -> 在 coarse_candidate_ids 范围内做 BM25 / vector chunk 精召回
  -> _aggregate_trials(...)
  -> _enrich_protocol_candidates(...)
  -> candidate_ranking
```

相关代码：

- `agent/graph/nodes.py`
  - `_retrieve_trial_candidates(...)`
  - `_assess_trial_eligibility_candidates(...)`
  - `protocol_node(...)`
- `agent/tools/trial_vector_retrieval_tools.py`
  - `build_protocol_trial_query_profile(...)`
  - `TrialChunkRetrievalTool.retrieve_coarse_from_structured_case(...)`
  - `TrialChunkRetrievalTool.retrieve_from_structured_case(...)`
  - `TrialChunkRetrievalTool._aggregate_trials(...)`
  - `_trial_level_rerank_payload(...)`
- `agent/trial_vector_kb.py`
  - `_split_eligibility_sections(...)`
  - `build_trial_record_from_xml_bytes(...)`
  - `build_trial_chunks(...)`
- `agent/protocol/`
  - `criteria_parser.py`
  - `evidence_retriever.py`
  - `criterion_judge.py`
  - `eligibility_aggregator.py`
  - `missing_data.py`
  - `pipeline.py`

当前默认规模：

```python
_PROTOCOL_TRIAL_COARSE_TOP_K = 30
_PROTOCOL_TRIAL_TOP_K = 10
eligibility_assessment_limit = 3
```

这适合冒烟测试，但对大规模 trial corpus 偏保守，需要配置化。

## 架构定位

`protocol` 在外层 workflow 里应保持一个节点，但内部建议升级为 protocol subgraph。

外层关系：

```text
clinical_assisstment
  -> calculator
  -> protocol
  -> reporter
```

其中 `protocol` 不只是 trial retrieval 节点，而是一个证据整合节点。当前阶段不把 `clinical_assisstment` 和 `calculator` 编入 protocol subgraph，也不由 protocol 调度它们；protocol 默认接收这些上游 agent 已经产出的结果，并消费、整合这些结果。

它应该包住以下能力：

```text
protocol:
  trial retrieval
  trial eligibility assessment
  medical knowledge retrieval
  calculator/risk evidence integration
  protocol trial selection
  treatment/trial recommendation bundle
```

也就是说，`protocol` 当前先把上游产物汇总起来：

1. 来自 `clinical_assisstment` 的患者结构化事实。
2. 来自 `calculator` 的风险评分、计算器结果、缺失输入和解释。
3. 来自 trial KB 的候选临床试验、入排标准和 trial 状态。
4. 来自医学知识区的疾病、治疗、指南、药物、生物标志物和试验解释知识。

边界约束：

1. `protocol` 不重新执行 `clinical_assisstment`。
2. `protocol` 不重新调度 `calculator`。
3. `protocol` 只读取 `structured_case_json`、`calculation_results`、`calculator_matches`、`calculation_bundle` 等已存在状态。
4. 后续如果需要 protocol 反向请求补充计算器或病例结构化信息，应通过明确的 follow-up task 或 missing question 输出，而不是在 protocol subgraph 内隐式重跑上游 agent。

这个定位下，`protocol` 的核心输出不是单纯“找到 trial”，而是：

```text
把患者事实、计算器风险证据、医学知识和临床试验资格评估汇成一个 protocol decision bundle
```

建议把 `protocol` 正式定义为一个小 graph，而不是继续把所有逻辑压进一个顺序函数。

外层 workflow 仍然只看到一个 `protocol_node`；但 `protocol_node` 内部运行 `protocol_subgraph`，先分发到三个 subagent，再由 aggregator 汇总：

```text
clinical_assisstment / calculator
  -> protocol_node
       -> run_protocol_subgraph
            ├─ trial_agent
            ├─ medical_knowledge_agent
            └─ patient_calculator_evidence_agent
            ↓
          protocol_aggregator
       -> protocol bundles
  -> reporter
```

三个 subagent 的边界：

| subagent | 输入 | 职责 | 输出 |
| --- | --- | --- | --- |
| `trial_agent` | `structured_case_json`、`department_tags`、可选 calculator 风险线索 | trial 粗召回、精召回、trial-level rerank、trial_record 获取、inclusion/exclusion 解析、top trial 资格评估 | `trial_retrieval_bundle`、`eligibility_assessment_bundle` |
| `medical_knowledge_agent` | 病例问题、top trial、unknown criteria、calculator/risk terms | 检索疾病、治疗、指南、药物、生物标志物和 trial 标准解释知识 | `medical_knowledge_bundle` |
| `patient_calculator_evidence_agent` | `structured_case_json`、`calculation_results`、`calculator_matches`、`calculation_bundle` | 整理患者事实、calculator 风险证据、缺失输入和可用于 criterion 判断的 patient evidence | `patient_evidence_bundle`、`calculator_evidence_bundle`、`missing_data_bundle` |

`protocol_aggregator` 的职责：

1. 合并 trial、医学知识、患者证据和 calculator 风险证据。
2. 根据 `eligibility_assessment_bundle` 和 trial 状态选择 trial / protocol branch。
3. 区分可推荐、需人工复核、证据不足和需要补充数据的场景。
4. 生成 `protocol_decision_bundle`、`treatment_bundle` 和 `protocol_recommendations`。
5. 把稳定字段写回外层 `GraphState`，供 `reporter` 检查和成文。

边界要求：

1. `trial_agent` 不能直接生成最终治疗建议，只产出 trial 证据和资格评估。
2. `medical_knowledge_agent` 不能直接判定 eligibility，只提供解释、归一化和背景证据。
3. `patient_calculator_evidence_agent` 不能重跑 `clinical_assisstment` 或 `calculator`，只整理已有上游结果。
4. `protocol_aggregator` 是唯一负责跨通道合并、选择和推荐的 protocol 内部节点。

第一版可以把三个 subagent 实现成普通 Python 节点或函数，不要求全部是 LLM agent。真正需要 LLM 的位置应限制在复杂 criterion 解析、复杂单条标准判断和必要的医学概念解释；检索、过滤、状态聚合和 schema 校验优先用规则和工具。

### 文献和 multi-agent 设计对齐

本方案不是把通用 multi-agent 形式硬套到临床流程，而是吸收现有 trial matching 和医学 multi-agent 文献中稳定的结构。

#### TrialGPT：retrieval、matching、ranking 三段式

TrialGPT 将 patient-to-trial matching 拆为三个模块：

```text
TrialGPT-Retrieval
  -> TrialGPT-Matching
  -> TrialGPT-Ranking
```

对应到 MadAI：

| TrialGPT | MadAI protocol subgraph |
| --- | --- |
| Retrieval：大规模 trial 过滤 | `trial_agent` 的 coarse retrieval / fine retrieval |
| Matching：criterion-level eligibility prediction | `trial_agent` 内的 criteria parsing、patient evidence matching、criterion judgment |
| Ranking：聚合标准级判断并排序 trial | `protocol_aggregator` 的 eligibility aggregation、trial selection |

对本项目的直接启发：

1. 大库上必须先 retrieval，再对少量 top trial 做 criterion-level matching。
2. ranking / selection 不能只看向量相似度，必须吃到 criterion-level 判断。
3. 标准级解释和证据位置是核心产物，不能只输出一个 trial 分数。

#### PRISM / OncoLLM：真实 EHR 和 criterion 解释

PRISM 关注真实患者记录和 trial inclusion/exclusion 的语义匹配，强调长 EHR、复杂肿瘤信息、标准级解释和医生评估。

对应到 MadAI：

1. `patient_calculator_evidence_agent` 不能只消费 `case_summary`，还要逐步扩展到 raw text、known facts、结构化输入、calculator results 和后续 EHR 文档。
2. `trial_agent` 的 criterion 判断必须保留 evidence spans。
3. 对复杂 oncology 场景，如治疗线数、分子标志物、CNS metastases、器官功能阈值，应允许规则失败后进入 LLM 辅助，但输出仍要严格 schema 化。

#### MedAgents：多学科专家协作

MedAgents 的核心价值是把医学推理拆成多个专业角色，再由协作机制汇总。它适合复杂临床推理，但不适合把检索、切分、聚合这些确定性步骤都交给自由讨论。

对应到 MadAI：

1. `medical_knowledge_agent` 后续可以按领域扩成 oncology / cardiology / genomics / pharmacology 等专家子角色。
2. 基础链路仍应保持工具和规则为骨架。
3. 专家 agent 的输出必须进入 `medical_knowledge_bundle` 或 `criterion_assessment`，不能绕过 aggregator 直接影响最终推荐。

#### MDAgents：按复杂度自适应协作

MDAgents 强调根据任务复杂度选择 solo 或 group collaboration。这个思想适合 MadAI 的 eligibility pipeline。

对应到 MadAI：

1. 简单年龄、性别、明确诊断、明确否定事实走规则，不调用 LLM。
2. 中等复杂度标准进入单条 criterion LLM 辅助。
3. 高复杂度标准，例如多治疗线组合、模糊时间窗、复杂 biomarker、药物禁忌，才触发医学知识检索和专家子 agent。
4. `protocol_aggregator` 应记录每条判断的 `decision_path`，区分 rule、retrieval、LLM-assisted 和 manual_review。

#### AgentClinic：不完整信息下的顺序决策

AgentClinic 强调临床任务不是静态问答，而是不完整信息下的工具使用和顺序决策。

对应到 MadAI：

1. `missing_data_bundle` 应成为 protocol subgraph 的正式输出，不是附属说明。
2. 当关键 criterion 缺证据时，系统应生成最小必要追问，而不是猜 eligibility。
3. protocol 输出应允许 `needs_data` / `manual_review`，由 reporter 清楚呈现下一步需要补什么。

#### MedAgentBoard：谨慎使用 multi-agent

MedAgentBoard 的结论提醒：multi-agent 并不总是优于单 LLM 或传统方法，额外复杂度必须由任务收益证明。

对应到 MadAI：

1. 不把所有节点都做成 LLM agent。
2. 不为了“multi-agent”增加无测试、无结构化输出的讨论轮。
3. 每个 subagent 必须有明确输入、输出、trace、失败模式和测试。
4. 多 agent 的收益要落在可观测指标上，例如 recall、criterion label accuracy、unknown recall、evidence span hit rate、missing question usefulness。

#### MadAI 采用的融合方式

最终不是简单复刻某一篇论文，而是采用一个混合式小 graph：

```text
evidence acquisition layer
  ├─ trial_agent
  ├─ medical_knowledge_agent
  └─ patient_calculator_evidence_agent

reasoning / matching layer
  ├─ criteria_parser
  ├─ patient evidence matcher
  └─ criterion_judge

aggregation / review layer
  ├─ eligibility_aggregator
  ├─ protocol_aggregator
  └─ reporter
```

这满足两个目标：

1. 把 multi-agent 的分工、并行、专家化和审查机制揉进 `protocol`。
2. 保留医疗系统需要的确定性、可追溯、可测试和保守失败行为。

### 计算器风险证据汇入

`calculator` 阶段的结果不应只作为 `recommendations.linked_calculators` 出现，而应进入 protocol 判断上下文。

建议新增内部 bundle：

```json
{
  "calculator_evidence_bundle": {
    "schema_version": 1,
    "completed_results": [],
    "partial_results": [],
    "estimated_results": [],
    "risk_evidence_items": [
      {
        "calculator": "",
        "category": "risk_score",
        "value": "",
        "unit": "",
        "status": "completed",
        "rationale": "",
        "usable_for": ["trial_query_profile", "eligibility_evidence", "treatment_recommendation"],
        "missing_inputs": []
      }
    ]
  }
}
```

用途：

1. 生成 trial query profile，例如根据风险分层、疾病严重度或评分结果扩展 trial intent。
2. 作为 patient evidence，例如 ECOG、CHA2DS2-VASc、MELD、Child-Pugh、Gleason、TNM 等可支持入排标准判断。
3. 支持 treatment recommendation，使推荐不仅链接 trial，也链接风险证据。
4. 生成 missing questions，如果计算器缺关键输入且该输入也影响 trial eligibility，则合并追问。

### 医学知识区汇入

`protocol` 还应有医学知识通道，不只依赖 trial KB。

建议新增：

```json
{
  "medical_knowledge_bundle": {
    "schema_version": 1,
    "queries": [],
    "retrieved_items": [],
    "knowledge_gaps": []
  }
}
```

医学知识区用于：

1. 解释 trial 标准中的医学概念。
2. 支持复杂标准抽参，例如 prior line therapy、stable CNS metastases、organ function threshold。
3. 支持治疗推荐中的 guideline / evidence 背景。
4. 为 reporter 提供可引用背景。

注意：

`medical_knowledge_bundle` 不能直接决定 eligibility。它只提供解释和背景，最终 criterion label 仍然必须来自 trial criterion 和患者证据比对。

## 总体链路

在线查询时不应对全库 trial 直接做资格评估。正确链路是：

```text
structured_case
  -> query_profile
  -> 全库粗召回 trial IDs
  -> 候选 trial 内精召回 chunks
  -> 聚合成 trial-level candidate_ranking
  -> 只对 top few trial 做 eligibility assessment
  -> parse inclusion/exclusion criteria
  -> retrieve patient evidence
  -> judge criterion
  -> aggregate trial eligibility
```

推荐默认规模：

```text
coarse_top_k = 100-300
fine_top_k = 10-20
eligibility_assessment_limit = 3-5
```

研究评估模式可放宽：

```text
coarse_top_k = 1000
fine_top_k = 50
eligibility_assessment_limit = 10
```

关键原则：

1. 粗召回负责 recall，不做资格判断。
2. 精召回负责 ranking，不做逐条入排标准结论。
3. 资格评估只在 top candidates 上运行。
4. 入排标准判断必须回到完整 `trial_record` 的原始 eligibility 字段，不依赖单个命中 chunk。
5. 无法可靠定位 inclusion/exclusion 时，必须保守输出 `unknown` 或人工复核提示。

## 阶段 A：离线 trial KB 构建优化

离线阶段应尽量把稳定信息预处理好，在线阶段只做病例相关计算。

输入：

```text
ClinicalTrials.gov XML corpus
```

输出：

```text
trial_record catalog
trial chunk index
trial_id -> chunk_ids
trial_id -> parsed eligibility sections
optional trial_id -> parsed criteria cache
```

需要保证的字段：

```json
{
  "nct_id": "NCT...",
  "eligibility_text": "...",
  "eligibility_inclusion_text": "...",
  "eligibility_exclusion_text": "...",
  "eligibility_unsplit_text": "",
  "eligibility_section_parse_status": "parsed",
  "eligibility_section_parse_warnings": []
}
```

`eligibility_section_parse_status` 建议取值：

| 状态 | 含义 | 在线处理 |
| --- | --- | --- |
| `parsed` | 明确切出 inclusion 和 exclusion | 正常解析 criteria |
| `inclusion_only` | 只找到 inclusion heading | exclusion 为空，保守评估 |
| `exclusion_only` | 只找到 exclusion heading | inclusion 为空，保守评估 |
| `unsplit` | 找不到可靠 section heading | 不要强行判断 inclusion/exclusion，标记需人工复核 |
| `empty` | eligibility text 为空 | 资格评估返回 `needs_data` |

需要增强：

- `agent/trial_vector_kb.py`
  - 增强 `_split_eligibility_sections(...)`
  - 保存 `eligibility_unsplit_text`
  - 保存 `eligibility_section_parse_status`
  - 保存 `eligibility_section_parse_warnings`

## 阶段 B：稳定定位 inclusion / exclusion

稳定定位 inclusion / exclusion 必须发生在离线 XML 构建阶段，而不是在线检索阶段。

当前实现主要识别：

```text
Inclusion Criteria
Exclusion Criteria
```

需要扩展 heading 识别：

```text
Inclusion Criteria
Key Inclusion Criteria
Main Inclusion Criteria
Inclusion
Criteria for Inclusion
Patients must meet all of the following

Exclusion Criteria
Key Exclusion Criteria
Main Exclusion Criteria
Exclusion
Criteria for Exclusion
Patients will be excluded if
```

切分规则：

1. 先规范化 XML textblock 的空白，但保留原始语义。
2. 用 heading 锚点定位 section 起止。
3. section 类型只由 heading 决定，不由向量检索结果决定。
4. 每个 section 保留 `section_type`、`heading_text`、`start_offset`、`end_offset`。
5. 如果无法可靠切分，写入 `eligibility_unsplit_text`，不要把整段默认当作 inclusion。

建议的数据结构：

```python
@dataclass(slots=True)
class EligibilitySection:
    nct_id: str
    section_type: Literal["inclusion", "exclusion", "unknown"]
    heading_text: str
    raw_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    parse_status: str = "parsed"
    warnings: list[str] = field(default_factory=list)
```

验收测试：

1. 标准 `Inclusion Criteria` / `Exclusion Criteria` 能正确切分。
2. `Key Inclusion Criteria` / `Key Exclusion Criteria` 能正确切分。
3. 只有 inclusion heading 时，status 为 `inclusion_only`。
4. 只有 exclusion heading 时，status 为 `exclusion_only`。
5. 没有 heading 时，status 为 `unsplit`，不强行生成 exclusion。
6. inclusion/exclusion 顺序反过来时也能正确切分。

## 阶段 C：粗召回优化

粗召回目标是从全库快速缩小 trial 范围。

输入：

```text
structured_case -> query_profile -> query_text
```

粗召回应使用：

1. 疾病词：diagnosis / condition terms
2. 干预词：drug / procedure / treatment terms
3. 意图词：treatment / prevention / screening / supportive care
4. 人口学：age / gender
5. 阴性事实：must_not profile terms
6. department tags 或 disease domain filters

输出：

```json
{
  "coarse_candidate_ids": [],
  "coarse_candidate_ranking": [],
  "query_profile": {},
  "backend_used": "vector | bm25"
}
```

优化点：

1. 将 `coarse_top_k` 做成配置项，而不是固定常量。
2. 大库默认优先使用 vector coarse recall，vector 不可用时回退 BM25。
3. 粗召回可以取 `top_k * 4` 个 chunk，再聚合回 trial，避免一个 trial 的多个 chunk 占满结果。
4. 粗召回结果只负责提供 candidate IDs，不直接产生资格结论。

建议配置：

```text
MEDAI_PROTOCOL_TRIAL_COARSE_TOP_K=300
MEDAI_PROTOCOL_TRIAL_TOP_K=20
MEDAI_PROTOCOL_ELIGIBILITY_LIMIT=5
```

需要修改：

- `agent/graph/nodes.py`
  - `_PROTOCOL_TRIAL_COARSE_TOP_K`
  - `_PROTOCOL_TRIAL_TOP_K`
  - 新增 eligibility limit 配置读取
- `agent/tools/trial_vector_retrieval_tools.py`
  - 保持 `retrieve_coarse_from_structured_case(...)` 只返回 coarse ranking 和 IDs

## 阶段 D：精召回与 trial-level rerank 优化

精召回只在 `coarse_candidate_ids` 范围内运行。

输入：

```text
coarse_candidate_ids
query_text
query_profile
```

过程：

```text
candidate_nct_ids -> candidate_chunk_ids
  -> BM25 chunk retrieval
  -> vector chunk retrieval
  -> aggregate chunks by nct_id
  -> trial-level rerank
```

精召回应重点利用 chunk 类型：

```text
overview
eligibility_inclusion
eligibility_exclusion
arms_interventions
description
outcomes
```

排序信号：

1. top chunk score
2. 多 chunk support score
3. 是否命中 eligibility chunk
4. 是否同时命中 overview + eligibility
5. 是否命中 arms/interventions
6. condition / intervention / intent terms 覆盖度
7. trial status bonus / penalty
8. must_not conflicts

输出：

```json
{
  "bm25_top5": [],
  "vector_top5": [],
  "candidate_ranking": []
}
```

注意：

`candidate_ranking` 是进入资格评估的候选排序，但它不是最终资格结论。资格结论必须由后续 `eligibility_assessment_bundle` 产生。

## 阶段 E：抽参比对优化

抽参比对只对精召回后的 top trial 运行。

推荐入口：

```python
assess_trial_eligibility_candidates(
    structured_case=...,
    calculation_results=...,
    calculator_matches=...,
    trial_bundle=trial_retrieval_bundle,
    trial_retriever=trial_retriever,
    limit=eligibility_limit,
)
```

单个 trial 的处理过程：

```text
candidate nct_id
  -> get full trial_record
  -> read eligibility_inclusion_text / eligibility_exclusion_text
  -> parse_trial_criteria(...)
  -> build patient evidence index
  -> find evidence for each criterion
  -> judge criterion
  -> aggregate trial status
  -> generate missing questions
```

抽参分两类：

1. trial 标准参数：

```json
{
  "condition": "age",
  "operator": ">=",
  "value": "18 years",
  "required_evidence_type": "demographic"
}
```

2. 患者证据参数：

```json
{
  "source": "raw_text",
  "text": "62-year-old male",
  "value": "62",
  "unit": "years"
}
```

比对规则：

| 标准类型 | 患者证据 | 输出 |
| --- | --- | --- |
| inclusion: age >= 18 | age 62 | `met` |
| inclusion: ECOG 0-1 | ECOG 2 | `not_met` |
| exclusion: active brain metastases | no brain metastases | `not_met` |
| exclusion: pregnancy | no pregnancy evidence | `unknown` |
| exclusion: active infection | active infection evidence | `met` |

关键语义：

```text
inclusion.met 是好事
inclusion.not_met 是阻断风险
exclusion.not_met 是好事
exclusion.met 是阻断风险
unknown 表示缺证据，不允许猜
```

## 阶段 F：criteria 解析缓存

trial criteria 是 trial 固定属性，不随患者病例变化。大规模运行时应缓存：

```text
parse_trial_criteria(trial_record)
```

缓存 key：

```text
nct_id + source_snapshot_date + source_archive + eligibility_text_hash
```

缓存内容：

```json
{
  "nct_id": "NCT...",
  "eligibility_text_hash": "...",
  "criteria": [],
  "section_parse_status": "parsed"
}
```

收益：

1. 多个病例评估同一 trial 时不重复解析标准。
2. 可离线统计解析质量。
3. 可提前发现大量 `unsplit` 或复杂标准。

## 阶段 G：质量监控与验收指标

检索指标：

1. coarse recall：相关 trial 是否进入 `coarse_candidate_ids`
2. fine ranking：相关 trial 是否进入 `candidate_ranking` top N
3. chunk coverage：是否命中 eligibility / overview / arms_interventions

include/exclude 定位指标：

1. section parse success rate
2. unsplit rate
3. inclusion/exclusion reversal error rate
4. empty eligibility rate

资格评估指标：

1. criterion label accuracy
2. unknown recall
3. blocker precision
4. evidence span hit rate
5. aggregate_status accuracy
6. missing_questions usefulness

最低验收：

1. 大库查询不会对全库 trial 执行 criteria 判断。
2. `coarse_candidate_ids`、`candidate_ranking`、`eligibility_assessment_bundle` 三层结果都可观测。
3. include/exclude 的来源字段可追溯。
4. `unsplit` eligibility 不会被强行当作 inclusion/exclusion。
5. top trial 的每条 criterion 都有 `raw_text`、`label`、`evidence_spans`、`rationale`、`missing_data`。

## 实施顺序

第 0 批：搭建 protocol subgraph 框架，不改变行为。

1. 新增 `agent/protocol/state.py`，定义 `ProtocolGraphState`。
2. 新增 `agent/protocol/config.py`，定义 `ProtocolGraphConfig`，包含 `coarse_top_k`、`fine_top_k`、`eligibility_limit`。
3. 新增 `agent/protocol/graph.py`，提供 `run_protocol_subgraph(...)`。
4. 在 `agent/protocol/graph.py` 中明确三个 subagent 函数边界：
   - `run_trial_agent(...)`
   - `run_medical_knowledge_agent(...)`
   - `run_patient_calculator_evidence_agent(...)`
5. 新增 `run_protocol_aggregator(...)`，第一版只做现有字段透传和 bundle 汇总，不改变推荐策略。
6. `run_protocol_subgraph(...)` 的输入只来自外层已有状态：`structured_case_json`、`calculation_results`、`calculator_matches`、`calculation_bundle`、`department_tags`、`trial_retriever`、`medical_knowledge_retriever`。
7. 新增 `patient_evidence_bundle` 和 `calculator_evidence_bundle` 空壳，把现有病例事实、`calculation_results`、`calculator_matches` 转成 protocol 内部证据。
8. 新增 `medical_knowledge_bundle` 空壳，第一版可返回空列表或 `not_configured`，但保留接口。
9. 新增 `protocol_decision_bundle`，聚合三路 subagent 的状态、warnings、关键输出 ID 和 trace summary。
10. `protocol_node` 调用 `run_protocol_subgraph(...)`，再把稳定结果写回外层 `GraphState`。
11. 行为保持不变，现有测试继续通过。

第一批：稳定 include/exclude 和配置项。

1. 增强 `_split_eligibility_sections(...)`。
2. 增加 `eligibility_unsplit_text`、`eligibility_section_parse_status`、`eligibility_section_parse_warnings`。
3. 增加 section 切分单元测试。
4. 将 `coarse_top_k`、`fine_top_k`、`eligibility_limit` 做成配置。

第二批：增强 criteria 抽参。

1. 扩展年龄、性别、ECOG、Karnofsky、CNS 转移、妊娠、常见 lab 解析。
2. 增加 `source_field`、`section_type`、`section_parse_status` 到 criterion。
3. 对 `unsplit` 标准保守输出 `unknown`。
4. 增加真实 eligibility fixture 测试。

第三批：接入医学知识区和计算器风险证据。

1. 将 `calculator_evidence_bundle` 接入 trial query profile、患者证据索引和 recommendation bundle。
2. 将医学知识检索接入 `medical_knowledge_bundle`。
3. 优先支持疾病、治疗、药物、指南、生物标志物和评分解释。
4. 在 treatment recommendation 和 reporter 中引用医学知识证据。
5. `medical_knowledge_agent` 输出只进入解释、归一化、背景和 recommendation evidence，不直接覆盖 criterion label。

第四批：缓存和观测。

1. 增加 criteria parse cache。
2. 在 `trial_retrieval_bundle` 中暴露 coarse/fine 召回统计。
3. 在 `eligibility_assessment_bundle` 中暴露 parse warning 统计。
4. 在 `protocol_decision_bundle` 中暴露三个 subagent 的 status、latency、warnings、output counts。
5. 增加大库冒烟脚本，检查延迟和 top N 覆盖。

第五批：复杂标准和 LLM 辅助。

1. 仅对规则无法解析的复杂标准调用 LLM。
2. LLM 只做单条标准结构化解析或单条判断。
3. LLM 输出必须经过 JSON schema 校验。
4. 证据不足时仍然优先 `unknown`。
5. 按复杂度分流：简单规则、单条 LLM 辅助、专家知识检索、人工复核四类路径必须可追踪。

## 第一批完成定义

第 0 批完成定义：

1. `protocol_node` 内部通过 `run_protocol_subgraph(...)` 执行。
2. `ProtocolGraphState` 和 `ProtocolGraphConfig` 存在。
3. `run_trial_agent(...)`、`run_medical_knowledge_agent(...)`、`run_patient_calculator_evidence_agent(...)` 和 `run_protocol_aggregator(...)` 的接口存在。
4. `patient_evidence_bundle`、`calculator_evidence_bundle`、`medical_knowledge_bundle` 和 `protocol_decision_bundle` 在 protocol 内部 state 中存在。
5. 外层输出仍保留 `trial_retrieval_bundle`、`eligibility_assessment_bundle`、`treatment_bundle`。
6. 三个 subagent 的输出均有 `schema_version`、`status`、`warnings` 或等价可观测字段。
7. 现有 `tests/test_trial_retrieval_tools.py` 和 `tests/test_protocol_eligibility_pipeline.py` 通过。

第一批完成定义：

1. `build_trial_record_from_xml_bytes(...)` 输出 section parse status。
2. `eligibility_unsplit_text` 不为空时，系统不会强行把它当作 inclusion 或 exclusion。
3. `tests/test_trial_vector_kb.py` 覆盖常见 inclusion/exclusion heading 变体。
4. `protocol_node` 的粗召回、精召回和 eligibility limit 可通过环境变量配置。
5. 第 0 批测试继续通过。
