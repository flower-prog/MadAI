# 轻量 Trial 检索、精召回与 Eligibility 语义判断方案

## 目标

本方案用于把 ClinicalTrials.gov XML trial 数据接入 `protocol`，完成从病例到候选 trial 的多阶段召回和资格判断。

核心原则：

1. 不做全库复杂 eligibility 规则解析器。
2. XML 预处理只做浅结构化，保留原始 `eligibility/criteria/textblock`。
3. 粗召回和精召回使用 BM25、向量检索、结构化字段和轻量 rerank。
4. 只对 top candidates 使用 LLM 做病例与入排标准的语义对照。
5. parser 服务于检索，不承担最终医学判定。

## 背景判断

ClinicalTrials.gov XML 中，trial metadata 比较稳定，但入排标准本身通常是自由文本：

```xml
<eligibility>
  <criteria>
    <textblock>
      Inclusion Criteria:

        - ...

      Exclusion Criteria:

        - ...
    </textblock>
  </criteria>
  <gender>All</gender>
  <minimum_age>65 Years</minimum_age>
  <maximum_age>N/A</maximum_age>
  <healthy_volunteers>Accepts Healthy Volunteers</healthy_volunteers>
</eligibility>
```

因此不建议一开始把每条标准抽成复杂 JSON，例如疾病类型、治疗线数、实验室阈值、时间窗、证据类型等。这样会把全库预处理变成一个高复杂度医学 NLP 项目，而且错误会污染后续召回。

本阶段只需要保留可稳定提取的字段：

```json
{
  "nct_id": "NCTxxxx",
  "title": "...",
  "overall_status": "...",
  "study_type": "...",
  "phase": "...",
  "primary_purpose": "...",
  "conditions": ["..."],
  "condition_mesh_terms": ["..."],
  "keywords": ["..."],
  "interventions": ["..."],
  "intervention_types": ["..."],
  "brief_summary": "...",
  "detailed_description": "...",
  "eligibility_text": "原始 criteria textblock",
  "inclusion_text": "能稳定切分时保留，否则为空",
  "exclusion_text": "能稳定切分时保留，否则为空",
  "gender": "All",
  "minimum_age": "18 Years",
  "maximum_age": "N/A",
  "age_floor_years": 18.0,
  "age_ceiling_years": null,
  "countries": ["United States"],
  "source_url": "https://clinicaltrials.gov/show/NCTxxxx"
}
```

## 总体链路

```text
病例原文
  -> 病例浅结构化
  -> protocol 生成 trial search profile 和 patient evidence card
  -> XML 浅结构化 trial KB
  -> 多 query 粗召回 top 1000-2000
  -> trial-level 聚合与规则精排 top 300-500
  -> reranker 或轻量 LLM 精召回 top 100
  -> LLM eligibility judge 评估 top 20-50
  -> 聚合 retrieval score、eligibility assessment、missing information
  -> 输出最终 trial/protocol 建议
```

## 阶段 1：XML 浅结构化

### 必做字段

从 XML 中抽取稳定字段：

```text
id:
  id_info/nct_id
  required_header/url

title:
  brief_title
  official_title
  acronym

metadata:
  overall_status
  study_type
  phase
  primary_purpose
  allocation
  intervention_model
  masking
  enrollment

condition:
  condition
  condition_browse/mesh_term
  keyword

intervention:
  intervention/intervention_name
  intervention/intervention_type
  intervention/description
  intervention/other_name
  intervention_browse/mesh_term

text:
  brief_summary/textblock
  detailed_description/textblock
  eligibility/criteria/textblock

eligibility metadata:
  eligibility/gender
  eligibility/minimum_age
  eligibility/maximum_age
  eligibility/healthy_volunteers

location:
  location_countries/country
  location/facility/address/city
  location/facility/address/state
  location/facility/address/country
```

### 可选字段

能稳定切分时，额外保存：

```json
{
  "inclusion_text": "...",
  "exclusion_text": "...",
  "eligibility_section_parse_status": "parsed | inclusion_only | exclusion_only | unsplit | empty",
  "eligibility_section_parse_warnings": []
}
```

要求：

1. 不把切分结果当作医学事实。
2. 如果无法稳定识别 inclusion/exclusion heading，保留 `eligibility_text`，不要强拆。
3. 不在全库阶段把自由文本拆成复杂 criteria schema。

### 召回文本

为每个 trial 构造几个检索文本：

```text
overview_text:
  title + condition + intervention + brief_summary + study_type + phase + status

description_text:
  detailed_description

eligibility_text:
  inclusion_text + exclusion_text + gender + minimum_age + maximum_age

intervention_text:
  intervention_name + intervention_description + arm_group

trial_card_text:
  overview_text + eligibility_text + intervention_text 的压缩版
```

## 阶段 2：病例侧 Trial Search Profile

`protocol` 不应直接拿完整病例原文做唯一 query。应先生成两份结构。

### Retrieval Query Profile

用于召回和精排：

```json
{
  "retrieval_queries": [
    "severe aortic stenosis bicuspid aortic valve clinical trial",
    "critical aortic stenosis left ventricular dysfunction ejection fraction 25 trial",
    "aortic valve replacement bicuspid aortic valve trial",
    "TAVR surgical valve replacement severe aortic stenosis eligibility",
    "heart failure reduced ejection fraction severe aortic stenosis intervention"
  ],
  "primary_condition_terms": [
    "severe aortic stenosis",
    "critical aortic stenosis",
    "bicuspid aortic valve"
  ],
  "intervention_or_context_terms": [
    "aortic valve replacement",
    "valve replacement",
    "TAVR",
    "surgical valve replacement"
  ],
  "clinical_context_terms": [
    "left ventricular dysfunction",
    "ejection fraction 25%",
    "heart failure",
    "shortness of breath",
    "lower extremity edema"
  ],
  "demographics": {
    "age": 48,
    "sex": "Male"
  },
  "negative_or_qualifying_facts": [
    "no flow-limiting coronary artery disease",
    "mild pulmonary hypertension"
  ]
}
```

### Patient Evidence Card

用于 top trial 的 LLM eligibility 判断：

```json
{
  "patient_summary": "48-year-old male with hypertension, hyperlipidemia, tobacco use, bicuspid aortic valve, critical severe aortic stenosis, EF 25%, progressive dyspnea and edema, being evaluated for valve replacement. Cath showed no flow-limiting CAD.",
  "age": 48,
  "sex": "Male",
  "positive_facts": [
    "critical severe aortic stenosis",
    "bicuspid aortic valve",
    "EF 25%",
    "heart failure symptoms",
    "planned/evaluated for valve replacement"
  ],
  "negative_facts": [
    "no flow-limiting coronary artery disease"
  ],
  "unknown_facts": [
    "renal function",
    "current medications",
    "prior valve intervention",
    "infection status"
  ]
}
```

## 阶段 3：粗召回到 1000-2000

输入：

1. `retrieval_queries`
2. `primary_condition_terms`
3. `intervention_or_context_terms`
4. `clinical_context_terms`
5. 年龄、性别等轻结构化字段

检索方式：

1. 每个 query 分别跑 BM25。
2. 每个 query 分别跑向量检索。
3. 检索字段覆盖 `overview_text`、`description_text`、`eligibility_text`、`intervention_text`。
4. 使用 RRF 合并不同 query 和不同检索通道。
5. 聚合到 `nct_id`，保留命中的 top chunks。

粗召回不要使用过严 hard filter。

建议：

```text
可以硬过滤或强惩罚：
  年龄明显不符
  性别明显不符

只做 boost，不做 hard must：
  conditions
  interventions
  keywords
  eligibility_text 命中
  study_type
  phase
  primary_purpose

按使用场景加权：
  Recruiting / Not yet recruiting / Enrolling by invitation
  Active, not recruiting / Completed
  Terminated / Withdrawn / Suspended
```

输出：

```json
{
  "coarse_candidates": [
    {
      "nct_id": "NCTxxxx",
      "coarse_score": 0.83,
      "matched_chunks": ["..."],
      "matched_queries": ["..."],
      "match_sources": ["bm25", "vector"]
    }
  ],
  "target_count": 1000
}
```

## 阶段 4：规则和聚合精排到 300-500

对粗召回结果做 trial-level 聚合，不调用 LLM。

建议打分：

```text
fine_score =
  0.35 * vector_score
+ 0.25 * bm25_score
+ 0.15 * disease_anchor_score
+ 0.10 * intervention_anchor_score
+ 0.05 * eligibility_anchor_score
+ 0.05 * demographic_compatibility_score
+ 0.05 * status_study_type_score
- 0.20 * obvious_mismatch_penalty
```

各项定义：

```text
vector_score:
  patient query vs trial_card_text embedding similarity

bm25_score:
  query variants vs overview / eligibility / intervention BM25

disease_anchor_score:
  病例核心疾病词是否出现在 title / conditions / summary / eligibility

intervention_anchor_score:
  治疗或场景词是否出现，如 valve replacement / TAVR

eligibility_anchor_score:
  病例关键事实是否出现在 inclusion_text 或 eligibility_text

demographic_compatibility_score:
  年龄、性别是否明显符合

status_study_type_score:
  Interventional、Recruiting、Active 等加权

obvious_mismatch_penalty:
  年龄不符、性别不符、疾病完全不相关、健康志愿者-only 等
```

输出 top 300-500。

## 阶段 5：精召回到 100 以内

目标：从 top 300-500 压到 top 100，仍然不做最终入组判断。

### 方案 A：Cross-Encoder / Reranker

如果有可用 reranker，优先使用：

```text
query:
  patient trial search profile

document:
  trial_card_text + eligibility_text
```

优点：

1. 稳定。
2. 便宜。
3. 易并发。
4. 不需要复杂 prompt。

### 方案 B：轻量 LLM Rerank

如果没有 reranker，可以用 LLM 做轻量筛选。

每批 5-10 个 trial，请 LLM 判断是否值得进入 eligibility 精判，而不是判断最终入组。

输入：

```json
{
  "patient_evidence_card": {},
  "trials": [
    {
      "nct_id": "NCTxxxx",
      "title": "...",
      "conditions": ["..."],
      "interventions": ["..."],
      "brief_summary": "...",
      "gender": "...",
      "minimum_age": "...",
      "maximum_age": "...",
      "inclusion_text": "...",
      "exclusion_text": "..."
    }
  ]
}
```

输出：

```json
{
  "results": [
    {
      "nct_id": "NCTxxxx",
      "relevance": "high",
      "score": 0.86,
      "keep_for_eligibility_review": true,
      "matched_reasons": [
        "trial concerns severe aortic stenosis",
        "trial intervention is valve replacement/TAVR"
      ],
      "obvious_mismatch": [],
      "notes": "Eligibility requires manual review."
    }
  ]
}
```

并发建议：

```text
batch_size: 5-10 trials
parallelism: 3-5
target_output: top 100
```

## 阶段 6：Top Trial Eligibility Judge

只对 top 20-50 做真正 eligibility 判断。

输入每个 trial 的必要字段：

```json
{
  "nct_id": "...",
  "title": "...",
  "status": "...",
  "study_type": "...",
  "conditions": ["..."],
  "interventions": ["..."],
  "gender": "...",
  "minimum_age": "...",
  "maximum_age": "...",
  "brief_summary": "...",
  "inclusion_text": "...",
  "exclusion_text": "..."
}
```

LLM 输出：

```json
{
  "nct_id": "NCTxxxx",
  "trial_relevance": "high | medium | low",
  "eligibility_assessment": "likely_eligible | possible_with_missing_info | likely_excluded | not_relevant",
  "inclusion_matches": [
    {
      "criterion_or_text": "...",
      "patient_evidence": "...",
      "status": "met | unknown | not_met"
    }
  ],
  "exclusion_risks": [
    {
      "criterion_or_text": "...",
      "patient_evidence": "...",
      "status": "present | absent | unknown"
    }
  ],
  "missing_information": [
    "renal function",
    "current medications"
  ],
  "short_reason": "..."
}
```

并发建议：

```text
batch_size: 5-10 trials
parallelism: 3-5
target_input: top 20-50
```

## 阶段 7：最终聚合

最终输出不能只看 LLM 判断，也不能只看召回分。应组合：

1. retrieval score
2. rerank score
3. trial relevance
4. eligibility assessment
5. exclusion risks
6. missing information
7. trial lifecycle status

推荐状态：

```text
candidate_recommended:
  高相关，未发现明确排除项，关键信息基本满足或缺失较少

candidate_possible_missing_info:
  高相关，但存在关键未知项，需要补充信息

candidate_manual_review:
  相关但 eligibility 复杂或冲突不明确

candidate_likely_excluded:
  相关但存在明确排除风险

candidate_not_relevant:
  疾病、治疗场景或人群明显不匹配
```

## 与当前代码的关系

当前已有基础：

1. `agent/trial_vector_kb.py` 已经支持 XML 到 trial record / trial chunk。
2. `agent/tools/trial_vector_retrieval_tools.py` 已经有 query profile、coarse retrieval、trial-level rerank 的雏形。
3. `agent/protocol/criteria_parser.py` 已经有少量规则解析能力。

本方案建议调整方向：

1. 保留现有 XML 浅解析和 chunk 构建。
2. 不把 `criteria_parser.py` 扩展成全库复杂医学规则引擎。
3. 增强 `build_protocol_trial_query_profile`，输出更清晰的 `retrieval_queries` 和 `patient_evidence_card`。
4. 增强 trial-level 聚合精排，把粗召回 1000-2000 压到 300-500。
5. 新增 reranker 或轻量 LLM rerank，把 300-500 压到 100。
6. 新增 top trial LLM eligibility judge，只处理 top 20-50。

## 实施顺序

### Step 1：确认 XML flat schema

产物：

```text
trial_record.jsonl
trial_chunk.jsonl
```

要求：

1. 保留 `eligibility_text` 原文。
2. 保留 `inclusion_text` / `exclusion_text` 的 best-effort 切分。
3. 保留 age/gender/status/study_type/condition/intervention 等字段。

### Step 2：病例侧生成 query profile

产物：

```json
{
  "retrieval_queries": [],
  "primary_condition_terms": [],
  "intervention_or_context_terms": [],
  "clinical_context_terms": [],
  "demographics": {},
  "negative_or_qualifying_facts": [],
  "patient_evidence_card": {}
}
```

### Step 3：多 query 粗召回

目标：

```text
全库 -> top 1000-2000 nct_id
```

实现：

1. BM25 多 query。
2. 向量多 query。
3. RRF 合并。
4. 聚合到 trial。

### Step 4：规则聚合精排

目标：

```text
top 1000-2000 -> top 300-500
```

实现：

1. disease anchor coverage。
2. intervention/context anchor coverage。
3. eligibility text hit。
4. demographic compatibility。
5. status/study_type weighting。
6. obvious mismatch penalty。

### Step 5：Rerank 到 top 100

优先级：

1. 有 reranker 时用 cross-encoder。
2. 没有 reranker 时用轻量 LLM rerank。

目标：

```text
top 300-500 -> top 100
```

### Step 6：LLM eligibility judge

目标：

```text
top 20-50 -> eligibility assessment
```

要求：

1. batch size 5-10。
2. 并发 3-5。
3. 输出严格 JSON。
4. 不要求 LLM 给最终治疗建议，只给 trial relevance、inclusion match、exclusion risk、missing information。

### Step 7：Protocol 聚合输出

输出：

```json
{
  "trial_candidates": [],
  "eligibility_assessments": [],
  "recommended_trials": [],
  "manual_review_trials": [],
  "missing_information": [],
  "retrieval_diagnostics": {}
}
```

## 不做事项

本阶段明确不做：

1. 全库复杂 eligibility criterion schema。
2. 全库 LLM 解析 trial 入排标准。
3. 用规则强行判断复杂治疗线数、器官功能、分子标志物。
4. 把 exclusion_text 命中直接等同于患者排除。
5. 把 LLM rerank 结果直接当最终 eligibility。

## 验收标准

功能验收：

1. 给定一个病例，protocol 能生成多条 trial retrieval query。
2. 能从 trial KB 粗召回 1000-2000 个候选。
3. 能通过非 LLM 聚合精排到 300-500。
4. 能通过 reranker 或轻量 LLM 精排到 100。
5. 能对 top 20-50 批量调用 LLM eligibility judge。
6. 最终输出每个 trial 的 relevance、eligibility assessment、exclusion risks、missing information。

质量验收：

1. top 100 中应明显减少疾病/治疗场景不相关 trial。
2. top 20-50 中应包含可人工解释的 trial supporting text。
3. 明显年龄/性别不符 trial 应被过滤或强降权。
4. LLM 判断必须引用病例证据和 trial eligibility 原文。
5. 不能因为 eligibility parser 误拆而丢失原始文本。

## 简短结论

最合适的路线是：

```text
浅 parser + 多 query 粗召回 + trial-level 精排 + reranker/轻量 LLM 到 top 100 + top trial LLM eligibility judge
```

不要把主要复杂度放在全库规则解析器上。全库阶段只做稳定字段和原文保留；真正复杂的语义判断只在少量 top trial 上做。
