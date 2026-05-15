# MedAI 治疗方案数据包

## 简介

这个目录保存的是 MedAI 的治疗方案 / 临床试验 payload，而不是 calculator 语料。

核心目标是把 ClinicalTrials.gov XML 按科室拆分成可直接给 protocol 层消费的 `treatment_trials.json`。每个科室目录下都有一个同名文件，例如：

- `传染科/treatment_trials.json`
- `内科/treatment_trials.json`
- `肿瘤科/treatment_trials.json`

这些文件里的每条记录都以 `NCT ID` 作为顶层 key，value 是一个 trial payload 对象。

当前目录中的主要文件：

- `xml/`：复制过来的原始 ClinicalTrials.gov XML
- `medai_smoke.jsonl`：一条 trial 一行的中间产物，保留原始 XML 内容和补充标签
- `medai_index.tsv`：一条 trial 一行的索引表
- `department_payload_index.tsv`：一条 trial 在一个科室中的落盘索引
- `<科室>/treatment_trials.json`：按科室拆分后的最终 trial payload

## 生成方式

这套数据由 [`scripts/build_treatment_department_payloads.py`](../../scripts/build_treatment_department_payloads.py) 确定性生成，不是逐条由 LLM 写出的自由文本。

生成流程概括如下：

1. 从 `medai_index.tsv` 读取 trial 的科室分配与 XML 路径。
2. 从 `xml/*.xml` 解析 trial 元数据，例如标题、疾病、干预、eligibility、phase。
3. 根据 `overall_status` 做 MedAI 内部状态映射，生成 `status`、`status_reason`、`actions`、`enrollment_open`。
4. 按科室写入 `<科室>/treatment_trials.json`。

## 顶层结构

`treatment_trials.json` 的顶层结构是一个对象，而不是数组：

```json
{
  "NCT00201448": {
    "name": "Evaluation of the Safety and Immune Responses of the Towne Strain of CMV in Seronegative Women",
    "strategy": "trial_match",
    "...": "..."
  },
  "NCT00068809": {
    "name": "4-Day-A-Week Treatment Plan for HIV Infected Adolescents",
    "strategy": "trial_match",
    "...": "..."
  }
}
```

说明：

- 顶层 key：`NCT ID`
- 顶层 value：该 trial 在当前科室下的 payload

## 字段来源分类

为方便阅读，下面的“来源”使用以下缩写：

- `XML`：来自 ClinicalTrials.gov XML
- `INDEX`：来自 `medai_index.tsv`
- `RULE`：根据 `overall_status` 的规则映射生成
- `TEMPLATE`：模板字符串直接拼出
- `ASSEMBLED`：由多个来源拼装

## `treatment_trials.json` 字段映射

| 字段 | 类型 | 含义 | 来源 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 当前条目的显示名 | `ASSEMBLED` | 优先取 `brief_title`，其次 `official_title`，最后回退到 `nct_id` |
| `strategy` | `str` | MedAI 内部策略类型 | `RULE` | 当前脚本固定为 `trial_match` |
| `source` | `str` | 数据来源标识 | `TEMPLATE` | 当前固定为 `clinicaltrials.gov` |
| `status` | `str` | MedAI 内部状态 | `RULE` | 由 `overall_status` 映射为 `trial_matched`、`manual_review` 或 `abandoned` |
| `rationale` | `str` | 短审计说明 | `TEMPLATE` | 说明原始 `overall_status` 如何映射到当前 `status`，以及写入的是哪个科室 |
| `linked_calculators` | `list[str]` | 关联 calculator 列表 | `TEMPLATE` | 当前生成脚本里固定为 `[]`，属于预留字段 |
| `linked_trials` | `list[str]` | 关联 trial 列表 | `ASSEMBLED` | 当前通常只包含当前 `nct_id` |
| `actions` | `list[str]` | 后续动作建议 | `RULE` | 随 `status` 一起生成，用于提醒 protocol 层如何处理 |
| `nct_id` | `str` | ClinicalTrials.gov 试验编号 | `INDEX` | 当前条目的主标识 |
| `department_tag` | `str` | 当前写入的科室标签 | `INDEX` | 例如 `传染科` 文件中的记录，这个字段就是 `传染科` |
| `department_role` | `str` | 当前科室角色 | `ASSEMBLED` | 取值通常为 `primary` 或 `secondary` |
| `department_tags` | `list[str]` | 该 trial 的全部科室标签 | `ASSEMBLED` | 由主科室和次科室列表拼出 |
| `primary_department` | `str` | 主科室 | `INDEX` | 直接来自索引 |
| `secondary_departments` | `list[str]` | 次科室列表 | `INDEX` | 直接来自索引 |
| `overall_status` | `str` | ClinicalTrials.gov 原始试验状态 | `XML` | 例如 `Completed`、`Terminated` |
| `status_reason` | `str` | 对 `status` 的详细解释 | `RULE` | 解释为什么当前 `overall_status` 会被映射成这个 MedAI 状态 |
| `enrollment_open` | `bool` | 是否视为仍开放或准备开放入组 | `RULE` | 只对开放类状态标为 `true` |
| `brief_title` | `str` | 试验短标题 | `XML` | 来自 XML 的 `brief_title` |
| `official_title` | `str` | 试验正式标题 | `XML` | 来自 XML 的 `official_title` |
| `conditions` | `list[str]` | 试验涉及疾病或条件 | `XML` | 来自 XML 的 `condition` 节点集合 |
| `mesh_terms` | `list[str]` | MeSH 术语 | `XML` | 来自 XML 的 `condition_browse/mesh_term` |
| `keywords` | `list[str]` | 关键词 | `XML` | 来自 XML 的 `keyword` 节点集合 |
| `interventions` | `list[str]` | 干预措施 | `XML` | 来自 XML 的 `intervention/intervention_name` |
| `brief_summary` | `str` | 试验摘要 | `XML` | 来自 XML 的 `brief_summary/textblock`，已做空白清洗 |
| `eligibility_text` | `str` | 入排标准全文 | `XML` | 来自 XML 的 `eligibility/criteria/textblock` |
| `study_type` | `str` | 研究类型 | `XML` | 例如 `Interventional` |
| `phase` | `str` | 试验分期 | `XML` | 例如 `Phase 2`、`Phase 3` |
| `primary_purpose` | `str` | 研究主要目的 | `XML` | 例如 `Treatment`、`Prevention` |
| `copied_xml_relpath` | `str` | 包内 XML 相对路径 | `INDEX` | 例如 `xml/NCT00201448.xml` |
| `source_xml_path` | `str` | 上游原始 XML 路径 | `INDEX` | 便于追溯到源数据集位置 |

## 常见字段关系

### `name`、`brief_title`、`official_title`

- `brief_title`：XML 原始短标题
- `official_title`：XML 原始正式标题
- `name`：MedAI 统一展示名，按 `brief_title -> official_title -> nct_id` 回退

### `department_tag`、`department_tags`

- `department_tag`：当前对象所在文件对应的科室
- `department_tags`：该 trial 归属的全部科室

一个 trial 可以同时出现在多个科室文件里，只是每个文件中的 `department_tag` 和 `department_role` 不同。

### `rationale`、`status_reason`

- `rationale`：短审计句，适合日志和快速查看
- `status_reason`：更详细的人类可读解释

### `nct_id`、`linked_trials`

- `nct_id`：当前对象的主键
- `linked_trials`：推荐层面的关联 trial 列表

当前生成脚本里两者通常是一对一对应，但 `linked_trials` 是为未来多 trial 关联保留的结构。

## MedAI 状态映射

当前映射规则如下：

| `overall_status` | `status` | `enrollment_open` | 含义 |
| --- | --- | --- | --- |
| `Recruiting` | `trial_matched` | `true` | 可视为活跃候选 trial |
| `Not yet recruiting` | `trial_matched` | `true` | 即将开放，也视为候选 |
| `Enrolling by invitation` | `trial_matched` | `true` | 定向入组，也视为开放类 |
| `Completed` | `trial_matched` | `false` | 不能视为当前开放 trial，但可保留为 trial-backed evidence |
| `Active, not recruiting` | `trial_matched` | `false` | 同上，作为证据保留 |
| `Terminated` | `abandoned` | `false` | 不应作为当前可推荐 trial |
| `Withdrawn` | `abandoned` | `false` | 不应作为当前可推荐 trial |
| `Suspended` | `abandoned` | `false` | 不应作为当前可推荐 trial |
| `No longer available` | `abandoned` | `false` | 不应作为当前可推荐 trial |
| `Temporarily not available` | `abandoned` | `false` | 不应作为当前可推荐 trial |
| `Unknown status` | `manual_review` | `false` | 状态不清楚，需要人工审查 |

## 和 `计算器科室` 的区别

这个目录的对象更像“trial evidence payload”，包含：

- trial 原始元数据
- 科室落盘信息
- MedAI 自己的状态映射和动作建议

相比之下，`数据/计算器科室/*/riskcalcs.json` 更像 calculator 摘要卡片，重点是工具说明和计算说明，而不是 trial lifecycle 与科室落盘逻辑。
