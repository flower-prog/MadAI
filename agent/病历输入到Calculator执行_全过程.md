# 病历输入到 Calculator 执行全过程

本文档描述当前版本的 MedAI 流程，不再使用旧的“四类标签”或 `patient_strata / calculator_categories / protocol_targets` 链路。

## 当前主链路

`orchestrator -> clinical_assisstment -> protocol -> reporter`

- `orchestrator`
  负责初始化工作流、确定 `department_tags`、写入 workflow contract，并把任务交给下游。
- `clinical_assisstment`
  负责把病例整理为 `structured_case`，提炼 `problem_list`、生成 `case_summary` 和 `progressive_queries`，然后调度 calculator 子 agent。
- `calculator`
  作为 `clinical_assisstment` 的子 agent，负责 retrieval、eligibility 判断和实际 calculator 执行。
- `protocol`
  基于 `structured_case` 与 `calculation_bundle` 生成治疗建议、临床试验方向或 fallback 建议。
- `reporter`
  汇总医生可读报告，检查结果是否成立；如果不成立，返回 blocking feedback 触发下一轮，最多三轮。

## 入口状态

图入口会把请求规范为 `GraphState`，核心字段包括：

- `request`
- `patient_case`
- `clinical_tool_job`
- `department_tags`
- `structured_case_json`
- `problem_list`
- `retrieval_queries`
- `calculation_tasks`
- `calculation_results`
- `treatment_recommendations`

`clinical_tool_job` 当前只依赖：

- `mode`
- `text`
- `case_summary`
- `risk_hints`
- `retrieval_queries`
- `top_k`
- `risk_count`
- `max_selected_tools`
- `retriever_backend`
- `riskcalcs_path`
- `pmid_metadata_path`
- `llm_model`
- `temperature`
- `max_rounds`

## clinical_assisstment 节点做什么

这个节点是上半程的核心，主要做五件事：

1. 如果是 `patient_note` 模式且还没有 `risk_hints`，先调用 `generate_risk_hints(...)`
2. 从病例文本或结构化输入中派生 `problem_list`
3. 调用 `build_case_summary(...)` 生成压缩版病例摘要
4. 调用 `_build_query_set(...)` 生成 `progressive_queries`
5. 把 `case_summary` 和 `retrieval_queries` 回填到 `clinical_tool_job`，再交给 `ClinicalToolAgent`

这里的关键点是：

- 现在不再给病例打四类标签
- 也不再把分类结果写回 `clinical_tool_job`
- 下游检索主要依赖原始病例文本、`case_summary` 和 staged queries

## progressive_queries 的作用

系统不会只发出一条检索语句，而是按模式构造一组 staged queries。

### patient_note 模式

会优先生成：

- `case_summary_dense`
- `problem_anchor_*`
- `risk_hint_*`

### question / baseline 模式

会优先保留：

- 原始问题文本
- 压缩后的 `case_summary`
- 主问题锚点 `problem_anchor`

这样做的目的不是“多加一个列表好看”，而是把一个长而杂乱的病例拆成几条职责明确的检索输入：

- 一条保留全局语义
- 一条保留主要问题
- 若干条保留风险短句

这样比只扔整段病历去检索更稳定，也更容易调试。

## ClinicalToolAgent 做什么

`ClinicalToolAgent` 收到 `clinical_tool_job` 以后，会继续执行：

1. 用 `case_summary` 或原始文本构造候选 calculator 池
2. 对 `progressive_queries` 逐条检索并融合排序
3. 对候选 calculator 做 eligibility 判断
4. 对能执行的 calculator 进行参数提取和执行
5. 把执行 trace、候选列表和结果返回给 graph

## protocol 节点做什么

`protocol` 不再依赖任何旧分类字段，只看：

- `structured_case`
- `calculation_results`
- `calculator_matches`

它负责输出：

- 明确治疗建议
- 相似病例 fallback
- 直接保守建议

## reporter 节点做什么

`reporter` 会检查：

- `structured_case` 是否存在
- `calculation_bundle` 是否完整
- `protocol` 是否给出了建议
- 当前轮是否存在计算错误或阻断问题

如果不通过，就把 blocking feedback 返回 `orchestrator`，最多三轮。

## 为什么新增这些列表

如果你问的是 `problem_list`、`progressive_queries` 这些“增设列表”的意义，答案很直接：

- `problem_list` 的意义是把原始病例切成几个可检索、可追踪的问题片段
- `progressive_queries` 的意义是把“单条大查询”改成“多条分工明确的小查询”
- 这样做的价值在于提升 retrieval 稳定性、降低噪声、方便 trace 和调试

如果某个列表只是为了凑分类层、但不直接服务检索、执行或报告，那它就没有保留必要。这也是这次删除四类标签链路的原因。
