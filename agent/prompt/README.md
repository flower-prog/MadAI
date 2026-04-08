# MedAI Agent Prompts

本目录保存 MedAI 的 XML 风格 system prompts。

核心运行链路：

- `orchestrator -> clinical_assisstment -> protocol -> reporter`

当前实际使用的 prompt：

- `orchestrator.md`：只负责工作合同、路由与回退入口
- `clinical_assisstment.md`：病例 JSON 化、检索查询、计算任务规划
- `calculator.md`：`clinical_assisstment` 的子 agent，负责具体计算与单参数估计后的计算
- `protocol.md`：治疗方案 / 临床试验 / 相似病例 fallback / 保守建议
- `reporter.md`：医生报告 + 合理性审查 + 最多三轮迭代控制

说明：

- 当前 graph 只有 4 个顶层 agent：`orchestrator`、`clinical_assisstment`、`protocol`、`reporter`
- 另有 1 个子 agent：`calculator`
- 旧兼容 prompt 已删除；相关语义已并入当前主 prompt 中

补充说明：

- prompt 文件主体统一采用 XML 结构，便于角色、职责、规则和输出契约显式表达。
- runtime context 由 `agent.prompt.loader.render_agent_prompt(...)` 追加到 prompt 末尾，并同样包装为 XML 风格块。
