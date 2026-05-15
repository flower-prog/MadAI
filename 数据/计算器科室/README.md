# MedAI 计算器科室数据包

## 简介

这个目录保存的是按科室拆分后的 calculator payload。每个科室目录下有一个 `riskcalcs.json`，例如：

- `传染科/riskcalcs.json`
- `内科/riskcalcs.json`
- `肿瘤科/riskcalcs.json`

这些文件中的顶层 key 是 `PMID`，value 是对应 calculator 的说明对象。

这类文件和 `数据/治疗方案/*/treatment_trials.json` 的定位不同：

- `riskcalcs.json`：面向 calculator 检索、选择和执行
- `treatment_trials.json`：面向 trial / protocol recommendation

## 顶层结构

`riskcalcs.json` 的顶层结构同样是对象，而不是数组：

```json
{
  "16729309": {
    "title": "FIB-4 Index for Liver Fibrosis",
    "purpose": "This calculator is used to predict significant liver fibrosis in patients with HIV/HCV coinfection.",
    "...": "..."
  },
  "16505659": {
    "title": "Candida Score Calculator",
    "purpose": "This calculator is used to decide early antifungal treatment when candidal infection is suspected.",
    "...": "..."
  }
}
```

说明：

- 顶层 key：`PMID`
- 顶层 value：calculator payload

## 当前目录中的字段

当前仓库中的科室切分 `riskcalcs.json` 与主 calculator 库同构，单条记录常见字段如下：

| 字段 | 类型 | 含义 | 备注 |
| --- | --- | --- | --- |
| `title` | `str` | calculator 名称 | 检索和展示时最重要的标题字段 |
| `purpose` | `str` | calculator 的用途 | 描述这个工具用来判断什么问题 |
| `specialty` | `str` | 适用专科 | 例如 `Infectious Disease`、`Cardiology` |
| `eligibility` | `str` | 适用人群或适用条件 | 说明哪些患者适合使用这个工具 |
| `size` | `int` | 文本长度或 payload 大小统计 | 属于辅助元数据，不是临床结论 |
| `computation` | `str` | 计算公式或计算说明 | 常包含公式、说明文字，很多条目中还带 Python 代码块 |
| `interpretation` | `str` | 结果解释 | 说明高分/低分/阈值代表什么 |
| `utility` | `str` | 工具效能或使用价值说明 | 常包含 AUROC、敏感度、特异度等背景说明 |
| `example` | `str` | 示例输入和示例计算 | 常带自然语言例子或代码片段 |
| `citation` | `int` | 引用次数 | 学术影响力相关元数据 |
| `citations_per_year` | `float` | 年均引用次数 | 学术影响力相关元数据 |

## 字段语义补充

### `computation`

这是最关键的执行字段之一。它通常包含：

- 公式描述
- 参数说明
- Python 代码块

运行时参数抽取会直接从 `computation` 和 `example` 中提取参数名，因此这个字段不只是展示文本，还会影响 execution 前的参数解析。

### `interpretation`

这个字段主要用于回答“算出来以后怎么解读”。例如：

- 低风险 / 中风险 / 高风险
- 某个 cutoff 的临床含义
- 是否建议进一步检查

### `utility`

这个字段更偏证据背景，常见内容包括：

- 模型区分度
- 适用队列
- 论文中报告的性能指标

### `example`

这个字段常包含：

- 一个虚构患者例子
- 一组示例输入
- 对应的示例计算代码

它既方便人工阅读，也会被部分参数解析逻辑利用。

## 当前运行时怎么用这些文件

需要特别说明一点：

当前检索层在按科室过滤 calculator 时，主要读取的是各科室 `riskcalcs.json` 的顶层 `PMID` 键集合，用它来构造“当前科室允许的候选 calculator 池”。

也就是说，按科室过滤这一层最直接依赖的是：

- 哪些 PMID 出现在这个科室文件里

而不是逐条深度消费文件中的全部字段值。

相关代码路径：

- `agent/tools/retrieval_tools.py` 中的 `_load_department_pmids(...)`
- `agent/tools/retrieval_tools.py` 中的 `_resolve_department_candidate_pmids(...)`

与此同时，完整 calculator payload 在构建执行上下文时会使用主 calculator 库字段，例如：

- `title`
- `purpose`
- `specialty`
- `eligibility`
- `computation`
- `interpretation`
- `utility`
- `example`

这些字段会被合并进 `calculator_payload`，供检索排序、参数抽取和最终执行使用。

## 和 `治疗方案` 目录的区别

`riskcalcs.json` 更像“calculator 卡片库”，重点是：

- 这是哪个工具
- 适用于谁
- 怎么算
- 算完怎么解释

`treatment_trials.json` 更像“trial evidence payload”，重点是：

- trial 属于哪个科室
- 当前 trial 状态是什么
- 能否当作当前候选或只能当证据
- 对 protocol 层应给出什么动作建议
