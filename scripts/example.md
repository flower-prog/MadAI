# `scripts/` 目录脚本用法

以下命令默认都在项目根目录 `MadAI` 下执行：

```bash
cd /home/yuanzy/MadAI
```

推荐统一用 `uv run python` 运行脚本。

## 运行前准备

1. 安装依赖。

```bash
uv sync --extra hybrid
```

2. 配置根目录 `.env`。

- 工作流类脚本会自动读取 `MedAI/.env`
- `run_baselines.py` 至少需要可用的 `OPENAI_API_KEY`
- 如果走 Azure，还需要 `OPENAI_ENDPOINT`，模型名可以放在 `OPENAI_MODEL` / `BASIC_MODEL` / `CLINICAL_TOOL_AGENT_MODEL` / `CODING_MODEL`

3. 确认语料路径可用。

- 默认数据集：`数据/riskqa.json`
- 默认病例文件：`数据/病例.txt`
- 默认计算器语料：`数据/riskcalcs.json`
- `pmid2info.json` 如果不能被自动发现，需要手动传 `--pmid-metadata-path`

## 脚本之间的常见关系

最常见的一条链路是：

1. 用 `try_single_case_workflow.py` 先跑通单条样例
2. 用 `run_riskqa_parallel.py` 批量跑 RiskQA
3. 用 `evaluate.py` 评估 `run_riskqa_parallel.py` 产出的答案文件
4. 用 `run_baselines.py` 跑直接问模型的 baseline，和工作流结果对比

`build_treatment_department_payloads.py` 是另一条独立的数据构建脚本，和 RiskQA 评测链路无关。

---

## 1. `try_single_case_workflow.py`

用途：运行单条病例或单条问题，适合做冒烟测试、调试参数、看工作流输出结构。

默认输入 / 输出：

- 输入：`数据/病例.txt`
- 输出：`outputs/try_single_case_workflow_result.json`

最常用命令：

```bash
uv run python scripts/try_single_case_workflow.py
```

按问题模式运行：

```bash
uv run python scripts/try_single_case_workflow.py \
  --case-text "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA; which stroke risk calculator should be used and what would it compute?" \
  --mode question
```

打印完整 JSON：

```bash
uv run python scripts/try_single_case_workflow.py --show-json
```

显式指定语料路径：

```bash
uv run python scripts/try_single_case_workflow.py \
  --riskcalcs-path 数据/riskcalcs.json \
  --pmid-metadata-path /path/to/pmid2info.json
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--case-text` | 直接在命令行传病例/问题文本 |
| `--case-file` | 从文件读取文本；未传 `--case-text` 时使用 |
| `--mode` | `patient_note` 或 `question`，默认 `patient_note` |
| `--top-k` | 检索召回数量，未传则读取环境变量或默认值 |
| `--riskcalcs-path` | 手动指定 `riskcalcs.json` |
| `--pmid-metadata-path` | 手动指定 `pmid2info.json` |
| `--output` | 保存完整结果 JSON 的路径 |
| `--show-json` | 终端打印完整结果 JSON，而不是摘要 |
| `--debug` | 打开调试日志 |

适用场景：

- 先验证环境、模型、检索链路能否跑通
- 调 `--top-k`、病例文本、语料路径
- 观察 `run_workflow()` 的完整输出结构

---

## 2. `run_riskqa_parallel.py`

用途：并发跑完整个 RiskQA 数据集，把每道题的结果写成文本记录块，供后续评估。

默认输入 / 输出：

- 输入：`数据/riskqa.json`
- 输出：`outputs/riskqa/riskqa_answers.txt`

最常用命令：

```bash
uv run python scripts/run_riskqa_parallel.py --limit 100 --workers 2 --retriever-backend hybrid
```

只跑一小段索引用于调试：

```bash
uv run python scripts/run_riskqa_parallel.py \
  --start-index 100 \
  --end-index 120 \
  --workers 4
```

切换模型与输出文件：

```bash
uv run python scripts/run_riskqa_parallel.py \
  --limit 50 \
  --llm-model gpt-5.4 \
  --output outputs/riskqa/riskqa_answers_gpt54.txt
```

显式指定语料路径：

```bash
uv run python scripts/run_riskqa_parallel.py \
  --riskcalcs-path 数据/riskcalcs.json \
  --pmid-metadata-path /path/to/pmid2info.json
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--dataset` | RiskQA 数据集路径，默认 `数据/riskqa.json` |
| `--output` | 输出文本路径，默认 `outputs/riskqa/riskqa_answers.txt` |
| `--workers` | 并发线程数，默认 `4` |
| `--start-index` | 从哪个数据集索引开始 |
| `--end-index` | 跑到哪个索引之前结束 |
| `--limit` | 切片后最多处理多少题 |
| `--top-k` | 检索 top-k 覆盖默认值 |
| `--llm-model` | 临时覆盖工作流里的模型 |
| `--retriever-backend` | `keyword` / `vector` / `hybrid`，默认 `hybrid` |
| `--riskcalcs-path` | 手动指定 `riskcalcs.json` |
| `--pmid-metadata-path` | 手动指定 `pmid2info.json` |
| `--debug` | 打开工作流调试日志 |

输出文件格式说明：

- 结果不是纯 JSON 数组，而是一个 `.txt`
- 每道题会写成一个 `=== RISKQA RESULT ... ===` 包裹的 JSON block
- 这个格式就是给 `scripts/evaluate.py` 和 `scripts/riskqa_support.py` 解析用的

适用场景：

- 跑完整个 RiskQA 基准
- 跑数据切片做局部验证
- 生成后续评估所需的答案文件

---

## 3. `evaluate.py`

用途：读取 `run_riskqa_parallel.py` 生成的答案文本，和 `riskqa.json` 标准答案对比，输出准确率报告。

默认输入 / 输出：

- 输入答案：`outputs/riskqa/riskqa_answers.txt`
- 输入数据集：`数据/riskqa.json`
- 默认报告：`<answers>.evaluation.txt`

最常用命令：

```bash
uv run python scripts/evaluate.py
```

评估自定义结果文件：

```bash
uv run python scripts/evaluate.py \
  --answers outputs/riskqa/riskqa_answers_gpt54.txt
```

自定义报告路径和错误样本数量：

```bash
uv run python scripts/evaluate.py \
  --answers outputs/riskqa/riskqa_answers.txt \
  --report-path outputs/riskqa/riskqa_eval_report.txt \
  --wrong-limit 50
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--answers` | 待评估的答案文本文件 |
| `--dataset` | 标准数据集路径 |
| `--report-path` | 报告输出路径；未传则自动生成为 `<answers>.evaluation.txt` |
| `--wrong-limit` | 报告里最多列出多少条错误样本，默认 `20` |

报告内容包括：

- 总记录数
- 可比较样本数
- 正确数 / 错误数
- 准确率
- 部分错误样本列表

---

## 4. `run_baselines.py`

用途：不走 MedAI workflow，直接把 RiskQA 题目喂给大模型回答，作为 baseline。

默认输入 / 输出目录：

- 输入：`数据/riskqa.json`
- 输出目录：`outputs/riskqa`

这个脚本会在输出目录下生成 4 份文件：

- `<model_slug>_riskqa_baseline_results.json`：缓存文件，已跑过的题会记录在这里
- `<model_slug>_riskqa_baseline_answers.json`：兼容旧打分脚本的答案文件
- `<model_slug>_riskqa_baseline_comparison.json`：逐题对比结果
- `<model_slug>_riskqa_baseline_summary.json`：汇总统计

最常用命令：

```bash
uv run python scripts/run_baselines.py
```

显式指定模型：

```bash
uv run python scripts/run_baselines.py gpt-5.4
```

只跑前 100 题：

```bash
uv run python scripts/run_baselines.py \
  --max-questions 100 \
  --workers 4
```

从指定索引继续跑：

```bash
uv run python scripts/run_baselines.py \
  --start-index 200 \
  --max-questions 100
```

忽略缓存重跑：

```bash
uv run python scripts/run_baselines.py --overwrite
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `model` | 位置参数，可直接传模型名；不传则从 `.env` 读取 |
| `--dataset` | RiskQA 数据集路径 |
| `--output-dir` | 输出目录 |
| `--max-questions` | 最多跑多少题 |
| `--start-index` | 从哪个索引开始 |
| `--sleep-seconds` | 每次请求后 sleep，便于限流 |
| `--workers` | 并发线程数，默认 `4` |
| `--overwrite` | 忽略缓存，强制重跑 |

环境变量说明：

- 直接 OpenAI / 兼容接口：至少要有 `OPENAI_API_KEY`
- 如果设置了 `OPENAI_ENDPOINT`，脚本会按 Azure OpenAI 路径初始化客户端
- 模型名优先级：位置参数 `model` > `.env` 中的 `OPENAI_MODEL` / `BASIC_MODEL` / `CLINICAL_TOOL_AGENT_MODEL` / `CODING_MODEL`

补充说明：

- 这个脚本自带缓存机制，同一个输出目录下再次运行时，会跳过已存在结果
- 如果要做严格复现实验，建议加 `--overwrite`

---

## 5. `build_treatment_department_payloads.py`

用途：读取 `数据/治疗方案` 下已有的临床试验 XML 和索引文件，按科室生成 `treatment_trials.json` 负载，并同步更新索引与清单文件。

默认输入目录：

- `数据/治疗方案`

最常用命令：

```bash
uv run python scripts/build_treatment_department_payloads.py
```

指定其他治疗方案目录：

```bash
uv run python scripts/build_treatment_department_payloads.py \
  --package-dir /path/to/treatment_package
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--package-dir` | 治疗方案包目录，默认 `数据/治疗方案` |

脚本会读取或依赖这些文件：

- `medai_index.tsv`
- `medai_smoke.jsonl`（不存在时可为空，但有的话会被重写）
- `xml/*.xml`

脚本会重写或生成这些文件：

- `medai_smoke.jsonl`
- `medai_index.tsv`
- `department_payload_index.tsv`
- `manifest.json`
- `README.md`
- 各科室目录下的 `treatment_trials.json`

适用场景：

- 重新整理治疗方案分科负载
- 从 XML 元数据重新生成科室级试验 payload
- 更新 `manifest.json` 里的分布统计信息

注意：

- 这个脚本会原地覆盖生成文件，运行前最好确认目录内容是否需要保留
- 它不依赖 LLM，主要是 XML 解析和本地文件重建

---

## 6. `riskqa_support.py`

用途：内部辅助模块，不是命令行入口。

主要能力：

- 加载 `riskqa.json`
- 组装题目文本
- 解析 `run_riskqa_parallel.py` 生成的结果块
- 从模型回答中提取选项字母

典型用法：

```python
from scripts.riskqa_support import load_dataset, load_record_blocks
```

适合在 notebook、临时脚本、测试代码里复用，不建议直接 `python scripts/riskqa_support.py`。

---

## 7. `__init__.py`

用途：仅用于把 `scripts` 目录标记成 Python 包，没有单独运行价值。

---

## 常见命令速查

单条病例冒烟测试：

```bash
uv run python scripts/try_single_case_workflow.py
```

批量跑 RiskQA：

```bash
uv run python scripts/run_riskqa_parallel.py --limit 100 --workers 2 --retriever-backend hybrid
```

评估批量结果：

```bash
uv run python scripts/evaluate.py --answers outputs/riskqa/riskqa_answers.txt
```

跑直接问模型的 baseline：

```bash
uv run python scripts/run_baselines.py gpt-5.4 --max-questions 100
```

重建治疗方案分科 payload：

```bash
uv run python scripts/build_treatment_department_payloads.py
```
