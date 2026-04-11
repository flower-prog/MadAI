# MedAI

MedAI 是一个面向临床文本的 calculator workflow。它会读取病例或问题文本，检索合适的风险评分工具，抽取参数，并返回结构化执行结果与临床回答摘要。

当前仓库的命令行入口已经接好，最常用的入口有两个：

- `main.py`：通用 CLI，适合直接输入问题或病历文本。
- `scripts/try_single_case_workflow.py`：跑仓库自带的单条样例病历，适合先验证流程是否能通。

## 运行前准备

运行本项目前，至少需要准备好下面几项：

- Python `3.12+`
- 一套可用的 LLM 凭证
- RiskCalc 语料文件

### 1. Python

项目在 `pyproject.toml` 中声明的最低版本是 `Python >= 3.12`。

### 2. LLM 凭证

运行 calculator workflow 时，至少要配置下面两种路径中的一种：

- OpenAI / OpenAI-compatible
- Azure OpenAI

最少需要的变量如下。

#### OpenAI / OpenAI-compatible

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

#### Azure OpenAI

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`

`.env.example` 里已经给出了完整模板。

### 3. 语料文件

workflow 会用到两个核心文件：

- `riskcalcs.json`
- `pmid2info.json`

代码默认会按下面顺序尝试查找：

1. `MedAI\数据\`
2. `MedAI\data\`
3. 同级目录下的 `Clinical-Tool-Learning\mimic_evaluation\`
4. 同级目录下的 `Clinical-Tool-Learning\riskqa_evaluation\`

按当前仓库布局：

- `riskcalcs.json` 可以从 `.\数据\riskcalcs.json` 找到
- `pmid2info.json` 会从同级目录的 `..\Clinical-Tool-Learning\mimic_evaluation\dataset\pmid2info.json` 自动发现

如果你把 `MedAI` 单独拷走运行，没有保留旁边的 `Clinical-Tool-Learning`，就需要在命令里显式传 `--pmid-metadata-path`。

## 推荐安装方式

### 方式一：使用 `uv`（推荐）

仓库里有 `pyproject.toml` 和 `uv.lock`，推荐直接用 `uv`。

```powershell
cd D:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI
uv sync --extra hybrid
```

说明：

- `hybrid` extra 会安装 `faiss-cpu`、`torch`、`transformers`
- CLI 默认的 `--retriever-backend` 是 `hybrid`
- 如果你只装核心依赖，不装 hybrid 依赖，检索阶段大概率跑不通

安装完成后，推荐用 `uv run` 直接执行命令，不需要手动激活虚拟环境。

### 方式二：使用 `venv + pip`

如果你不用 `uv`，也可以手动创建虚拟环境。

```powershell
cd D:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[hybrid]
```

如果你只想按传统方式安装，也可以：

```powershell
pip install -r requirements.txt
```

`requirements.txt` 已经包含 hybrid retrieval 依赖，但 `pip install -e .[hybrid]` 更贴近项目本身的包定义。

## 环境变量配置

先复制模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，至少填一套可用配置。

一个最小 OpenAI 配置示例：

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.4
CLINICAL_TOOL_AGENT_MODEL=gpt-5.4
MEDAI_RETRIEVAL_TOP_K=5
```

项目会在导入阶段自动加载根目录下的 `.env`，所以通常不需要手动 `set` 环境变量。

## 如何运行

### 1. 先确认 CLI 可用

```powershell
uv run python main.py --help
```

如果你已经有可用的本地虚拟环境，也可以直接：

```powershell
.\.venv\Scripts\python.exe .\main.py --help
```

### 2. 以 question 模式运行

适合输入“应该用哪个 calculator”和“会算出什么”的问题。

```powershell
uv run python main.py "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA; which stroke risk calculator should be used and what would it compute?" --mode question --output summary
```

如果希望保留完整 JSON 结果：

```powershell
uv run python main.py "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA; which stroke risk calculator should be used and what would it compute?" --mode question --output json --save-json .\outputs\question_run.json
```

### 3. 以 patient_note 模式运行

适合直接喂整段病历文本，让系统自己做摘要、检索和 calculator 执行。

```powershell
uv run python main.py "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA. Evaluate stroke risk and suggest the most appropriate calculator." --mode patient_note --output summary
```

`patient_note` 模式下还支持：

- `--top-k`：控制召回数量
- `--max-selected-tools`：限制最多执行多少个 calculator
- `--debug`：输出更详细日志

例如：

```powershell
uv run python main.py "78-year-old male with atrial fibrillation, hypertension, diabetes, and prior TIA. Evaluate stroke risk and suggest the most appropriate calculator." --mode patient_note --top-k 10 --max-selected-tools 3 --debug
```

### 4. 跑仓库自带的样例病历

这个脚本更适合第一次验证。默认会读取：

- `.\数据\病例.txt`

并把完整结果保存到：

- `.\outputs\try_single_case_workflow_result.json`

运行命令：

```powershell
uv run python .\scripts\try_single_case_workflow.py --mode patient_note
```

如果想直接在终端打印完整 JSON：

```powershell
uv run python .\scripts\try_single_case_workflow.py --mode patient_note --show-json
```

如果想临时替换病例文本：

```powershell
uv run python .\scripts\try_single_case_workflow.py --case-text "65-year-old with CHF and CKD, evaluate the relevant risk calculator." --mode patient_note
```

### 5. 显式指定语料路径

如果默认路径没有找到，直接把两个文件路径传进去。

```powershell
uv run python main.py "78-year-old male with atrial fibrillation and prior TIA" --mode question `
  --riskcalcs-path .\数据\riskcalcs.json `
  --pmid-metadata-path ..\Clinical-Tool-Learning\mimic_evaluation\dataset\pmid2info.json `
  --output summary
```

## 常用参数

`main.py` 最常用的参数如下：

| 参数 | 作用 |
| --- | --- |
| `text` | 直接传入病例文本或问题文本 |
| `--mode {question,patient_note}` | 选择运行模式 |
| `--top-k` | 控制检索召回数量 |
| `--max-selected-tools` | `patient_note` 模式下最多执行多少个 calculator |
| `--riskcalcs-path` | 手动指定 `riskcalcs.json` |
| `--pmid-metadata-path` | 手动指定 `pmid2info.json` |
| `--output {summary,json}` | 输出摘要或完整 JSON |
| `--save-json` | 把完整结果写入文件 |
| `--debug` | 打开 debug 日志 |

## 测试

一个最轻量的回归测试是：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_baselines.py --help
```

这个测试会验证：

- `main.py` / `agent/workflow.py` 的 CLI 能正常启动
- `run_workflow(...)` 和 `run_clinical_tool_workflow(...)` 能正确构造 state
- `python agent/workflow.py --help` 可以直接运行

## 常见问题

### 1. 提示缺少 `OPENAI_API_KEY` 或 `AZURE_OPENAI_API_KEY`

说明 `.env` 没配好，或者当前 shell 没读到根目录下的 `.env`。先确认：

- `MedAI\.env` 存在
- 至少填了一套可用 provider 配置
- 没有把 key 留空

### 2. 提示找不到 `pmid2info.json`

说明当前目录结构不满足自动发现规则。解决方式有两种：

- 保持 `MedAI` 和 `Clinical-Tool-Learning` 同级
- 在命令中显式传 `--pmid-metadata-path`

### 3. `faiss` / `torch` / `transformers` 导入失败

说明 hybrid retrieval 依赖没有装完整。优先重新执行：

```powershell
uv sync --extra hybrid
```

如果你是 `pip` 安装，重新执行：

```powershell
pip install -e .[hybrid]
```

### 4. 直接运行 `python main.py` 没有看到你想要的输入

`main.py` 的 positional argument 是病例/问题文本。如果你不传文本，CLI 会回退到内置 demo text。实际使用时建议始终显式传入自己的文本。

## 已验证的命令

当前仓库里已经验证过下面这些命令可以正常启动：

```powershell
.\.venv\Scripts\python.exe .\main.py --help
.\.venv\Scripts\python.exe .\scripts\try_single_case_workflow.py --help
.\.venv\Scripts\python.exe .\scripts\run_baselines.py --help
```

如果你只是想先确认环境是否基本可用，优先跑上面这三个。
