# Clinical Tool Agent Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the tool-calling agent behavior from `Clinical-Tool-Learning` into the `MedAI` framework so MedAI can retrieve, select, and execute clinical calculators inside its own workflow.

**Architecture:** Add a first-class clinical tool agent module under `MedAI/agent/tools`, wire it into the existing `agent/graph` scaffold via a typed request payload, and expose a usable workflow entrypoint in `agent/workflow.py`. Keep the source calculator corpus external by default so MedAI can operate on the sibling `Clinical-Tool-Learning` assets without copying large JSON files.

**Tech Stack:** Python, dataclasses, optional OpenAI/Azure OpenAI chat client, optional MedCPT + FAISS retrieval, keyword fallback retrieval, `unittest`.

---

### Task 1: Define the MedAI-side migration surface

**Files:**
- Modify: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\graph\types.py`
- Modify: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\graph\nodes.py`
- Modify: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\graph\builder.py`

**Step 1: Add a typed clinical tool job payload**

Extend the graph state with a `ClinicalToolJob` dataclass that can carry:
- mode (`patient_note` or `question`)
- input text
- optional risk hints
- calculator corpus paths
- retrieval backend
- LLM model override
- execution loop parameters

**Step 2: Update state coercion**

Teach `ensure_state()` to accept `clinical_tool_job` dictionaries and normalize them into the dataclass.

**Step 3: Update graph semantics**

Keep the old placeholder path as fallback, but allow the calculator node to switch into real clinical tool mode whenever `clinical_tool_job` is present.

### Task 2: Implement the migrated clinical tool agent

**Files:**
- Create: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\tools\__init__.py`
- Create: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\tools\code_execution.py`
- Create: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\tools\clinical_tool_agent.py`

**Step 1: Add Python code execution helpers**

Implement reusable helpers to:
- extract fenced Python blocks
- execute generated code
- capture stdout/stderr/tracebacks

**Step 2: Add calculator corpus loaders**

Implement a loader that reads:
- `riskcalcs.json`
- `pmid2info.json`

and formats calculator text for LLM prompts.

**Step 3: Add retrieval**

Implement:
- keyword retrieval as the default zero-download backend
- optional MedCPT + FAISS retrieval when dependencies are available

**Step 4: Add selection and execution loops**

Mirror AgentMD behavior:
- patient-note mode: risk generation -> candidate retrieval -> eligibility filtering -> calculator execution
- question mode: candidate retrieval -> single-tool selection -> calculator execution

### Task 3: Wire the tool agent into MedAI workflow

**Files:**
- Modify: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\agent\workflow.py`

**Step 1: Replace broken `src.*` imports**

Make `workflow.py` use local `agent.graph` modules instead of missing `src.*` paths.

**Step 2: Expose a MedAI-native entrypoint**

Support both:
- generic workflow execution
- direct clinical tool runs via `clinical_tool_job`

**Step 3: Preserve simple defaults**

Avoid requiring the unfinished `src.tools` ecosystem so the migrated agent is independently runnable.

### Task 4: Validate with tests

**Files:**
- Create: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\tests\test_clinical_tool_agent.py`

**Step 1: Test utility behavior**

Add tests for:
- Python code block extraction
- execution output capture
- keyword retrieval ordering

**Step 2: Test graph integration**

Inject a fake clinical tool runner through `tool_registry` and confirm:
- planner builds the clinical-tool plan
- calculator stores migrated results
- tester marks the run completed

### Task 5: Document usage

**Files:**
- Modify: `d:\虚拟C盘\ibp\journal_club\2026.3.7\MedAI\README.md`

**Step 1: Add migration note**

Document that MedAI now supports the AgentMD-style clinical calculator tool agent.

**Step 2: Add example invocation**

Show how to call `run_agent_workflow(..., clinical_tool_job=...)` with source corpus paths that point to the sibling `Clinical-Tool-Learning` checkout.
