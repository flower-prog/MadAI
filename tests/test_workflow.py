from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import workflow as workflow_module


class _FakeGraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def invoke(self, input, config=None):
        self.calls.append({"input": input, "config": config})
        return {
            "status": "completed",
            "echo_request": input["request"],
            "echo_messages": list(input["messages"]),
            "echo_clinical_tool_job": dict(input.get("clinical_tool_job") or {}),
        }


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_graph = workflow_module.graph
        self.fake_graph = _FakeGraph()
        workflow_module.graph = self.fake_graph

    def tearDown(self) -> None:
        workflow_module.graph = self.original_graph

    def test_run_agent_workflow_builds_state_and_invokes_graph(self) -> None:
        result = workflow_module.run_agent_workflow(
            "请处理这个病例",
            patient_case={"structured_inputs": {"age": 78}},
            clinical_tool_job={"mode": "question", "text": "病例文本"},
            tool_registry={"demo": object()},
            messages=[{"role": "user", "content": "病例文本"}],
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        recorded = self.fake_graph.calls[0]
        state = recorded["input"]
        self.assertEqual(state["request"], "请处理这个病例")
        self.assertEqual(state["messages"], [{"role": "user", "content": "病例文本"}])
        self.assertEqual(state["patient_case"], {"structured_inputs": {"age": 78}})
        self.assertEqual(state["clinical_tool_job"], {"mode": "question", "text": "病例文本"})
        self.assertIn("configurable", recorded["config"])
        self.assertIn("thread_id", recorded["config"]["configurable"])

    def test_run_clinical_tool_workflow_wraps_job_payload(self) -> None:
        result = workflow_module.run_clinical_tool_workflow(
            "78-year-old with AF and prior TIA",
            mode="question",
            structured_case={"raw_text": "78-year-old with AF and prior TIA"},
            risk_hints=["stroke risk"],
            retrieval_queries=[{"stage": "question_anchor", "text": "AF stroke risk"}],
            top_k=7,
            max_selected_tools=3,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        state = self.fake_graph.calls[0]["input"]
        clinical_tool_job = state["clinical_tool_job"]
        self.assertEqual(state["request"], "Run MedAI clinical tool workflow in question mode.")
        self.assertEqual(state["messages"], [{"role": "user", "content": "78-year-old with AF and prior TIA"}])
        self.assertEqual(clinical_tool_job["mode"], "question")
        self.assertEqual(clinical_tool_job["text"], "78-year-old with AF and prior TIA")
        self.assertEqual(clinical_tool_job["structured_case"], {"raw_text": "78-year-old with AF and prior TIA"})
        self.assertEqual(clinical_tool_job["risk_hints"], ["stroke risk"])
        self.assertEqual(clinical_tool_job["retrieval_queries"], [{"stage": "question_anchor", "text": "AF stroke risk"}])
        self.assertEqual(clinical_tool_job["top_k"], 7)
        self.assertEqual(clinical_tool_job["max_selected_tools"], 3)

    def test_run_workflow_accepts_case_text_as_unified_entrypoint(self) -> None:
        result = workflow_module.run_workflow(
            case_text="65-year-old with CHF and CKD",
            mode="patient_note",
            structured_case={"raw_text": "65-year-old with CHF and CKD"},
            risk_hints=["heart failure risk"],
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        state = self.fake_graph.calls[0]["input"]
        clinical_tool_job = state["clinical_tool_job"]
        self.assertEqual(state["request"], "Run MedAI clinical tool workflow in patient_note mode.")
        self.assertEqual(state["messages"], [{"role": "user", "content": "65-year-old with CHF and CKD"}])
        self.assertEqual(clinical_tool_job["mode"], "patient_note")
        self.assertEqual(clinical_tool_job["structured_case"], {"raw_text": "65-year-old with CHF and CKD"})
        self.assertEqual(clinical_tool_job["risk_hints"], ["heart failure risk"])

    def test_run_workflow_uses_env_retrieval_top_k_when_not_explicitly_provided(self) -> None:
        with patch.dict(os.environ, {"MEDAI_RETRIEVAL_TOP_K": "6"}, clear=False):
            result = workflow_module.run_workflow(
                case_text="65-year-old with CHF and CKD",
                mode="patient_note",
                structured_case={"raw_text": "65-year-old with CHF and CKD"},
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        state = self.fake_graph.calls[0]["input"]
        clinical_tool_job = state["clinical_tool_job"]
        self.assertEqual(clinical_tool_job["top_k"], 6)

    def test_run_workflow_uses_default_retrieval_top_k_of_thirty_when_not_overridden(self) -> None:
        with patch.dict(os.environ, {"MEDAI_RETRIEVAL_TOP_K": "", "MEDAI_TOP_K": ""}, clear=False):
            result = workflow_module.run_workflow(
                case_text="65-year-old with CHF and CKD",
                mode="patient_note",
                structured_case={"raw_text": "65-year-old with CHF and CKD"},
            )

        self.assertEqual(result["status"], "completed")
        state = self.fake_graph.calls[0]["input"]
        clinical_tool_job = state["clinical_tool_job"]
        self.assertEqual(clinical_tool_job["top_k"], 30)

    def test_run_workflow_rejects_string_risk_hints(self) -> None:
        with self.assertRaisesRegex(ValueError, "risk_hints must be a JSON array / list"):
            workflow_module.run_workflow(
                case_text="65-year-old with CHF and CKD",
                mode="patient_note",
                risk_hints="heart failure risk",
            )

    def test_run_workflow_rejects_non_mapping_retrieval_queries(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieval_queries\\[0\\] must be a JSON object / dict"):
            workflow_module.run_workflow(
                case_text="65-year-old with CHF and CKD",
                mode="patient_note",
                retrieval_queries=["heart failure risk"],
            )

    def test_run_workflow_accepts_existing_state_payload(self) -> None:
        result = workflow_module.run_workflow(
            state={
                "request": "继续这个病例流程",
                "messages": [{"role": "user", "content": "已有状态"}],
                "next_agent": "protocol",
                "tool_registry": {"existing": "value"},
            },
            tool_registry={"injected": "runner"},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        recorded = self.fake_graph.calls[0]
        state = recorded["input"]
        self.assertEqual(state["request"], "继续这个病例流程")
        self.assertEqual(state["next_agent"], "protocol")
        self.assertEqual(state["tool_registry"]["existing"], "value")
        self.assertEqual(state["tool_registry"]["injected"], "runner")

    def test_run_agent_workflow_accepts_state_payload(self) -> None:
        result = workflow_module.run_agent_workflow(
            state={
                "request": "从中间状态继续",
                "messages": [{"role": "user", "content": "继续"}],
                "next_agent": "reporter",
            }
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        state = self.fake_graph.calls[0]["input"]
        self.assertEqual(state["request"], "从中间状态继续")
        self.assertEqual(state["next_agent"], "reporter")

    def test_run_agent_workflow_resumes_from_snapshot_and_reuses_thread_id(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / "workflow_resume_snapshot"
        temp_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = temp_root / "resume_snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "snapshot_version": 1,
                    "thread_id": "resume-thread-001",
                    "sequence": 7,
                    "phase": "node",
                    "agent_name": "clinical_assisstment",
                    "state": {
                        "request": "从快照恢复",
                        "messages": [{"role": "user", "content": "已有病例"}],
                        "next_agent": "protocol",
                        "tool_registry": {"registered_tools": ["demo_tool"]},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        snapshot_dir = temp_root / "snapshots"
        result = workflow_module.run_agent_workflow(
            resume_snapshot=str(snapshot_path),
            resume_mode="continue",
            snapshot_dir=str(snapshot_dir),
            tool_registry={"injected": "runner"},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        recorded = self.fake_graph.calls[0]
        state = recorded["input"]
        self.assertEqual(state["request"], "从快照恢复")
        self.assertEqual(state["next_agent"], "protocol")
        self.assertEqual(state["tool_registry"], {"injected": "runner"})
        self.assertEqual(recorded["config"]["configurable"]["thread_id"], "resume-thread-001")
        self.assertEqual(
            recorded["config"]["configurable"]["snapshot_dir"],
            str(snapshot_dir.resolve()),
        )

    def test_run_agent_workflow_restart_mode_resets_iteration_state(self) -> None:
        temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / "workflow_restart_snapshot"
        temp_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = temp_root / "restart_snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "snapshot_version": 1,
                    "thread_id": "resume-thread-002",
                    "sequence": 9,
                    "phase": "final",
                    "agent_name": "reporter",
                    "state": {
                        "request": "重跑这个病例",
                        "next_agent": "FINISH",
                        "status": "failed",
                        "errors": ["previous failure"],
                        "reporter_attempts": 3,
                        "review_passed": True,
                        "abandoned": True,
                        "final_output": {"old": "result"},
                        "plan": [
                            {
                                "step_id": 1,
                                "agent_name": "orchestrator",
                                "title": "old step",
                                "description": "old description",
                                "status": "completed",
                                "result": "done",
                            }
                        ],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result = workflow_module.run_agent_workflow(
            resume_snapshot=str(snapshot_path),
            resume_mode="restart",
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.fake_graph.calls), 1)

        state = self.fake_graph.calls[0]["input"]
        self.assertEqual(state["request"], "重跑这个病例")
        self.assertEqual(state["next_agent"], "orchestrator")
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["reporter_attempts"], 0)
        self.assertFalse(state["review_passed"])
        self.assertFalse(state["abandoned"])
        self.assertEqual(state["errors"], [])
        self.assertEqual(state["final_output"], {})
        self.assertEqual(state["plan"][0].status, "pending")
        self.assertIsNone(state["plan"][0].result)

    def test_workflow_script_supports_direct_execution(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "agent" / "workflow.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("Run the MedAI workflow", completed.stdout)


if __name__ == "__main__":
    unittest.main()
