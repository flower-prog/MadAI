from __future__ import annotations

import unittest

from agent.graph.nodes import clinical_assisstment_node, orchestrator_node, protocol_node, reporter_node
from agent.graph.types import ClinicalToolJob, GraphState, RetrievalQuery, ensure_state
from agent.graph.types import TreatmentRecommendation


class _FakeChatClient:
    def __init__(self, response: str | list[str]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, *, model=None, temperature=0.0) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
            }
        )
        if isinstance(self.response, list):
            index = min(len(self.calls) - 1, len(self.response) - 1)
            return self.response[index]
        return self.response


class _SelectionAwareClinicalToolRunner:
    def __init__(self) -> None:
        self.plan_calls: list[ClinicalToolJob] = []
        self.run_calls: list[ClinicalToolJob] = []

    def plan_selection(self, job: ClinicalToolJob) -> dict[str, object]:
        self.plan_calls.append(job)
        selected_tool = {
            "pmid": "123456",
            "title": "Demo Stroke Risk Calculator",
            "purpose": "Estimate stroke risk for the current case.",
            "parameter_names": ["age", "stroke_history"],
            "reason": "Selected by the parent clinical_assisstment node.",
        }
        candidate = dict(selected_tool)
        candidate["query_text"] = "atrial fibrillation prior stroke age"
        return {
            "mode": job.mode,
            "risk_hints": list(job.risk_hints),
            "retrieval_batches": [{"channel": "coarse", "result_count": 1}],
            "retrieved_tools": [dict(selected_tool)],
            "candidate_ranking": [dict(selected_tool)],
            "selection_candidates": [candidate],
            "bm25_raw_top5": [{"pmid": "123456", "channel": "bm25", "rank": 1}],
            "vector_raw_top5": [{"pmid": "123456", "channel": "vector", "rank": 1}],
            "recommended_pmids": ["123456"],
            "selected_tool": dict(selected_tool),
            "selection_mode": "model_selected",
            "dispatch_query_text": "atrial fibrillation prior stroke age",
            "dispatch_query_source": "selected_candidate.query_text",
        }

    def run(self, job: ClinicalToolJob) -> dict[str, object]:
        self.run_calls.append(job)
        return {
            "mode": job.mode,
            "trace": {"tool_calls": []},
            "selected_tool": dict(job.selected_tool or {}),
            "retrieved_tools": list(job.selection_context.get("retrieved_tools") or []),
            "candidate_ranking": list(job.selection_context.get("candidate_ranking") or []),
            "executions": [
                {
                    "pmid": job.selected_tool_pmid,
                    "status": "completed",
                    "result": 3,
                    "final_text": "Demo score is 3.",
                }
            ]
            if job.selected_tool_pmid
            else [],
            "execution": {
                "pmid": job.selected_tool_pmid,
                "status": "completed",
                "result": 3,
                "final_text": "Demo score is 3.",
            }
            if job.selected_tool_pmid
            else {"status": "skipped", "final_text": "No calculator selected."},
            "selection_mode": str(job.selection_context.get("selection_mode") or ""),
        }


class OrchestratorStateTests(unittest.TestCase):
    def test_ensure_state_loads_orchestrator_result_and_department(self) -> None:
        state = ensure_state(
            {
                "request": "整理病例",
                "department": "内科",
                "orchestrator_result": {"department_tags": ["内科", "传染科"]},
            }
        )

        self.assertEqual(state.department, "内科")
        self.assertEqual(state.orchestrator_result["department_tags"], ["内科", "传染科"])

    def test_orchestrator_node_writes_result_to_main_state(self) -> None:
        state = GraphState(
            request="整理病例",
            orchestrator_result={"department_tags": ["精神心理科", "儿科"]},
        )

        result = orchestrator_node(state)

        self.assertEqual(result.department, "精神心理科")
        self.assertEqual(result.department_tags, ["精神心理科", "儿科"])
        self.assertEqual(result.orchestrator_result["department"], "精神心理科")
        self.assertEqual(result.orchestrator_result["department_tags"], ["精神心理科", "儿科"])

    def test_orchestrator_node_classifies_department_via_chat_client_when_missing(self) -> None:
        chat_client = _FakeChatClient(
            """
            {
              "structured_case": {
                "raw_request": "请整理这个病例",
                "raw_text": "患者发热、咳嗽 3 天",
                "case_summary": "发热咳嗽病例",
                "problem_list": ["发热", "咳嗽"],
                "known_facts": ["发热 3 天"],
                "missing_information": ["年龄"]
              },
              "department": "内科",
              "department_tags": ["内科"],
              "mode": "baseline",
              "notes": ["demo"]
            }
            """
        )
        state = GraphState(
            request="请整理这个病例",
            messages=[{"role": "user", "content": "患者发热、咳嗽 3 天"}],
            tool_registry={"orchestrator_chat_client": chat_client},
        )

        result = orchestrator_node(state)

        self.assertEqual(result.department, "内科")
        self.assertEqual(result.department_tags, ["内科"])
        self.assertEqual(result.orchestrator_result["department"], "内科")
        self.assertEqual(result.structured_case_json["case_summary"], "发热咳嗽病例")
        self.assertEqual(len(chat_client.calls), 1)

    def test_orchestrator_retries_until_valid_department_response(self) -> None:
        chat_client = _FakeChatClient(
            [
                "",
                "not json",
                """
                {
                  "department": "内科",
                  "department_tags": ["内科"]
                }
                """,
            ]
        )
        state = GraphState(
            request="患者发热、咳嗽 3 天",
            messages=[{"role": "user", "content": "患者发热、咳嗽 3 天"}],
            tool_registry={"orchestrator_chat_client": chat_client},
        )

        result = orchestrator_node(state)

        self.assertEqual(result.department, "内科")
        self.assertEqual(result.department_tags, ["内科"])
        self.assertEqual(result.orchestrator_result["department"], "内科")
        self.assertEqual(len(chat_client.calls), 3)

    def test_graph_state_no_longer_has_intermediate_field(self) -> None:
        state = ensure_state({"request": "整理病例"})
        self.assertFalse(hasattr(state, "intermediate"))

    def test_clinical_assisstment_writes_department_into_structured_case(self) -> None:
        state = GraphState(
            request="患者发热、咳嗽 3 天",
            department="内科",
            department_tags=["内科"],
        )

        result = clinical_assisstment_node(state)

        self.assertEqual(result.structured_case_json["department"], "内科")
        self.assertEqual(result.structured_case_json["department_tags"], ["内科"])

    def test_clinical_assisstment_preserves_seeded_query_plan(self) -> None:
        state = GraphState(
            request="患者房颤伴TIA病史，评估卒中风险。",
            department="内科",
            department_tags=["内科"],
            clinical_tool_job=ClinicalToolJob(
                mode="patient_note",
                text="患者房颤伴TIA病史，评估卒中风险。",
                case_summary="房颤伴TIA病史，卒中风险评估",
                structured_case={
                    "raw_text": "患者房颤伴TIA病史，评估卒中风险。",
                    "problem_list": ["房颤伴TIA病史", "既往高血压和糖尿病", "当前未使用华法林"],
                    "known_facts": ["既往TIA", "高血压", "糖尿病"],
                },
                risk_hints=["房颤伴TIA病史"],
                retrieval_queries=[
                    RetrievalQuery(
                        stage="case_summary_dense",
                        text="房颤伴TIA病史，卒中风险评估",
                        intent="case_summary_dense",
                        priority=1,
                    ),
                    RetrievalQuery(
                        stage="problem_anchor_1",
                        text="房颤伴TIA病史",
                        intent="problem_anchor",
                        priority=2,
                    ),
                ],
            ),
            tool_registry={
                "clinical_tool_agent": lambda job: {
                    "trace": {"tool_calls": []},
                    "retrieved_tools": [],
                    "candidate_ranking": [],
                }
            },
        )

        result = clinical_assisstment_node(state)

        self.assertEqual(result.problem_list, ["房颤伴TIA病史", "既往高血压和糖尿病", "当前未使用华法林"])
        self.assertEqual(result.structured_case_json["case_summary"], "房颤伴TIA病史，卒中风险评估")
        self.assertEqual(result.structured_case_json["known_facts"], ["既往TIA", "高血压", "糖尿病"])
        self.assertEqual(
            [query.text for query in result.retrieval_queries],
            ["房颤伴TIA病史，卒中风险评估", "房颤伴TIA病史"],
        )

    def test_clinical_assisstment_parent_selects_pmid_before_child_execution(self) -> None:
        runner = _SelectionAwareClinicalToolRunner()
        state = GraphState(
            request="房颤伴既往卒中病史，评估卒中风险。",
            department="内科",
            department_tags=["内科"],
            clinical_tool_job=ClinicalToolJob(
                mode="question",
                text="房颤伴既往卒中病史，评估卒中风险。",
                risk_hints=["卒中风险"],
            ),
            tool_registry={"clinical_tool_agent": runner},
        )

        result = clinical_assisstment_node(state)

        self.assertEqual(len(runner.plan_calls), 1)
        self.assertEqual(len(runner.run_calls), 1)
        self.assertEqual(result.clinical_tool_job.selected_tool_pmid, "123456")
        self.assertEqual(result.clinical_tool_job.selected_tool["pmid"], "123456")
        self.assertEqual(
            result.clinical_tool_job.dispatch_query_text,
            "atrial fibrillation prior stroke age",
        )
        self.assertEqual(
            result.clinical_tool_job.selection_context["selection_mode"],
            "parent_selected",
        )
        self.assertEqual(
            runner.run_calls[0].dispatch_query_text,
            "atrial fibrillation prior stroke age",
        )
        self.assertEqual(runner.run_calls[0].selected_tool_pmid, "123456")
        self.assertEqual(
            runner.run_calls[0].selection_context["dispatch_query_source"],
            "selected_candidate.query_text",
        )
        self.assertEqual(
            result.final_output["clinical_tool_agent"]["selected_tool"]["pmid"],
            "123456",
        )

    def test_protocol_and_reporter_store_results_on_main_state(self) -> None:
        state = GraphState(
            request="整理病例",
            structured_case_json={"case_summary": "summary"},
            calculation_bundle={"mode": "baseline"},
            calculation_results=[],
            calculator_matches=[],
        )

        protocol_state = protocol_node(state)
        self.assertTrue(protocol_state.treatment_bundle)

        protocol_state.treatment_recommendations = [
            TreatmentRecommendation(
                name="direct treatment advice",
                strategy="direct_advice",
                source="no_calculation_signal",
                status="advice_only",
                rationale="demo",
                actions=["collect more data"],
            )
        ]
        protocol_state.treatment_bundle = {"recommendations": [{"name": "direct treatment advice"}]}

        reporter_state = reporter_node(protocol_state)
        self.assertTrue(reporter_state.reporter_result)
        self.assertTrue(reporter_state.review_report)
        self.assertIsInstance(reporter_state.clinical_answer, list)


if __name__ == "__main__":
    unittest.main()
