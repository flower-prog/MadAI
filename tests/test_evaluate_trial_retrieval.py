from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from agent.graph.types import GraphState
from agent.protocol.evaluation import (
    evaluate_case_result,
    load_eval_cases,
    run_protocol_trial_eval,
    summarize_eval_results,
)


def _eval_case(case_id: str = "case_001") -> dict[str, object]:
    return {
        "case_id": case_id,
        "description": "Melanoma trial eval case.",
        "structured_case": {
            "raw_text": "62-year-old male with metastatic melanoma and ECOG 1.",
            "case_summary": "Metastatic melanoma.",
            "problem_list": ["metastatic melanoma"],
            "known_facts": ["ECOG 1"],
            "department_tags": ["肿瘤科"],
        },
        "expected": {
            "must_retrieve_nct_ids": ["NCTGOOD001"],
            "acceptable_conditions": ["melanoma"],
            "acceptable_interventions": ["pembrolizumab"],
        },
    }


def _fake_protocol_runner(state: GraphState) -> GraphState:
    del state.clinical_tool_job
    state.final_output["trial_retrieval_bundle"] = {
        "query_text": "metastatic melanoma ECOG 1",
        "query_profile": {
            "primary_condition": "melanoma",
            "expanded_terms": {
                "conditions": ["melanoma", "metastatic melanoma"],
                "biomarkers": [],
                "treatments": ["pembrolizumab"],
                "treatment_context": [],
                "eligibility": ["ECOG 1"],
                "negative": [],
            },
            "retrieval_queries": [
                {
                    "name": "case_summary",
                    "text": "Metastatic melanoma.",
                    "source_terms": ["case_summary"],
                },
                {
                    "name": "condition",
                    "text": "metastatic melanoma clinical trial",
                    "source_terms": ["conditions"],
                },
                {
                    "name": "raw_fallback",
                    "text": "62-year-old male with metastatic melanoma and ECOG 1.",
                    "source_terms": ["raw_text"],
                },
            ],
            "trial_retrieval_context": {
                "raw_text": "62-year-old male with metastatic melanoma and ECOG 1.",
                "case_summary": "Metastatic melanoma.",
                "source_terms": {
                    "problem_list": ["metastatic melanoma"],
                    "known_facts": ["ECOG 1"],
                    "structured_inputs": [],
                },
                "expanded_terms": {
                    "conditions": ["melanoma", "metastatic melanoma"],
                    "biomarkers": [],
                    "treatments": ["pembrolizumab"],
                    "treatment_context": [],
                    "eligibility": ["ECOG 1"],
                    "negative": [],
                },
                "retrieval_queries": [
                    {
                        "name": "case_summary",
                        "text": "Metastatic melanoma.",
                        "source_terms": ["case_summary"],
                    },
                    {
                        "name": "condition",
                        "text": "metastatic melanoma clinical trial",
                        "source_terms": ["conditions"],
                    },
                    {
                        "name": "raw_fallback",
                        "text": "62-year-old male with metastatic melanoma and ECOG 1.",
                        "source_terms": ["raw_text"],
                    },
                ],
            },
        },
        "candidate_ranking": [
            {
                "nct_id": "NCTGOOD001",
                "title": "Pembrolizumab for Melanoma",
                "conditions": ["Melanoma"],
                "interventions": ["Pembrolizumab"],
                "overall_status": "Recruiting",
                "enrollment_open": True,
            }
        ],
    }
    state.final_output["eligibility_assessment_bundle"] = {
        "assessed_trial_count": 1,
        "assessed_trials": [
            {
                "nct_id": "NCTGOOD001",
                "eligibility_section_parse_status": "split",
                "criteria": [
                    {
                        "criterion_id": "NCTGOOD001::inclusion::001",
                        "label": "met",
                        "raw_text": "Age >= 18 years",
                    },
                    {
                        "criterion_id": "NCTGOOD001::exclusion::001",
                        "label": "unknown",
                        "raw_text": "Adequate organ function",
                    },
                ],
                "blocking_criteria": [],
                "unknown_criteria": ["NCTGOOD001::exclusion::001"],
                "missing_questions": [{"question": "Please provide recent organ function labs."}],
            }
        ],
    }
    state.final_output["patient_evidence_bundle"] = {"evidence_span_count": 1}
    state.final_output["calculator_evidence_bundle"] = {"risk_evidence_items": []}
    state.final_output["medical_knowledge_bundle"] = {"status": "not_configured"}
    state.final_output["missing_data_bundle"] = {"missing_question_count": 1}
    state.final_output["protocol_decision_bundle"] = {"status": "completed"}
    return state


class TrialRetrievalEvaluationTests(unittest.TestCase):
    def test_load_eval_cases_reads_jsonl_and_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(_eval_case("case_001"), ensure_ascii=False),
                        json.dumps(_eval_case("case_002"), ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_eval_cases(path, limit=1)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["case_id"], "case_001")

    def test_evaluate_case_result_detects_retrieval_hits_and_unknowns(self) -> None:
        bundles = _fake_protocol_runner(GraphState(request="demo")).final_output

        metrics = evaluate_case_result(_eval_case(), bundles)

        self.assertEqual(metrics["retrieved_trial_count"], 1)
        self.assertEqual(metrics["must_retrieve_recall_at_10"], 1.0)
        self.assertEqual(metrics["condition_match_count_at_10"], 1)
        self.assertEqual(metrics["intervention_match_count_at_10"], 1)
        self.assertEqual(metrics["open_or_recruiting_count_at_10"], 1)
        self.assertEqual(metrics["criteria_count"], 2)
        self.assertEqual(metrics["met_count"], 1)
        self.assertEqual(metrics["unknown_count"], 1)
        self.assertEqual(metrics["missing_question_count"], 1)
        self.assertEqual(metrics["retrieval_query_count"], 3)
        self.assertTrue(metrics["has_raw_fallback_query"])
        self.assertEqual(metrics["expanded_condition_count"], 2)
        self.assertEqual(metrics["expanded_treatment_count"], 1)
        self.assertEqual(metrics["expanded_eligibility_count"], 1)
        self.assertNotIn("no_candidates", metrics["failure_buckets"])

    def test_evaluate_case_result_buckets_no_candidates_and_miss(self) -> None:
        metrics = evaluate_case_result(
            _eval_case(),
            {
                "trial_retrieval_bundle": {"candidate_ranking": []},
                "eligibility_assessment_bundle": {"assessed_trials": []},
            },
        )

        self.assertIn("no_candidates", metrics["failure_buckets"])
        self.assertIn("must_retrieve_miss", metrics["failure_buckets"])
        self.assertEqual(metrics["must_retrieve_recall_at_20"], 0.0)

    def test_summarize_eval_results_aggregates_counts(self) -> None:
        summary = summarize_eval_results(
            [
                {"status": "completed", "retrieved_trial_count": 2, "failure_buckets": ["too_many_unknowns"]},
                {"status": "failed", "retrieved_trial_count": 0, "failure_buckets": ["protocol_exception"]},
            ]
        )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["completed_case_count"], 1)
        self.assertEqual(summary["failed_case_count"], 1)
        self.assertEqual(summary["metrics"]["mean_retrieved_trial_count"], 1.0)
        self.assertIn("mean_retrieval_query_count", summary["metrics"])
        self.assertEqual(summary["failure_buckets"]["protocol_exception"], 1)

    def test_run_protocol_trial_eval_writes_summary_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"
            trace_dir = Path(temp_dir) / "traces"

            summary = run_protocol_trial_eval(
                [_eval_case()],
                output_path=output_path,
                trace_dir=trace_dir,
                protocol_runner=_fake_protocol_runner,
            )

            case_trace_dir = trace_dir / "case_001"
            self.assertTrue(output_path.exists())
            self.assertTrue((case_trace_dir / "input_case.json").exists())
            self.assertTrue((case_trace_dir / "trial_retrieval_bundle.json").exists())
            self.assertTrue((case_trace_dir / "eligibility_assessment_bundle.json").exists())
            self.assertTrue((case_trace_dir / "trial_retrieval_context.json").exists())
            self.assertTrue((case_trace_dir / "retrieval_queries.json").exists())
            self.assertTrue((case_trace_dir / "expanded_terms.json").exists())
            self.assertTrue((case_trace_dir / "case_metrics.json").exists())
            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["cases"][0]["must_retrieve_recall_at_10"], 1.0)
            self.assertEqual(summary["cases"][0]["retrieval_query_count"], 3)


if __name__ == "__main__":
    unittest.main()
