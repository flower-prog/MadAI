from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent import workflow as workflow_module
from agent.graph.nodes import protocol_node, reporter_node
from agent.graph.types import CalculationArtifact, GraphState, TreatmentRecommendation
from agent.tools.trial_retrieval_tools import TrialKeywordRetriever, create_trial_retrieval_tool


def _trial_payload(
    nct_id: str,
    title: str,
    *,
    department_tag: str,
    department_role: str = "primary",
    status: str = "trial_matched",
    overall_status: str = "Recruiting",
    enrollment_open: bool = True,
    brief_summary: str = "",
    status_reason: str = "",
    conditions: list[str] | None = None,
    keywords: list[str] | None = None,
    interventions: list[str] | None = None,
    actions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": title,
        "strategy": "trial_match",
        "source": "clinicaltrials.gov",
        "status": status,
        "rationale": f"{title} rationale",
        "linked_calculators": [],
        "linked_trials": [nct_id],
        "actions": list(actions or [f"Review {title} manually."]),
        "nct_id": nct_id,
        "department_tag": department_tag,
        "department_role": department_role,
        "department_tags": [department_tag],
        "primary_department": department_tag,
        "secondary_departments": [],
        "overall_status": overall_status,
        "status_reason": status_reason or f"{title} status reason",
        "enrollment_open": enrollment_open,
        "brief_title": title,
        "official_title": title,
        "conditions": list(conditions or []),
        "mesh_terms": list(conditions or []),
        "keywords": list(keywords or []),
        "interventions": list(interventions or []),
        "brief_summary": brief_summary or f"{title} summary",
        "eligibility_text": f"{title} eligibility",
        "study_type": "Interventional",
        "phase": "Phase 2",
        "primary_purpose": "Treatment",
    }


def _write_department_payload(
    root: Path,
    department_tag: str,
    payload: dict[str, dict[str, object]],
) -> None:
    department_dir = root / department_tag
    department_dir.mkdir(parents=True, exist_ok=True)
    (department_dir / "treatment_trials.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class _FakeTrialVectorRetriever:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        del query
        rows = [dict(row) for row in self.rows]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("nct_id") or "") in candidate_pmids]
        return rows[:top_k]


class _FakeTrialRetriever:
    def __init__(self, bundle: dict[str, object]) -> None:
        self.bundle = dict(bundle)
        self.calls: list[dict[str, object]] = []

    def retrieve_from_structured_case(self, structured_case, **kwargs):
        self.calls.append(
            {
                "structured_case": dict(structured_case or {}),
                "kwargs": dict(kwargs),
            }
        )
        return dict(self.bundle)


class _SpySubsetBM25:
    def __init__(self, score_by_index: dict[int, float] | None = None) -> None:
        self.score_by_index = dict(score_by_index or {})
        self.calls: list[dict[str, object]] = []

    def score_subset(
        self,
        query: str,
        *,
        document_indexes: list[int] | None = None,
    ) -> dict[int, float]:
        self.calls.append(
            {
                "query": query,
                "document_indexes": list(document_indexes) if document_indexes is not None else None,
            }
        )
        target_indexes = list(document_indexes or [])
        return {
            index: float(self.score_by_index.get(index, 0.0))
            for index in target_indexes
        }


class TrialRetrievalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"trial-retrieval-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_loader_dedupes_trials_and_prefers_requested_department_payload(self) -> None:
        _write_department_payload(
            self.temp_root,
            "内科",
            {
                "NCT0001": _trial_payload(
                    "NCT0001",
                    "Internal Primary Title",
                    department_tag="内科",
                    department_role="primary",
                )
            },
        )
        _write_department_payload(
            self.temp_root,
            "肿瘤科",
            {
                "NCT0001": _trial_payload(
                    "NCT0001",
                    "Oncology Preferred Title",
                    department_tag="肿瘤科",
                    department_role="secondary",
                )
            },
        )

        preferred_tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="肿瘤科",
            backend="keyword",
        )
        default_tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="",
            backend="keyword",
        )

        self.assertEqual(preferred_tool.catalog.get("NCT0001").title, "Oncology Preferred Title")
        self.assertEqual(default_tool.catalog.get("NCT0001").title, "Internal Primary Title")
        self.assertEqual(
            set(preferred_tool.catalog.get("NCT0001").department_tags),
            {"内科", "肿瘤科"},
        )

    def test_retrieval_uses_department_candidate_pool_before_full_corpus(self) -> None:
        _write_department_payload(
            self.temp_root,
            "内科",
            {
                "NCT1001": _trial_payload(
                    "NCT1001",
                    "Atrial Fibrillation Internal Trial",
                    department_tag="内科",
                    conditions=["Atrial Fibrillation"],
                    keywords=["atrial fibrillation", "stroke prevention"],
                )
            },
        )
        _write_department_payload(
            self.temp_root,
            "肿瘤科",
            {
                "NCT2001": _trial_payload(
                    "NCT2001",
                    "Lung Cancer Oncology Trial",
                    department_tag="肿瘤科",
                    conditions=["Lung Cancer"],
                    keywords=["lung cancer", "immunotherapy"],
                )
            },
        )

        tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="内科",
            backend="keyword",
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": "Atrial fibrillation with prior TIA",
                "problem_list": ["atrial fibrillation"],
                "known_facts": ["prior TIA"],
                "department_tags": ["内科"],
            },
            department_tags=["内科"],
        )

        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCT1001")
        self.assertFalse(bundle["fallback_to_full_catalog"])
        returned_ids = {row["nct_id"] for row in bundle["candidate_ranking"]}
        self.assertEqual(returned_ids, {"NCT1001"})

    def test_retrieval_falls_back_to_full_catalog_when_department_pool_is_empty(self) -> None:
        _write_department_payload(self.temp_root, "内科", {})
        _write_department_payload(
            self.temp_root,
            "肿瘤科",
            {
                "NCT2001": _trial_payload(
                    "NCT2001",
                    "Lung Cancer Oncology Trial",
                    department_tag="肿瘤科",
                    conditions=["Lung Cancer"],
                    keywords=["lung cancer", "immunotherapy"],
                )
            },
        )

        tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="内科",
            backend="keyword",
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": "Lung cancer patient for immunotherapy",
                "problem_list": ["lung cancer"],
                "known_facts": ["immunotherapy"],
                "department_tags": ["内科"],
            },
            department_tags=["内科"],
        )

        self.assertTrue(bundle["fallback_to_full_catalog"])
        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCT2001")

    def test_trial_keyword_retriever_scores_only_candidate_subset(self) -> None:
        _write_department_payload(
            self.temp_root,
            "内科",
            {
                "NCT1001": _trial_payload(
                    "NCT1001",
                    "Trial 1001",
                    department_tag="内科",
                    conditions=["Disease A"],
                    keywords=["disease a"],
                ),
                "NCT2001": _trial_payload(
                    "NCT2001",
                    "Trial 2001",
                    department_tag="内科",
                    conditions=["Disease B"],
                    keywords=["disease b"],
                ),
            },
        )

        tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="内科",
            backend="keyword",
        )
        retriever = TrialKeywordRetriever(tool.catalog)
        spy_bm25 = _SpySubsetBM25({1: 4.2})
        retriever._bm25 = spy_bm25

        rows = retriever.retrieve(
            "Disease B patient",
            top_k=5,
            candidate_ids=["NCT2001"],
        )

        self.assertEqual(rows[0]["nct_id"], "NCT2001")
        self.assertEqual(spy_bm25.calls[0]["document_indexes"], [1])
        self.assertEqual(
            retriever.retrieve(
                "Disease B patient",
                top_k=5,
                candidate_ids=[],
            ),
            [],
        )

    def test_two_stage_retrieval_returns_bm25_top5_vector_top5_and_merged_union(self) -> None:
        _write_department_payload(
            self.temp_root,
            "肿瘤科",
            {
                "NCT3001": _trial_payload(
                    "NCT3001",
                    "Melanoma Checkpoint Trial",
                    department_tag="肿瘤科",
                    keywords=["melanoma", "checkpoint"],
                    interventions=["nivolumab"],
                ),
                "NCT3002": _trial_payload(
                    "NCT3002",
                    "Melanoma Adoptive Cell Trial",
                    department_tag="肿瘤科",
                    keywords=["melanoma", "cell therapy"],
                    interventions=["TIL therapy"],
                ),
            },
        )

        tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="肿瘤科",
            backend="hybrid",
            vector_retriever=_FakeTrialVectorRetriever(
                [
                    {"nct_id": "NCT3002", "title": "Melanoma Adoptive Cell Trial", "score": 0.95},
                    {"nct_id": "NCT3001", "title": "Melanoma Checkpoint Trial", "score": 0.61},
                ]
            ),
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": "Melanoma patient considering nivolumab based treatment",
                "problem_list": ["melanoma"],
                "known_facts": ["checkpoint inhibitor"],
                "department_tags": ["肿瘤科"],
            },
            department_tags=["肿瘤科"],
        )

        self.assertTrue(bundle["coarse_candidate_ids"])
        self.assertTrue(bundle["bm25_top5"])
        self.assertTrue(bundle["vector_top5"])
        self.assertEqual(bundle["vector_top5"][0]["nct_id"], "NCT3002")
        merged_ids = [row["nct_id"] for row in bundle["candidate_ranking"]]
        self.assertIn("NCT3001", merged_ids)
        self.assertIn("NCT3002", merged_ids)

    def test_status_priority_prefers_open_trial_when_scores_tie(self) -> None:
        shared_summary = "lung cancer immunotherapy study"
        _write_department_payload(
            self.temp_root,
            "肿瘤科",
            {
                "NCT4001": _trial_payload(
                    "NCT4001",
                    "Open Trial",
                    department_tag="肿瘤科",
                    status="trial_matched",
                    overall_status="Recruiting",
                    enrollment_open=True,
                    brief_summary=shared_summary,
                    keywords=["lung cancer", "immunotherapy"],
                ),
                "NCT4002": _trial_payload(
                    "NCT4002",
                    "Closed Evidence Trial",
                    department_tag="肿瘤科",
                    status="trial_matched",
                    overall_status="Completed",
                    enrollment_open=False,
                    brief_summary=shared_summary,
                    keywords=["lung cancer", "immunotherapy"],
                ),
                "NCT4003": _trial_payload(
                    "NCT4003",
                    "Manual Review Trial",
                    department_tag="肿瘤科",
                    status="manual_review",
                    overall_status="Unknown status",
                    enrollment_open=False,
                    brief_summary=shared_summary,
                    keywords=["lung cancer", "immunotherapy"],
                ),
                "NCT4004": _trial_payload(
                    "NCT4004",
                    "Abandoned Trial",
                    department_tag="肿瘤科",
                    status="abandoned",
                    overall_status="Terminated",
                    enrollment_open=False,
                    brief_summary=shared_summary,
                    keywords=["lung cancer", "immunotherapy"],
                ),
            },
        )

        tool = create_trial_retrieval_tool(
            department_root=self.temp_root,
            preferred_department="肿瘤科",
            backend="keyword",
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": shared_summary,
                "problem_list": ["lung cancer"],
                "known_facts": ["immunotherapy"],
                "department_tags": ["肿瘤科"],
            },
            department_tags=["肿瘤科"],
        )

        self.assertEqual(
            [row["nct_id"] for row in bundle["candidate_ranking"][:4]],
            ["NCT4001", "NCT4002", "NCT4003", "NCT4004"],
        )

    def test_protocol_node_populates_trial_bundle_and_links_top_trials(self) -> None:
        trial_bundle = {
            "query_text": "melanoma",
            "backend_used": "hybrid",
            "available_backends": ["bm25", "vector", "hybrid"],
            "department_tags": ["肿瘤科"],
            "fallback_to_full_catalog": False,
            "coarse_candidate_ids": ["NCT5001", "NCT5002"],
            "bm25_top5": [],
            "vector_top5": [],
            "candidate_ranking": [
                {
                    "nct_id": "NCT5001",
                    "title": "Primary Trial",
                    "status": "trial_matched",
                    "enrollment_open": True,
                    "actions": ["Confirm site enrollment."],
                },
                {
                    "nct_id": "NCT5002",
                    "title": "Secondary Trial",
                    "status": "manual_review",
                    "enrollment_open": False,
                    "actions": ["Review manually."],
                },
            ],
        }
        fake_retriever = _FakeTrialRetriever(trial_bundle)
        state = GraphState(
            request="整理病例",
            structured_case_json={
                "case_summary": "melanoma",
                "problem_list": ["melanoma"],
                "known_facts": ["checkpoint inhibitor"],
            },
            calculation_results=[
                CalculationArtifact(
                    name="risk_result",
                    category="risk_score",
                    status="completed",
                    linked_calculator="PMID-1",
                )
            ],
            calculation_bundle={"mode": "baseline"},
            tool_registry={"trial_retriever": fake_retriever},
            department="肿瘤科",
            department_tags=["肿瘤科"],
        )

        result = protocol_node(state)

        self.assertEqual(result.trial_retrieval_bundle["coarse_candidate_ids"], ["NCT5001", "NCT5002"])
        self.assertEqual(result.treatment_bundle["trial_candidate_ids"], ["NCT5001", "NCT5002"])
        self.assertEqual(result.treatment_bundle["trial_candidates"][0]["nct_id"], "NCT5001")
        self.assertEqual(result.treatment_bundle["trial_selection"]["selected_trial"]["nct_id"], "NCT5001")
        self.assertEqual(result.treatment_bundle["trial_selection"]["trial_status_assessment"]["overall_status"], "")
        self.assertTrue(result.treatment_bundle["trial_selection"]["selection_reason"])
        self.assertEqual(result.treatment_recommendations[0].linked_trials, ["NCT5001", "NCT5002"])
        self.assertEqual(result.protocol_recommendations[0].linked_trials, ["NCT5001", "NCT5002"])
        self.assertEqual(len(fake_retriever.calls), 1)

    def test_protocol_node_adds_trial_candidate_review_without_calculation_signal(self) -> None:
        fake_retriever = _FakeTrialRetriever(
            {
                "query_text": "severe asthma",
                "backend_used": "bm25",
                "available_backends": ["bm25"],
                "department_tags": ["内科"],
                "fallback_to_full_catalog": False,
                "coarse_candidate_ids": ["NCT6001"],
                "bm25_top5": [],
                "vector_top5": [],
                "candidate_ranking": [
                    {
                        "nct_id": "NCT6001",
                        "title": "Asthma Trial",
                        "status": "manual_review",
                        "enrollment_open": False,
                        "actions": ["Review eligibility."],
                    }
                ],
            }
        )
        state = GraphState(
            request="整理病例",
            structured_case_json={"case_summary": "severe asthma"},
            calculation_results=[],
            calculator_matches=[],
            calculation_bundle={"mode": "baseline"},
            tool_registry={"trial_retriever": fake_retriever},
            department="内科",
            department_tags=["内科"],
        )

        result = protocol_node(state)

        self.assertEqual(result.treatment_recommendations[0].strategy, "trial_candidate_review")
        self.assertEqual(result.treatment_recommendations[0].source, "trial_retrieval")
        self.assertEqual(result.treatment_recommendations[0].linked_trials, ["NCT6001"])
        self.assertEqual(result.treatment_bundle["trial_candidate_ids"], ["NCT6001"])
        self.assertEqual(result.treatment_bundle["trial_selection"]["selected_trial"]["nct_id"], "NCT6001")
        self.assertIn(
            "Review eligibility.",
            result.treatment_bundle["trial_selection"]["eligibility_assessment"]["next_checks"],
        )

    def test_protocol_node_defaults_to_trial_chunk_retriever_for_xml_trial_kb(self) -> None:
        fake_retriever = _FakeTrialRetriever(
            {
                "query_text": "melanoma",
                "backend_used": "hybrid",
                "available_backends": ["bm25", "vector", "hybrid"],
                "department_tags": ["肿瘤科"],
                "fallback_to_full_catalog": False,
                "coarse_candidate_ids": ["NCTXML001"],
                "bm25_top5": [],
                "vector_top5": [],
                "candidate_ranking": [
                    {
                        "nct_id": "NCTXML001",
                        "title": "XML Trial",
                        "status": "trial_matched",
                        "enrollment_open": True,
                        "actions": ["Review eligibility."],
                    }
                ],
            }
        )
        state = GraphState(
            request="整理病例",
            structured_case_json={
                "case_summary": "melanoma",
                "problem_list": ["melanoma"],
                "known_facts": ["checkpoint inhibitor"],
            },
            calculation_results=[],
            calculator_matches=[],
            calculation_bundle={"mode": "baseline"},
            department="肿瘤科",
            department_tags=["肿瘤科"],
        )

        with patch("agent.graph.nodes.create_trial_chunk_retrieval_tool", return_value=fake_retriever) as mocked_factory:
            result = protocol_node(state)

        mocked_factory.assert_called_once_with(backend="hybrid", vector_store="auto")
        self.assertEqual(result.treatment_bundle["trial_candidate_ids"], ["NCTXML001"])
        self.assertEqual(result.treatment_bundle["trial_candidates"][0]["nct_id"], "NCTXML001")

    def test_reporter_and_cli_summary_expose_linked_trials_and_trial_candidates(self) -> None:
        state = GraphState(
            request="整理病例",
            structured_case_json={"case_summary": "summary"},
            calculation_bundle={"mode": "baseline"},
            treatment_recommendations=[
                TreatmentRecommendation(
                    name="trial candidate review",
                    strategy="trial_candidate_review",
                    source="trial_retrieval",
                    status="manual_review",
                    linked_trials=["NCT7001"],
                    actions=["Review eligibility."],
                )
            ],
            treatment_bundle={
                "recommendations": [{"name": "trial candidate review"}],
                "trial_candidates": [
                    {
                        "nct_id": "NCT7001",
                        "title": "Visible Trial",
                        "status": "trial_matched",
                        "enrollment_open": True,
                    }
                ],
                "trial_candidate_ids": ["NCT7001"],
            },
        )

        reporter_state = reporter_node(state)
        self.assertEqual(
            reporter_state.reporter_result["recommendation_summary"][0]["linked_trials"],
            ["NCT7001"],
        )
        self.assertEqual(
            reporter_state.reporter_result["trial_candidate_summary"][0]["nct_id"],
            "NCT7001",
        )

        summary = workflow_module._extract_cli_summary(
            {
                "status": "completed",
                "review_passed": True,
                "clinical_answer": reporter_state.clinical_answer,
                "final_output": {
                    "treatment_bundle": reporter_state.treatment_bundle,
                    "clinical_answer": reporter_state.clinical_answer,
                },
                "errors": [],
            }
        )
        self.assertEqual(summary["clinical_answer"][0]["linked_trials"], ["NCT7001"])
        self.assertEqual(summary["trial_candidates"][0]["nct_id"], "NCT7001")

    def test_trial_retrieval_degrades_to_bm25_when_vector_backend_is_unavailable(self) -> None:
        _write_department_payload(
            self.temp_root,
            "肿瘤科",
            {
                "NCT8001": _trial_payload(
                    "NCT8001",
                    "BM25 Only Trial",
                    department_tag="肿瘤科",
                    keywords=["glioma", "radiation"],
                )
            },
        )

        with patch("agent.tools.trial_retrieval_tools.MedCPTRetriever", side_effect=RuntimeError("missing deps")):
            tool = create_trial_retrieval_tool(
                department_root=self.temp_root,
                preferred_department="肿瘤科",
                backend="hybrid",
                vector_retriever=None,
            )

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": "glioma with radiation planning",
                "problem_list": ["glioma"],
                "known_facts": ["radiation"],
                "department_tags": ["肿瘤科"],
            },
            department_tags=["肿瘤科"],
        )

        self.assertEqual(bundle["backend_used"], "bm25")
        self.assertEqual(bundle["vector_top5"], [])
        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCT8001")


if __name__ == "__main__":
    unittest.main()
