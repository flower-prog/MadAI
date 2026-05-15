from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.protocol.medical_knowledge import retrieve_medical_knowledge_for_protocol
from agent.protocol.medical_phrase_parser import parse_medical_phrases_for_protocol
from agent.protocol.query_planner import build_protocol_medical_queries
from agent.tools.medical_knowledge_tools import LiveMedicalKnowledgeRetriever


class ProtocolMedicalKnowledgeTests(unittest.TestCase):
    def test_protocol_query_planner_derives_queries_from_case_trial_and_calculator(self) -> None:
        phrase_bundle = parse_medical_phrases_for_protocol(
            structured_case={
                "problem_list": ["atrial fibrillation", "hypertension", "diabetes", "prior TIA"],
            },
            trial_retrieval_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTAF001",
                        "conditions": ["Atrial Fibrillation"],
                        "interventions": ["Anticoagulation"],
                    }
                ]
            },
            eligibility_assessment_bundle={
                "assessed_trials": [
                    {
                        "nct_id": "NCTAF001",
                        "criteria": [
                            {
                                "label": "unknown",
                                "raw_text": "No concurrent warfarin",
                            },
                            {
                                "label": "unknown",
                                "raw_text": "QTc < 500 msec",
                            },
                        ],
                    }
                ]
            },
            limit=5,
        )
        query_bundle = build_protocol_medical_queries(
            structured_case={
                "problem_list": ["atrial fibrillation", "hypertension", "diabetes", "prior TIA"],
            },
            trial_retrieval_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTAF001",
                        "conditions": ["Atrial Fibrillation"],
                        "interventions": ["Anticoagulation"],
                    }
                ]
            },
            eligibility_assessment_bundle={
                "assessed_trials": [
                    {
                        "nct_id": "NCTAF001",
                        "criteria": [
                            {
                                "label": "unknown",
                                "raw_text": "No concurrent warfarin",
                            },
                            {
                                "label": "unknown",
                                "raw_text": "QTc < 500 msec",
                            },
                        ],
                    }
                ]
            },
            calculator_evidence_bundle={
                "risk_evidence_items": [
                    {
                        "calculator": "CHA2DS2-VASc",
                        "category": "stroke_risk",
                    }
                ]
            },
            medical_phrase_bundle=phrase_bundle,
            limit=8,
        )

        queries = query_bundle["queries"]
        query_text = "\n".join(item["query"] for item in queries)
        self.assertEqual(query_bundle["schema_version"], 1)
        self.assertEqual(query_bundle["status"], "completed")
        self.assertTrue(any(item["purpose"] == "case_trial_overlap" for item in queries))
        self.assertTrue(any(item["purpose"] == "calculator_interpretation" for item in queries))
        self.assertIn("No concurrent warfarin", query_text)
        self.assertIn("QTc < 500 msec", query_text)
        self.assertTrue(
            any(
                item["linked_nct_id"] == "NCTAF001" and item["linked_criterion"] == "No concurrent warfarin"
                for item in queries
            )
        )
        self.assertTrue(any(item["concepts"] for item in queries))

    def test_medical_phrase_parser_extracts_protocol_concepts(self) -> None:
        bundle = parse_medical_phrases_for_protocol(
            structured_case={
                "problem_list": ["metastatic melanoma"],
            },
            eligibility_assessment_bundle={
                "assessed_trials": [
                    {
                        "nct_id": "NCTMEL001",
                        "criteria": [
                            {
                                "raw_text": "No active autoimmune disease requiring systemic therapy",
                            },
                            {
                                "raw_text": "ECOG performance status 0-2",
                            },
                        ],
                    }
                ]
            },
        )

        concept_names = {item["canonical_name"] for item in bundle["concepts"]}
        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["status"], "completed")
        self.assertEqual(bundle["backend_used"], "rule")
        self.assertIn("Autoimmune Disease", concept_names)
        self.assertIn("Systemic Therapy", concept_names)
        self.assertIn("ECOG Performance Status", concept_names)
        self.assertTrue(
            any(
                "current systemic immunosuppressive therapy" in item.get("required_patient_evidence", [])
                for item in bundle["parsed_phrases"]
            )
        )

    def test_retrieval_module_returns_stable_empty_bundle_without_retriever(self) -> None:
        bundle = retrieve_medical_knowledge_for_protocol(
            structured_case={
                "problem_list": ["metastatic melanoma"],
            },
            trial_retrieval_bundle={
                "candidate_ranking": [
                    {
                        "nct_id": "NCTMEL001",
                        "conditions": ["Melanoma"],
                        "interventions": ["Pembrolizumab"],
                    }
                ]
            },
            eligibility_assessment_bundle={
                "assessed_trials": [
                    {
                        "criteria": [
                            {
                                "label": "unknown",
                                "raw_text": "Active autoimmune disease requiring systemic therapy",
                            }
                        ]
                    }
                ]
            },
            calculator_evidence_bundle={
                "risk_evidence_items": [
                    {
                        "calculator": "ECOG performance status",
                        "category": "performance_status",
                    }
                ]
            },
        )

        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["status"], "not_configured")
        self.assertEqual(bundle["backend_used"], "none")
        self.assertTrue(bundle["queries"])
        self.assertEqual(bundle["retrieved_items"], [])
        self.assertTrue(bundle["knowledge_gaps"])
        self.assertTrue(
            any("Active autoimmune disease" in item["query"] for item in bundle["queries"])
        )

    def test_medical_knowledge_retriever_receives_concepts_and_filters(self) -> None:
        class RecordingRetriever:
            def __init__(self) -> None:
                self.calls = []

            def retrieve(self, query: str, *, top_k: int, concepts=None, filters=None):
                self.calls.append(
                    {
                        "query": query,
                        "top_k": top_k,
                        "concepts": list(concepts or []),
                        "filters": dict(filters or {}),
                    }
                )
                if concepts:
                    return [
                        {
                            "title": "Autoimmune disease eligibility context",
                            "text": "Active autoimmune disease may require criterion-specific review.",
                        }
                    ]
                return []

        retriever = RecordingRetriever()
        bundle = retrieve_medical_knowledge_for_protocol(
            structured_case={"problem_list": ["metastatic melanoma"]},
            trial_retrieval_bundle={"candidate_ranking": []},
            medical_phrase_bundle={
                "concepts": [
                    {
                        "canonical_name": "Autoimmune Disease",
                        "semantic_type": "condition",
                    }
                ]
            },
            retriever=retriever,
        )

        concept_calls = [item for item in retriever.calls if item["concepts"]]
        self.assertEqual(bundle["status"], "completed")
        self.assertTrue(concept_calls)
        self.assertEqual(concept_calls[0]["concepts"][0]["canonical_name"], "Autoimmune Disease")
        self.assertIn("source_type", concept_calls[0]["filters"])
        self.assertTrue(bundle["retrieved_items"])

    def test_live_medical_knowledge_retriever_aggregates_source_tools(self) -> None:
        class FakeSource:
            def __init__(self, source_name: str) -> None:
                self.source_name = source_name
                self.calls = []

            def retrieve(self, query: str, *, top_k: int, concepts=None, filters=None):
                self.calls.append(
                    {
                        "query": query,
                        "top_k": top_k,
                        "concepts": list(concepts or []),
                        "filters": dict(filters or {}),
                    }
                )
                return [
                    {
                        "source": self.source_name,
                        "source_type": "test",
                        "title": f"{self.source_name} result",
                    }
                ]

        pubmed = FakeSource("pubmed")
        wikidata = FakeSource("wikidata")
        retriever = LiveMedicalKnowledgeRetriever(sources=[pubmed, wikidata], max_sources=2)

        rows = retriever.retrieve(
            "atrial fibrillation anticoagulation",
            top_k=5,
            concepts=[{"canonical_name": "Atrial Fibrillation"}],
            filters={"source_type": "clinical_context"},
        )

        self.assertEqual({item["source"] for item in rows}, {"pubmed", "wikidata"})
        self.assertEqual(pubmed.calls[0]["concepts"][0]["canonical_name"], "Atrial Fibrillation")
        self.assertEqual(wikidata.calls[0]["filters"]["source_type"], "clinical_context")
        self.assertTrue(all(item["retrieved_live"] for item in rows))


if __name__ == "__main__":
    unittest.main()
