from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.tools.trial_vector_retrieval_tools import (
    build_protocol_trial_query_profile,
    TrialChunkCatalog,
    TrialChunkKeywordRetriever,
    create_trial_chunk_retrieval_tool,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class _FakeTrialChunkVectorRetriever:
    def __init__(self, rows: list[dict[str, object]], *, raise_on_retrieve: bool = False) -> None:
        self.rows = [dict(row) for row in rows]
        self.raise_on_retrieve = bool(raise_on_retrieve)
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        candidate_pmids: set[str] | list[str] | tuple[str, ...] | None = None,
        payload_filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "candidate_ids": set(candidate_ids or []),
                "candidate_pmids": set(candidate_pmids or []),
                "payload_filters": dict(payload_filters or {}),
            }
        )
        if self.raise_on_retrieve:
            raise RuntimeError("vector backend unavailable")
        normalized_candidate_ids = {
            str(item).strip()
            for item in list(candidate_ids if candidate_ids is not None else candidate_pmids or [])
            if str(item).strip()
        }
        rows = [dict(row) for row in self.rows]
        if normalized_candidate_ids:
            rows = [
                row
                for row in rows
                if str(row.get("chunk_id") or row.get("document_id") or "").strip() in normalized_candidate_ids
            ]
        return rows[:top_k]


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


class TrialVectorRetrievalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"trial-vector-retrieval-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_protocol_trial_query_profile_rewrites_calculator_case_into_trial_search_profile(self) -> None:
        profile = build_protocol_trial_query_profile(
            {
                "raw_text": (
                    "A 78-year-old male patient presents to your clinic for a routine check-up. "
                    "He has a history of hypertension and recently experienced a transient ischemic attack. "
                    "He does not have diabetes or congestive heart failure. "
                    "He is not currently prescribed warfarin. "
                    "Based on these factors, what is the estimated stroke rate per 100 patient-years "
                    "without antithrombotic therapy for this patient?"
                ),
                "case_summary": "Older male with hypertension and prior TIA, not on warfarin.",
                "problem_list": ["hypertension", "transient ischemic attack"],
                "known_facts": ["no diabetes", "no congestive heart failure", "not on warfarin"],
            }
        )

        self.assertEqual(profile["trial_condition_terms"][0], "atrial fibrillation")
        self.assertIn("transient ischemic attack", profile["trial_condition_terms"])
        self.assertIn("stroke prevention", profile["trial_intent_terms"])
        self.assertIn("anticoagulation", profile["trial_intervention_terms"])
        self.assertIn("diabetes", profile["patient_negative_terms"])
        self.assertIn("warfarin", profile["patient_negative_terms"])
        self.assertEqual(profile["age_years"], 78)
        self.assertEqual(profile["gender"], "Male")
        self.assertEqual(profile["payload_filters"]["must"][0]["field"], "condition_terms")
        self.assertEqual(profile["payload_filters"]["must"][0]["values"], ["atrial fibrillation"])
        self.assertEqual(profile["payload_filters"]["must_not"][0]["values"], ["diabetes", "congestive heart failure"])
        self.assertIn("clinical trial eligibility search", profile["query_text"])
        self.assertNotIn("estimated stroke rate per 100 patient-years", profile["query_text"])

    def test_trial_chunk_retrieval_aggregates_chunks_back_to_trial_candidates(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCTMEL001",
                    "display_title": "Pembrolizumab for Metastatic Melanoma",
                    "brief_title": "Pembrolizumab for Metastatic Melanoma",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "conditions": ["Metastatic Melanoma"],
                    "interventions": ["Pembrolizumab"],
                    "source_url": "https://clinicaltrials.gov/show/NCTMEL001",
                },
                {
                    "nct_id": "NCTLUNG001",
                    "display_title": "Osimertinib for EGFR Lung Cancer",
                    "brief_title": "Osimertinib for EGFR Lung Cancer",
                    "overall_status": "Recruiting",
                    "phase": "Phase 3",
                    "study_type": "Interventional",
                    "conditions": ["Non-small Cell Lung Cancer"],
                    "interventions": ["Osimertinib"],
                    "source_url": "https://clinicaltrials.gov/show/NCTLUNG001",
                },
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCTMEL001::overview::0",
                    "nct_id": "NCTMEL001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "title: Pembrolizumab for Metastatic Melanoma\nconditions: metastatic melanoma\ninterventions: pembrolizumab",
                    "embedding_text": "title: Pembrolizumab for Metastatic Melanoma\nconditions: metastatic melanoma\ninterventions: pembrolizumab\nsummary: immune checkpoint inhibitor trial",
                    "source_fields": ["display_title", "conditions", "interventions", "brief_summary"],
                    "token_estimate": 30,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCTMEL001::eligibility_inclusion::0",
                    "nct_id": "NCTMEL001",
                    "chunk_type": "eligibility_inclusion",
                    "sequence": 0,
                    "text": "Inclusion criteria: histologically confirmed metastatic melanoma; prior PD-1 exposure not allowed.",
                    "embedding_text": "title: Pembrolizumab for Metastatic Melanoma\nchunk type: eligibility_inclusion\ninclusion criteria: histologically confirmed metastatic melanoma; prior PD-1 exposure not allowed.",
                    "source_fields": ["eligibility_text", "eligibility_inclusion_text"],
                    "token_estimate": 28,
                    "rank_weight": 1.2,
                },
                {
                    "chunk_id": "NCTLUNG001::overview::0",
                    "nct_id": "NCTLUNG001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "title: Osimertinib for EGFR Lung Cancer\nconditions: EGFR mutated lung cancer\ninterventions: osimertinib",
                    "embedding_text": "title: Osimertinib for EGFR Lung Cancer\nconditions: EGFR mutated lung cancer\ninterventions: osimertinib\nsummary: targeted therapy trial",
                    "source_fields": ["display_title", "conditions", "interventions", "brief_summary"],
                    "token_estimate": 28,
                    "rank_weight": 1.0,
                },
            ],
        )

        tool = create_trial_chunk_retrieval_tool(output_root=self.temp_root, backend="keyword")
        bundle = tool.retrieve_trials_from_structured_case(
            {
                "case_summary": "Metastatic melanoma patient considered for pembrolizumab trial.",
                "problem_list": ["metastatic melanoma"],
                "known_facts": ["pembrolizumab candidate"],
            },
            top_k=3,
            chunk_top_k=10,
        )

        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCTMEL001")
        self.assertEqual(bundle["candidate_ranking"][0]["display_title"], "Pembrolizumab for Metastatic Melanoma")
        self.assertTrue(bundle["candidate_ranking"][0]["matched_chunks"])
        self.assertIn("metastatic melanoma", bundle["candidate_ranking"][0]["best_evidence_text"].lower())
        self.assertIn("conditions", bundle["candidate_ranking"][0]["matched_fields"])

    def test_trial_chunk_retrieval_respects_candidate_nct_filter(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCT1000",
                    "display_title": "Trial 1000",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "conditions": ["Disease A"],
                    "interventions": ["Drug A"],
                },
                {
                    "nct_id": "NCT2000",
                    "display_title": "Trial 2000",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "conditions": ["Disease B"],
                    "interventions": ["Drug B"],
                },
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCT1000::overview::0",
                    "nct_id": "NCT1000",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Disease A with Drug A",
                    "embedding_text": "Disease A with Drug A",
                    "source_fields": ["conditions", "interventions"],
                    "token_estimate": 4,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCT2000::overview::0",
                    "nct_id": "NCT2000",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Disease B with Drug B",
                    "embedding_text": "Disease B with Drug B",
                    "source_fields": ["conditions", "interventions"],
                    "token_estimate": 4,
                    "rank_weight": 1.0,
                },
            ],
        )

        tool = create_trial_chunk_retrieval_tool(output_root=self.temp_root, backend="keyword")
        bundle = tool.retrieve_trials_from_structured_case(
            {
                "case_summary": "Disease B patient",
                "problem_list": ["Disease B"],
                "known_facts": ["Drug B"],
            },
            top_k=5,
            chunk_top_k=10,
            candidate_nct_ids=["NCT2000"],
        )

        self.assertEqual([row["nct_id"] for row in bundle["candidate_ranking"]], ["NCT2000"])

    def test_trial_chunk_keyword_retriever_scores_only_candidate_subset(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {"nct_id": "NCT1000", "display_title": "Trial 1000"},
                {"nct_id": "NCT2000", "display_title": "Trial 2000"},
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCT1000::overview::0",
                    "nct_id": "NCT1000",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Disease A with Drug A",
                    "embedding_text": "Disease A with Drug A",
                    "source_fields": ["conditions", "interventions"],
                    "token_estimate": 4,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCT2000::overview::0",
                    "nct_id": "NCT2000",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Disease B with Drug B",
                    "embedding_text": "Disease B with Drug B",
                    "source_fields": ["conditions", "interventions"],
                    "token_estimate": 4,
                    "rank_weight": 1.0,
                },
            ],
        )

        catalog = TrialChunkCatalog.from_output_root(self.temp_root)
        retriever = TrialChunkKeywordRetriever(catalog)
        spy_bm25 = _SpySubsetBM25({1: 3.5})
        retriever._bm25 = spy_bm25

        rows = retriever.retrieve(
            "Disease B patient",
            top_k=5,
            candidate_ids=["NCT2000::overview::0"],
        )

        self.assertEqual(rows[0]["chunk_id"], "NCT2000::overview::0")
        self.assertEqual(
            spy_bm25.calls[0]["document_indexes"],
            [1],
        )
        self.assertEqual(
            retriever.retrieve(
                "Disease B patient",
                top_k=5,
                candidate_ids=[],
            ),
            [],
        )

    def test_protocol_style_trial_retrieval_uses_vector_coarse_recall_then_reranks_trials(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCTMEL001",
                    "display_title": "Pembrolizumab for Metastatic Melanoma",
                    "brief_title": "Pembrolizumab for Metastatic Melanoma",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "primary_purpose": "Treatment",
                    "conditions": ["Metastatic Melanoma"],
                    "interventions": ["Pembrolizumab"],
                    "brief_summary": "Checkpoint inhibitor trial for metastatic melanoma.",
                    "source_url": "https://clinicaltrials.gov/show/NCTMEL001",
                },
                {
                    "nct_id": "NCTLUNG001",
                    "display_title": "Archived Lung Cancer Trial",
                    "brief_title": "Archived Lung Cancer Trial",
                    "overall_status": "Completed",
                    "phase": "Phase 3",
                    "study_type": "Interventional",
                    "primary_purpose": "Treatment",
                    "conditions": ["Lung Cancer"],
                    "interventions": ["Osimertinib"],
                    "brief_summary": "Completed targeted therapy protocol.",
                    "source_url": "https://clinicaltrials.gov/show/NCTLUNG001",
                },
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCTMEL001::overview::0",
                    "nct_id": "NCTMEL001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Pembrolizumab metastatic melanoma trial",
                    "embedding_text": "Pembrolizumab metastatic melanoma checkpoint inhibitor trial",
                    "source_fields": ["display_title", "conditions", "interventions"],
                    "token_estimate": 12,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCTMEL001::eligibility_inclusion::0",
                    "nct_id": "NCTMEL001",
                    "chunk_type": "eligibility_inclusion",
                    "sequence": 0,
                    "text": "Inclusion criteria include histologically confirmed metastatic melanoma.",
                    "embedding_text": "eligibility metastatic melanoma pembrolizumab checkpoint inhibitor",
                    "source_fields": ["eligibility_text", "eligibility_inclusion_text"],
                    "token_estimate": 14,
                    "rank_weight": 1.2,
                },
                {
                    "chunk_id": "NCTLUNG001::overview::0",
                    "nct_id": "NCTLUNG001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Osimertinib lung cancer trial",
                    "embedding_text": "Osimertinib targeted therapy lung cancer trial",
                    "source_fields": ["display_title", "conditions", "interventions"],
                    "token_estimate": 10,
                    "rank_weight": 1.0,
                },
            ],
        )

        tool = create_trial_chunk_retrieval_tool(
            output_root=self.temp_root,
            backend="hybrid",
            vector_retriever=_FakeTrialChunkVectorRetriever(
                [
                    {
                        "document_id": "NCTMEL001::eligibility_inclusion::0",
                        "chunk_id": "NCTMEL001::eligibility_inclusion::0",
                        "nct_id": "NCTMEL001",
                        "title": "Pembrolizumab for Metastatic Melanoma [eligibility_inclusion]",
                        "summary": "metastatic melanoma checkpoint inhibitor",
                        "purpose": "eligibility inclusion",
                        "eligibility": "metastatic melanoma",
                        "score": 0.98,
                    },
                    {
                        "document_id": "NCTLUNG001::overview::0",
                        "chunk_id": "NCTLUNG001::overview::0",
                        "nct_id": "NCTLUNG001",
                        "title": "Archived Lung Cancer Trial [overview]",
                        "summary": "lung cancer targeted therapy",
                        "purpose": "overview",
                        "score": 0.75,
                    },
                ]
            ),
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": "Metastatic melanoma patient considered for pembrolizumab treatment.",
                "problem_list": ["metastatic melanoma"],
                "known_facts": ["pembrolizumab", "checkpoint inhibitor"],
            },
            top_k=5,
            coarse_top_k=5,
        )

        self.assertIn("query_profile", bundle)
        self.assertTrue(bundle["coarse_candidate_ranking"])
        self.assertEqual(bundle["coarse_candidate_ids"][0], "NCTMEL001")
        self.assertTrue(bundle["bm25_top5"])
        self.assertTrue(bundle["vector_top5"])
        self.assertEqual(bundle["vector_top5"][0]["nct_id"], "NCTMEL001")
        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCTMEL001")
        self.assertEqual(bundle["candidate_ranking"][0]["status"], "trial_matched")
        self.assertTrue(bundle["candidate_ranking"][0]["enrollment_open"])
        self.assertTrue(bundle["candidate_ranking"][0]["actions"])
        self.assertTrue(tool.vector_retriever.calls)
        self.assertEqual(
            tool.vector_retriever.calls[0]["payload_filters"]["must"][0]["field"],
            "condition_terms",
        )
        completed_row = next(row for row in bundle["candidate_ranking"] if row["nct_id"] == "NCTLUNG001")
        self.assertEqual(completed_row["status"], "trial_matched")
        self.assertFalse(completed_row["enrollment_open"])

    def test_protocol_style_trial_retrieval_falls_back_to_bm25_when_vector_runtime_fails(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCTFALL001",
                    "display_title": "Glioma Radiation Trial",
                    "brief_title": "Glioma Radiation Trial",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "primary_purpose": "Treatment",
                    "conditions": ["Glioma"],
                    "interventions": ["Radiation Therapy"],
                    "brief_summary": "Radiation-focused glioma trial.",
                }
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCTFALL001::overview::0",
                    "nct_id": "NCTFALL001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Glioma radiation treatment trial",
                    "embedding_text": "Glioma radiation treatment trial",
                    "source_fields": ["conditions", "interventions"],
                    "token_estimate": 8,
                    "rank_weight": 1.0,
                }
            ],
        )

        tool = create_trial_chunk_retrieval_tool(
            output_root=self.temp_root,
            backend="hybrid",
            vector_retriever=_FakeTrialChunkVectorRetriever([], raise_on_retrieve=True),
        )

        coarse_bundle = tool.retrieve_coarse_from_structured_case(
            {
                "case_summary": "Glioma with radiation planning",
                "problem_list": ["glioma"],
                "known_facts": ["radiation"],
            },
            top_k=5,
        )
        self.assertEqual(coarse_bundle["backend_used"], "bm25")
        self.assertEqual(coarse_bundle["coarse_candidate_ids"], ["NCTFALL001"])

        bundle = tool.retrieve_from_structured_case(
            {
                "case_summary": "Glioma with radiation planning",
                "problem_list": ["glioma"],
                "known_facts": ["radiation"],
            },
            top_k=5,
            coarse_top_k=5,
        )

        self.assertEqual(bundle["backend_used"], "bm25")
        self.assertEqual(bundle["coarse_candidate_ids"], ["NCTFALL001"])
        self.assertTrue(bundle["bm25_top5"])
        self.assertEqual(bundle["vector_top5"], [])
        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCTFALL001")

    def test_trial_level_fine_rerank_prefers_compatible_trial_over_single_high_scoring_conflict(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCTGOOD001",
                    "display_title": "Open AF Stroke Prevention Trial",
                    "brief_title": "Open AF Stroke Prevention Trial",
                    "overall_status": "Recruiting",
                    "phase": "Phase 3",
                    "study_type": "Interventional",
                    "primary_purpose": "Prevention",
                    "conditions": ["Atrial Fibrillation"],
                    "condition_terms": ["Atrial Fibrillation", "Stroke"],
                    "interventions": ["Warfarin Program"],
                    "intervention_terms": ["Warfarin", "Anticoagulation"],
                    "gender": "All",
                    "age_floor_years": 50,
                    "age_ceiling_years": 90,
                    "brief_summary": "Anticoagulation for stroke prevention in atrial fibrillation.",
                },
                {
                    "nct_id": "NCTBAD001",
                    "display_title": "AF Trial With Diabetes Focus",
                    "brief_title": "AF Trial With Diabetes Focus",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "primary_purpose": "Prevention",
                    "conditions": ["Atrial Fibrillation", "Diabetes"],
                    "condition_terms": ["Atrial Fibrillation", "Diabetes"],
                    "interventions": ["Warfarin Program"],
                    "intervention_terms": ["Warfarin"],
                    "gender": "Female",
                    "age_floor_years": 18,
                    "age_ceiling_years": 60,
                    "brief_summary": "Focused on diabetic patients with atrial fibrillation.",
                },
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCTGOOD001::eligibility_inclusion::0",
                    "nct_id": "NCTGOOD001",
                    "chunk_type": "eligibility_inclusion",
                    "sequence": 0,
                    "text": "Inclusion criteria: atrial fibrillation with stroke prevention anticoagulation planning.",
                    "embedding_text": "atrial fibrillation stroke prevention anticoagulation eligibility",
                    "source_fields": ["eligibility_text", "eligibility_inclusion_text"],
                    "token_estimate": 12,
                    "rank_weight": 1.2,
                },
                {
                    "chunk_id": "NCTGOOD001::arms_interventions::0",
                    "nct_id": "NCTGOOD001",
                    "chunk_type": "arms_interventions",
                    "sequence": 0,
                    "text": "Intervention arm uses warfarin anticoagulation for prevention.",
                    "embedding_text": "warfarin anticoagulation prevention intervention arm",
                    "source_fields": ["interventions", "intervention_descriptions"],
                    "token_estimate": 10,
                    "rank_weight": 0.85,
                },
                {
                    "chunk_id": "NCTGOOD001::overview::0",
                    "nct_id": "NCTGOOD001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Overview: open prevention trial for atrial fibrillation.",
                    "embedding_text": "open prevention atrial fibrillation overview",
                    "source_fields": ["display_title", "conditions", "interventions"],
                    "token_estimate": 8,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCTBAD001::eligibility_inclusion::0",
                    "nct_id": "NCTBAD001",
                    "chunk_type": "eligibility_inclusion",
                    "sequence": 0,
                    "text": "High scoring eligibility text for diabetic female AF patients.",
                    "embedding_text": "atrial fibrillation diabetes female eligibility high score",
                    "source_fields": ["eligibility_text", "eligibility_inclusion_text"],
                    "token_estimate": 10,
                    "rank_weight": 1.2,
                },
            ],
        )

        tool = create_trial_chunk_retrieval_tool(
            output_root=self.temp_root,
            backend="vector",
            vector_retriever=_FakeTrialChunkVectorRetriever(
                [
                    {
                        "document_id": "NCTBAD001::eligibility_inclusion::0",
                        "chunk_id": "NCTBAD001::eligibility_inclusion::0",
                        "nct_id": "NCTBAD001",
                        "title": "AF Trial With Diabetes Focus [eligibility_inclusion]",
                        "summary": "diabetes female atrial fibrillation eligibility",
                        "purpose": "eligibility inclusion",
                        "score": 0.99,
                    },
                    {
                        "document_id": "NCTGOOD001::eligibility_inclusion::0",
                        "chunk_id": "NCTGOOD001::eligibility_inclusion::0",
                        "nct_id": "NCTGOOD001",
                        "title": "Open AF Stroke Prevention Trial [eligibility_inclusion]",
                        "summary": "atrial fibrillation anticoagulation prevention eligibility",
                        "purpose": "eligibility inclusion",
                        "score": 0.98,
                    },
                    {
                        "document_id": "NCTGOOD001::arms_interventions::0",
                        "chunk_id": "NCTGOOD001::arms_interventions::0",
                        "nct_id": "NCTGOOD001",
                        "title": "Open AF Stroke Prevention Trial [arms_interventions]",
                        "summary": "warfarin anticoagulation prevention intervention arm",
                        "purpose": "arms interventions",
                        "score": 0.97,
                    },
                    {
                        "document_id": "NCTGOOD001::overview::0",
                        "chunk_id": "NCTGOOD001::overview::0",
                        "nct_id": "NCTGOOD001",
                        "title": "Open AF Stroke Prevention Trial [overview]",
                        "summary": "open prevention atrial fibrillation overview",
                        "purpose": "overview",
                        "score": 0.96,
                    },
                ]
            ),
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "raw_text": (
                    "A 78-year-old male patient with hypertension recently experienced a transient ischemic attack. "
                    "He does not have diabetes or congestive heart failure and is not currently prescribed warfarin."
                ),
                "case_summary": "Older male with hypertension and prior TIA, not on warfarin.",
                "problem_list": ["hypertension", "transient ischemic attack"],
                "known_facts": ["no diabetes", "no congestive heart failure", "not on warfarin"],
            },
            top_k=5,
            coarse_top_k=5,
            chunk_top_k=10,
            backend="vector",
        )

        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCTGOOD001")
        self.assertGreater(bundle["candidate_ranking"][0]["query_signal_bonus"], 0.0)
        self.assertTrue(bundle["candidate_ranking"][0]["coverage_chunk_types"])

        conflicting_row = next(row for row in bundle["candidate_ranking"] if row["nct_id"] == "NCTBAD001")
        self.assertIn("diabetes", conflicting_row["must_not_conflicts"])
        self.assertGreater(conflicting_row["eligibility_penalty"], 0.0)
        self.assertTrue(conflicting_row["eligibility_conflicts"])
        self.assertEqual(conflicting_row["score_breakdown"]["stage"], "fine")

    def test_trial_level_fine_rerank_requires_disease_and_intervention_alignment_when_both_exist(self) -> None:
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCTREHAB001",
                    "display_title": "Stroke Rehabilitation With tDCS",
                    "brief_title": "Stroke Rehabilitation With tDCS",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "primary_purpose": "Treatment",
                    "conditions": ["Stroke Rehabilitation"],
                    "condition_terms": ["Stroke", "Stroke Rehabilitation"],
                    "interventions": ["tDCS", "Rehabilitation Training"],
                    "intervention_terms": ["tDCS", "Rehabilitation Training"],
                    "brief_summary": "Neurologic rehabilitation study using tDCS after stroke.",
                },
                {
                    "nct_id": "NCTCARD001",
                    "display_title": "Cardiac Surgery LAA Exclusion Trial",
                    "brief_title": "Cardiac Surgery LAA Exclusion Trial",
                    "overall_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "primary_purpose": "Treatment",
                    "conditions": ["Left Atrial Appendage Exclusion"],
                    "condition_terms": ["Left Atrial Appendage Exclusion"],
                    "interventions": ["AtriCure LAA Exclusion System"],
                    "intervention_terms": ["AtriCure LAA Exclusion System"],
                    "brief_summary": "Cardiac surgery trial for LAA exclusion in AF.",
                },
            ],
        )
        _write_jsonl(
            self.temp_root / "trial_chunk.jsonl",
            [
                {
                    "chunk_id": "NCTREHAB001::overview::0",
                    "nct_id": "NCTREHAB001",
                    "chunk_type": "overview",
                    "sequence": 0,
                    "text": "Stroke rehabilitation trial using tDCS and rehabilitation training.",
                    "embedding_text": "stroke rehabilitation tdcs rehabilitation training neurology trial",
                    "source_fields": ["display_title", "conditions", "interventions"],
                    "token_estimate": 12,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCTREHAB001::arms_interventions::0",
                    "nct_id": "NCTREHAB001",
                    "chunk_type": "arms_interventions",
                    "sequence": 0,
                    "text": "Intervention arm includes rehabilitation training with adjunctive tDCS.",
                    "embedding_text": "rehabilitation training tdcs intervention arm stroke recovery",
                    "source_fields": ["interventions", "intervention_descriptions"],
                    "token_estimate": 12,
                    "rank_weight": 0.9,
                },
                {
                    "chunk_id": "NCTCARD001::eligibility_inclusion::0",
                    "nct_id": "NCTCARD001",
                    "chunk_type": "eligibility_inclusion",
                    "sequence": 0,
                    "text": "Eligibility includes cardiac surgery patients with previous stroke or atrial fibrillation.",
                    "embedding_text": "previous stroke atrial fibrillation cardiac surgery eligibility",
                    "source_fields": ["eligibility_text", "eligibility_inclusion_text"],
                    "token_estimate": 12,
                    "rank_weight": 1.2,
                },
                {
                    "chunk_id": "NCTCARD001::arms_interventions::0",
                    "nct_id": "NCTCARD001",
                    "chunk_type": "arms_interventions",
                    "sequence": 0,
                    "text": "Intervention arm uses the AtriCure LAA Exclusion System during surgery.",
                    "embedding_text": "atricure laa exclusion system cardiac surgery intervention arm",
                    "source_fields": ["interventions", "intervention_descriptions"],
                    "token_estimate": 12,
                    "rank_weight": 0.9,
                },
            ],
        )

        tool = create_trial_chunk_retrieval_tool(
            output_root=self.temp_root,
            backend="vector",
            vector_retriever=_FakeTrialChunkVectorRetriever(
                [
                    {
                        "document_id": "NCTCARD001::eligibility_inclusion::0",
                        "chunk_id": "NCTCARD001::eligibility_inclusion::0",
                        "nct_id": "NCTCARD001",
                        "title": "Cardiac Surgery LAA Exclusion Trial [eligibility_inclusion]",
                        "summary": "previous stroke cardiac surgery eligibility",
                        "purpose": "eligibility inclusion",
                        "score": 0.99,
                    },
                    {
                        "document_id": "NCTCARD001::arms_interventions::0",
                        "chunk_id": "NCTCARD001::arms_interventions::0",
                        "nct_id": "NCTCARD001",
                        "title": "Cardiac Surgery LAA Exclusion Trial [arms_interventions]",
                        "summary": "atricure laa exclusion system intervention arm",
                        "purpose": "arms interventions",
                        "score": 0.98,
                    },
                    {
                        "document_id": "NCTREHAB001::overview::0",
                        "chunk_id": "NCTREHAB001::overview::0",
                        "nct_id": "NCTREHAB001",
                        "title": "Stroke Rehabilitation With tDCS [overview]",
                        "summary": "stroke rehabilitation tdcs",
                        "purpose": "overview",
                        "score": 0.95,
                    },
                    {
                        "document_id": "NCTREHAB001::arms_interventions::0",
                        "chunk_id": "NCTREHAB001::arms_interventions::0",
                        "nct_id": "NCTREHAB001",
                        "title": "Stroke Rehabilitation With tDCS [arms_interventions]",
                        "summary": "rehabilitation training tdcs intervention arm",
                        "purpose": "arms interventions",
                        "score": 0.94,
                    },
                ]
            ),
        )

        bundle = tool.retrieve_from_structured_case(
            {
                "raw_text": (
                    "Patient is recovering after ischemic stroke and is receiving rehabilitation training "
                    "with tDCS localization treatment."
                ),
                "case_summary": "Post-stroke rehabilitation with tDCS.",
                "problem_list": ["post-stroke functional impairment"],
                "known_facts": [
                    "rehabilitation training provided",
                    "tDCS localization treatment provided",
                ],
            },
            top_k=5,
            coarse_top_k=5,
            chunk_top_k=10,
            backend="vector",
        )

        self.assertIn("stroke", [term.casefold() for term in bundle["query_profile"]["trial_condition_terms"]])
        self.assertTrue(bundle["query_profile"]["trial_intervention_terms"])
        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCTREHAB001")

        cardiac_row = next(row for row in bundle["candidate_ranking"] if row["nct_id"] == "NCTCARD001")
        self.assertIn(
            "No trial intervention term matched the case treatment focus.",
            cardiac_row["eligibility_conflicts"],
        )
        self.assertGreater(cardiac_row["eligibility_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
