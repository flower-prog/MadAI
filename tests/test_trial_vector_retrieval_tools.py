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


from agent.tools.trial_vector_retrieval_tools import create_trial_chunk_retrieval_tool


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

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        candidate_pmids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, object]]:
        del query
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


class TrialVectorRetrievalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"trial-vector-retrieval-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

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

        self.assertEqual(bundle["coarse_candidate_ids"][0], "NCTMEL001")
        self.assertTrue(bundle["bm25_top5"])
        self.assertTrue(bundle["vector_top5"])
        self.assertEqual(bundle["vector_top5"][0]["nct_id"], "NCTMEL001")
        self.assertEqual(bundle["candidate_ranking"][0]["nct_id"], "NCTMEL001")
        self.assertEqual(bundle["candidate_ranking"][0]["status"], "trial_matched")
        self.assertTrue(bundle["candidate_ranking"][0]["enrollment_open"])
        self.assertTrue(bundle["candidate_ranking"][0]["actions"])
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


if __name__ == "__main__":
    unittest.main()
