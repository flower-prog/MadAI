from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.retrieval import (
    QdrantTrialChunkIndexManager,
    QdrantTrialChunkVectorRetriever,
    build_qdrant_trial_chunk_payload,
)
from agent.retrieval.qdrant.trial_chunk import _qdrant_point_id
from agent.tools.trial_vector_retrieval_tools import (
    TrialChunkCatalog,
    create_trial_chunk_retrieval_tool,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class _FakeModels:
    class Distance:
        DOT = "dot"

    class PayloadSchemaType:
        KEYWORD = "keyword"
        INTEGER = "integer"
        FLOAT = "float"

    class VectorParams:
        def __init__(self, *, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, *, id: str, vector: list[float], payload: dict[str, object]) -> None:
            self.id = id
            self.vector = list(vector)
            self.payload = dict(payload)

    class HasIdCondition:
        def __init__(self, *, has_id: list[str]) -> None:
            self.has_id = list(has_id)

    class MatchAny:
        def __init__(self, *, any: list[str] | None = None, values: list[str] | None = None) -> None:
            self.any = list(any or values or [])

    class Range:
        def __init__(self, *, gte: float | None = None, lte: float | None = None) -> None:
            self.gte = gte
            self.lte = lte

    class FieldCondition:
        def __init__(self, *, key: str, match=None, range=None) -> None:
            self.key = key
            self.match = match
            self.range = range

    class Filter:
        def __init__(
            self,
            *,
            must: list[object] | None = None,
            should: list[object] | None = None,
            must_not: list[object] | None = None,
        ) -> None:
            self.must = list(must or [])
            self.should = list(should or [])
            self.must_not = list(must_not or [])


class _FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def vector_size(self) -> int:
        return 3

    def encode_documents(self, documents) -> list[list[float]]:
        rows = [str(getattr(document, "chunk_id", "") or "") for document in list(documents or [])]
        self.document_calls.append(rows)
        return [[float(index + 1), 0.0, 0.5] for index, _ in enumerate(rows)]

    def encode_query(self, query: str) -> list[float]:
        self.query_calls.append(str(query))
        return [0.1, 0.2, 0.3]


class _FakeQdrantClient:
    def __init__(self, *, hits: list[object] | None = None) -> None:
        self.collections: set[str] = set()
        self.create_collection_calls: list[dict[str, object]] = []
        self.delete_collection_calls: list[str] = []
        self.create_payload_index_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []
        self.hits = list(hits or [])

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.collections.add(collection_name)
        self.create_collection_calls.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
            }
        )

    def delete_collection(self, *, collection_name: str) -> None:
        self.collections.discard(collection_name)
        self.delete_collection_calls.append(collection_name)

    def create_payload_index(self, *, collection_name: str, field_name: str, field_schema=None, field_type=None) -> None:
        self.create_payload_index_calls.append(
            {
                "collection_name": collection_name,
                "field_name": field_name,
                "field_schema": field_schema,
                "field_type": field_type,
            }
        )

    def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points": list(points),
                "wait": wait,
            }
        )

    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        query_filter: object,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> list[object]:
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query_vector": list(query_vector),
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return list(self.hits[:limit])


class TrialQdrantToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"trial-qdrant-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            self.temp_root / "trial_record.jsonl",
            [
                {
                    "nct_id": "NCTMEL001",
                    "display_title": "Pembrolizumab for Metastatic Melanoma",
                    "brief_title": "Pembrolizumab for Metastatic Melanoma",
                    "official_title": "Pembrolizumab for Metastatic Melanoma",
                    "overall_status": "Recruiting",
                    "normalized_status": "Recruiting",
                    "phase": "Phase 2",
                    "study_type": "Interventional",
                    "primary_purpose": "Treatment",
                    "conditions": ["Metastatic Melanoma"],
                    "condition_terms": ["Metastatic Melanoma", "Melanoma"],
                    "interventions": ["Pembrolizumab"],
                    "intervention_terms": ["Pembrolizumab", "PD-1 inhibitor"],
                    "gender": "All",
                    "age_floor_years": 18.0,
                    "age_ceiling_years": 75.0,
                    "source_url": "https://clinicaltrials.gov/show/NCTMEL001",
                    "source_corpus": "totrials_2021_2022",
                    "source_archive": "batch001.zip",
                    "source_member_path": "NCTMEL001.xml",
                    "xml_sha256": "abc123",
                    "has_results_references": True,
                }
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
                    "text": "title: Pembrolizumab for Metastatic Melanoma\nconditions: metastatic melanoma",
                    "embedding_text": "title: Pembrolizumab for Metastatic Melanoma\nconditions: metastatic melanoma\ninterventions: pembrolizumab",
                    "source_fields": ["display_title", "conditions", "interventions"],
                    "token_estimate": 24,
                    "rank_weight": 1.0,
                },
                {
                    "chunk_id": "NCTMEL001::eligibility_inclusion::0",
                    "nct_id": "NCTMEL001",
                    "chunk_type": "eligibility_inclusion",
                    "sequence": 0,
                    "text": "Inclusion criteria: histologically confirmed metastatic melanoma.",
                    "embedding_text": "title: Pembrolizumab for Metastatic Melanoma\nchunk type: eligibility_inclusion\ninclusion criteria: histologically confirmed metastatic melanoma.",
                    "source_fields": ["eligibility_text", "eligibility_inclusion_text"],
                    "token_estimate": 22,
                    "rank_weight": 1.2,
                },
            ],
        )
        self.catalog = TrialChunkCatalog.from_output_root(self.temp_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_build_qdrant_trial_chunk_payload_keeps_record_and_chunk_metadata(self) -> None:
        document = self.catalog.get("NCTMEL001::eligibility_inclusion::0")
        self.assertIsNotNone(document)

        payload = build_qdrant_trial_chunk_payload(document)

        self.assertEqual(payload["chunk_id"], "NCTMEL001::eligibility_inclusion::0")
        self.assertEqual(payload["nct_id"], "NCTMEL001")
        self.assertEqual(payload["chunk_type"], "eligibility_inclusion")
        self.assertEqual(payload["display_title"], "Pembrolizumab for Metastatic Melanoma")
        self.assertEqual(payload["phase"], "Phase 2")
        self.assertEqual(payload["conditions"], ["Metastatic Melanoma"])
        self.assertEqual(payload["intervention_terms"], ["Pembrolizumab", "PD-1 inhibitor"])
        self.assertEqual(payload["age_floor_years"], 18.0)
        self.assertTrue(payload["has_results_references"])

    def test_index_manager_creates_collection_indexes_and_upserts_batches(self) -> None:
        fake_client = _FakeQdrantClient()
        fake_embedder = _FakeEmbedder()
        manager = QdrantTrialChunkIndexManager(
            self.catalog,
            collection_name="trial_chunks_test",
            client=fake_client,
            models=_FakeModels,
            embedding_function=fake_embedder,
        )

        summary = manager.sync_catalog(recreate=True, batch_size=1)

        self.assertEqual(summary["point_count"], 2)
        self.assertEqual(summary["batch_count"], 2)
        self.assertEqual(len(fake_client.create_collection_calls), 1)
        vector_params = fake_client.create_collection_calls[0]["vectors_config"]
        self.assertEqual(vector_params.size, 3)
        self.assertEqual(vector_params.distance, _FakeModels.Distance.DOT)
        indexed_fields = {row["field_name"] for row in fake_client.create_payload_index_calls}
        self.assertIn("nct_id", indexed_fields)
        self.assertIn("chunk_type", indexed_fields)
        self.assertIn("condition_terms", indexed_fields)
        self.assertEqual(
            [point.id for call in fake_client.upsert_calls for point in call["points"]],
            [
                _qdrant_point_id("NCTMEL001::overview::0"),
                _qdrant_point_id("NCTMEL001::eligibility_inclusion::0"),
            ],
        )

    def test_qdrant_retriever_builds_has_id_filter_and_serializes_hits(self) -> None:
        fake_client = _FakeQdrantClient(
            hits=[
                SimpleNamespace(
                    id="NCTMEL001::overview::0",
                    score=0.92,
                    payload={
                        "chunk_id": "NCTMEL001::overview::0",
                        "nct_id": "NCTMEL001",
                        "chunk_type": "overview",
                        "title": "Pembrolizumab for Metastatic Melanoma [overview]",
                        "summary": "metastatic melanoma trial overview",
                        "purpose": "overview",
                        "source_fields": ["conditions", "interventions"],
                        "rank_weight": 1.0,
                    },
                )
            ]
        )
        fake_embedder = _FakeEmbedder()
        retriever = QdrantTrialChunkVectorRetriever(
            self.catalog,
            collection_name="trial_chunks_test",
            client=fake_client,
            models=_FakeModels,
            embedding_function=fake_embedder,
        )

        rows = retriever.retrieve(
            "metastatic melanoma pembrolizumab",
            top_k=5,
            candidate_ids={"NCTMEL001::overview::0"},
        )

        self.assertEqual(rows[0]["chunk_id"], "NCTMEL001::overview::0")
        self.assertEqual(rows[0]["nct_id"], "NCTMEL001")
        self.assertEqual(rows[0]["score"], 0.92)
        self.assertEqual(fake_embedder.query_calls, ["metastatic melanoma pembrolizumab"])
        query_filter = fake_client.search_calls[0]["query_filter"]
        self.assertIsInstance(query_filter, _FakeModels.Filter)
        self.assertEqual(query_filter.must[0].has_id, [_qdrant_point_id("NCTMEL001::overview::0")])

    def test_qdrant_retriever_merges_payload_filters_with_candidate_filter(self) -> None:
        fake_client = _FakeQdrantClient(
            hits=[
                SimpleNamespace(
                    id="NCTMEL001::overview::0",
                    score=0.92,
                    payload={
                        "chunk_id": "NCTMEL001::overview::0",
                        "nct_id": "NCTMEL001",
                        "chunk_type": "overview",
                        "title": "Pembrolizumab for Metastatic Melanoma [overview]",
                        "summary": "metastatic melanoma trial overview",
                        "purpose": "overview",
                        "source_fields": ["conditions", "interventions"],
                        "rank_weight": 1.0,
                        "gender": "All",
                        "age_floor_years": 18.0,
                        "age_ceiling_years": 75.0,
                    },
                )
            ]
        )
        retriever = QdrantTrialChunkVectorRetriever(
            self.catalog,
            collection_name="trial_chunks_test",
            client=fake_client,
            models=_FakeModels,
            embedding_function=_FakeEmbedder(),
        )

        retriever.retrieve(
            "metastatic melanoma pembrolizumab",
            top_k=5,
            candidate_ids={"NCTMEL001::overview::0"},
            payload_filters={
                "must": [{"field": "condition_terms", "values": ["Melanoma"]}],
                "should": [{"field": "intervention_terms", "values": ["Pembrolizumab"]}],
                "must_not": [{"field": "condition_terms", "values": ["Lung Cancer"]}],
                "gender": "Male",
                "age_years": 60,
            },
        )

        query_filter = fake_client.search_calls[0]["query_filter"]
        self.assertIsInstance(query_filter, _FakeModels.Filter)
        self.assertEqual(query_filter.must[0].has_id, [_qdrant_point_id("NCTMEL001::overview::0")])
        field_conditions = [item for item in query_filter.must[1:] if isinstance(item, _FakeModels.FieldCondition)]
        self.assertTrue(any(item.key == "condition_terms" and item.match.any == ["Melanoma"] for item in field_conditions))
        self.assertTrue(any(item.key == "gender" and item.match.any == ["All", "Male"] for item in field_conditions))
        self.assertTrue(any(item.key == "age_floor_years" and item.range.lte == 60 for item in field_conditions))
        self.assertTrue(any(item.key == "age_ceiling_years" and item.range.gte == 60 for item in field_conditions))
        self.assertTrue(
            any(
                isinstance(item, _FakeModels.FieldCondition)
                and item.key == "intervention_terms"
                and item.match.any == ["Pembrolizumab"]
                for item in query_filter.should
            )
        )
        self.assertTrue(
            any(
                isinstance(item, _FakeModels.FieldCondition)
                and item.key == "condition_terms"
                and item.match.any == ["Lung Cancer"]
                for item in query_filter.must_not
            )
        )

    def test_trial_chunk_factory_can_wire_qdrant_vector_store(self) -> None:
        sentinel_vector_retriever = object()
        with patch(
            "agent.tools.trial_vector_retrieval_tools.create_qdrant_trial_chunk_retriever",
            return_value=sentinel_vector_retriever,
        ) as mocked_factory:
            tool = create_trial_chunk_retrieval_tool(
                output_root=self.temp_root,
                backend="hybrid",
                vector_store="qdrant",
                qdrant_collection_name="trial_chunks_test",
            )

        self.assertIs(tool.vector_retriever, sentinel_vector_retriever)
        mocked_factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
