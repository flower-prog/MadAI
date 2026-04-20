from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent.retrieval import (
    DEFAULT_QDRANT_COLLECTION_NAME,
    FieldedBM25Index,
    MedCPTRetriever,
    TrialChunkCatalog,
    TrialChunkDocument,
    build_structured_query_text,
    create_qdrant_trial_chunk_retriever,
    create_structured_retriever,
    resolve_trial_qdrant_runtime_config,
    resolve_trial_vector_output_root,
)

from .execution_tools import tool


_CACHE_LOCK = threading.RLock()
_TRIAL_CHUNK_RETRIEVER_CACHE: dict[tuple[str, str, str, str, str, str], "TrialChunkRetrievalTool"] = {}
_DEFAULT_PROTOCOL_CHUNK_TOP_K = 40
_OPEN_TRIAL_STATUSES = {
    "Recruiting",
    "Not yet recruiting",
    "Enrolling by invitation",
}
_EVIDENCE_TRIAL_STATUSES = {
    "Active, not recruiting",
    "Completed",
}
_ABANDONED_TRIAL_STATUSES = {
    "Terminated",
    "Withdrawn",
    "Suspended",
    "No longer available",
    "Temporarily not available",
}


def _normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_backend_name(value: str | None) -> str:
    normalized = str(value or "hybrid").strip().lower()
    if not normalized:
        normalized = "hybrid"
    return {
        "keyword": "bm25",
        "auto": "hybrid",
        "medcpt": "vector",
    }.get(normalized, normalized)


def _normalize_vector_store_name(value: str | None) -> str:
    normalized = str(value or "faiss").strip().lower()
    if not normalized:
        normalized = "faiss"
    return {
        "default": "auto",
        "automatic": "auto",
        "local": "faiss",
        "medcpt": "faiss",
        "qdrant_local": "qdrant",
    }.get(normalized, normalized)


def _status_priority(status: str, enrollment_open: bool) -> int:
    normalized_status = str(status or "").strip().lower()
    if normalized_status == "trial_matched":
        return 3 if enrollment_open else 2
    if normalized_status == "manual_review":
        return 1
    return 0


def _map_protocol_trial_status(record: Mapping[str, Any]) -> dict[str, Any]:
    overall_status = _normalize_whitespace(record.get("overall_status"))
    if overall_status in _OPEN_TRIAL_STATUSES:
        return {
            "status": "trial_matched",
            "enrollment_open": True,
            "status_reason": (
                "The trial is actively open or preparing to open enrollment, so it can be surfaced as a direct trial candidate."
            ),
            "actions": [
                "Review detailed inclusion and exclusion criteria against the current case.",
                "Confirm site-level enrollment availability before surfacing this as a live trial option.",
            ],
        }
    if overall_status in _EVIDENCE_TRIAL_STATUSES:
        return {
            "status": "trial_matched",
            "enrollment_open": False,
            "status_reason": (
                "The trial is not currently open for enrollment, but it remains useful as protocol or evidence support."
            ),
            "actions": [
                "Use the trial as protocol or evidence support instead of assuming active enrollment.",
                "If a current trial option is needed, search for an active successor or related study before surfacing.",
            ],
        }
    if overall_status in _ABANDONED_TRIAL_STATUSES:
        return {
            "status": "abandoned",
            "enrollment_open": False,
            "status_reason": (
                "The trial lifecycle indicates that it is halted or unavailable, so it should not be treated as a current treatment-trial option."
            ),
            "actions": [
                "Do not surface this trial as a direct option for the current patient.",
                "Keep it only as historical context if the protocol remains clinically informative.",
            ],
        }
    if overall_status == "Unknown status":
        return {
            "status": "manual_review",
            "enrollment_open": False,
            "status_reason": (
                "The trial lifecycle is unclear, so manual review is required before using it in protocol recommendations."
            ),
            "actions": [
                "Review the full XML and the current ClinicalTrials.gov record manually before use.",
                "Do not promote this trial to a direct recommendation until the lifecycle status is confirmed.",
            ],
        }
    return {
        "status": "manual_review",
        "enrollment_open": False,
        "status_reason": (
            "The trial status does not map cleanly to a stable MedAI recommendation state, so it should remain under manual review."
        ),
        "actions": [
            "Review the trial lifecycle manually before using it in MedAI.",
            "Document any case-specific rationale before surfacing this trial to clinicians.",
        ],
    }


def _default_qdrant_runtime_config() -> dict[str, Any]:
    return resolve_trial_qdrant_runtime_config()


def _normalize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    raw_scores = {
        str(row.get("document_id") or row.get("chunk_id") or "").strip(): float(row.get("score") or 0.0)
        for row in list(rows or [])
        if str(row.get("document_id") or row.get("chunk_id") or "").strip()
    }
    if not raw_scores:
        return {}

    values = list(raw_scores.values())
    max_score = max(values)
    min_score = min(values)
    if max_score == min_score:
        if max_score <= 0.0:
            return {document_id: 0.0 for document_id in raw_scores}
        return {document_id: 1.0 for document_id in raw_scores}

    denominator = max_score - min_score
    return {
        document_id: (score - min_score) / denominator
        for document_id, score in raw_scores.items()
    }


class TrialChunkKeywordRetriever:
    FIELD_WEIGHTS: dict[str, float] = {
        "title": 2.8,
        "retrieval_text": 4.0,
        "purpose": 0.4,
        "eligibility": 0.8,
        "summary": 1.2,
    }

    def __init__(self, catalog: TrialChunkCatalog) -> None:
        self.catalog = catalog
        self._search_rows: list[dict[str, Any]] = []
        bm25_documents: list[dict[str, str]] = []
        for document in catalog.documents():
            self._search_rows.append(
                {
                    "document": document,
                    "title": document.title,
                    "retrieval_text": document.retrieval_text,
                    "purpose": document.purpose,
                    "eligibility": document.eligibility,
                    "summary": document.summary,
                }
            )
            bm25_documents.append(
                {
                    "title": document.title,
                    "retrieval_text": document.retrieval_text,
                    "purpose": document.purpose,
                    "eligibility": document.eligibility,
                    "summary": document.summary,
                }
            )
        self._bm25 = FieldedBM25Index(bm25_documents, field_weights=self.FIELD_WEIGHTS)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        candidate_pmids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_candidate_ids = {
            str(item).strip()
            for item in list(candidate_ids if candidate_ids is not None else candidate_pmids or [])
            if str(item).strip()
        }
        filtered_rows = (
            [
                row
                for row in self._search_rows
                if str(row["document"].chunk_id) in normalized_candidate_ids
            ]
            if normalized_candidate_ids
            else list(self._search_rows)
        )
        if not str(query or "").strip():
            return []

        score_by_id = {
            str(self._search_rows[index]["document"].chunk_id): float(score)
            for index, score in enumerate(self._bm25.score(query))
            if index < len(self._search_rows)
        }

        scored_rows: list[tuple[float, TrialChunkDocument]] = []
        for row in filtered_rows:
            score = float(score_by_id.get(str(row["document"].chunk_id)) or 0.0)
            if score <= 0.0:
                continue
            scored_rows.append((score, row["document"]))

        scored_rows.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].chunk_id))
        return [self._serialize(document, score) for score, document in scored_rows[: max(int(top_k), 1)]]

    @staticmethod
    def _serialize(document: TrialChunkDocument, score: float) -> dict[str, Any]:
        payload = document.to_brief()
        payload["score"] = float(score)
        return payload


class TrialChunkRetrievalTool:
    def __init__(
        self,
        catalog: TrialChunkCatalog,
        *,
        vector_retriever: Any | None = None,
        backend: str = "hybrid",
    ) -> None:
        self.catalog = catalog
        self.vector_retriever = vector_retriever
        self.keyword_retriever = TrialChunkKeywordRetriever(catalog)
        self._retriever = create_structured_retriever(
            catalog,
            bm25_retriever=self.keyword_retriever,
            vector_retriever=vector_retriever,
            query_builder=build_structured_query_text,
            default_backend=backend,
            id_field="chunk_id",
        )
        self.default_backend = self._retriever.resolve_backend(backend)

    @property
    def available_backends(self) -> tuple[str, ...]:
        return self._retriever.available_backends

    def resolve_backend(self, backend: str | None = None) -> str:
        return self._retriever.resolve_backend(backend or self.default_backend)

    def _retrieve_query_bundle(
        self,
        query_text: str,
        *,
        top_k: int,
        candidate_chunk_ids: set[str] | None,
        backend: str,
    ) -> dict[str, Any]:
        try:
            return self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=candidate_chunk_ids,
                backend=backend,
                include_scores=True,
            )
        except Exception:
            if backend == "bm25":
                raise
            return self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=candidate_chunk_ids,
                backend="bm25",
                include_scores=True,
            )

    def _channel_rows(
        self,
        query_text: str,
        *,
        top_k: int,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None,
        backend: str,
    ) -> list[dict[str, Any]]:
        candidate_chunk_ids = self._resolve_candidate_chunk_ids(candidate_nct_ids)
        if backend == "vector" and self.vector_retriever is None:
            return []
        try:
            bundle = self._retriever.retrieve_from_query(
                query_text,
                top_k=max(int(top_k), 1),
                candidate_ids=candidate_chunk_ids,
                backend=backend,
                include_scores=True,
            )
        except Exception:
            return []
        return [dict(row) for row in list(bundle.get("candidate_ranking") or [])]

    def _enrich_protocol_candidates(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for row in list(rows or []):
            nct_id = str(row.get("nct_id") or "").strip()
            if not nct_id:
                continue
            record = self.catalog.get_record(nct_id) or {"nct_id": nct_id}
            status_meta = _map_protocol_trial_status(record)
            display_title = _normalize_whitespace(
                record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
            )
            enriched.append(
                {
                    **dict(row),
                    "nct_id": nct_id,
                    "title": display_title,
                    "name": display_title,
                    "display_title": display_title,
                    "brief_summary": _normalize_whitespace(record.get("brief_summary")),
                    "status": str(status_meta.get("status") or ""),
                    "enrollment_open": bool(status_meta.get("enrollment_open")),
                    "status_reason": _normalize_whitespace(status_meta.get("status_reason")),
                    "actions": list(status_meta.get("actions") or []),
                    "overall_status": _normalize_whitespace(record.get("overall_status") or row.get("overall_status")),
                    "study_type": _normalize_whitespace(record.get("study_type") or row.get("study_type")),
                    "phase": _normalize_whitespace(record.get("phase") or row.get("phase")),
                    "primary_purpose": _normalize_whitespace(record.get("primary_purpose")),
                    "conditions": list(record.get("conditions") or row.get("conditions") or []),
                    "interventions": list(record.get("interventions") or row.get("interventions") or []),
                    "source_url": _normalize_whitespace(record.get("source_url") or row.get("source_url")),
                }
            )
        enriched.sort(
            key=lambda row: (
                -_status_priority(str(row.get("status") or ""), bool(row.get("enrollment_open"))),
                -float(row.get("score") or 0.0),
                -len(list(row.get("matched_chunks") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("nct_id") or ""),
            )
        )
        return enriched

    @tool(
        name="trial_chunk_retriever",
        description=(
            "Retrieve protocol-stage trial chunks from the local XML-derived trial vector KB. "
            "Use this when the caller wants chunk evidence instead of department payload retrieval."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
            "candidate_nct_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_chunks(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 20,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_chunks_from_structured_case(
            structured_case,
            top_k=top_k,
            candidate_nct_ids=candidate_nct_ids,
            backend=backend,
        )

    @tool(
        name="trial_vector_candidate_retriever",
        description=(
            "Aggregate chunk-level retrieval over the XML-derived trial KB into trial-level candidates "
            "with matched_chunks and best_evidence_text."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
            "chunk_top_k": "int",
            "candidate_nct_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_trials(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        chunk_top_k: int = 40,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_trials_from_structured_case(
            structured_case,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            candidate_nct_ids=candidate_nct_ids,
            backend=backend,
        )

    def retrieve_chunks_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 20,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        query_text = self._build_query_text(case_payload)
        resolved_backend = self.resolve_backend(backend)
        candidate_chunk_ids = self._resolve_candidate_chunk_ids(candidate_nct_ids)
        if not query_text:
            return {
                "query_text": "",
                "backend_used": resolved_backend,
                "available_backends": list(self.available_backends),
                "hits": [],
            }

        payload = self._retrieve_query_bundle(
            query_text,
            top_k=max(int(top_k), 1),
            candidate_chunk_ids=candidate_chunk_ids,
            backend=resolved_backend,
        )
        hits = self._hydrate_chunk_rows(list(payload.get("candidate_ranking") or []))
        return {
            "query_text": query_text,
            "backend_used": str(payload.get("backend_used") or resolved_backend),
            "available_backends": list(payload.get("available_backends") or self.available_backends),
            "hits": hits,
        }

    def retrieve_trials_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        chunk_top_k: int = 40,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        query_text = self._build_query_text(case_payload)
        resolved_backend = self.resolve_backend(backend)
        candidate_chunk_ids = self._resolve_candidate_chunk_ids(candidate_nct_ids)

        if not query_text:
            return {
                "query_text": "",
                "backend_used": resolved_backend,
                "available_backends": list(self.available_backends),
                "bm25_chunks": [],
                "vector_chunks": [],
                "candidate_ranking": [],
            }

        bm25_rows = self._channel_rows(
            query_text,
            top_k=max(int(chunk_top_k), 1),
            candidate_nct_ids=candidate_nct_ids,
            backend="bm25",
        )

        vector_rows: list[dict[str, Any]] = []
        if self.vector_retriever is not None and resolved_backend != "bm25":
            vector_rows = self._channel_rows(
                query_text,
                top_k=max(int(chunk_top_k), 1),
                candidate_nct_ids=candidate_nct_ids,
                backend="vector",
            )

        candidate_ranking = self._aggregate_trials(
            bm25_rows=bm25_rows,
            vector_rows=vector_rows,
            limit=max(int(top_k), 1),
        )

        return {
            "query_text": query_text,
            "backend_used": "bm25" if not vector_rows else "hybrid",
            "available_backends": list(self.available_backends),
            "bm25_chunks": self._hydrate_chunk_rows(bm25_rows[:5]),
            "vector_chunks": self._hydrate_chunk_rows(vector_rows[:5]),
            "candidate_ranking": candidate_ranking,
        }

    def retrieve_coarse_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        del department_tags
        case_payload = dict(structured_case or {})
        query_text = self._build_query_text(case_payload)
        resolved_backend = self.resolve_backend(backend)
        if not query_text:
            return {
                "query_text": "",
                "backend_used": "bm25" if resolved_backend == "bm25" or self.vector_retriever is None else "vector",
                "available_backends": list(self.available_backends),
                "department_tags": list(case_payload.get("department_tags") or []),
                "fallback_to_full_catalog": False,
                "candidate_ranking": [],
                "coarse_candidate_ids": [],
            }
        normalized_candidate_ids = {
            str(item).strip()
            for item in list(candidate_ids or [])
            if str(item).strip()
        } if candidate_ids is not None else None
        coarse_backend = "vector" if self.vector_retriever is not None and resolved_backend != "bm25" else "bm25"
        coarse_payload = self._retrieve_query_bundle(
            query_text,
            top_k=max(int(top_k) * 4, int(top_k), 1),
            candidate_chunk_ids=self._resolve_candidate_chunk_ids(normalized_candidate_ids),
            backend=coarse_backend,
        )
        actual_backend = str(coarse_payload.get("backend_used") or coarse_backend).strip() or coarse_backend
        chunk_rows = [dict(row) for row in list(coarse_payload.get("candidate_ranking") or [])]
        coarse_ranked_rows = self._aggregate_trials(
            bm25_rows=chunk_rows if actual_backend == "bm25" else [],
            vector_rows=chunk_rows if actual_backend != "bm25" else [],
            limit=max(int(top_k), 1),
        )
        enriched_candidates = self._enrich_protocol_candidates(coarse_ranked_rows)
        return {
            "query_text": query_text,
            "backend_used": actual_backend,
            "available_backends": list(coarse_payload.get("available_backends") or self.available_backends),
            "department_tags": list(case_payload.get("department_tags") or []),
            "fallback_to_full_catalog": False,
            "candidate_ranking": [
                {
                    "nct_id": str(row.get("nct_id") or ""),
                    "title": str(row.get("title") or ""),
                    "status": str(row.get("status") or ""),
                    "enrollment_open": bool(row.get("enrollment_open")),
                }
                for row in enriched_candidates
            ],
            "coarse_candidate_ids": [
                str(row.get("nct_id") or "")
                for row in enriched_candidates
                if str(row.get("nct_id") or "").strip()
            ],
        }

    @tool(
        name="trial_coarse_retriever",
        description=(
            "Coarsely recall local trial candidates from the XML-derived trial KB. "
            "Return a minimal NCT/title bundle for protocol-stage candidate pooling."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "department_tags": "list[str]",
            },
            "top_k": "int",
            "department_tags": "list[str] | None",
            "candidate_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_coarse(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 30,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_coarse_from_structured_case(
            structured_case,
            top_k=top_k,
            department_tags=department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

    def retrieve_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        coarse_top_k: int = 30,
        chunk_top_k: int = _DEFAULT_PROTOCOL_CHUNK_TOP_K,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        normalized_department_tags = list(
            department_tags if department_tags is not None else case_payload.get("department_tags") or []
        )
        query_text = self._build_query_text(case_payload)
        resolved_backend = self.resolve_backend(backend)
        coarse_bundle = self.retrieve_coarse_from_structured_case(
            case_payload,
            top_k=coarse_top_k,
            department_tags=normalized_department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )
        coarse_candidate_ids = [
            str(item).strip()
            for item in list(coarse_bundle.get("coarse_candidate_ids") or [])
            if str(item).strip()
        ]
        if not query_text or not coarse_candidate_ids:
            return {
                "query_text": query_text,
                "backend_used": "bm25" if resolved_backend == "bm25" or self.vector_retriever is None else "hybrid",
                "available_backends": list(self.available_backends),
                "department_tags": normalized_department_tags,
                "fallback_to_full_catalog": False,
                "coarse_candidate_ids": coarse_candidate_ids,
                "bm25_top5": [],
                "vector_top5": [],
                "candidate_ranking": [],
            }

        bm25_rows = self._channel_rows(
            query_text,
            top_k=max(int(chunk_top_k), 1),
            candidate_nct_ids=coarse_candidate_ids,
            backend="bm25",
        )
        vector_rows: list[dict[str, Any]] = []
        if self.vector_retriever is not None and resolved_backend != "bm25":
            vector_rows = self._channel_rows(
                query_text,
                top_k=max(int(chunk_top_k), 1),
                candidate_nct_ids=coarse_candidate_ids,
                backend="vector",
            )

        bm25_top5 = self._enrich_protocol_candidates(
            self._aggregate_trials(
                bm25_rows=bm25_rows,
                vector_rows=[],
                limit=min(max(int(top_k), 1), 5),
            )
        )
        vector_top5 = self._enrich_protocol_candidates(
            self._aggregate_trials(
                bm25_rows=[],
                vector_rows=vector_rows,
                limit=min(max(int(top_k), 1), 5),
            )
        )
        candidate_ranking = self._enrich_protocol_candidates(
            self._aggregate_trials(
                bm25_rows=bm25_rows,
                vector_rows=vector_rows,
                limit=max(int(top_k), 1),
            )
        )

        return {
            "query_text": query_text,
            "backend_used": "bm25" if not vector_rows else "hybrid",
            "available_backends": list(self.available_backends),
            "department_tags": normalized_department_tags,
            "fallback_to_full_catalog": False,
            "coarse_candidate_ids": coarse_candidate_ids,
            "bm25_top5": bm25_top5,
            "vector_top5": vector_top5,
            "candidate_ranking": candidate_ranking,
        }

    @tool(
        name="trial_candidate_retriever",
        description=(
            "Run the protocol-stage two-stage trial retrieval over the XML-derived trial KB. "
            "Return coarse candidate ids, bm25 top5, vector top5, and the merged top candidate ranking."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "department_tags": "list[str]",
            },
            "top_k": "int",
            "coarse_top_k": "int",
            "chunk_top_k": "int",
            "department_tags": "list[str] | None",
            "candidate_ids": "list[str] | set[str] | None",
            "backend": "str | None",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_candidates(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        coarse_top_k: int = 30,
        chunk_top_k: int = _DEFAULT_PROTOCOL_CHUNK_TOP_K,
        department_tags: list[str] | None = None,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        return self.retrieve_from_structured_case(
            structured_case,
            top_k=top_k,
            coarse_top_k=coarse_top_k,
            chunk_top_k=chunk_top_k,
            department_tags=department_tags,
            candidate_ids=candidate_ids,
            backend=backend,
        )

    def _build_query_text(self, structured_case: Mapping[str, Any]) -> str:
        return build_structured_query_text(
            raw_text=structured_case.get("raw_text") or structured_case.get("raw_request") or "",
            case_summary=structured_case.get("case_summary"),
            problem_list=structured_case.get("problem_list"),
            known_facts=structured_case.get("known_facts"),
        )

    def _resolve_candidate_chunk_ids(
        self,
        candidate_nct_ids: list[str] | set[str] | tuple[str, ...] | None,
    ) -> set[str] | None:
        if candidate_nct_ids is None:
            return None
        normalized_nct_ids = {
            str(item).strip()
            for item in list(candidate_nct_ids or [])
            if str(item).strip()
        }
        candidate_chunk_ids: set[str] = set()
        for nct_id in normalized_nct_ids:
            candidate_chunk_ids.update(self.catalog.chunk_index_by_trial.get(nct_id, set()))
        return candidate_chunk_ids

    def _hydrate_chunk_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for row in list(rows or []):
            chunk_id = str(row.get("chunk_id") or row.get("document_id") or "").strip()
            if not chunk_id:
                continue
            document = self.catalog.get(chunk_id)
            if document is None:
                continue
            candidate = {
                "chunk_id": chunk_id,
                "nct_id": document.nct_id,
                "title": document.title,
                "chunk_type": document.chunk_type,
                "text": document.text,
                "source_fields": list(document.source_fields),
                "rank_weight": float(document.rank_weight),
                "score": float(row.get("score") or 0.0),
                "match_sources": sorted(set(list(row.get("match_sources") or []))),
            }
            hydrated.append(candidate)
        return hydrated

    def _aggregate_trials(
        self,
        *,
        bm25_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized_bm25_scores = _normalize_scores(bm25_rows)
        normalized_vector_scores = _normalize_scores(vector_rows)
        grouped_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        merged: dict[str, dict[str, Any]] = {}

        for row in list(bm25_rows or []):
            chunk_id = str(row.get("chunk_id") or row.get("document_id") or "").strip()
            if not chunk_id:
                continue
            merged.setdefault(chunk_id, {})
            merged[chunk_id]["bm25_row"] = dict(row)

        for row in list(vector_rows or []):
            chunk_id = str(row.get("chunk_id") or row.get("document_id") or "").strip()
            if not chunk_id:
                continue
            merged.setdefault(chunk_id, {})
            merged[chunk_id]["vector_row"] = dict(row)

        for chunk_id, raw_bundle in merged.items():
            document = self.catalog.get(chunk_id)
            if document is None:
                continue
            combined_score = float(normalized_bm25_scores.get(chunk_id) or 0.0) + float(
                normalized_vector_scores.get(chunk_id) or 0.0
            )
            weighted_score = combined_score * float(document.rank_weight)
            chunk_payload = {
                "chunk_id": chunk_id,
                "chunk_type": document.chunk_type,
                "score": float(weighted_score),
                "bm25_score": float((raw_bundle.get("bm25_row") or {}).get("score") or 0.0),
                "vector_score": float((raw_bundle.get("vector_row") or {}).get("score") or 0.0),
                "match_sources": sorted(
                    {
                        *list((raw_bundle.get("bm25_row") or {}).get("match_sources") or []),
                        *list((raw_bundle.get("vector_row") or {}).get("match_sources") or []),
                    }
                ),
                "source_fields": list(document.source_fields),
                "text": document.text,
            }
            grouped_chunks[document.nct_id].append(chunk_payload)

        ranked_trials: list[dict[str, Any]] = []
        for nct_id, chunk_rows in grouped_chunks.items():
            record = self.catalog.get_record(nct_id) or {"nct_id": nct_id}
            chunk_rows.sort(
                key=lambda row: (
                    -float(row.get("score") or 0.0),
                    str(row.get("chunk_type") or ""),
                    str(row.get("chunk_id") or ""),
                )
            )
            best_chunk = chunk_rows[0]
            bonus = sum(float(row.get("score") or 0.0) for row in chunk_rows[1:3]) * 0.15
            matched_fields = sorted(
                {
                    field_name
                    for row in chunk_rows[:3]
                    for field_name in list(row.get("source_fields") or [])
                    if str(field_name).strip()
                }
            )
            ranked_trials.append(
                {
                    "nct_id": nct_id,
                    "display_title": _normalize_whitespace(
                        record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
                    ),
                    "title": _normalize_whitespace(
                        record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
                    ),
                    "score": float(best_chunk.get("score") or 0.0) + float(bonus),
                    "matched_chunks": [
                        {
                            "chunk_id": str(row.get("chunk_id") or ""),
                            "chunk_type": str(row.get("chunk_type") or ""),
                            "score": float(row.get("score") or 0.0),
                            "match_sources": list(row.get("match_sources") or []),
                            "source_fields": list(row.get("source_fields") or []),
                        }
                        for row in chunk_rows[:3]
                    ],
                    "best_evidence_text": str(best_chunk.get("text") or ""),
                    "matched_fields": matched_fields,
                    "overall_status": _normalize_whitespace(record.get("overall_status")),
                    "phase": _normalize_whitespace(record.get("phase")),
                    "study_type": _normalize_whitespace(record.get("study_type")),
                    "conditions": list(record.get("conditions") or []),
                    "interventions": list(record.get("interventions") or []),
                    "source_url": _normalize_whitespace(record.get("source_url")),
                }
            )

        ranked_trials.sort(
            key=lambda row: (
                -float(row.get("score") or 0.0),
                -len(list(row.get("matched_chunks") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("nct_id") or ""),
            )
        )
        return ranked_trials[: max(int(limit), 1)]


def create_trial_chunk_retrieval_tool(
    *,
    output_root: str | Path | None = None,
    backend: str = "hybrid",
    vector_retriever: Any | None = None,
    vector_store: str = "faiss",
    qdrant_collection_name: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_path: str | Path | None = None,
    qdrant_client: Any | None = None,
) -> TrialChunkRetrievalTool:
    resolved_root = resolve_trial_vector_output_root(output_root)
    normalized_backend = _normalize_backend_name(backend)
    normalized_vector_store = _normalize_vector_store_name(vector_store)
    qdrant_defaults = _default_qdrant_runtime_config()
    if normalized_vector_store == "auto":
        normalized_vector_store = "qdrant" if qdrant_defaults["enabled"] else "faiss"
    resolved_qdrant_collection_name = str(
        qdrant_collection_name
        or qdrant_defaults.get("collection_name")
        or DEFAULT_QDRANT_COLLECTION_NAME
    )
    resolved_qdrant_url = qdrant_url or qdrant_defaults.get("url")
    resolved_qdrant_api_key = qdrant_api_key or qdrant_defaults.get("api_key")
    resolved_qdrant_path = qdrant_path or qdrant_defaults.get("path")
    cache_key = (
        str(resolved_root),
        normalized_backend,
        normalized_vector_store,
        str(resolved_qdrant_collection_name or ""),
        str(resolved_qdrant_url or ""),
        str(resolved_qdrant_path or ""),
    )
    should_cache = vector_retriever is None and qdrant_client is None and not resolved_qdrant_api_key
    if should_cache:
        with _CACHE_LOCK:
            cached = _TRIAL_CHUNK_RETRIEVER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    catalog = TrialChunkCatalog.from_output_root(resolved_root)
    resolved_vector_retriever = vector_retriever
    if resolved_vector_retriever is None and normalized_backend in {"vector", "hybrid"}:
        if normalized_vector_store == "qdrant":
            try:
                resolved_vector_retriever = create_qdrant_trial_chunk_retriever(
                    catalog,
                    collection_name=resolved_qdrant_collection_name,
                    url=resolved_qdrant_url,
                    api_key=resolved_qdrant_api_key,
                    path=resolved_qdrant_path,
                    client=qdrant_client,
                )
            except Exception:
                try:
                    resolved_vector_retriever = MedCPTRetriever(catalog)
                except Exception:
                    resolved_vector_retriever = None
        else:
            try:
                resolved_vector_retriever = MedCPTRetriever(catalog)
            except Exception:
                resolved_vector_retriever = None

    tool_instance = TrialChunkRetrievalTool(
        catalog,
        vector_retriever=resolved_vector_retriever,
        backend="bm25" if resolved_vector_retriever is None else normalized_backend,
    )
    if should_cache:
        with _CACHE_LOCK:
            _TRIAL_CHUNK_RETRIEVER_CACHE.setdefault(cache_key, tool_instance)
    return tool_instance


__all__ = [
    "TrialChunkCatalog",
    "TrialChunkDocument",
    "TrialChunkKeywordRetriever",
    "TrialChunkRetrievalTool",
    "create_trial_chunk_retrieval_tool",
]
