from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import math
import re
from typing import Any


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Mapping):
        raw_items = list(value.values())
    elif isinstance(value, Iterable):
        raw_items = list(value)
    else:
        raw_items = [value]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = _normalize_whitespace(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def build_structured_query_text(
    *,
    raw_text: Any = "",
    case_summary: Any = None,
    problem_list: Any = None,
    known_facts: Any = None,
) -> str:
    parts: list[str] = []

    normalized_raw_text = _normalize_whitespace(raw_text)
    normalized_case_summary = _normalize_whitespace(case_summary)
    normalized_problem_list = _coerce_text_list(problem_list)
    normalized_known_facts = _coerce_text_list(known_facts)

    if normalized_case_summary:
        parts.append(normalized_case_summary)
    elif normalized_raw_text:
        parts.append(normalized_raw_text)

    if normalized_problem_list:
        parts.append("problem_list: " + " ; ".join(normalized_problem_list))

    if normalized_known_facts:
        parts.append("known_facts: " + " ; ".join(normalized_known_facts))

    if normalized_raw_text and normalized_raw_text not in parts:
        parts.append(normalized_raw_text)

    return "\n".join(part for part in parts if str(part).strip())


def _normalize_backend_name(backend: str | None, *, default: str = "hybrid") -> str:
    normalized = str(backend or default).strip().lower()
    if not normalized:
        normalized = default
    return {
        "auto": "hybrid",
        "keyword": "bm25",
        "parameter": "bm25",
        "medcpt": "vector",
    }.get(normalized, normalized)


def _normalize_candidate_ids(
    candidate_ids: list[str] | set[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if candidate_ids is None:
        return None
    normalized = {
        str(item).strip()
        for item in list(candidate_ids)
        if str(item).strip()
    }
    return normalized or set()


def _document_field(document: Any, name: str) -> Any:
    if isinstance(document, Mapping):
        return document.get(name)
    return getattr(document, name, None)


@dataclass(slots=True)
class StructuredRetrievalDocument:
    document_id: str
    title: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_brief(self, *, id_field: str = "document_id") -> dict[str, Any]:
        payload = {
            "document_id": self.document_id,
            "title": self.title,
            "metadata": dict(self.metadata),
        }
        if id_field != "document_id":
            payload[id_field] = self.document_id
        return payload


class StructuredRetriever:
    def __init__(
        self,
        catalog: Any,
        *,
        bm25_retriever: Any | None,
        vector_retriever: Any | None = None,
        query_builder: Callable[..., str] | None = None,
        default_backend: str = "hybrid",
        id_field: str = "document_id",
    ) -> None:
        self.catalog = catalog
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.query_builder = query_builder or build_structured_query_text
        self.id_field = str(id_field or "document_id").strip() or "document_id"
        self.default_backend = self.resolve_backend(default_backend)

    @property
    def available_backends(self) -> tuple[str, ...]:
        if self.vector_retriever is None:
            return ("bm25",)
        return ("bm25", "vector", "hybrid")

    def resolve_backend(self, backend: str | None = None) -> str:
        normalized = _normalize_backend_name(
            backend,
            default=self.default_backend if hasattr(self, "default_backend") else "hybrid",
        )
        if normalized not in {"bm25", "vector", "hybrid"}:
            normalized = "bm25"
        if normalized in {"vector", "hybrid"} and self.vector_retriever is None:
            return "bm25"
        return normalized

    def retrieve_from_query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        resolved_backend = self.resolve_backend(backend)
        normalized_candidate_ids = _normalize_candidate_ids(candidate_ids)
        requested_top_k = max(int(top_k), 1)

        bm25_rows: list[dict[str, Any]] = []
        vector_rows: list[dict[str, Any]] = []
        if resolved_backend in {"bm25", "hybrid"}:
            bm25_rows = self._retrieve_rows(
                retriever=self.bm25_retriever,
                query_text=query_text,
                top_k=requested_top_k,
                candidate_ids=normalized_candidate_ids,
                source_name="bm25",
            )
        if resolved_backend in {"vector", "hybrid"}:
            vector_rows = self._retrieve_rows(
                retriever=self.vector_retriever,
                query_text=query_text,
                top_k=requested_top_k,
                candidate_ids=normalized_candidate_ids,
                source_name="vector",
            )

        if resolved_backend == "bm25":
            ranking_rows = list(bm25_rows)
        elif resolved_backend == "vector":
            ranking_rows = list(vector_rows)
        else:
            ranking_rows = self._fuse_rows(
                bm25_rows=bm25_rows,
                vector_rows=vector_rows,
                top_k=requested_top_k,
            )

        public_bm25_rows = [
            self._public_row(row, include_scores=include_scores)
            for row in bm25_rows[:requested_top_k]
        ]
        public_vector_rows = [
            self._public_row(row, include_scores=include_scores)
            for row in vector_rows[:requested_top_k]
        ]
        public_ranking_rows = [
            self._public_row(row, include_scores=include_scores)
            for row in ranking_rows[:requested_top_k]
        ]

        return {
            "query_text": str(query_text or ""),
            "backend_used": resolved_backend,
            "available_backends": list(self.available_backends),
            "bm25_hits": public_bm25_rows,
            "vector_hits": public_vector_rows,
            "candidate_ranking": public_ranking_rows,
            "hits": public_ranking_rows,
            "retrieved_ids": [
                str(row.get("document_id") or "").strip()
                for row in public_ranking_rows
                if str(row.get("document_id") or "").strip()
            ],
        }

    def retrieve_from_structured_case(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        backend: str | None = None,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        case_payload = dict(structured_case or {})
        if isinstance(case_payload.get("structured_case"), Mapping):
            case_payload = dict(case_payload["structured_case"])
        query_text = self.query_builder(
            raw_text=case_payload.get("raw_text") or case_payload.get("raw_request") or "",
            case_summary=case_payload.get("case_summary"),
            problem_list=case_payload.get("problem_list"),
            known_facts=case_payload.get("known_facts"),
        )
        return self.retrieve_from_query(
            query_text,
            top_k=top_k,
            candidate_ids=candidate_ids,
            backend=backend,
            include_scores=include_scores,
        )

    def _retrieve_rows(
        self,
        *,
        retriever: Any | None,
        query_text: str,
        top_k: int,
        candidate_ids: set[str] | None,
        source_name: str,
    ) -> list[dict[str, Any]]:
        if retriever is None:
            return []
        try:
            rows = retriever.retrieve(
                query_text,
                top_k=top_k,
                candidate_ids=candidate_ids,
            )
        except TypeError:
            try:
                rows = retriever.retrieve(
                    query_text,
                    top_k=top_k,
                    candidate_pmids=candidate_ids,
                )
            except TypeError:
                rows = retriever.retrieve(query_text, top_k=top_k)
                if candidate_ids is not None:
                    rows = [
                        row
                        for row in list(rows)
                        if self._resolve_document_id(row) in candidate_ids
                    ]

        serialized_rows: list[dict[str, Any]] = []
        for row in list(rows):
            document_id = self._resolve_document_id(row)
            if not document_id:
                continue
            if candidate_ids is not None and document_id not in candidate_ids:
                continue
            serialized = self._serialize_row(document_id, row)
            serialized["query_text"] = str(query_text or "")
            serialized["match_sources"] = sorted(
                {
                    *list(serialized.get("match_sources") or []),
                    source_name,
                }
            )
            serialized_rows.append(serialized)
        return serialized_rows[:top_k]

    def _resolve_document_id(self, row: Mapping[str, Any]) -> str:
        candidate_keys = (
            self.id_field,
            "document_id",
            "calc_id",
            "pmid",
            "trial_id",
            "id",
        )
        for key in candidate_keys:
            value = str(row.get(key) or "").strip()
            if value:
                return value
        return ""

    def _serialize_row(self, document_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = self._catalog_payload(document_id)
        payload["document_id"] = document_id
        if self.id_field != "document_id":
            payload[self.id_field] = document_id

        title = _normalize_whitespace(row.get("title") or payload.get("title"))
        if title:
            payload["title"] = title
        payload["score"] = float(row.get("score") or 0.0)

        for field_name in ("summary", "purpose", "eligibility"):
            field_value = _normalize_whitespace(row.get(field_name) or payload.get(field_name))
            if field_value:
                payload[field_name] = field_value

        metadata = dict(payload.get("metadata") or {})
        for field_name, value in dict(row).items():
            if field_name in {
                "document_id",
                self.id_field,
                "calc_id",
                "pmid",
                "trial_id",
                "id",
                "title",
                "score",
            }:
                continue
            if field_name in {"summary", "purpose", "eligibility"}:
                continue
            metadata.setdefault(field_name, value)
        payload["metadata"] = metadata
        return payload

    def _catalog_payload(self, document_id: str) -> dict[str, Any]:
        document = None
        if self.catalog is not None:
            try:
                document = self.catalog.get(document_id)
            except Exception:
                document = None

        if document is None:
            return StructuredRetrievalDocument(document_id=document_id).to_brief(id_field=self.id_field)

        if hasattr(document, "to_brief"):
            try:
                raw_payload = dict(document.to_brief() or {})
            except Exception:
                raw_payload = {}
        else:
            raw_payload = {}

        if not raw_payload:
            raw_payload = {
                "title": _normalize_whitespace(_document_field(document, "title")),
                "summary": _normalize_whitespace(
                    _document_field(document, "summary")
                    or _document_field(document, "purpose")
                    or _document_field(document, "content")
                ),
            }

        payload = {
            "document_id": document_id,
            "title": _normalize_whitespace(raw_payload.get("title")),
            "summary": _normalize_whitespace(
                raw_payload.get("summary")
                or raw_payload.get("purpose")
                or raw_payload.get("eligibility")
            ),
            "purpose": _normalize_whitespace(raw_payload.get("purpose")),
            "eligibility": _normalize_whitespace(raw_payload.get("eligibility")),
            "metadata": {},
        }
        if self.id_field != "document_id":
            payload[self.id_field] = document_id
        for field_name, value in raw_payload.items():
            if field_name in payload:
                continue
            payload["metadata"][field_name] = value
        return payload

    @staticmethod
    def _normalize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
        raw_scores = {
            str(row.get("document_id") or ""): float(row.get("score") or 0.0)
            for row in rows
            if str(row.get("document_id") or "").strip()
        }
        if not raw_scores:
            return {}
        values = list(raw_scores.values())
        max_score = max(values)
        min_score = min(values)
        if math.isclose(max_score, min_score):
            if max_score <= 0.0:
                return {document_id: 0.0 for document_id in raw_scores}
            return {document_id: 1.0 for document_id in raw_scores}
        denominator = max_score - min_score
        return {
            document_id: (score - min_score) / denominator
            for document_id, score in raw_scores.items()
        }

    def _fuse_rows(
        self,
        *,
        bm25_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        normalized_bm25_scores = self._normalize_scores(bm25_rows)
        normalized_vector_scores = self._normalize_scores(vector_rows)

        fused: dict[str, dict[str, Any]] = {}
        for source_name, rows, normalized_scores in (
            ("bm25", bm25_rows, normalized_bm25_scores),
            ("vector", vector_rows, normalized_vector_scores),
        ):
            for row in rows:
                document_id = str(row.get("document_id") or "").strip()
                if not document_id:
                    continue
                current = fused.setdefault(document_id, dict(row))
                current["score"] = (
                    float(current.get("score") or 0.0)
                    + float(normalized_scores.get(document_id) or 0.0)
                )
                current["match_sources"] = sorted(
                    {
                        *list(current.get("match_sources") or []),
                        source_name,
                    }
                )
                metadata = dict(current.get("metadata") or {})
                raw_scores = dict(metadata.get("raw_scores") or {})
                raw_scores[source_name] = float(row.get("score") or 0.0)
                metadata["raw_scores"] = raw_scores
                current["metadata"] = metadata

        ranked = sorted(
            fused.values(),
            key=lambda row: (
                -float(row.get("score") or 0.0),
                -len(list(row.get("match_sources") or [])),
                str(row.get("title") or "").lower(),
                str(row.get("document_id") or ""),
            ),
        )
        return ranked[: max(int(top_k), 1)]

    def _public_row(self, row: Mapping[str, Any], *, include_scores: bool) -> dict[str, Any]:
        payload = {
            "document_id": str(row.get("document_id") or "").strip(),
            "title": _normalize_whitespace(row.get("title")),
            "match_sources": sorted(set(list(row.get("match_sources") or []))),
        }
        if self.id_field != "document_id":
            payload[self.id_field] = payload["document_id"]
        for field_name in ("summary", "purpose", "eligibility"):
            field_value = _normalize_whitespace(row.get(field_name))
            if field_value:
                payload[field_name] = field_value
        metadata = dict(row.get("metadata") or {})
        if metadata:
            payload["metadata"] = metadata
        if include_scores:
            payload["score"] = float(row.get("score") or 0.0)
        return payload


def create_structured_retriever(
    catalog: Any,
    *,
    bm25_retriever: Any | None,
    vector_retriever: Any | None = None,
    query_builder: Callable[..., str] | None = None,
    default_backend: str = "hybrid",
    id_field: str = "document_id",
) -> StructuredRetriever:
    return StructuredRetriever(
        catalog,
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        query_builder=query_builder,
        default_backend=default_backend,
        id_field=id_field,
    )
