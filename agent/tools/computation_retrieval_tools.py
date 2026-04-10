from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from .execution_tools import tool
from .retrieval_tools import (
    KeywordToolRetriever,
    MedCPTRetriever,
    _catalog_runtime_cache_key,
    _clean_text,
    _coerce_text_list,
    _load_riskcalcs_payload_index,
    _sanitize_parameter_names,
    build_case_query_text,
    extract_parameter_names_from_computation,
)


_COMPUTATION_TOOL_CACHE: dict[str, "RiskCalcComputationRetrievalTool"] = {}
_CACHE_LOCK = threading.RLock()


@dataclass(slots=True)
class _ComputationParameterDocument:
    pmid: str
    display_title: str
    parameter_names: list[str]
    retrieval_text: str
    calculator_payload: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    purpose: str = ""
    eligibility: str = ""
    abstract: str = ""

    def to_brief(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.display_title,
            "purpose": "",
            "eligibility": "",
            "taxonomy": {},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.display_title,
            "parameter_names": list(self.parameter_names),
            "retrieval_text": self.retrieval_text,
            "calculator_payload": dict(self.calculator_payload),
        }


class _ComputationParameterCatalog:
    def __init__(self, documents: dict[str, _ComputationParameterDocument], *, runtime_cache_key: str) -> None:
        self._documents = dict(documents)
        self.runtime_cache_key = runtime_cache_key

    def documents(self) -> list[_ComputationParameterDocument]:
        return list(self._documents.values())

    def get(self, pmid: str) -> _ComputationParameterDocument:
        return self._documents[str(pmid)]

    def dense_index_cache_paths(self, *, query_model_name: str, doc_model_name: str):
        del query_model_name, doc_model_name
        return None


def extract_case_parameter_terms(structured_case: Any) -> list[str]:
    case_payload = dict(structured_case or {})
    if isinstance(case_payload.get("structured_case"), dict):
        case_payload = dict(case_payload["structured_case"])

    parameter_candidates: list[Any] = []
    for field_name in ("structured_inputs", "interval_inputs"):
        mapping = case_payload.get(field_name)
        if isinstance(mapping, dict):
            parameter_candidates.extend(str(key) for key in mapping.keys())

    parameter_candidates.extend(_coerce_text_list(case_payload.get("known_facts")))
    parameter_candidates.extend(_coerce_text_list(case_payload.get("problem_list")))
    return _sanitize_parameter_names(parameter_candidates)


def build_case_parameter_query_text(structured_case: Any) -> tuple[str, list[str]]:
    case_payload = dict(structured_case or {})
    if isinstance(case_payload.get("structured_case"), dict):
        case_payload = dict(case_payload["structured_case"])

    parameter_terms = extract_case_parameter_terms(case_payload)
    if parameter_terms:
        return " ; ".join(parameter_terms), parameter_terms

    fallback_query = build_case_query_text(
        raw_text=case_payload.get("raw_text") or case_payload.get("raw_request") or "",
        case_summary=case_payload.get("case_summary"),
        problem_list=case_payload.get("problem_list"),
        known_facts=case_payload.get("known_facts"),
    )
    return fallback_query, parameter_terms


class RiskCalcComputationRetrievalTool:
    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self._riskcalcs_payload_index = _load_riskcalcs_payload_index(catalog)
        self._documents = self._build_documents()
        self._parameter_catalog = _ComputationParameterCatalog(
            self._documents,
            runtime_cache_key=f"{_catalog_runtime_cache_key(catalog)}:computation-parameters",
        )
        self.keyword_retriever = KeywordToolRetriever(self._parameter_catalog)
        try:
            self.vector_retriever = MedCPTRetriever(self._parameter_catalog)
        except Exception:
            self.vector_retriever = None

    @property
    def available_backends(self) -> tuple[str, ...]:
        if self.vector_retriever is None:
            return ("bm25",)
        return ("bm25", "vector")

    @tool(
        name="riskcalc_computation_retriever",
        description=(
            "Within a coarse PMID pool, use only computation-derived function parameters to score the full "
            "candidate set, rerank it by parameter similarity, and expose per-channel context hits with full "
            "calculator payloads."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
                "structured_inputs": "dict[str, Any]",
                "interval_inputs": "dict[str, Any]",
            },
            "candidate_pmids": "list[str] | set[str]",
            "top_k_per_channel": "int",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
        },
    )
    def retrieve_from_structured_case(
        self,
        structured_case: dict[str, Any],
        *,
        candidate_pmids: list[str] | set[str] | tuple[str, ...],
        top_k_per_channel: int = 3,
    ) -> dict[str, Any]:
        query_text, case_parameter_terms = build_case_parameter_query_text(structured_case)
        ordered_candidate_pmids = [
            str(pmid).strip()
            for pmid in list(candidate_pmids or [])
            if str(pmid).strip()
        ]
        scoped_pmids = set(ordered_candidate_pmids)
        if not ordered_candidate_pmids:
            return {
                "query_text": query_text,
                "case_parameter_terms": case_parameter_terms,
                "available_backends": list(self.available_backends),
                "candidate_pmids": [],
                "bm25_raw_top3": [],
                "vector_raw_top3": [],
                "bm25_raw_top5": [],
                "vector_raw_top5": [],
                "retrieved_tools": [],
                "candidate_ranking": [],
                "recommended_pmids": [],
            }

        # Refinement should behave like a gentle reranker over the full coarse pool,
        # not a second hard filter. Preserve the legacy top_k_per_channel input as a
        # lower bound for returned context, but always score the entire scoped pool.
        retrieval_limit = max(int(top_k_per_channel), len(ordered_candidate_pmids), 1)
        bm25_rows = self.keyword_retriever.retrieve(
            query_text,
            top_k=retrieval_limit,
            candidate_pmids=scoped_pmids,
        )
        if self.vector_retriever is not None:
            vector_rows = self.vector_retriever.retrieve(
                query_text,
                top_k=retrieval_limit,
                candidate_pmids=scoped_pmids,
            )
        else:
            vector_rows = []

        bm25_raw = self._build_raw_hits(bm25_rows, channel="bm25", query_text=query_text)
        vector_raw = self._build_raw_hits(vector_rows, channel="vector", query_text=query_text)
        candidate_ranking = self._build_candidate_ranking(
            ordered_candidate_pmids=ordered_candidate_pmids,
            case_parameter_terms=case_parameter_terms,
            bm25_raw=bm25_raw,
            vector_raw=vector_raw,
        )

        return {
            "query_text": query_text,
            "case_parameter_terms": list(case_parameter_terms),
            "available_backends": list(self.available_backends),
            "backend_used": "hybrid" if self.vector_retriever is not None else "bm25",
            "candidate_pmids": list(ordered_candidate_pmids),
            "bm25_top3": self._build_context_hits(bm25_raw[:3]),
            "vector_top3": self._build_context_hits(vector_raw[:3]),
            "bm25_raw_top3": bm25_raw[:3],
            "vector_raw_top3": vector_raw[:3],
            "bm25_raw_top5": list(bm25_raw[:5]),
            "vector_raw_top5": list(vector_raw[:5]),
            "model_context": {
                "bm25_top3": self._build_context_hits(bm25_raw[:3]),
                "vector_top3": self._build_context_hits(vector_raw[:3]),
            },
            "retrieved_tools": list(candidate_ranking),
            "candidate_ranking": list(candidate_ranking),
            "recommended_pmids": [str(item.get("pmid") or "").strip() for item in candidate_ranking if str(item.get("pmid") or "").strip()],
        }

    def _build_documents(self) -> dict[str, _ComputationParameterDocument]:
        documents: dict[str, _ComputationParameterDocument] = {}
        payload_index = dict(self._riskcalcs_payload_index or {})
        if payload_index:
            for pmid, payload in payload_index.items():
                document = self._build_document(str(pmid), dict(payload or {}))
                if document is not None:
                    documents[str(pmid)] = document
            return documents

        documents_method = getattr(self.catalog, "documents", None)
        if not callable(documents_method):
            return documents
        for document in list(documents_method() or []):
            pmid = str(getattr(document, "pmid", "") or "").strip()
            if not pmid:
                continue
            payload = {}
            if hasattr(document, "to_dict"):
                try:
                    payload = dict(document.to_dict() or {})
                except Exception:
                    payload = {}
            if not payload:
                payload = {
                    "pmid": pmid,
                    "title": str(getattr(document, "title", "") or "").strip(),
                    "computation": str(getattr(document, "computation", "") or "").strip(),
                }
            built = self._build_document(pmid, payload)
            if built is not None:
                documents[pmid] = built
        return documents

    def _build_document(self, pmid: str, payload: dict[str, Any]) -> _ComputationParameterDocument | None:
        computation = str(payload.get("computation") or "").strip()
        parameter_names = _sanitize_parameter_names(
            extract_parameter_names_from_computation(
                computation,
                example=str(payload.get("example") or ""),
            )
        )
        if not parameter_names:
            return None

        title = _clean_text(payload.get("title"))
        normalized_payload = dict(payload)
        normalized_payload["pmid"] = pmid
        normalized_payload["title"] = title
        return _ComputationParameterDocument(
            pmid=pmid,
            display_title=title,
            parameter_names=parameter_names,
            retrieval_text="inputs: " + ", ".join(parameter_names),
            calculator_payload=normalized_payload,
        )

    def _build_raw_hits(
        self,
        rows: list[dict[str, Any]],
        *,
        channel: str,
        query_text: str,
    ) -> list[dict[str, Any]]:
        raw_hits: list[dict[str, Any]] = []
        for rank, row in enumerate(list(rows or []), start=1):
            pmid = str(row.get("pmid") or "").strip()
            if not pmid:
                continue
            document = self._documents.get(pmid)
            if document is None:
                continue
            raw_hits.append(
                {
                    "rank": rank,
                    "channel": channel,
                    "pmid": pmid,
                    "title": document.display_title,
                    "score": float(row.get("score") or 0.0),
                    "parameter_names": list(document.parameter_names),
                    "query_text": query_text,
                    "calculator_payload": dict(document.calculator_payload),
                }
            )
        return raw_hits

    @staticmethod
    def _build_context_hits(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        hits: dict[str, dict[str, Any]] = {}
        for row in list(rows or []):
            pmid = str(row.get("pmid") or "").strip()
            if not pmid:
                continue
            hits[pmid] = {
                "title": str(row.get("title") or "").strip(),
                "parameter_names": list(row.get("parameter_names") or []),
            }
        return hits

    @staticmethod
    def _normalize_parameter_term(value: Any) -> str:
        return str(_clean_text(value) or "").strip().lower().replace("_", " ")

    def _build_candidate_ranking(
        self,
        *,
        ordered_candidate_pmids: list[str],
        case_parameter_terms: list[str],
        bm25_raw: list[dict[str, Any]],
        vector_raw: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_pmid: dict[str, dict[str, Any]] = {}
        coarse_order = {
            str(pmid): index
            for index, pmid in enumerate(list(ordered_candidate_pmids or []), start=1)
            if str(pmid).strip()
        }
        normalized_case_terms = {
            self._normalize_parameter_term(term)
            for term in list(case_parameter_terms or [])
            if self._normalize_parameter_term(term)
        }
        bm25_by_pmid = {
            str(row.get("pmid") or "").strip(): dict(row)
            for row in list(bm25_raw or [])
            if str(row.get("pmid") or "").strip()
        }
        vector_by_pmid = {
            str(row.get("pmid") or "").strip(): dict(row)
            for row in list(vector_raw or [])
            if str(row.get("pmid") or "").strip()
        }
        bm25_max_score = max((float(row.get("score") or 0.0) for row in bm25_raw), default=0.0)
        vector_max_score = max((float(row.get("score") or 0.0) for row in vector_raw), default=0.0)
        bm25_pool_size = max(len(list(bm25_raw or [])), 1)
        vector_pool_size = max(len(list(vector_raw or [])), 1)

        for pmid in list(ordered_candidate_pmids or []):
            normalized_pmid = str(pmid or "").strip()
            if not normalized_pmid:
                continue

            document = self._documents.get(normalized_pmid)
            bm25_entry = bm25_by_pmid.get(normalized_pmid)
            vector_entry = vector_by_pmid.get(normalized_pmid)
            fallback_payload = dict(
                (bm25_entry or {}).get("calculator_payload")
                or (vector_entry or {}).get("calculator_payload")
                or {}
            )
            calculator_payload = dict(getattr(document, "calculator_payload", {}) or fallback_payload)
            parameter_names = list(
                getattr(document, "parameter_names", None)
                or (bm25_entry or {}).get("parameter_names")
                or (vector_entry or {}).get("parameter_names")
                or []
            )
            normalized_parameter_names = {
                self._normalize_parameter_term(name)
                for name in list(parameter_names or [])
                if self._normalize_parameter_term(name)
            }
            matched_case_parameters = [
                name
                for name in list(parameter_names or [])
                if self._normalize_parameter_term(name) in normalized_case_terms
            ]
            parameter_overlap_count = len(matched_case_parameters)
            parameter_overlap_ratio = (
                parameter_overlap_count / max(len(normalized_parameter_names), 1)
                if normalized_parameter_names
                else 0.0
            )

            source_channels: list[str] = []
            if bm25_entry is not None:
                source_channels.append("bm25")
            if vector_entry is not None:
                source_channels.append("vector")

            bm25_rank = int((bm25_entry or {}).get("rank") or 0)
            vector_rank = int((vector_entry or {}).get("rank") or 0)
            bm25_score = float((bm25_entry or {}).get("score") or 0.0)
            vector_score = float((vector_entry or {}).get("score") or 0.0)
            bm25_rank_signal = (
                (bm25_pool_size - bm25_rank + 1) / bm25_pool_size
                if bm25_rank > 0
                else 0.0
            )
            vector_rank_signal = (
                (vector_pool_size - vector_rank + 1) / vector_pool_size
                if vector_rank > 0
                else 0.0
            )
            bm25_score_signal = bm25_score / bm25_max_score if bm25_max_score > 0 else 0.0
            vector_score_signal = vector_score / vector_max_score if vector_max_score > 0 else 0.0
            rerank_score = (
                (parameter_overlap_count * 2.0)
                + (parameter_overlap_ratio * 1.5)
                + (len(source_channels) * 0.75)
                + bm25_rank_signal
                + vector_rank_signal
                + (bm25_score_signal * 0.35)
                + (vector_score_signal * 0.35)
            )

            by_pmid[normalized_pmid] = {
                "pmid": normalized_pmid,
                "title": _clean_text(
                    calculator_payload.get("title")
                    or getattr(document, "display_title", "")
                    or (bm25_entry or {}).get("title")
                    or (vector_entry or {}).get("title")
                ),
                "purpose": _clean_text(
                    calculator_payload.get("purpose")
                    or getattr(document, "purpose", "")
                ),
                "specialty": _clean_text(calculator_payload.get("specialty")),
                "eligibility": _clean_text(
                    calculator_payload.get("eligibility")
                    or getattr(document, "eligibility", "")
                ),
                "parameter_names": list(parameter_names),
                "matched_case_parameters": matched_case_parameters,
                "parameter_overlap_count": parameter_overlap_count,
                "parameter_overlap_ratio": parameter_overlap_ratio,
                "source_channels": source_channels,
                "bm25_rank": bm25_rank or None,
                "vector_rank": vector_rank or None,
                "bm25_score": bm25_score if bm25_entry is not None else None,
                "vector_score": vector_score if vector_entry is not None else None,
                "score": rerank_score,
                "coarse_rank": coarse_order.get(normalized_pmid, len(coarse_order) + 1),
                "calculator_payload": calculator_payload,
            }

        ranked = list(by_pmid.values())
        ranked.sort(
            key=lambda candidate: (
                -float(candidate.get("score") or 0.0),
                -len(list(candidate.get("source_channels") or [])),
                -int(candidate.get("parameter_overlap_count") or 0),
                int(candidate.get("bm25_rank") or 10**9),
                int(candidate.get("vector_rank") or 10**9),
                int(candidate.get("coarse_rank") or 10**9),
                str(candidate.get("title") or "").lower(),
                str(candidate.get("pmid") or ""),
            )
        )
        return ranked


def create_computation_retrieval_tool(catalog: Any) -> RiskCalcComputationRetrievalTool:
    cache_key = _catalog_runtime_cache_key(catalog)
    with _CACHE_LOCK:
        cached = _COMPUTATION_TOOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tool_instance = RiskCalcComputationRetrievalTool(catalog)
    with _CACHE_LOCK:
        cached = _COMPUTATION_TOOL_CACHE.setdefault(cache_key, tool_instance)
    return cached


__all__ = [
    "RiskCalcComputationRetrievalTool",
    "build_case_parameter_query_text",
    "create_computation_retrieval_tool",
    "extract_case_parameter_terms",
]
