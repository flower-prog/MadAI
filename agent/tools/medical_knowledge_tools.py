from __future__ import annotations

from collections.abc import Mapping
import json
import os
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .execution_tools import tool


DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS = 6.0
DEFAULT_LIVE_MEDICAL_KNOWLEDGE_MAX_SOURCES = 2
_PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
_DEFAULT_USER_AGENT = "MedAI live medical knowledge retriever"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_top_k(value: int | None, *, default: int = 5) -> int:
    try:
        return max(int(value or default), 1)
    except (TypeError, ValueError):
        return default


def _json_get(url: str, *, timeout: float, user_agent: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _concept_terms(concepts: Any) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in list(concepts or []):
        if isinstance(item, Mapping):
            candidates = (
                item.get("canonical_name"),
                item.get("canonical"),
                item.get("name"),
                item.get("text"),
            )
        else:
            candidates = (item,)
        for candidate in candidates:
            term = _normalize_text(candidate)
            key = term.casefold()
            if term and key not in seen:
                seen.add(key)
                terms.append(term)
    return terms


def _query_with_concepts(query: str, concepts: Any) -> str:
    base_query = _normalize_text(query)
    terms = _concept_terms(concepts)[:4]
    if not terms:
        return base_query
    concept_clause = " OR ".join(terms)
    if not base_query:
        return concept_clause
    return f"({base_query}) AND ({concept_clause})"


class PubMedRealtimeSearchTool:
    """Small PubMed E-utilities client used as a live evidence source."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS,
        user_agent: str | None = None,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.user_agent = user_agent or os.getenv("MEDAI_HTTP_USER_AGENT") or _DEFAULT_USER_AGENT
        self.email = email or os.getenv("NCBI_EMAIL") or os.getenv("ENTREZ_EMAIL") or ""
        self.api_key = api_key or os.getenv("NCBI_API_KEY") or os.getenv("ENTREZ_API_KEY") or ""

    @tool(
        name="pubmed_realtime_search",
        description="Search PubMed live via NCBI E-utilities for protocol-facing medical evidence.",
        input_schema={
            "query": "str",
            "top_k": "int",
            "concepts": "list[dict] | list[str] | None",
            "filters": "dict | None",
        },
    )
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        concepts: list[dict[str, Any]] | list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        requested_top_k = _coerce_top_k(top_k)
        term = _query_with_concepts(query, concepts)
        if not term:
            return []

        search_params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": str(requested_top_k),
            "sort": "relevance",
        }
        if self.email:
            search_params["email"] = self.email
        if self.api_key:
            search_params["api_key"] = self.api_key
        search_payload = _json_get(
            f"{_PUBMED_ESEARCH_URL}?{urlencode(search_params)}",
            timeout=self.timeout_seconds,
            user_agent=self.user_agent,
        )
        pmids = [
            str(item).strip()
            for item in list(search_payload.get("esearchresult", {}).get("idlist") or [])
            if str(item).strip()
        ][:requested_top_k]
        if not pmids:
            return []

        summary_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if self.email:
            summary_params["email"] = self.email
        if self.api_key:
            summary_params["api_key"] = self.api_key
        summary_payload = _json_get(
            f"{_PUBMED_ESUMMARY_URL}?{urlencode(summary_params)}",
            timeout=self.timeout_seconds,
            user_agent=self.user_agent,
        )
        result_map = dict(summary_payload.get("result") or {})
        rows: list[dict[str, Any]] = []
        for rank, pmid in enumerate(pmids, start=1):
            item = result_map.get(pmid)
            if not isinstance(item, Mapping):
                continue
            title = _normalize_text(item.get("title"))
            journal = _normalize_text(item.get("source"))
            pubdate = _normalize_text(item.get("pubdate"))
            text_parts = [title]
            if journal or pubdate:
                text_parts.append(" ".join(part for part in (journal, pubdate) if part))
            rows.append(
                {
                    "source": "pubmed",
                    "source_type": "literature",
                    "backend": "ncbi_eutilities",
                    "rank": rank,
                    "pmid": pmid,
                    "title": title,
                    "text": ". ".join(part for part in text_parts if part),
                    "journal": journal,
                    "publication_date": pubdate,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "filters": dict(filters or {}),
                }
            )
        return rows


class WikidataKnowledgeGraphTool:
    """Live Wikidata entity linker for biomedical concepts."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS,
        user_agent: str | None = None,
        language: str = "en",
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.user_agent = user_agent or os.getenv("MEDAI_HTTP_USER_AGENT") or _DEFAULT_USER_AGENT
        self.language = _normalize_text(language) or "en"

    @tool(
        name="wikidata_entity_graph_search",
        description="Search Wikidata live for biomedical entities and graph identifiers.",
        input_schema={
            "query": "str",
            "top_k": "int",
            "concepts": "list[dict] | list[str] | None",
            "filters": "dict | None",
        },
    )
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        concepts: list[dict[str, Any]] | list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        requested_top_k = _coerce_top_k(top_k)
        search_terms = _concept_terms(concepts)
        if not search_terms:
            search_terms = [_normalize_text(query)]

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for term in search_terms[: max(requested_top_k, 1)]:
            if not term:
                continue
            params = {
                "action": "wbsearchentities",
                "format": "json",
                "language": self.language,
                "uselang": self.language,
                "type": "item",
                "limit": str(max(1, min(requested_top_k, 7))),
                "search": term,
            }
            payload = _json_get(
                f"{_WIKIDATA_SEARCH_URL}?{urlencode(params)}",
                timeout=self.timeout_seconds,
                user_agent=self.user_agent,
            )
            for item in list(payload.get("search") or []):
                if not isinstance(item, Mapping):
                    continue
                entity_id = _normalize_text(item.get("id"))
                if not entity_id or entity_id in seen:
                    continue
                seen.add(entity_id)
                label = _normalize_text(item.get("label"))
                description = _normalize_text(item.get("description"))
                rows.append(
                    {
                        "source": "wikidata",
                        "source_type": "knowledge_graph",
                        "backend": "wikidata_api",
                        "rank": len(rows) + 1,
                        "entity_id": entity_id,
                        "title": label or entity_id,
                        "text": ". ".join(part for part in (label, description) if part),
                        "description": description,
                        "matched_term": term,
                        "url": f"https://www.wikidata.org/wiki/{entity_id}",
                        "filters": dict(filters or {}),
                    }
                )
                if len(rows) >= requested_top_k:
                    return rows
        return rows


class LiveMedicalKnowledgeRetriever:
    """Aggregate multiple live sources behind the protocol medical knowledge interface."""

    def __init__(
        self,
        *,
        sources: list[Any] | None = None,
        timeout_seconds: float = DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS,
        max_sources: int = DEFAULT_LIVE_MEDICAL_KNOWLEDGE_MAX_SOURCES,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_sources = max(int(max_sources), 1)
        self.sources = list(sources or [
            PubMedRealtimeSearchTool(timeout_seconds=self.timeout_seconds),
            WikidataKnowledgeGraphTool(timeout_seconds=self.timeout_seconds),
        ])[: self.max_sources]

    @tool(
        name="live_medical_knowledge_retriever",
        description=(
            "Retrieve protocol medical knowledge from live online sources, including "
            "PubMed literature and Wikidata biomedical entities."
        ),
        input_schema={
            "query": "str",
            "top_k": "int",
            "concepts": "list[dict] | list[str] | None",
            "filters": "dict | None",
        },
    )
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        concepts: list[dict[str, Any]] | list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        requested_top_k = _coerce_top_k(top_k)
        rows: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        source_top_k = max(1, requested_top_k)
        started_at = time.monotonic()

        for source in self.sources:
            retrieve = getattr(source, "retrieve", None)
            if not callable(retrieve):
                continue
            source_name = source.__class__.__name__
            try:
                source_rows = retrieve(
                    query,
                    top_k=source_top_k,
                    concepts=concepts,
                    filters=filters,
                )
            except (TimeoutError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                warnings.append(
                    {
                        "source": source_name,
                        "warning": "live_source_unavailable",
                        "message": _normalize_text(exc),
                    }
                )
                continue
            for item in list(source_rows or []):
                if not isinstance(item, Mapping):
                    continue
                payload = dict(item)
                payload.setdefault("query", query)
                payload.setdefault("retrieved_live", True)
                rows.append(payload)

        for warning in warnings:
            rows.append(
                {
                    "source": warning["source"],
                    "source_type": "diagnostic",
                    "status": warning["warning"],
                    "text": warning["message"],
                    "query": query,
                    "retrieved_live": True,
                }
            )
        for rank, item in enumerate(rows, start=1):
            item.setdefault("aggregate_rank", rank)
            item.setdefault("latency_ms", int((time.monotonic() - started_at) * 1000))
        return rows[:requested_top_k]


def create_live_medical_knowledge_retriever(
    *,
    timeout_seconds: float | None = None,
    max_sources: int | None = None,
) -> LiveMedicalKnowledgeRetriever:
    timeout = timeout_seconds
    if timeout is None:
        raw_timeout = os.getenv("MEDAI_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS")
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS
        except ValueError:
            timeout = DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS

    source_count = max_sources
    if source_count is None:
        raw_source_count = os.getenv("MEDAI_LIVE_MEDICAL_KNOWLEDGE_MAX_SOURCES")
        try:
            source_count = int(raw_source_count) if raw_source_count else DEFAULT_LIVE_MEDICAL_KNOWLEDGE_MAX_SOURCES
        except ValueError:
            source_count = DEFAULT_LIVE_MEDICAL_KNOWLEDGE_MAX_SOURCES

    return LiveMedicalKnowledgeRetriever(
        timeout_seconds=float(timeout),
        max_sources=max(int(source_count), 1),
    )


__all__ = [
    "DEFAULT_LIVE_MEDICAL_KNOWLEDGE_TIMEOUT_SECONDS",
    "LiveMedicalKnowledgeRetriever",
    "PubMedRealtimeSearchTool",
    "WikidataKnowledgeGraphTool",
    "create_live_medical_knowledge_retriever",
]
