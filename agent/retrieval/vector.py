from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
import threading
from typing import Any


_CACHE_LOCK = threading.RLock()
_MEDCPT_RESOURCE_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _document_field(document: Any, name: str) -> Any:
    if isinstance(document, Mapping):
        return document.get(name)
    return getattr(document, name, None)


def _document_to_brief(document: Any) -> dict[str, Any]:
    if hasattr(document, "to_brief"):
        payload = document.to_brief()
        if isinstance(payload, dict):
            return dict(payload)
    return {
        "pmid": str(_document_field(document, "pmid") or ""),
        "title": _normalize_whitespace(_document_field(document, "title")),
        "purpose": _normalize_whitespace(_document_field(document, "purpose")),
        "eligibility": _normalize_whitespace(_document_field(document, "eligibility")),
        "taxonomy": dict(_document_field(document, "taxonomy") or {}),
    }


def _flatten_dense_vector(vector: Any) -> list[float]:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    while values and isinstance(values[0], list | tuple):
        values = values[0]
    return [float(item) for item in list(values or [])]


def _rank_dense_subset(
    *,
    index: Any,
    query_embedding: Any,
    pmids: list[str],
    pmid_to_index: Mapping[str, int],
    candidate_pmids: set[str],
) -> list[tuple[str, float]] | None:
    if not candidate_pmids:
        return []
    if not hasattr(index, "reconstruct"):
        return None

    query_vector = _flatten_dense_vector(query_embedding)
    if not query_vector:
        return []

    scored_rows: list[tuple[str, float, int]] = []
    for order, pmid in enumerate(list(pmids or [])):
        if pmid not in candidate_pmids:
            continue
        vector_index = pmid_to_index.get(pmid)
        if vector_index is None:
            continue
        try:
            document_vector = _flatten_dense_vector(index.reconstruct(int(vector_index)))
        except Exception:
            return None
        score = sum(
            float(query_value) * float(doc_value)
            for query_value, doc_value in zip(query_vector, document_vector, strict=False)
        )
        scored_rows.append((pmid, float(score), order))

    scored_rows.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [(pmid, score) for pmid, score, _ in scored_rows]


def _retrieve_dense_scored_pmids(
    *,
    index: Any,
    query_embedding: Any,
    pmids: list[str],
    pmid_to_index: Mapping[str, int],
    candidate_pmids: set[str] | None,
    top_k: int,
) -> list[tuple[str, float]]:
    normalized_candidate_pmids = (
        {str(pmid).strip() for pmid in list(candidate_pmids or []) if str(pmid).strip()}
        if candidate_pmids is not None
        else None
    )
    if normalized_candidate_pmids is not None and not normalized_candidate_pmids:
        return []

    if normalized_candidate_pmids is not None and len(normalized_candidate_pmids) < len(pmids):
        subset_rows = _rank_dense_subset(
            index=index,
            query_embedding=query_embedding,
            pmids=pmids,
            pmid_to_index=pmid_to_index,
            candidate_pmids=normalized_candidate_pmids,
        )
        if subset_rows is not None:
            return subset_rows[:top_k]

    search_k = (
        min(top_k, len(pmids))
        if normalized_candidate_pmids is None or len(normalized_candidate_pmids) >= len(pmids)
        else len(pmids)
    )
    if search_k <= 0:
        return []

    scores, indices = index.search(query_embedding, k=search_k)
    results: list[tuple[str, float]] = []
    for score, index_value in zip(scores[0], indices[0], strict=False):
        pmid = pmids[int(index_value)]
        if normalized_candidate_pmids is not None and pmid not in normalized_candidate_pmids:
            continue
        results.append((pmid, float(score)))
        if len(results) >= top_k:
            break
    return results


class MedCPTRetriever:
    QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
    DOC_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, catalog: Any) -> None:
        try:
            import faiss  # type: ignore
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("MedCPT retrieval requires faiss, torch, and transformers.") from exc

        self.catalog = catalog
        self._faiss = faiss
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._query_encoder, self._doc_encoder, self._tokenizer = self._load_shared_resources(
            AutoModel=AutoModel,
            AutoTokenizer=AutoTokenizer,
        )
        self._pmids = [doc.pmid for doc in catalog.documents()]
        self._pmid_to_index = {str(pmid): index for index, pmid in enumerate(self._pmids)}
        self._inference_lock = threading.RLock()
        self._index = self._load_or_build_index()

    def _load_shared_resources(self, *, AutoModel: Any, AutoTokenizer: Any) -> tuple[Any, Any, Any]:
        cache_key = (self._device, self.QUERY_MODEL_NAME, self.DOC_MODEL_NAME)
        with _CACHE_LOCK:
            cached = _MEDCPT_RESOURCE_CACHE.get(cache_key)
            if cached is not None:
                return cached

        query_encoder = AutoModel.from_pretrained(self.QUERY_MODEL_NAME).to(self._device)
        doc_encoder = AutoModel.from_pretrained(self.DOC_MODEL_NAME).to(self._device)
        tokenizer = AutoTokenizer.from_pretrained(self.QUERY_MODEL_NAME)
        if hasattr(query_encoder, "eval"):
            query_encoder.eval()
        if hasattr(doc_encoder, "eval"):
            doc_encoder.eval()

        with _CACHE_LOCK:
            cached = _MEDCPT_RESOURCE_CACHE.setdefault(cache_key, (query_encoder, doc_encoder, tokenizer))
        return cached

    def _load_or_build_index(self):
        cache_paths = self.catalog.dense_index_cache_paths(
            query_model_name=self.QUERY_MODEL_NAME,
            doc_model_name=self.DOC_MODEL_NAME,
        )
        if cache_paths is not None:
            index_path, pmids_path = cache_paths
            try:
                if index_path.exists() and pmids_path.exists():
                    cached_pmids = json.loads(pmids_path.read_text(encoding="utf-8"))
                    if list(cached_pmids) == self._pmids:
                        return self._faiss.read_index(str(index_path))
            except Exception:
                pass

        index = self._build_index()
        if cache_paths is not None:
            index_path, pmids_path = cache_paths
            try:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                self._faiss.write_index(index, str(index_path))
                pmids_path.write_text(json.dumps(self._pmids, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return index

    def _build_index(self):  # pragma: no cover - depends on local environment
        tool_texts = []
        for doc in self.catalog.documents():
            tool_texts.append([doc.title, doc.retrieval_text or doc.abstract or doc.purpose])

        embeddings = []
        with self._torch.no_grad():
            for start in range(0, len(tool_texts), 16):
                encoded = self._tokenizer(
                    tool_texts[start : start + 16],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded.to(self._device)
                model_output = self._doc_encoder(**encoded)
                embeddings.extend(model_output.last_hidden_state[:, 0, :].detach().cpu())

        matrix = self._torch.stack(embeddings).numpy()
        index = self._faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, Any]]:  # pragma: no cover
        with self._inference_lock:
            with self._torch.no_grad():
                encoded = self._tokenizer(
                    [query],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded.to(self._device)
                model_output = self._query_encoder(**encoded)
                query_embedding = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()

        results = []
        scored_pmids = _retrieve_dense_scored_pmids(
            index=self._index,
            query_embedding=query_embedding,
            pmids=self._pmids,
            pmid_to_index=getattr(
                self,
                "_pmid_to_index",
                {str(pmid): index for index, pmid in enumerate(self._pmids)},
            ),
            candidate_pmids=candidate_pmids,
            top_k=top_k,
        )
        for pmid, score in scored_pmids:
            doc = self.catalog.get(pmid)
            payload = _document_to_brief(doc)
            payload["score"] = float(score)
            results.append(payload)
        return results


class HybridRetriever:
    def __init__(
        self,
        catalog: Any,
        *,
        keyword_retriever: Any | None = None,
        dense_retriever: MedCPTRetriever | None = None,
        rrf_k: int = 60,
        fusion_depth: int = 10,
        keyword_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.catalog = catalog
        self.keyword_retriever = keyword_retriever
        self.dense_retriever = dense_retriever or MedCPTRetriever(catalog)
        self.rrf_k = max(int(rrf_k), 1)
        self.fusion_depth = max(int(fusion_depth), 1)
        self.keyword_weight = max(float(keyword_weight), 0.0)
        self.dense_weight = max(float(dense_weight), 0.0)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        depth = max(int(top_k), self.fusion_depth)
        keyword_results = self._safe_retrieve(
            self.keyword_retriever,
            query,
            top_k=depth,
            candidate_pmids=candidate_pmids,
        )
        dense_results = self._safe_retrieve(
            self.dense_retriever,
            query,
            top_k=depth,
            candidate_pmids=candidate_pmids,
        )

        normalized_keyword_scores = self._normalize_scores(keyword_results)
        normalized_dense_scores = self._normalize_scores(dense_results)
        active_total_weight = 0.0
        if keyword_results:
            active_total_weight += self.keyword_weight
        if dense_results:
            active_total_weight += self.dense_weight

        fused: dict[str, dict[str, Any]] = {}
        for channel_name, results, channel_weight, normalized_scores in (
            ("keyword", keyword_results, self.keyword_weight, normalized_keyword_scores),
            ("vector", dense_results, self.dense_weight, normalized_dense_scores),
        ):
            for row in results:
                pmid = str(row.get("pmid") or "")
                if not pmid:
                    continue
                entry = fused.setdefault(
                    pmid,
                    {
                        **self._coerce_payload(row),
                        "score": 0.0,
                        "raw_scores": {},
                        "normalized_scores": {},
                        "match_sources": [],
                    },
                )
                raw_score = float(row.get("score") or 0.0)
                normalized_score = float(normalized_scores.get(pmid) or 0.0)
                entry["score"] = float(entry["score"]) + (channel_weight * normalized_score)
                entry["raw_scores"] = {
                    **dict(entry.get("raw_scores") or {}),
                    channel_name: raw_score,
                }
                entry["normalized_scores"] = {
                    **dict(entry.get("normalized_scores") or {}),
                    channel_name: normalized_score,
                }
                entry["match_sources"] = sorted(
                    {
                        *list(entry.get("match_sources") or []),
                        channel_name,
                    }
                )

        if active_total_weight > 0:
            for entry in fused.values():
                entry["score"] = float(entry["score"]) / active_total_weight

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -float(item["score"]),
                -len(list(item.get("match_sources") or [])),
                str(item.get("title") or "").lower(),
                str(item["pmid"]),
            ),
        )
        return ranked[:top_k]

    @staticmethod
    def _normalize_scores(results: list[dict[str, Any]]) -> dict[str, float]:
        raw_scores = {
            str(row.get("pmid") or ""): float(row.get("score") or 0.0)
            for row in results
            if str(row.get("pmid") or "").strip()
        }
        if not raw_scores:
            return {}
        values = list(raw_scores.values())
        max_score = max(values)
        min_score = min(values)
        if math.isclose(max_score, min_score):
            if max_score <= 0.0:
                return {pmid: 0.0 for pmid in raw_scores}
            return {pmid: 1.0 for pmid in raw_scores}
        denominator = max_score - min_score
        return {
            pmid: (score - min_score) / denominator
            for pmid, score in raw_scores.items()
        }

    @staticmethod
    def _safe_retrieve(retriever: Any, query: str, *, top_k: int, candidate_pmids: set[str] | None) -> list[dict[str, Any]]:
        if retriever is None:
            return []
        if candidate_pmids is not None:
            try:
                return retriever.retrieve(query, top_k=top_k, candidate_pmids=candidate_pmids)
            except TypeError:
                rows = retriever.retrieve(query, top_k=top_k)
                return [row for row in list(rows) if str(row.get("pmid") or "") in candidate_pmids]
        return retriever.retrieve(query, top_k=top_k)

    def _coerce_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        pmid = str(row.get("pmid") or "")
        document = self.catalog.get(pmid)
        payload = _document_to_brief(document)
        for field in ("title", "purpose", "eligibility"):
            value = str(row.get(field) or "").strip()
            if value:
                payload[field] = value
        return payload
