from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from typing import Any
import uuid


DEFAULT_QDRANT_COLLECTION_NAME = "trial_chunks_medcpt"
_CACHE_LOCK = threading.RLock()
_MEDCPT_RESOURCE_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}
_TRIAL_CHUNK_POINT_ID_NAMESPACE = uuid.UUID("4a2c57f1-bf41-4f4b-91db-59d12f35e6a8")


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_text_list(values: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        normalized = _normalize_whitespace(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _normalize_candidate_ids(
    candidate_ids: set[str] | list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if candidate_ids is None:
        return None
    normalized = [
        str(item).strip()
        for item in list(candidate_ids or [])
        if str(item).strip()
    ]
    return normalized or []


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _qdrant_point_id(raw_id: Any) -> str:
    normalized = _normalize_whitespace(raw_id)
    if not normalized:
        normalized = "__empty__"
    return str(uuid.uuid5(_TRIAL_CHUNK_POINT_ID_NAMESPACE, normalized))


def _load_qdrant_dependencies() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models as qdrant_models  # type: ignore
    except Exception:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http import models as qdrant_models  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Qdrant trial retrieval requires qdrant-client.") from exc
    return QdrantClient, qdrant_models


def _create_qdrant_client(
    *,
    url: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
) -> Any:
    QdrantClient, _ = _load_qdrant_dependencies()
    if path is not None and str(path).strip():
        return QdrantClient(path=str(Path(path).expanduser().resolve()))
    return QdrantClient(url=str(url or "http://localhost:6333"), api_key=api_key)


def _payload_schema(models: Any, schema_name: str) -> Any:
    schema_type = getattr(models, "PayloadSchemaType", None)
    if schema_type is None:
        return schema_name
    normalized = str(schema_name or "").strip().upper()
    return getattr(schema_type, normalized, schema_name)


def _collection_exists(client: Any, collection_name: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name))
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _create_payload_index(client: Any, *, collection_name: str, field_name: str, field_schema: Any) -> None:
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    except TypeError:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_type=field_schema,
        )


class MedCPTTrialChunkEmbeddingFunction:
    QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
    DOC_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

    def __init__(
        self,
        *,
        query_model_name: str | None = None,
        doc_model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Qdrant trial retrieval requires torch and transformers.") from exc

        self.query_model_name = str(query_model_name or self.QUERY_MODEL_NAME)
        self.doc_model_name = str(doc_model_name or self.DOC_MODEL_NAME)
        self._torch = torch
        self._device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._query_encoder, self._doc_encoder, self._tokenizer = self._load_shared_resources(
            AutoModel=AutoModel,
            AutoTokenizer=AutoTokenizer,
        )
        self._inference_lock = threading.RLock()

    def _load_shared_resources(self, *, AutoModel: Any, AutoTokenizer: Any) -> tuple[Any, Any, Any]:
        cache_key = (self._device, self.query_model_name, self.doc_model_name)
        with _CACHE_LOCK:
            cached = _MEDCPT_RESOURCE_CACHE.get(cache_key)
            if cached is not None:
                return cached

        query_encoder = AutoModel.from_pretrained(self.query_model_name).to(self._device)
        doc_encoder = AutoModel.from_pretrained(self.doc_model_name).to(self._device)
        tokenizer = AutoTokenizer.from_pretrained(self.query_model_name)
        if hasattr(query_encoder, "eval"):
            query_encoder.eval()
        if hasattr(doc_encoder, "eval"):
            doc_encoder.eval()

        with _CACHE_LOCK:
            cached = _MEDCPT_RESOURCE_CACHE.setdefault(cache_key, (query_encoder, doc_encoder, tokenizer))
        return cached

    @property
    def vector_size(self) -> int:
        hidden_size = getattr(getattr(self._doc_encoder, "config", None), "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        fallback = self.encode_query("trial search embedding size probe")
        if not fallback:
            raise RuntimeError("Unable to determine MedCPT embedding size.")
        return len(fallback)

    def encode_query(self, query: str) -> list[float]:
        normalized_query = _normalize_whitespace(query)
        if not normalized_query:
            return []
        with self._inference_lock:
            with self._torch.no_grad():
                encoded = self._tokenizer(
                    [normalized_query],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded.to(self._device)
                model_output = self._query_encoder(**encoded)
                vector = model_output.last_hidden_state[:, 0, :].detach().cpu().tolist()[0]
        return [float(item) for item in list(vector or [])]

    def encode_documents(self, documents: Sequence[Any]) -> list[list[float]]:
        if not list(documents or []):
            return []

        tool_texts: list[list[str]] = []
        for document in list(documents or []):
            title = _normalize_whitespace(getattr(document, "title", ""))
            retrieval_text = _normalize_whitespace(
                getattr(document, "retrieval_text", "")
                or getattr(document, "text", "")
                or getattr(document, "summary", "")
                or getattr(document, "purpose", "")
            )
            tool_texts.append([title, retrieval_text])

        embeddings: list[list[float]] = []
        with self._inference_lock:
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
                    batch_vectors = model_output.last_hidden_state[:, 0, :].detach().cpu().tolist()
                    embeddings.extend(
                        [[float(item) for item in list(vector or [])] for vector in list(batch_vectors or [])]
                    )
        return embeddings


def build_qdrant_trial_chunk_payload(document: Any) -> dict[str, Any]:
    record = dict(getattr(document, "record_payload", {}) or {})
    display_title = _normalize_whitespace(
        record.get("display_title") or record.get("brief_title") or record.get("official_title") or getattr(document, "trial_title", "")
    )
    return {
        "chunk_id": str(getattr(document, "chunk_id", "") or ""),
        "document_id": str(getattr(document, "chunk_id", "") or ""),
        "nct_id": str(getattr(document, "nct_id", "") or ""),
        "trial_id": str(getattr(document, "nct_id", "") or ""),
        "chunk_type": _normalize_whitespace(getattr(document, "chunk_type", "")),
        "sequence": _coerce_int(getattr(document, "sequence", None)),
        "rank_weight": float(getattr(document, "rank_weight", 1.0) or 1.0),
        "token_estimate": int(getattr(document, "token_estimate", 0) or 0),
        "title": _normalize_whitespace(getattr(document, "title", "")),
        "trial_title": _normalize_whitespace(getattr(document, "trial_title", "")),
        "display_title": display_title,
        "brief_title": _normalize_whitespace(record.get("brief_title")),
        "official_title": _normalize_whitespace(record.get("official_title")),
        "summary": _normalize_whitespace(getattr(document, "summary", "")),
        "purpose": _normalize_whitespace(getattr(document, "purpose", "")),
        "eligibility": _normalize_whitespace(getattr(document, "eligibility", "")),
        "text": _normalize_whitespace(getattr(document, "text", "")),
        "embedding_text": _normalize_whitespace(
            getattr(document, "retrieval_text", "")
            or getattr(document, "embedding_text", "")
            or getattr(document, "text", "")
        ),
        "source_fields": _normalize_text_list(getattr(document, "source_fields", [])),
        "overall_status": _normalize_whitespace(record.get("overall_status")),
        "normalized_status": _normalize_whitespace(record.get("normalized_status")),
        "phase": _normalize_whitespace(record.get("phase")),
        "study_type": _normalize_whitespace(record.get("study_type")),
        "primary_purpose": _normalize_whitespace(record.get("primary_purpose")),
        "conditions": _normalize_text_list(record.get("conditions", [])),
        "condition_terms": _normalize_text_list(record.get("condition_terms", [])),
        "interventions": _normalize_text_list(record.get("interventions", [])),
        "intervention_terms": _normalize_text_list(record.get("intervention_terms", [])),
        "gender": _normalize_whitespace(record.get("gender")),
        "age_floor_years": _coerce_float(record.get("age_floor_years")),
        "age_ceiling_years": _coerce_float(record.get("age_ceiling_years")),
        "has_results_references": bool(record.get("has_results_references")),
        "source_url": _normalize_whitespace(record.get("source_url")),
        "source_corpus": _normalize_whitespace(record.get("source_corpus")),
        "source_archive": _normalize_whitespace(record.get("source_archive")),
        "source_member_path": _normalize_whitespace(record.get("source_member_path")),
        "xml_sha256": _normalize_whitespace(record.get("xml_sha256")),
    }


class QdrantTrialChunkIndexManager:
    def __init__(
        self,
        catalog: Any,
        *,
        collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME,
        client: Any | None = None,
        models: Any | None = None,
        url: str | None = None,
        api_key: str | None = None,
        path: str | Path | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.catalog = catalog
        self.collection_name = str(collection_name or DEFAULT_QDRANT_COLLECTION_NAME).strip() or DEFAULT_QDRANT_COLLECTION_NAME
        _, resolved_models = _load_qdrant_dependencies() if models is None else (None, models)
        self.models = resolved_models
        self.client = client or _create_qdrant_client(url=url, api_key=api_key, path=path)
        self.embedding_function = embedding_function or MedCPTTrialChunkEmbeddingFunction()

    def ensure_collection(self, *, recreate: bool = False) -> dict[str, Any]:
        existed_before = _collection_exists(self.client, self.collection_name)
        if recreate and existed_before:
            self.client.delete_collection(collection_name=self.collection_name)
            existed_before = False
        if not existed_before:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=int(self.embedding_function.vector_size),
                    distance=self.models.Distance.DOT,
                ),
            )
        self.ensure_payload_indexes()
        return {
            "collection_name": self.collection_name,
            "recreated": bool(recreate),
            "created": not existed_before,
        }

    def ensure_payload_indexes(self) -> list[str]:
        indexed_fields = [
            ("nct_id", "keyword"),
            ("trial_id", "keyword"),
            ("chunk_type", "keyword"),
            ("overall_status", "keyword"),
            ("normalized_status", "keyword"),
            ("phase", "keyword"),
            ("study_type", "keyword"),
            ("primary_purpose", "keyword"),
            ("gender", "keyword"),
            ("source_corpus", "keyword"),
            ("source_archive", "keyword"),
            ("source_fields", "keyword"),
            ("conditions", "keyword"),
            ("condition_terms", "keyword"),
            ("interventions", "keyword"),
            ("intervention_terms", "keyword"),
            ("sequence", "integer"),
            ("token_estimate", "integer"),
            ("rank_weight", "float"),
            ("age_floor_years", "float"),
            ("age_ceiling_years", "float"),
        ]
        created_fields: list[str] = []
        for field_name, schema_name in indexed_fields:
            _create_payload_index(
                self.client,
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=_payload_schema(self.models, schema_name),
            )
            created_fields.append(field_name)
        return created_fields

    def sync_catalog(
        self,
        *,
        recreate: bool = False,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        collection_bundle = self.ensure_collection(recreate=recreate)
        documents = list(self.catalog.documents())
        total_points = 0
        batch_count = 0
        requested_batch_size = max(int(batch_size), 1)

        for start in range(0, len(documents), requested_batch_size):
            batch_documents = documents[start : start + requested_batch_size]
            vectors = self.embedding_function.encode_documents(batch_documents)
            if len(vectors) != len(batch_documents):
                raise RuntimeError(
                    "Embedding function returned a different number of vectors than trial chunk documents."
                )
            points = [
                self.models.PointStruct(
                    id=_qdrant_point_id(getattr(document, "chunk_id", "") or ""),
                    vector=list(vector or []),
                    payload=build_qdrant_trial_chunk_payload(document),
                )
                for document, vector in zip(batch_documents, vectors, strict=False)
                if str(getattr(document, "chunk_id", "") or "").strip()
            ]
            if not points:
                continue
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            total_points += len(points)
            batch_count += 1

        return {
            **collection_bundle,
            "output_root": str(getattr(self.catalog, "output_root", "")),
            "trial_count": len(getattr(self.catalog, "_record_by_id", {}) or {}),
            "chunk_count": len(documents),
            "point_count": total_points,
            "batch_count": batch_count,
            "batch_size": requested_batch_size,
        }


class QdrantTrialChunkVectorRetriever:
    def __init__(
        self,
        catalog: Any,
        *,
        collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME,
        client: Any | None = None,
        models: Any | None = None,
        url: str | None = None,
        api_key: str | None = None,
        path: str | Path | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.catalog = catalog
        self.collection_name = str(collection_name or DEFAULT_QDRANT_COLLECTION_NAME).strip() or DEFAULT_QDRANT_COLLECTION_NAME
        _, resolved_models = _load_qdrant_dependencies() if models is None else (None, models)
        self.models = resolved_models
        self.client = client or _create_qdrant_client(url=url, api_key=api_key, path=path)
        self.embedding_function = embedding_function or MedCPTTrialChunkEmbeddingFunction()

    def _candidate_filter_components(
        self,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None,
    ) -> dict[str, list[Any]]:
        normalized_candidate_ids = _normalize_candidate_ids(candidate_ids)
        if normalized_candidate_ids is None:
            return {}
        if not normalized_candidate_ids:
            return {
                "must": [self.models.HasIdCondition(has_id=[_qdrant_point_id("__empty__")])],
                "should": [],
                "must_not": [],
            }
        return {
            "must": [self.models.HasIdCondition(has_id=[_qdrant_point_id(item) for item in normalized_candidate_ids])],
            "should": [],
            "must_not": [],
        }

    @staticmethod
    def _coerce_filter_values(values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            raw_values = [values]
        elif isinstance(values, Iterable):
            raw_values = list(values)
        else:
            raw_values = [values]

        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            text = _normalize_whitespace(value)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def _build_match_any(self, values: list[str]) -> Any:
        match_any_cls = getattr(self.models, "MatchAny", None)
        if match_any_cls is None:
            return list(values)
        for kwargs in ({"any": list(values)}, {"values": list(values)}):
            try:
                return match_any_cls(**kwargs)
            except TypeError:
                continue
        try:
            return match_any_cls(list(values))
        except TypeError:
            return list(values)

    def _build_range(self, **kwargs: float) -> Any:
        range_cls = getattr(self.models, "Range", None)
        if range_cls is None:
            return dict(kwargs)
        try:
            return range_cls(**kwargs)
        except TypeError:
            return dict(kwargs)

    def _build_field_condition(
        self,
        *,
        field_name: str,
        match: Any | None = None,
        range_value: Any | None = None,
    ) -> Any:
        field_condition_cls = getattr(self.models, "FieldCondition", None)
        if field_condition_cls is None:
            return {
                "key": str(field_name),
                "match": match,
                "range": range_value,
            }
        for kwargs in (
            {"key": str(field_name), "match": match, "range": range_value},
            {"field": str(field_name), "match": match, "range": range_value},
            {"key": str(field_name), "match": match},
            {"field": str(field_name), "match": match},
            {"key": str(field_name), "range": range_value},
            {"field": str(field_name), "range": range_value},
        ):
            cleaned_kwargs = {key: value for key, value in kwargs.items() if value is not None}
            try:
                return field_condition_cls(**cleaned_kwargs)
            except TypeError:
                continue
        return {
            "key": str(field_name),
            "match": match,
            "range": range_value,
        }

    def _match_filter_entry(self, entry: Mapping[str, Any]) -> Any | None:
        field_name = _normalize_whitespace(entry.get("field"))
        values = self._coerce_filter_values(entry.get("values"))
        if not field_name or not values:
            return None
        return self._build_field_condition(
            field_name=field_name,
            match=self._build_match_any(values),
        )

    def _payload_filter_components(
        self,
        payload_filters: Mapping[str, Any] | None,
    ) -> dict[str, list[Any]]:
        components: dict[str, list[Any]] = {
            "must": [],
            "should": [],
            "must_not": [],
        }
        if not isinstance(payload_filters, Mapping):
            return components

        for clause_name in ("must", "should", "must_not"):
            for raw_entry in list(payload_filters.get(clause_name) or []):
                if not isinstance(raw_entry, Mapping):
                    continue
                condition = self._match_filter_entry(raw_entry)
                if condition is None:
                    continue
                components[clause_name].append(condition)

        gender = _normalize_whitespace(payload_filters.get("gender"))
        if gender.casefold() in {"male", "female"}:
            allowed_gender_values = ["All", gender.title()]
            components["must"].append(
                self._build_field_condition(
                    field_name="gender",
                    match=self._build_match_any(allowed_gender_values),
                )
            )

        age_years = _coerce_float(payload_filters.get("age_years"))
        if age_years is not None:
            components["must"].append(
                self._build_field_condition(
                    field_name="age_floor_years",
                    range_value=self._build_range(lte=float(age_years)),
                )
            )
            components["must"].append(
                self._build_field_condition(
                    field_name="age_ceiling_years",
                    range_value=self._build_range(gte=float(age_years)),
                )
            )

        return components

    @staticmethod
    def _merge_filter_components(*component_sets: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        merged: dict[str, list[Any]] = {
            "must": [],
            "should": [],
            "must_not": [],
        }
        for component_set in component_sets:
            for clause_name in merged:
                merged[clause_name].extend(list(component_set.get(clause_name) or []))
        return merged

    def _build_filter(self, components: Mapping[str, list[Any]]) -> Any | None:
        must = list(components.get("must") or [])
        should = list(components.get("should") or [])
        must_not = list(components.get("must_not") or [])
        if not must and not should and not must_not:
            return None
        filter_cls = getattr(self.models, "Filter", None)
        if filter_cls is None:
            return {
                "must": must,
                "should": should,
                "must_not": must_not,
            }
        for kwargs in (
            {"must": must, "should": should, "must_not": must_not},
            {"must": must, "should": should},
            {"must": must},
        ):
            cleaned_kwargs = {
                key: value
                for key, value in kwargs.items()
                if value
            }
            try:
                return filter_cls(**cleaned_kwargs)
            except TypeError:
                continue
        return filter_cls(must=must)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        candidate_pmids: set[str] | list[str] | tuple[str, ...] | None = None,
        payload_filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = _normalize_whitespace(query)
        if not normalized_query:
            return []

        query_vector = self.embedding_function.encode_query(normalized_query)
        if not query_vector:
            return []

        query_filter = self._build_filter(
            self._merge_filter_components(
                self._candidate_filter_components(
                    candidate_ids if candidate_ids is not None else candidate_pmids
                ),
                self._payload_filter_components(payload_filters),
            )
        )
        limit = max(int(top_k), 1)

        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=list(query_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        else:  # pragma: no cover - compatibility fallback
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            hits = getattr(response, "points", response)

        rows: list[dict[str, Any]] = []
        for hit in list(hits or []):
            payload = dict(getattr(hit, "payload", {}) or {})
            chunk_id = str(payload.get("chunk_id") or getattr(hit, "id", "") or "").strip()
            if not chunk_id:
                continue
            chunk_type = _normalize_whitespace(payload.get("chunk_type"))
            summary = _normalize_whitespace(payload.get("summary") or payload.get("text"))
            rows.append(
                {
                    "document_id": chunk_id,
                    "chunk_id": chunk_id,
                    "pmid": chunk_id,
                    "nct_id": _normalize_whitespace(payload.get("nct_id")),
                    "title": _normalize_whitespace(
                        payload.get("title") or payload.get("trial_title") or payload.get("display_title") or chunk_id
                    ),
                    "summary": summary,
                    "purpose": _normalize_whitespace(payload.get("purpose") or chunk_type.replace("_", " ")),
                    "eligibility": _normalize_whitespace(
                        payload.get("eligibility") or (summary if chunk_type.startswith("eligibility") else "")
                    ),
                    "chunk_type": chunk_type,
                    "source_fields": list(payload.get("source_fields") or []),
                    "rank_weight": float(payload.get("rank_weight") or 1.0),
                    "overall_status": _normalize_whitespace(payload.get("overall_status")),
                    "study_type": _normalize_whitespace(payload.get("study_type")),
                    "phase": _normalize_whitespace(payload.get("phase")),
                    "primary_purpose": _normalize_whitespace(payload.get("primary_purpose")),
                    "conditions": _normalize_text_list(payload.get("conditions", [])),
                    "interventions": _normalize_text_list(payload.get("interventions", [])),
                    "gender": _normalize_whitespace(payload.get("gender")),
                    "age_floor_years": _coerce_float(payload.get("age_floor_years")),
                    "age_ceiling_years": _coerce_float(payload.get("age_ceiling_years")),
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                }
            )
        return rows


def create_qdrant_trial_chunk_retriever(
    catalog: Any,
    *,
    collection_name: str = DEFAULT_QDRANT_COLLECTION_NAME,
    client: Any | None = None,
    models: Any | None = None,
    url: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
    embedding_function: Any | None = None,
) -> QdrantTrialChunkVectorRetriever:
    return QdrantTrialChunkVectorRetriever(
        catalog,
        collection_name=collection_name,
        client=client,
        models=models,
        url=url,
        api_key=api_key,
        path=path,
        embedding_function=embedding_function,
    )


__all__ = [
    "DEFAULT_QDRANT_COLLECTION_NAME",
    "MedCPTTrialChunkEmbeddingFunction",
    "QdrantTrialChunkIndexManager",
    "QdrantTrialChunkVectorRetriever",
    "build_qdrant_trial_chunk_payload",
    "create_qdrant_trial_chunk_retriever",
]
