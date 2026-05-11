from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CACHE_LOCK = threading.RLock()
_TRIAL_CHUNK_CATALOG_CACHE: dict[str, "TrialChunkCatalog"] = {}


def _normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def resolve_trial_vector_output_root(output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        project_root / "vector_stores" / "trials" / "full" / "kb",
        project_root / "vector_stores" / "trials" / "part1_caseprobe" / "kb",
        project_root / "vector_stores" / "trials" / "smoke" / "kb",
        project_root / "outputs" / "trial_vector_kb_full",
        project_root / "outputs" / "trial_vector_kb_part1_caseprobe",
        project_root / "outputs" / "trial_vector_kb",
        project_root / "data" / "trial_vector_kb",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass(slots=True)
class TrialChunkDocument:
    chunk_id: str
    nct_id: str
    title: str
    chunk_type: str
    sequence: int = 0
    text: str = ""
    embedding_text: str = ""
    source_fields: list[str] = field(default_factory=list)
    rank_weight: float = 1.0
    token_estimate: int = 0
    trial_title: str = ""
    record_payload: dict[str, Any] = field(default_factory=dict)
    pmid: str = field(init=False)
    retrieval_text: str = field(init=False)
    summary: str = field(init=False)
    purpose: str = field(init=False)
    eligibility: str = field(init=False)

    def __post_init__(self) -> None:
        self.pmid = self.chunk_id
        self.retrieval_text = _normalize_whitespace(self.embedding_text or self.text)
        self.summary = _normalize_whitespace(self.text)
        self.purpose = self.chunk_type.replace("_", " ")
        self.eligibility = self.summary if self.chunk_type.startswith("eligibility") else ""

    def to_brief(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "nct_id": self.nct_id,
            "trial_id": self.nct_id,
            "title": self.title,
            "summary": self.summary,
            "purpose": self.purpose,
            "eligibility": self.eligibility,
            "chunk_type": self.chunk_type,
            "sequence": int(self.sequence),
            "text": self.text,
            "source_fields": list(self.source_fields),
            "rank_weight": float(self.rank_weight),
            "token_estimate": int(self.token_estimate),
            "trial_title": self.trial_title,
        }


def _record_trial_title(record: Mapping[str, Any], nct_id: str) -> str:
    return _normalize_whitespace(
        record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
    )


def _build_trial_chunk_document(
    payload: Mapping[str, Any],
    *,
    record_by_id: Mapping[str, dict[str, Any]],
) -> TrialChunkDocument | None:
    chunk_id = str(payload.get("chunk_id") or "").strip()
    nct_id = str(payload.get("nct_id") or "").strip()
    if not chunk_id or not nct_id:
        return None

    record = dict(record_by_id.get(nct_id, {}) or {})
    trial_title = _record_trial_title(record, nct_id)
    chunk_type = _normalize_whitespace(payload.get("chunk_type"))
    return TrialChunkDocument(
        chunk_id=chunk_id,
        nct_id=nct_id,
        title=f"{trial_title} [{chunk_type}]",
        chunk_type=chunk_type,
        sequence=int(payload.get("sequence") or 0),
        text=_normalize_whitespace(payload.get("text")),
        embedding_text=_normalize_whitespace(payload.get("embedding_text")),
        source_fields=list(payload.get("source_fields") or []),
        rank_weight=float(payload.get("rank_weight") or 1.0),
        token_estimate=int(payload.get("token_estimate") or 0),
        trial_title=trial_title,
        record_payload=record,
    )


class TrialChunkCatalog:
    def __init__(
        self,
        *,
        output_root: Path,
        records: list[dict[str, Any]],
        chunks: list[TrialChunkDocument] | None = None,
        record_path: Path | None = None,
        chunk_path: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.record_path = Path(record_path or self.output_root / "trial_record.jsonl").resolve()
        self.chunk_path = Path(chunk_path or self.output_root / "trial_chunk.jsonl").resolve()
        self.manifest_path = Path(manifest_path or self.output_root / "manifest.json").resolve()
        self._records = [dict(record) for record in records]
        self._record_by_id = {
            str(record.get("nct_id") or "").strip(): dict(record)
            for record in self._records
            if str(record.get("nct_id") or "").strip()
        }
        self._chunks = list(chunks) if chunks is not None else None
        self._chunk_by_id: dict[str, TrialChunkDocument] = {}
        self.chunk_index_by_trial: dict[str, set[str]] = defaultdict(set)
        self._chunk_count_cache = len(self._chunks) if self._chunks is not None else None
        if self._chunks is not None:
            self._rebuild_chunk_indexes()
        self.runtime_cache_key = f"trial-chunk-catalog:{self.output_root}"

    def documents(self) -> list[TrialChunkDocument]:
        self._ensure_chunks_loaded()
        return list(self._chunks or [])

    def get(self, chunk_id: str) -> TrialChunkDocument | None:
        self._ensure_chunks_loaded()
        return self._chunk_by_id.get(str(chunk_id or "").strip())

    def get_record(self, nct_id: str) -> dict[str, Any] | None:
        record = self._record_by_id.get(str(nct_id or "").strip())
        if record is None:
            return None
        return dict(record)

    def trial_count(self) -> int:
        return len(self._record_by_id)

    def document_count(self) -> int:
        if self._chunk_count_cache is not None:
            return max(int(self._chunk_count_cache), 0)

        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            raw_chunk_count = manifest.get("trial_chunk_count")
            if raw_chunk_count is not None:
                try:
                    self._chunk_count_cache = max(int(raw_chunk_count), 0)
                    return self._chunk_count_cache
                except (TypeError, ValueError):
                    pass

        chunk_count = 0
        for _ in self.iter_documents():
            chunk_count += 1
        self._chunk_count_cache = chunk_count
        return chunk_count

    def document_at(self, index: int) -> TrialChunkDocument | None:
        target_index = int(index)
        if target_index < 0:
            return None
        if self._chunks is not None:
            if target_index >= len(self._chunks):
                return None
            return self._chunks[target_index]
        for document in self.iter_documents(start_offset=target_index, stop_offset=target_index + 1):
            return document
        return None

    def iter_documents(
        self,
        *,
        start_offset: int = 0,
        stop_offset: int | None = None,
    ) -> Iterator[TrialChunkDocument]:
        normalized_start = max(int(start_offset), 0)
        normalized_stop = None if stop_offset is None else max(int(stop_offset), 0)
        if normalized_stop is not None and normalized_stop <= normalized_start:
            return

        if self._chunks is not None:
            for document in list(self._chunks[normalized_start:normalized_stop]):
                yield document
            return

        if not self.chunk_path.exists():
            return

        document_index = 0
        with self.chunk_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                document = _build_trial_chunk_document(
                    dict(json.loads(line)),
                    record_by_id=self._record_by_id,
                )
                if document is None:
                    continue
                current_index = document_index
                document_index += 1
                if current_index < normalized_start:
                    continue
                if normalized_stop is not None and current_index >= normalized_stop:
                    break
                yield document

    def dense_index_cache_paths(self, *, query_model_name: str, doc_model_name: str) -> tuple[Path, Path]:
        cache_root = self.output_root / ".cache"
        suffix = f"{query_model_name}__{doc_model_name}".replace("/", "__")
        return (
            cache_root / f"trial_chunks.{suffix}.faiss",
            cache_root / f"trial_chunks.{suffix}.pmids.json",
        )

    @classmethod
    def from_output_root(
        cls,
        output_root: str | Path | None = None,
        *,
        load_documents: bool = True,
    ) -> "TrialChunkCatalog":
        resolved_root = resolve_trial_vector_output_root(output_root)
        cache_key = str(resolved_root)
        with _CACHE_LOCK:
            cached = _TRIAL_CHUNK_CATALOG_CACHE.get(cache_key)
        if cached is not None:
            if load_documents:
                cached._ensure_chunks_loaded()
            return cached

        record_path = resolved_root / "trial_record.jsonl"
        chunk_path = resolved_root / "trial_chunk.jsonl"
        manifest_path = resolved_root / "manifest.json"
        records: list[dict[str, Any]] = []
        chunks: list[TrialChunkDocument] | None = [] if load_documents else None

        if record_path.exists():
            with record_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

        record_by_id = {
            str(record.get("nct_id") or "").strip(): dict(record)
            for record in records
            if str(record.get("nct_id") or "").strip()
        }
        if load_documents and chunk_path.exists():
            with chunk_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    document = _build_trial_chunk_document(
                        dict(json.loads(line)),
                        record_by_id=record_by_id,
                    )
                    if document is None:
                        continue
                    chunks.append(document)

        catalog = cls(
            output_root=resolved_root,
            records=records,
            chunks=chunks,
            record_path=record_path,
            chunk_path=chunk_path,
            manifest_path=manifest_path,
        )
        with _CACHE_LOCK:
            cached = _TRIAL_CHUNK_CATALOG_CACHE.setdefault(cache_key, catalog)
        if cached is not catalog and load_documents:
            cached._ensure_chunks_loaded()
        return cached

    def _rebuild_chunk_indexes(self) -> None:
        self._chunk_by_id = {}
        self.chunk_index_by_trial = defaultdict(set)
        for chunk in list(self._chunks or []):
            self._chunk_by_id[chunk.chunk_id] = chunk
            self.chunk_index_by_trial[str(chunk.nct_id)].add(chunk.chunk_id)

    def _ensure_chunks_loaded(self) -> None:
        if self._chunks is not None:
            return
        self._chunks = list(self.iter_documents())
        self._chunk_count_cache = len(self._chunks)
        self._rebuild_chunk_indexes()


__all__ = [
    "TrialChunkCatalog",
    "TrialChunkDocument",
    "resolve_trial_vector_output_root",
]
