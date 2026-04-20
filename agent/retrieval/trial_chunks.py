from __future__ import annotations

import json
import threading
from collections import defaultdict
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


class TrialChunkCatalog:
    def __init__(
        self,
        *,
        output_root: Path,
        records: list[dict[str, Any]],
        chunks: list[TrialChunkDocument],
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self._records = [dict(record) for record in records]
        self._record_by_id = {
            str(record.get("nct_id") or "").strip(): dict(record)
            for record in self._records
            if str(record.get("nct_id") or "").strip()
        }
        self._chunks = list(chunks)
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self.chunk_index_by_trial: dict[str, set[str]] = defaultdict(set)
        for chunk in self._chunks:
            self.chunk_index_by_trial[str(chunk.nct_id)].add(chunk.chunk_id)
        self.runtime_cache_key = f"trial-chunk-catalog:{self.output_root}"

    def documents(self) -> list[TrialChunkDocument]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> TrialChunkDocument | None:
        return self._chunk_by_id.get(str(chunk_id or "").strip())

    def get_record(self, nct_id: str) -> dict[str, Any] | None:
        record = self._record_by_id.get(str(nct_id or "").strip())
        if record is None:
            return None
        return dict(record)

    def dense_index_cache_paths(self, *, query_model_name: str, doc_model_name: str) -> tuple[Path, Path]:
        cache_root = self.output_root / ".cache"
        suffix = f"{query_model_name}__{doc_model_name}".replace("/", "__")
        return (
            cache_root / f"trial_chunks.{suffix}.faiss",
            cache_root / f"trial_chunks.{suffix}.pmids.json",
        )

    @classmethod
    def from_output_root(cls, output_root: str | Path | None = None) -> "TrialChunkCatalog":
        resolved_root = resolve_trial_vector_output_root(output_root)
        cache_key = str(resolved_root)
        with _CACHE_LOCK:
            cached = _TRIAL_CHUNK_CATALOG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        record_path = resolved_root / "trial_record.jsonl"
        chunk_path = resolved_root / "trial_chunk.jsonl"
        records: list[dict[str, Any]] = []
        chunks: list[TrialChunkDocument] = []

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
        if chunk_path.exists():
            with chunk_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = dict(json.loads(line))
                    chunk_id = str(payload.get("chunk_id") or "").strip()
                    nct_id = str(payload.get("nct_id") or "").strip()
                    if not chunk_id or not nct_id:
                        continue
                    record = record_by_id.get(nct_id, {})
                    trial_title = _normalize_whitespace(
                        record.get("display_title") or record.get("brief_title") or record.get("official_title") or nct_id
                    )
                    chunk_type = _normalize_whitespace(payload.get("chunk_type"))
                    chunks.append(
                        TrialChunkDocument(
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
                            record_payload=dict(record),
                        )
                    )

        catalog = cls(output_root=resolved_root, records=records, chunks=chunks)
        with _CACHE_LOCK:
            _TRIAL_CHUNK_CATALOG_CACHE.setdefault(cache_key, catalog)
        return catalog


__all__ = [
    "TrialChunkCatalog",
    "TrialChunkDocument",
    "resolve_trial_vector_output_root",
]
