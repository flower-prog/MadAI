from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent.retrieval.trial_chunks import TrialChunkCatalog, resolve_trial_vector_output_root
from agent.trial_vector_kb import DEFAULT_CHUNK_CHAR_LIMIT, build_trial_vector_kb

from .runtime import public_trial_qdrant_runtime_config, resolve_trial_qdrant_runtime_config
from .trial_chunk import QdrantTrialChunkIndexManager


def _normalize_input_paths(values: Sequence[str | Path] | None) -> list[Path] | None:
    if values is None:
        return None
    normalized = [
        Path(value).expanduser().resolve()
        for value in list(values or [])
        if str(value).strip()
    ]
    return normalized or None


def _ensure_trial_chunk_kb_exists(output_dir: Path) -> None:
    record_path = output_dir / "trial_record.jsonl"
    chunk_path = output_dir / "trial_chunk.jsonl"
    if record_path.exists() and chunk_path.exists():
        return
    raise FileNotFoundError(
        f"Trial chunk KB is missing under {output_dir}. Expected both {record_path.name} and {chunk_path.name}."
    )


def sync_trial_chunk_kb_to_qdrant(
    *,
    output_dir: str | Path | None = None,
    collection_name: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
    recreate: bool = False,
    batch_size: int = 64,
    qdrant_client: Any | None = None,
    models: Any | None = None,
    embedding_function: Any | None = None,
) -> dict[str, Any]:
    resolved_output_dir = resolve_trial_vector_output_root(output_dir)
    _ensure_trial_chunk_kb_exists(resolved_output_dir)
    runtime_config = resolve_trial_qdrant_runtime_config(
        url=url,
        api_key=api_key,
        path=path,
        collection_name=collection_name,
        enable_default_path=True,
    )
    catalog = TrialChunkCatalog.from_output_root(resolved_output_dir)
    manager = QdrantTrialChunkIndexManager(
        catalog,
        collection_name=str(runtime_config["collection_name"] or ""),
        client=qdrant_client,
        models=models,
        url=runtime_config.get("url"),
        api_key=runtime_config.get("api_key"),
        path=runtime_config.get("path"),
        embedding_function=embedding_function,
    )
    sync_summary = manager.sync_catalog(
        recreate=bool(recreate),
        batch_size=max(int(batch_size), 1),
    )
    return {
        "output_dir": str(resolved_output_dir),
        "qdrant_config": public_trial_qdrant_runtime_config(runtime_config),
        "qdrant_sync": sync_summary,
    }


def build_and_sync_trial_chunk_kb_to_qdrant(
    *,
    input_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    limit: int = 0,
    chunk_char_limit: int = DEFAULT_CHUNK_CHAR_LIMIT,
    collection_name: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
    recreate: bool = False,
    batch_size: int = 64,
    qdrant_client: Any | None = None,
    models: Any | None = None,
    embedding_function: Any | None = None,
) -> dict[str, Any]:
    normalized_input_paths = _normalize_input_paths(input_paths)
    build_manifest = build_trial_vector_kb(
        input_paths=normalized_input_paths,
        output_dir=output_dir,
        limit=max(int(limit), 0),
        chunk_char_limit=max(int(chunk_char_limit), 1),
    )
    sync_bundle = sync_trial_chunk_kb_to_qdrant(
        output_dir=build_manifest.get("output_dir") or output_dir,
        collection_name=collection_name,
        url=url,
        api_key=api_key,
        path=path,
        recreate=recreate,
        batch_size=batch_size,
        qdrant_client=qdrant_client,
        models=models,
        embedding_function=embedding_function,
    )
    return {
        "output_dir": str(sync_bundle.get("output_dir") or build_manifest.get("output_dir") or ""),
        "build_manifest": build_manifest,
        "qdrant_config": dict(sync_bundle.get("qdrant_config") or {}),
        "qdrant_sync": dict(sync_bundle.get("qdrant_sync") or {}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the XML-derived trial chunk KB and sync it into a Qdrant collection.",
    )
    parser.add_argument(
        "--input-path",
        action="append",
        dest="input_paths",
        help="Trial XML directory or zip path. Repeat the flag to provide multiple sources.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory that stores trial_record.jsonl and trial_chunk.jsonl.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip XML parsing and reuse an existing trial chunk KB under --output-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of source XML documents to parse.",
    )
    parser.add_argument(
        "--chunk-char-limit",
        type=int,
        default=DEFAULT_CHUNK_CHAR_LIMIT,
        help="Character limit applied when building chunked trial documents.",
    )
    parser.add_argument(
        "--collection-name",
        default="",
        help="Qdrant collection name. Falls back to MEDAI_TRIAL_QDRANT_COLLECTION or the default collection.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="",
        help="Remote Qdrant URL. Optional when using local on-disk Qdrant.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default="",
        help="Qdrant API key for remote deployments.",
    )
    parser.add_argument(
        "--qdrant-path",
        default="",
        help="Local Qdrant storage path. Defaults to /home/yuanzy/MadAI/outputs/qdrant_trial_chunks.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the target Qdrant collection before upserting points.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunk vectors to upsert per batch.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    kwargs = {
        "output_dir": args.output_dir or None,
        "collection_name": args.collection_name or None,
        "url": args.qdrant_url or None,
        "api_key": args.qdrant_api_key or None,
        "path": args.qdrant_path or None,
        "recreate": bool(args.recreate),
        "batch_size": max(int(args.batch_size), 1),
    }
    if args.skip_build:
        bundle = sync_trial_chunk_kb_to_qdrant(**kwargs)
    else:
        bundle = build_and_sync_trial_chunk_kb_to_qdrant(
            input_paths=args.input_paths,
            limit=max(int(args.limit), 0),
            chunk_char_limit=max(int(args.chunk_char_limit), 1),
            **kwargs,
        )

    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "build_and_sync_trial_chunk_kb_to_qdrant",
    "main",
    "sync_trial_chunk_kb_to_qdrant",
]
