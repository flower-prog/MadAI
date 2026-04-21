from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .trial_chunk import DEFAULT_QDRANT_COLLECTION_NAME


DEFAULT_TRIAL_QDRANT_SERVER_URL = "http://127.0.0.1:6333"


def _normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def default_trial_qdrant_server_url() -> str:
    return DEFAULT_TRIAL_QDRANT_SERVER_URL


def default_trial_qdrant_storage_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "outputs" / "qdrant_trial_chunks"


def resolve_trial_qdrant_runtime_config(
    *,
    url: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
    collection_name: str | None = None,
    enable_default_url: bool = True,
    enable_default_path: bool = False,
) -> dict[str, Any]:
    default_qdrant_path = default_trial_qdrant_storage_path()
    requested_url = _normalize_whitespace(url or os.getenv("MEDAI_TRIAL_QDRANT_URL")) or None
    resolved_api_key = _normalize_whitespace(api_key or os.getenv("MEDAI_TRIAL_QDRANT_API_KEY")) or None
    requested_path = _normalize_whitespace(path or os.getenv("MEDAI_TRIAL_QDRANT_PATH"))
    resolved_collection_name = (
        _normalize_whitespace(collection_name or os.getenv("MEDAI_TRIAL_QDRANT_COLLECTION"))
        or DEFAULT_QDRANT_COLLECTION_NAME
    )

    resolved_url = requested_url
    if resolved_url is None and not requested_path and enable_default_url:
        resolved_url = default_trial_qdrant_server_url()

    resolved_path: str | None = None
    if requested_path:
        resolved_path = str(Path(requested_path).expanduser().resolve())
    elif enable_default_path and not resolved_url:
        resolved_path = str(default_qdrant_path.resolve())

    return {
        "enabled": bool(resolved_url or resolved_path),
        "url": resolved_url,
        "api_key": resolved_api_key,
        "path": resolved_path,
        "collection_name": resolved_collection_name,
    }


def public_trial_qdrant_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(config.get("enabled")),
        "url": str(config.get("url") or ""),
        "path": str(config.get("path") or ""),
        "collection_name": str(config.get("collection_name") or DEFAULT_QDRANT_COLLECTION_NAME),
        "has_api_key": bool(config.get("api_key")),
    }


__all__ = [
    "DEFAULT_TRIAL_QDRANT_SERVER_URL",
    "default_trial_qdrant_server_url",
    "default_trial_qdrant_storage_path",
    "public_trial_qdrant_runtime_config",
    "resolve_trial_qdrant_runtime_config",
]
