from __future__ import annotations

from typing import Any, Dict, Optional


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def extract_token_usage(obj: Any) -> Optional[Dict[str, int]]:
    """
    Best-effort extraction of token usage from different LangChain message/response shapes.

    Returns a normalized dict:
      - prompt_tokens
      - completion_tokens
      - total_tokens
    """
    if obj is None:
        return None

    # LangChain AIMessage.usage_metadata (newer)
    usage = getattr(obj, "usage_metadata", None)
    if isinstance(usage, dict):
        pt = _as_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        ct = _as_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        tt = _as_int(usage.get("total_tokens"))
        if pt is not None or ct is not None or tt is not None:
            out: Dict[str, int] = {}
            if pt is not None:
                out["prompt_tokens"] = pt
            if ct is not None:
                out["completion_tokens"] = ct
            if tt is not None:
                out["total_tokens"] = tt
            else:
                if pt is not None and ct is not None:
                    out["total_tokens"] = pt + ct
            return out or None

    # response_metadata token_usage (OpenAI-compatible often)
    meta = getattr(obj, "response_metadata", None)
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage") or meta.get("usage_metadata")
        if isinstance(token_usage, dict):
            pt = _as_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
            ct = _as_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
            tt = _as_int(token_usage.get("total_tokens"))
            if pt is not None or ct is not None or tt is not None:
                out2: Dict[str, int] = {}
                if pt is not None:
                    out2["prompt_tokens"] = pt
                if ct is not None:
                    out2["completion_tokens"] = ct
                if tt is not None:
                    out2["total_tokens"] = tt
                else:
                    if pt is not None and ct is not None:
                        out2["total_tokens"] = pt + ct
                return out2 or None

    # Some wrappers store usage in additional_kwargs
    akw = getattr(obj, "additional_kwargs", None)
    if isinstance(akw, dict):
        token_usage = akw.get("token_usage") or akw.get("usage")
        if isinstance(token_usage, dict):
            pt = _as_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
            ct = _as_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
            tt = _as_int(token_usage.get("total_tokens"))
            if pt is not None or ct is not None or tt is not None:
                out3: Dict[str, int] = {}
                if pt is not None:
                    out3["prompt_tokens"] = pt
                if ct is not None:
                    out3["completion_tokens"] = ct
                if tt is not None:
                    out3["total_tokens"] = tt
                else:
                    if pt is not None and ct is not None:
                        out3["total_tokens"] = pt + ct
                return out3 or None

    return None

