from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatZhipuAI
import os
import logging
import math
import json
from typing import Any, Optional, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

try:
    from openai import RateLimitError as OpenAIRateLimitError
    from openai import APIConnectionError as OpenAIAPIConnectionError
    from openai import APITimeoutError as OpenAIAPITimeoutError
except Exception:  # pragma: no cover - openai package is a hard dependency in runtime
    OpenAIRateLimitError = Exception  # type: ignore[assignment]
    OpenAIAPIConnectionError = Exception  # type: ignore[assignment]
    OpenAIAPITimeoutError = Exception  # type: ignore[assignment]

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]

from src.config import (
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    CODING_MODEL,
    CODING_BASE_URL,
    CODING_API_KEY,
    DEEPSEEK_STREAMING,
)
from src.config.agents import LLMType


_DASHSCOPE_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
logger = logging.getLogger(__name__)


def _first_nonempty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if str(value).strip() == "":
            continue
        return str(value)
    return None


def _is_truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def create_zhipuai_llm(
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatZhipuAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if api_base:  # This will handle None or empty string
        llm_kwargs["api_base"] = api_base

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatZhipuAI(**llm_kwargs)


def create_qwen_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    streaming: bool = True,
    **kwargs,
) -> ChatOpenAI:
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
        **kwargs,
    }

    if base_url:
        llm_kwargs["base_url"] = base_url
    if api_key:
        llm_kwargs["api_key"] = api_key
    return ChatOpenAI(**llm_kwargs)


def _normalize_kimi_thinking_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    if v in ("enabled", "disabled"):
        return v
    if v in ("1", "true", "yes", "y", "on", "enable", "enabled"):
        return "enabled"
    if v in ("0", "false", "no", "n", "off", "disable", "disabled"):
        return "disabled"
    return None


def _resolve_kimi_thinking_mode(llm_type: Optional[LLMType] = None) -> Optional[str]:
    """
    Determine whether to enable or disable Kimi thinking mode based on env vars.
    Type-specific envs take precedence over global defaults.
    """
    candidates = []
    if llm_type:
        suffix = str(llm_type).upper()
        candidates.extend(
            [
                f"KIMI_THINKING_MODE_{suffix}",
                f"KIMI_THINKING_{suffix}",
            ]
        )
    candidates.extend(["KIMI_THINKING_MODE", "KIMI_THINKING_ENABLED"])
    for name in candidates:
        raw = os.getenv(name)
        normalized = _normalize_kimi_thinking_value(raw)
        if normalized:
            return normalized
    return None


def create_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    kimi_thinking: Optional[str] = None,
    **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    # #region agent log
    # #endregion
    # 检测 base_url 是否是 Moonshot/Kimi
    bu = (base_url or "").strip().lower()
    is_kimi = "moonshot" in bu or "api.moonshot.cn" in bu
    model_lower = model.lower()
    kimi_thinking_mode = _normalize_kimi_thinking_value(kimi_thinking)
    # #region agent log
    # #endregion
    
    # Kimi 不同模型有不同的 temperature 限制：
    # - kimi-k2.5 + thinking enabled:  只允许 temperature=1.0
    # - kimi-k2.5 + thinking disabled: 只允许 temperature=0.6
    # - kimi-k2-thinking 系列: 默认 1.0
    # - kimi-k2-turbo-preview, kimi-k2 系列: 默认 0.6
    # - moonshot-v1 系列: 默认 0.0
    if is_kimi:
        # 先确定 thinking mode（后续设置 temperature 需要依赖它）
        if kimi_thinking_mode is None:
            kimi_thinking_mode = _resolve_kimi_thinking_mode()
        thinking_enabled = kimi_thinking_mode == "enabled"

        if "k2.5" in model_lower:
            # kimi-k2.5 的 temperature 取决于 thinking mode
            temperature = 1.0 if thinking_enabled else 0.6
        elif "k2-thinking" in model_lower:
            temperature = 1.0
        elif "k2" in model_lower:
            # kimi-k2-turbo-preview 等 k2 系列，默认 0.6
            temperature = 0.6
        # moonshot-v1 系列保持 0.0
    
    # #region agent log
    # #endregion
    
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["base_url"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key
    
    if is_kimi:
        if kimi_thinking_mode:
            if "extra_body" not in llm_kwargs:
                llm_kwargs["extra_body"] = {}
            llm_kwargs["extra_body"]["thinking"] = {"type": kimi_thinking_mode}

    return ChatOpenAI(**llm_kwargs)


def create_azure_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    api_version: Optional[str] = None,
    **kwargs,
) -> AzureChatOpenAI:
    """
    Create an AzureChatOpenAI instance.
    The base_url should be the Azure OpenAI endpoint (e.g. https://<resource>.openai.azure.com/).
    The model parameter corresponds to deployment_name.
    """
    resolved_api_version = api_version or os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
    )
    
    # Keep explicit low-temperature routing deterministic by default.
    # Set AZURE_FORCE_TEMPERATURE_ONE=1 only when deployment requires it.
    force_temp_one = str(os.getenv("AZURE_FORCE_TEMPERATURE_ONE", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    resolved_temperature = temperature
    model_lower = (model or "").strip().lower()

    # Azure gpt-5 family rejects temperature=0.0 and only accepts default-like 1.0.
    if resolved_temperature == 0.0 and "gpt-5" in model_lower:
        resolved_temperature = 1.0
    if force_temp_one and resolved_temperature == 0.0:
        resolved_temperature = 1.0

    llm_kwargs = {
        "azure_deployment": model,  # In Azure, model usually refers to deployment name
        "azure_endpoint": base_url,
        "api_key": api_key,
        "api_version": resolved_api_version,
        **kwargs,
    }
    if resolved_temperature is not None:
        llm_kwargs["temperature"] = resolved_temperature

    return AzureChatOpenAI(**llm_kwargs)

def create_deepseek_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    streaming: bool = True,  # Enable streaming by default for better UX
    **kwargs,
) -> ChatDeepSeek:
    """
    Create a ChatDeepSeek instance with the specified configuration.

    Args:
        model: Model name
        base_url: API base URL
        api_key: API key
        temperature: Sampling temperature
        streaming: Whether to enable streaming (default: True)
                   Set to False if experiencing network stability issues like
                   'incomplete chunked read' errors
        **kwargs: Additional arguments passed to ChatDeepSeek

    Note:
        Streaming provides better UX (real-time token-by-token feedback) but may
        fail on unstable networks. If you encounter connection errors during
        long-running tasks, set streaming=False via DEEPSEEK_STREAMING env var.
    """

    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
        **kwargs,
    }
    # Only include base_url in the arguments if it's not None or empty
 

    if base_url:  # This will handle None or empty string
        llm_kwargs["api_base"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatDeepSeek(**llm_kwargs)


# -------------------- Provider Selection -------------------- #
def _infer_provider(provider_env: Optional[str], base_url: Optional[str]) -> str:
    """
    Decide which LangChain chat model wrapper to use for a given config.

    Returns: 'openai' or 'deepseek' (default).
    """
    if provider_env:
        p = str(provider_env).strip().lower()
        if p in ("openai", "openai_compatible", "kimi", "moonshot", "qwen"):
            return "openai"
        if p in ("deepseek",):
            return "deepseek"
    bu = (base_url or "").strip().lower()
    # Moonshot (Kimi) is OpenAI-compatible.
    if "moonshot" in bu or "api.moonshot.cn" in bu:
        return "openai"
    # Default to deepseek wrapper for DeepSeek endpoints.
    if "deepseek" in bu:
        return "deepseek"
    # Fallback: OpenAI-compatible is the safest generic default if base_url is set.
    if bu:
        if "azure.com" in bu:
            return "azure"
        return "openai"
    return "deepseek"


def _normalize_azure_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """
    Normalize Azure endpoint to root format:
    https://<resource>.cognitiveservices.azure.com/

    Accepts accidental full request URLs such as:
    https://.../openai/responses?api-version=...
    """
    raw = (endpoint or "").strip()
    if not raw:
        return None

    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))


def _split_csv_env(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _is_kimi_primary(
    *,
    provider_env: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
) -> bool:
    provider = (provider_env or "").strip().lower()
    if provider in {"kimi", "moonshot"}:
        return True

    bu = (base_url or "").strip().lower()
    if "moonshot" in bu or "api.moonshot.cn" in bu:
        return True

    model_lower = (model or "").strip().lower()
    return model_lower.startswith("kimi-")


def _resolve_azure_fallback_configs(llm_type: LLMType) -> list[dict[str, str]]:
    suffix = str(llm_type).upper()
    endpoints_csv = _first_nonempty(
        os.getenv(f"AZURE_FALLBACK_ENDPOINTS_{suffix}"),
        os.getenv("AZURE_FALLBACK_ENDPOINTS"),
    )
    keys_csv = _first_nonempty(
        os.getenv(f"AZURE_FALLBACK_API_KEYS_{suffix}"),
        os.getenv("AZURE_FALLBACK_API_KEYS"),
    )

    endpoint = _first_nonempty(
        os.getenv(f"AZURE_FALLBACK_ENDPOINT_{suffix}"),
        os.getenv("AZURE_FALLBACK_ENDPOINT"),
        os.getenv("AZURE_ENDPOINT"),
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_BASE_URL"),
    )
    api_key = _first_nonempty(
        os.getenv(f"AZURE_FALLBACK_API_KEY_{suffix}"),
        os.getenv("AZURE_FALLBACK_API_KEY"),
        os.getenv("SUBSCRIPTION_KEY"),
        os.getenv("subscription_key"),
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_API_KEY"),
    )
    model = _first_nonempty(
        os.getenv(f"AZURE_FALLBACK_MODEL_{suffix}"),
        os.getenv(f"AZURE_FALLBACK_DEPLOYMENT_{suffix}"),
        os.getenv("AZURE_FALLBACK_MODEL"),
        os.getenv("AZURE_FALLBACK_DEPLOYMENT"),
    )
    api_version = _first_nonempty(
        os.getenv(f"AZURE_FALLBACK_API_VERSION_{suffix}"),
        os.getenv("AZURE_FALLBACK_API_VERSION"),
        os.getenv("AZURE_OPENAI_API_VERSION"),
        "2024-12-01-preview",
    )

    if not model:
        return []

    endpoint_candidates = _split_csv_env(endpoints_csv)
    if endpoint:
        endpoint_candidates.append(endpoint)
    normalized_endpoints: list[str] = []
    for raw_endpoint in endpoint_candidates:
        normalized = _normalize_azure_endpoint(raw_endpoint)
        if not normalized:
            continue
        if normalized in normalized_endpoints:
            continue
        normalized_endpoints.append(normalized)
    if not normalized_endpoints:
        return []

    key_candidates = _split_csv_env(keys_csv)
    if api_key:
        key_candidates.append(api_key)
    deduped_keys: list[str] = []
    for key in key_candidates:
        if key in deduped_keys:
            continue
        deduped_keys.append(key)
    if not deduped_keys:
        return []

    # Support one key for all endpoints, or one-to-one key mapping.
    if len(deduped_keys) == 1 and len(normalized_endpoints) > 1:
        resolved_keys = [deduped_keys[0]] * len(normalized_endpoints)
    else:
        resolved_keys = deduped_keys[: len(normalized_endpoints)]
        if len(resolved_keys) < len(normalized_endpoints):
            logger.warning(
                "Azure fallback endpoints(%d) exceed keys(%d); truncating to paired subset.",
                len(normalized_endpoints),
                len(deduped_keys),
            )
            normalized_endpoints = normalized_endpoints[: len(resolved_keys)]

    return [
        {
            "endpoint": ep,
            "api_key": key,
            "model": model,
            "api_version": api_version or "2024-12-01-preview",
        }
        for ep, key in zip(normalized_endpoints, resolved_keys)
    ]


def _is_kimi_azure_fallback_enabled() -> bool:
    raw = os.getenv("KIMI_AZURE_FALLBACK_ENABLED")
    if raw is None:
        # default on when fallback credentials are configured
        return True
    return _is_truthy_env(raw)


def _is_kimi_azure_fallback_on_connect_enabled() -> bool:
    raw = os.getenv("KIMI_AZURE_FALLBACK_ON_CONNECT_ERROR")
    if raw is None:
        return True
    return _is_truthy_env(raw)


def _build_kimi_fallback_exceptions() -> tuple[type[BaseException], ...]:
    exceptions: list[type[BaseException]] = [OpenAIRateLimitError]
    if _is_kimi_azure_fallback_on_connect_enabled():
        exceptions.extend(
            [
                OpenAIAPIConnectionError,
                OpenAIAPITimeoutError,
                ConnectionError,
                TimeoutError,
            ]
        )

    deduped: list[type[BaseException]] = []
    for exc_cls in exceptions:
        if not isinstance(exc_cls, type):
            continue
        if not issubclass(exc_cls, BaseException):
            continue
        if exc_cls in deduped:
            continue
        deduped.append(exc_cls)
    return tuple(deduped)


def _maybe_attach_kimi_azure_fallback(
    llm: Any,
    *,
    llm_type: LLMType,
    provider_env: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
) -> Any:
    if not _is_kimi_azure_fallback_enabled():
        return llm
    if not _is_kimi_primary(
        provider_env=provider_env,
        base_url=base_url,
        model=model,
    ):
        return llm

    cfgs = _resolve_azure_fallback_configs(llm_type)
    if not cfgs:
        return llm

    fallback_llms = [
        create_azure_llm(
            model=cfg["model"],
            base_url=cfg["endpoint"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
            streaming=DEEPSEEK_STREAMING,
        )
        for cfg in cfgs
    ]
    fallback_exceptions = _build_kimi_fallback_exceptions()
    return llm.with_fallbacks(
        fallback_llms,
        exceptions_to_handle=fallback_exceptions,
    )


# Cache for LLM instances
_llm_cache: dict[LLMType, Any] = {}
_compression_llm_cache: Any = None


def _env_int_first(keys: Sequence[str], default: int, *, min_value: int = 1) -> int:
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            return max(min_value, int(text))
        except Exception:
            continue
    return max(min_value, int(default))


def _env_float_first(
    keys: Sequence[str],
    default: float,
    *,
    min_value: float,
    max_value: float,
) -> float:
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            value = float(text)
            return max(min_value, min(max_value, value))
        except Exception:
            continue
    return max(min_value, min(max_value, float(default)))


def _resolve_model_for_llm_type(llm_type: LLMType) -> str:
    if llm_type == "reasoning":
        return REASONING_MODEL
    if llm_type == "vision":
        return VL_MODEL
    if llm_type == "coding":
        return CODING_MODEL
    return BASIC_MODEL


def _default_model_context_windows(model_name: str) -> tuple[int, int]:
    """
    Return (input_window_tokens, output_window_tokens) for known models.
    Unknown models fall back conservatively to 128k/16k.
    """
    model_lower = str(model_name or "").strip().lower()
    if not model_lower:
        return 128_000, 16_384

    # User-confirmed window for Kimi k2.5.
    if "kimi-k2.5" in model_lower:
        return 262_144, 16_384
    if model_lower.startswith("kimi-"):
        return 128_000, 16_384

    # Azure/OpenAI GPT-5 family.
    if "gpt-5" in model_lower:
        return 272_000, 128_000
    if "gpt-4.1" in model_lower:
        return 1_000_000, 32_000
    if "gpt-4o" in model_lower:
        return 128_000, 16_384

    if "qwen" in model_lower:
        return 128_000, 16_384
    if "deepseek" in model_lower:
        return 128_000, 16_384
    return 128_000, 16_384


def _default_effective_input_tokens(model_name: str, input_window_tokens: int) -> int:
    """
    Effective usable prompt budget (after safety margin), not raw provider max.
    """
    model_lower = str(model_name or "").strip().lower()
    if "kimi-k2.5" in model_lower:
        return min(input_window_tokens, 200_000)
    if "gpt-5" in model_lower:
        return min(input_window_tokens, 200_000)
    # Conservative default for mixed providers.
    return max(16_384, int(input_window_tokens * 0.75))


def get_context_budget_for_llm_type(llm_type: LLMType) -> dict[str, Any]:
    """
    Compute model-aware context budget for a given llm_type.

    Environment overrides (priority: type-specific > global):
    - BIOINFO_MODEL_INPUT_WINDOW_TOKENS[_<TYPE>]
    - BIOINFO_MODEL_OUTPUT_WINDOW_TOKENS[_<TYPE>]
    - BIOINFO_EFFECTIVE_INPUT_TOKENS[_<TYPE>]
    - BIOINFO_CONTEXT_INTERNAL_OVERHEAD_TOKENS[_<TYPE>]
    - BIOINFO_COMPRESS_THRESHOLD[_<TYPE>]
    """
    suffix = str(llm_type).upper()
    model_name = _resolve_model_for_llm_type(llm_type)
    inferred_input, inferred_output = _default_model_context_windows(model_name)

    input_window_tokens = _env_int_first(
        [f"BIOINFO_MODEL_INPUT_WINDOW_TOKENS_{suffix}", "BIOINFO_MODEL_INPUT_WINDOW_TOKENS"],
        inferred_input,
    )
    output_window_tokens = _env_int_first(
        [f"BIOINFO_MODEL_OUTPUT_WINDOW_TOKENS_{suffix}", "BIOINFO_MODEL_OUTPUT_WINDOW_TOKENS"],
        inferred_output,
    )
    effective_input_tokens = _env_int_first(
        [f"BIOINFO_EFFECTIVE_INPUT_TOKENS_{suffix}", "BIOINFO_EFFECTIVE_INPUT_TOKENS"],
        _default_effective_input_tokens(model_name, input_window_tokens),
    )
    effective_input_tokens = min(effective_input_tokens, input_window_tokens)

    # Covers system prompt templates, LangGraph/tool schema overhead, and reserved buffer.
    internal_overhead_tokens = _env_int_first(
        [
            f"BIOINFO_CONTEXT_INTERNAL_OVERHEAD_TOKENS_{suffix}",
            "BIOINFO_CONTEXT_INTERNAL_OVERHEAD_TOKENS",
        ],
        12_000,
    )

    compress_threshold = _env_float_first(
        [f"BIOINFO_COMPRESS_THRESHOLD_{suffix}", "BIOINFO_COMPRESS_THRESHOLD"],
        0.80,
        min_value=0.10,
        max_value=0.98,
    )
    compress_trigger_tokens = max(1, int(effective_input_tokens * compress_threshold))

    return {
        "llm_type": llm_type,
        "model_name": model_name,
        "input_window_tokens": input_window_tokens,
        "output_window_tokens": output_window_tokens,
        "effective_input_tokens": effective_input_tokens,
        "internal_overhead_tokens": internal_overhead_tokens,
        "compress_threshold": compress_threshold,
        "compress_trigger_tokens": compress_trigger_tokens,
    }


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if isinstance(item, Mapping):
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val:
                    parts.append(text_val)
                    continue
                item_type = str(item.get("type") or "").lower()
                if item_type == "image_url":
                    parts.append("[image_url]")
                    continue
                if item_type == "image":
                    parts.append("[image]")
                    continue
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    parts.append(str(item))
                continue
            parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, Mapping):
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


def _extract_message_fields(message: Any) -> tuple[str, str, str]:
    role = ""
    name = ""
    content_obj: Any = ""
    if isinstance(message, Mapping):
        role = str(message.get("role") or message.get("type") or "")
        name = str(message.get("name") or "")
        content_obj = message.get("content", "")
    else:
        role = str(getattr(message, "type", "") or "")
        name = str(getattr(message, "name", "") or "")
        content_obj = getattr(message, "content", "")
    return role, name, _normalize_message_content(content_obj)


def _estimate_text_tokens_fallback(text: str) -> int:
    if not text:
        return 0
    total_chars = len(text)
    if total_chars <= 0:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = non_ascii / total_chars
    chars_per_token = 2.0 if ratio >= 0.35 else 3.8
    return max(1, int(math.ceil(total_chars / chars_per_token)))


def estimate_messages_tokens(messages: Sequence[Any], *, model_name: str) -> int:
    """
    Best-effort token estimate for chat messages (without hidden provider internals).
    """
    normalized_messages = [m for m in (messages or [])]
    if not normalized_messages:
        return 0

    enc = None
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None

    total = 0
    # ChatML-ish framing overhead.
    total += 2
    for message in normalized_messages:
        role, name, content = _extract_message_fields(message)
        if enc is not None:
            total += 4
            if role:
                total += len(enc.encode(role))
            if name:
                total += len(enc.encode(name))
            if content:
                total += len(enc.encode(content))
        else:
            total += 4
            total += _estimate_text_tokens_fallback(role)
            total += _estimate_text_tokens_fallback(name)
            total += _estimate_text_tokens_fallback(content)
    return max(0, int(total))


def estimate_context_usage_for_llm_type(
    messages: Sequence[Any],
    *,
    llm_type: LLMType,
) -> dict[str, Any]:
    """
    Estimate projected prompt-context usage for one LLM type.
    projected_prompt_tokens includes configurable internal overhead.
    """
    budget = get_context_budget_for_llm_type(llm_type)
    message_tokens = estimate_messages_tokens(messages, model_name=budget["model_name"])
    projected_prompt_tokens = int(message_tokens) + int(budget["internal_overhead_tokens"])
    effective_input_tokens = max(1, int(budget["effective_input_tokens"]))
    utilization_ratio = projected_prompt_tokens / effective_input_tokens
    return {
        **budget,
        "message_tokens": int(message_tokens),
        "projected_prompt_tokens": int(projected_prompt_tokens),
        "utilization_ratio": float(utilization_ratio),
    }


def get_llm_by_type(llm_type: LLMType) -> Any:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    # #region agent log
    # #endregion
    if llm_type in _llm_cache:
        # #region agent log
        # #endregion
        return _llm_cache[llm_type]

    kimi_thinking_pref = _resolve_kimi_thinking_mode(llm_type)

    if llm_type == "reasoning":
        stream_usage = os.getenv("BIOINFO_STREAM_USAGE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        provider = _infer_provider(
            os.getenv("REASONING_PROVIDER"), REASONING_BASE_URL
        )
        if provider == "openai":
            llm = create_openai_llm(
                model=REASONING_MODEL,
                base_url=REASONING_BASE_URL,
                api_key=REASONING_API_KEY,
                streaming=DEEPSEEK_STREAMING,
                stream_usage=stream_usage if DEEPSEEK_STREAMING else None,
                kimi_thinking=kimi_thinking_pref,
            )
        elif provider == "azure":
            llm = create_azure_llm(
                model=REASONING_MODEL,
                base_url=REASONING_BASE_URL,
                api_key=REASONING_API_KEY,
                streaming=DEEPSEEK_STREAMING,
            )
        else:
            llm = create_deepseek_llm(
                model=REASONING_MODEL,
                base_url=REASONING_BASE_URL,
                api_key=REASONING_API_KEY,
                streaming=DEEPSEEK_STREAMING,
            )
        llm = _maybe_attach_kimi_azure_fallback(
            llm,
            llm_type=llm_type,
            provider_env=os.getenv("REASONING_PROVIDER"),
            base_url=REASONING_BASE_URL,
            model=REASONING_MODEL,
        )
    elif llm_type == "basic":
        # #region agent log
        # #endregion
        stream_usage = os.getenv("BIOINFO_STREAM_USAGE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        provider = _infer_provider(os.getenv("BASIC_PROVIDER"), BASIC_BASE_URL)
        # #region agent log
        # #endregion
        if provider == "openai":
            llm = create_openai_llm(
                model=BASIC_MODEL,
                base_url=BASIC_BASE_URL,
                api_key=BASIC_API_KEY,
                streaming=DEEPSEEK_STREAMING,
                stream_usage=stream_usage if DEEPSEEK_STREAMING else None,
                kimi_thinking=kimi_thinking_pref,
            )
        elif provider == "azure":
            llm = create_azure_llm(
                model=BASIC_MODEL,
                base_url=BASIC_BASE_URL,
                api_key=BASIC_API_KEY,
                streaming=DEEPSEEK_STREAMING,
            )
        else:
            llm = create_deepseek_llm(
                model=BASIC_MODEL,
                base_url=BASIC_BASE_URL,
                api_key=BASIC_API_KEY,
                streaming=DEEPSEEK_STREAMING,
            )
        llm = _maybe_attach_kimi_azure_fallback(
            llm,
            llm_type=llm_type,
            provider_env=os.getenv("BASIC_PROVIDER"),
            base_url=BASIC_BASE_URL,
            model=BASIC_MODEL,
        )
    elif llm_type == "vision":

        llm = create_openai_llm(
            model=VL_MODEL,
            base_url=VL_BASE_URL,
            api_key=VL_API_KEY,
            kimi_thinking=kimi_thinking_pref,
        )
    elif llm_type == "coding":
        provider = _infer_provider(os.getenv("CODING_PROVIDER"), CODING_BASE_URL)
        if provider == "azure":
            llm = create_azure_llm(
                model=CODING_MODEL,
                base_url=CODING_BASE_URL,
                api_key=CODING_API_KEY,
                streaming=DEEPSEEK_STREAMING,
            )
        else:
            # Typically coding models on Azure are OpenAI compatible or fallback
            llm = create_openai_llm(
                model=CODING_MODEL,
                base_url=CODING_BASE_URL,
                api_key=CODING_API_KEY,
                streaming=DEEPSEEK_STREAMING,
                kimi_thinking=kimi_thinking_pref,
            )
        llm = _maybe_attach_kimi_azure_fallback(
            llm,
            llm_type=llm_type,
            provider_env=os.getenv("CODING_PROVIDER"),
            base_url=CODING_BASE_URL,
            model=CODING_MODEL,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    _llm_cache[llm_type] = llm
    return llm


def get_compression_llm() -> Any:
    """Get dedicated LLM instance for context compression.

    Priority:
    1. CONTEXT_COMPRESSION_* (explicit override)
    2. DashScope defaults (if DASHSCOPE_API_KEY exists): qwen-flash
    3. BASIC_* fallback
    """
    global _compression_llm_cache
    if _compression_llm_cache is not None:
        return _compression_llm_cache

    stream_usage = os.getenv("BIOINFO_STREAM_USAGE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    dashscope_key = _first_nonempty(os.getenv("DASHSCOPE_API_KEY"))
    dashscope_base = _first_nonempty(
        os.getenv("DASHSCOPE_API_BASE"),
        os.getenv("DASHSCOPE_BASE_URL"),
        _DASHSCOPE_COMPAT_BASE_URL,
    )

    model = _first_nonempty(os.getenv("CONTEXT_COMPRESSION_MODEL"))
    base_url = _first_nonempty(os.getenv("CONTEXT_COMPRESSION_BASE_URL"))
    api_key = _first_nonempty(os.getenv("CONTEXT_COMPRESSION_API_KEY"))
    provider_hint = _first_nonempty(os.getenv("CONTEXT_COMPRESSION_PROVIDER"))

    if not model:
        model = "qwen-flash" if dashscope_key else BASIC_MODEL
    if not base_url:
        model_lower = model.lower()
        if "qwen" in model_lower or (dashscope_key and model == "qwen-flash"):
            base_url = dashscope_base
        else:
            base_url = BASIC_BASE_URL
    if not api_key:
        if (base_url or "").find("dashscope.aliyuncs.com") >= 0:
            api_key = _first_nonempty(dashscope_key, BASIC_API_KEY)
        else:
            api_key = _first_nonempty(BASIC_API_KEY, dashscope_key)
    if not provider_hint:
        provider_hint = (
            "openai"
            if (base_url or "").find("dashscope.aliyuncs.com") >= 0
            else os.getenv("BASIC_PROVIDER")
        )

    provider = _infer_provider(provider_hint, base_url)
    kimi_thinking_pref = _resolve_kimi_thinking_mode("basic")
    qwen_thinking_enabled = _is_truthy_env(
        os.getenv("CONTEXT_COMPRESSION_ENABLE_THINKING")
    )
    extra_body = {"enable_thinking": True} if qwen_thinking_enabled else None

    if provider == "openai":
        kwargs = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "streaming": DEEPSEEK_STREAMING,
            "stream_usage": stream_usage if DEEPSEEK_STREAMING else None,
            "kimi_thinking": kimi_thinking_pref,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        llm = create_openai_llm(**kwargs)
    elif provider == "azure":
        llm = create_azure_llm(
            model=model,
            base_url=base_url,
            api_key=api_key,
            streaming=DEEPSEEK_STREAMING,
        )
    else:
        llm = create_deepseek_llm(
            model=model,
            base_url=base_url,
            api_key=api_key,
            streaming=DEEPSEEK_STREAMING,
        )

    _compression_llm_cache = llm
    return llm


if __name__ == "__main__":
    # stream = reasoning_llm.stream("what is mcp?")
    # full_response = ""
    # for chunk in stream:
    #     full_response += chunk.content
    # print(full_response)

    # Example run (requires env vars configured)
    vl_llm = get_llm_by_type("vision")
    basic_result = vl_llm.invoke("你有视觉理解能力吗")
    print("vl_llm 返回结果：", basic_result)
    print(f"vl_llm 类型是: {type(vl_llm)}")
