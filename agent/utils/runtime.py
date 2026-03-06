import functools
import logging
from typing import Optional


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def detect_sandbox_availability() -> bool:
    """
    Best-effort detection of whether the Docker sandbox image can be used.

    We try to import the docker SDK and ping the daemon once. Any failure
    (missing package, daemon unavailable, permission denied, etc.) is treated
    as "not available" so that the workflow can gracefully fall back to the
    local .venv execution path.
    """
    try:
        import docker  # type: ignore

        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception as exc:  # pragma: no cover - environment specific
            logger.warning("Docker sandbox unavailable: %s", exc)
            return False
    except ModuleNotFoundError:
        logger.info("docker SDK not installed; sandbox disabled.")
        return False


def format_env_note(sandbox_available: bool, manifest_path: Optional[str]) -> list[str]:
    """
    Build optional system messages describing environment hints that should be
    injected ahead of the user's query. This avoids silent failures when
    sandboxing is disabled or capsule metadata has been pre-computed.
    """
    notes: list[str] = []
    if not sandbox_available:
        notes.append(
            "System notice: Docker sandbox is unavailable; all commands will run "
            "directly on the host virtual environment. Please avoid assuming an "
            "isolated container."
        )
    if manifest_path:
        notes.append(
            f"System notice: Capsule file listing saved to {manifest_path}. "
            "Refer to this manifest before concluding that data is missing."
        )
    return notes
