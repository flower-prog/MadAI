from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in _TRUTHY


def _load_dotenv_defaults(dotenv_path: str) -> dict[str, str]:
    defaults: dict[str, str] = {}
    try:
        if not os.path.exists(dotenv_path):
            return defaults
        with open(dotenv_path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                defaults[key] = value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return defaults


def _set_if_unset_from_dotenv(dotenv_map: dict[str, str], key: str) -> None:
    if str(os.getenv(key, "")).strip():
        return
    val = str(dotenv_map.get(key, "")).strip()
    if val:
        os.environ[key] = val


def _run(
    cmd: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _is_local_searxng(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


def _searxng_health_ok(base_url: str, search_path: str, timeout: int = 8) -> bool:
    path = search_path if search_path.startswith("/") else f"/{search_path}"
    url = f"{base_url.rstrip('/')}{path}?q=BRCA1&format=json"
    req = Request(url, headers={"User-Agent": "bioagent-startup-check"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            return status < 400
    except (URLError, TimeoutError, ValueError, OSError):
        return False


def _ensure_sandbox_image(
    requested_image: str,
    fallback_image: str,
    default_local_image: str,
    *,
    strict: bool,
    print_fn=print,
) -> None:
    if not requested_image:
        return
    if shutil.which("docker") is None:
        msg = "docker command not found while checking SANDBOX_IMAGE"
        if strict:
            raise RuntimeError(msg)
        print_fn(f"[WARN] {msg}")
        return

    try:
        local = _run(["docker", "image", "inspect", requested_image], timeout=60)
    except Exception as exc:
        msg = f"failed to inspect local sandbox image '{requested_image}': {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        print_fn(f"[WARN] {msg}")
        return
    if local.returncode == 0:
        print_fn(f"Startup check: local sandbox image exists: {requested_image}")
        return

    print_fn(f"Startup check: local sandbox image missing: {requested_image}")

    if "/" in requested_image:
        pull = _run(["docker", "pull", requested_image], timeout=1200)
        if pull.returncode == 0:
            print_fn(f"Startup check: pulled sandbox image: {requested_image}")
            return
        msg = (pull.stdout or pull.stderr or "").strip()
        if strict:
            raise RuntimeError(f"failed to pull sandbox image '{requested_image}': {msg}")
        print_fn(f"[WARN] failed to pull sandbox image '{requested_image}': {msg}")
        return

    if (
        requested_image == default_local_image
        and fallback_image
        and fallback_image != requested_image
    ):
        pull_fb = _run(["docker", "pull", fallback_image], timeout=1200)
        if pull_fb.returncode != 0:
            msg = (pull_fb.stdout or pull_fb.stderr or "").strip()
            if strict:
                raise RuntimeError(
                    f"failed to pull fallback sandbox image '{fallback_image}': {msg}"
                )
            print_fn(f"[WARN] failed to pull fallback sandbox image '{fallback_image}': {msg}")
            return
        tag = _run(["docker", "tag", fallback_image, requested_image], timeout=120)
        if tag.returncode == 0:
            print_fn(
                f"Startup check: tagged fallback image {fallback_image} -> {requested_image}"
            )
            return
        msg = (tag.stdout or tag.stderr or "").strip()
        if strict:
            raise RuntimeError(
                f"pulled fallback image but failed to tag '{fallback_image}' to '{requested_image}': {msg}"
            )
        print_fn(f"[WARN] failed to tag fallback image '{fallback_image}' -> '{requested_image}': {msg}")
        return

    msg = f"sandbox image '{requested_image}' missing and no pull strategy succeeded"
    if strict:
        raise RuntimeError(msg)
    print_fn(f"[WARN] {msg}")


def _ensure_crawl4ai_available(*, strict: bool, print_fn=print) -> None:
    if importlib.util.find_spec("crawl4ai") is not None:
        return
    if not _env_flag("BIXBENCH_AUTO_INSTALL_CRAWL4AI", True):
        msg = "crawl4ai missing and auto install disabled (BIXBENCH_AUTO_INSTALL_CRAWL4AI=0)"
        if strict:
            raise RuntimeError(msg)
        print_fn(f"[WARN] {msg}")
        return

    print_fn("Startup check: crawl4ai missing, attempting installation...")
    proc = _run([sys.executable, "-m", "pip", "install", "crawl4ai"], timeout=1800)
    if proc.returncode != 0:
        msg = (proc.stdout or proc.stderr or "").strip()
        if strict:
            raise RuntimeError(f"failed to install crawl4ai: {msg}")
        print_fn(f"[WARN] failed to install crawl4ai: {msg}")
        return
    print_fn("Startup check: crawl4ai installed successfully")


def _docker_compose_up_searxng(project_root: str) -> tuple[bool, str]:
    compose_file = os.path.join(project_root, "docker", "searxng", "docker-compose.yml")
    if not os.path.exists(compose_file):
        return False, f"compose file not found: {compose_file}"

    candidates = [
        ["docker", "compose", "-f", compose_file, "up", "-d"],
        ["docker-compose", "-f", compose_file, "up", "-d"],
    ]
    last_error = "docker compose command unavailable"
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            proc = _run(cmd, timeout=300)
        except Exception as exc:
            last_error = str(exc)
            continue
        if proc.returncode == 0:
            return True, ""
        last_error = (proc.stdout or proc.stderr or "").strip() or f"exit={proc.returncode}"
    return False, last_error


def _ensure_searxng_available(project_root: str, *, strict: bool, print_fn=print) -> None:
    base_url = str(os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080")).strip()
    search_path = str(os.getenv("SEARXNG_SEARCH_PATH", "/search")).strip() or "/search"
    if not search_path.startswith("/"):
        search_path = f"/{search_path}"
        os.environ["SEARXNG_SEARCH_PATH"] = search_path

    if _searxng_health_ok(base_url, search_path):
        print_fn(f"Startup check: SearXNG reachable at {base_url.rstrip('/')}{search_path}")
        return

    if _is_local_searxng(base_url):
        ok, err = _docker_compose_up_searxng(project_root)
        if ok:
            retries = int(str(os.getenv("BIXBENCH_SEARXNG_HEALTH_RETRIES", "45")).strip() or 45)
            sleep_s = float(str(os.getenv("BIXBENCH_SEARXNG_HEALTH_SLEEP_SECONDS", "2")).strip() or 2.0)
            for _ in range(max(1, retries)):
                if _searxng_health_ok(base_url, search_path):
                    print_fn(f"Startup check: SearXNG started and healthy at {base_url.rstrip('/')}{search_path}")
                    return
                time.sleep(max(0.2, sleep_s))
            msg = f"SearXNG health check timed out at {base_url.rstrip('/')}{search_path}"
        else:
            msg = f"failed to start local SearXNG via docker compose: {err}"
    else:
        msg = f"SearXNG unreachable at remote endpoint {base_url.rstrip('/')}{search_path}"

    if strict:
        raise RuntimeError(msg)
    print_fn(f"[WARN] {msg}")


def run_startup_checks(
    project_root: str,
    *,
    strict: bool,
    print_fn=print,
) -> None:
    """Run start_backend-like startup checks for benchmark runners."""
    if not strict:
        return

    root = str(Path(project_root).resolve())
    dotenv_values = _load_dotenv_defaults(os.path.join(root, ".env"))

    # Fill open-source crawler/search vars from .env when not exported.
    for key in (
        "SEARXNG_BASE_URL",
        "SEARXNG_SEARCH_PATH",
        "JINA_API_KEY",
        "JINA_TIMEOUT",
        "CRAWLER_BACKENDS",
        "CRAWL4AI_HTTP_ONLY",
        "CRAWL4AI_VERIFY_SSL",
        "CRAWL4AI_ALLOW_INSECURE_RETRY",
        "SANDBOX_IMAGE",
        "SANDBOX_IMAGE_FALLBACK",
        "DEFAULT_SANDBOX_IMAGE_LOCAL",
        "REQUIRE_DOCKER",
    ):
        _set_if_unset_from_dotenv(dotenv_values, key)

    os.environ.setdefault("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
    os.environ.setdefault("SEARXNG_SEARCH_PATH", "/search")
    os.environ.setdefault("JINA_TIMEOUT", "30")
    os.environ.setdefault("CRAWLER_BACKENDS", "jina,crawl4ai,http")
    os.environ.setdefault("CRAWL4AI_HTTP_ONLY", "true")
    os.environ.setdefault("CRAWL4AI_VERIFY_SSL", "true")
    os.environ.setdefault("CRAWL4AI_ALLOW_INSECURE_RETRY", "true")
    os.environ.setdefault("EMBEDDING_BATCH_SIZE", "10")
    os.environ.setdefault("REQUIRE_DOCKER", "1")

    require_docker = _env_flag("REQUIRE_DOCKER", True)
    docker_exists = shutil.which("docker") is not None
    if require_docker and not docker_exists:
        raise RuntimeError(
            "strict startup check failed: docker is required (REQUIRE_DOCKER=1) but not found"
        )
    if docker_exists:
        _ensure_sandbox_image(
            requested_image=str(os.getenv("SANDBOX_IMAGE", "bioagent-sandbox:py312-r")).strip(),
            fallback_image=str(
                os.getenv("SANDBOX_IMAGE_FALLBACK", "ghcr.io/hszd-team/bioagent-sandbox:py312-r")
            ).strip(),
            default_local_image=str(os.getenv("DEFAULT_SANDBOX_IMAGE_LOCAL", "bioagent-sandbox:py312-r")).strip(),
            strict=strict,
            print_fn=print_fn,
        )

    _ensure_crawl4ai_available(strict=strict, print_fn=print_fn)
    _ensure_searxng_available(root, strict=strict, print_fn=print_fn)
    print_fn("Startup check: all strict checks passed")
