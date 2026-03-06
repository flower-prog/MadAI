import os
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def _resolve_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return None


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except Exception:
        return False


def snapshot_capsule_contents(
    case_dir: Optional[str],
    capsule_dir: Optional[str],
    *,
    max_entries: int = 200,
    manifest_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Persist a lightweight manifest of the capsule directory so downstream
    agents have a quick reference without repeatedly traversing the filesystem.

    Returns the manifest path if generation succeeded, otherwise None.
    """
    if not case_dir or not capsule_dir:
        return None

    case_path = _resolve_dir(case_dir)
    capsule_path = _resolve_dir(capsule_dir)
    if case_path is None or capsule_path is None:
        return None
    if not case_path.exists() or not case_path.is_dir():
        return None
    if not capsule_path.exists() or not capsule_path.is_dir():
        return None

    # Hard boundary: only snapshot files from within the declared case directory.
    if not _is_within(capsule_path, case_path):
        logger.warning(
            "Skip capsule manifest because capsule_dir is outside case_dir: case=%s capsule=%s",
            case_path,
            capsule_path,
        )
        return None

    manifest_root = _resolve_dir(manifest_dir) if manifest_dir else case_path
    if manifest_root is None:
        return None
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "capsule_manifest.txt"
    lines_written = 0
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            manifest.write(f"Case: {case_path}\n")
            manifest.write(f"Capsule: {capsule_path}\n")
            manifest.write("Relative path listing (truncated):\n")
            for root, _, files in os.walk(capsule_path):
                rel_root = os.path.relpath(root, capsule_path)
                if rel_root == ".":
                    rel_root = ""
                for file_name in sorted(files):
                    rel_path = Path(rel_root) / file_name if rel_root else file_name
                    manifest.write(f"- {rel_path}\n")
                    lines_written += 1
                    if lines_written >= max_entries:
                        manifest.write(
                            f"... listing truncated after {max_entries} entries ...\n"
                        )
                        return str(manifest_path)
        return str(manifest_path)
    except OSError:
        return None
