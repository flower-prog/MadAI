from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def iter_corpus_candidate_pairs(root: str | Path) -> tuple[tuple[Path, Path], ...]:
    root_path = Path(root).expanduser().resolve()
    clinical_tool_learning_root = root_path.parent / "Clinical-Tool-Learning"
    return (
        (
            root_path / "数据" / "riskcalcs.json",
            root_path / "数据" / "pmid2info.json",
        ),
        (
            root_path / "data" / "riskcalcs.json",
            root_path / "data" / "pmid2info.json",
        ),
        (
            clinical_tool_learning_root / "mimic_evaluation" / "tools" / "riskcalcs.json",
            clinical_tool_learning_root / "mimic_evaluation" / "dataset" / "pmid2info.json",
        ),
        (
            clinical_tool_learning_root / "riskqa_evaluation" / "tools" / "riskcalcs.json",
            clinical_tool_learning_root / "riskqa_evaluation" / "dataset" / "pmid2info.json",
        ),
    )


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_default_corpus_paths(root: str | Path) -> tuple[Path | None, Path | None]:
    candidates = iter_corpus_candidate_pairs(root)
    return (
        _first_existing(riskcalcs_path for riskcalcs_path, _ in candidates),
        _first_existing(pmid_metadata_path for _, pmid_metadata_path in candidates),
    )


def discover_complete_corpus_pair(root: str | Path) -> tuple[Path, Path]:
    for riskcalcs_path, pmid_metadata_path in iter_corpus_candidate_pairs(root):
        if riskcalcs_path.exists() and pmid_metadata_path.exists():
            return riskcalcs_path, pmid_metadata_path
    raise FileNotFoundError(
        "Unable to find RiskCalcs corpus files. "
        "Pass `riskcalcs_path` and `pmid_metadata_path` explicitly."
    )
