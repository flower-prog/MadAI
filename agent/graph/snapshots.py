from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Literal

from .types import GraphState, ensure_state, state_to_dict


SnapshotPhase = Literal["input", "node", "final"]
_SNAPSHOT_SCHEMA_VERSION = 1
_THREAD_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_thread_id(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "default-thread"
    sanitized = _THREAD_ID_SANITIZER.sub("_", text).strip("._")
    return sanitized or "default-thread"


def _normalize_loaded_state_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    state_payload = dict(raw_payload)
    tool_registry = state_payload.get("tool_registry")
    if isinstance(tool_registry, dict) and set(tool_registry.keys()) == {"registered_tools"}:
        state_payload["tool_registry"] = {}
    return state_payload


def _read_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot file must contain a JSON object: {path}")
    return payload


@dataclass(slots=True)
class LoadedStateSnapshot:
    path: Path
    state: GraphState
    thread_id: str | None = None
    sequence: int | None = None
    phase: str = ""
    agent_name: str | None = None
    timestamp: str | None = None


def load_state_snapshot(path_value: str | Path) -> LoadedStateSnapshot:
    path = Path(path_value).expanduser().resolve()
    payload = _read_json_file(path)

    if isinstance(payload.get("state"), dict):
        state_payload = dict(payload["state"])
    else:
        state_payload = payload

    normalized_state_payload = _normalize_loaded_state_payload(state_payload)
    state = ensure_state(normalized_state_payload)

    raw_thread_id = str(payload.get("thread_id") or "").strip() or None
    raw_phase = str(payload.get("phase") or "").strip()
    raw_agent_name = str(payload.get("agent_name") or "").strip() or None
    raw_timestamp = str(payload.get("timestamp") or "").strip() or None

    raw_sequence = payload.get("sequence")
    sequence: int | None = None
    if raw_sequence not in {None, ""}:
        try:
            sequence = int(raw_sequence)
        except (TypeError, ValueError):
            sequence = None

    return LoadedStateSnapshot(
        path=path,
        state=state,
        thread_id=raw_thread_id,
        sequence=sequence,
        phase=raw_phase,
        agent_name=raw_agent_name,
        timestamp=raw_timestamp,
    )


def prepare_state_for_resume(
    snapshot: LoadedStateSnapshot,
    *,
    mode: Literal["continue", "restart"] = "restart",
) -> GraphState:
    resumed_state = deepcopy(snapshot.state)

    if mode == "continue":
        if resumed_state.next_agent is None:
            resumed_state.next_agent = "orchestrator"
        return resumed_state

    resumed_state.status = "pending"
    resumed_state.next_agent = "orchestrator"
    resumed_state.review_passed = False
    resumed_state.abandoned = False
    resumed_state.reporter_attempts = 0
    resumed_state.errors = []
    resumed_state.safety_issues = []
    resumed_state.calculator_matches = []
    resumed_state.calculation_tasks = []
    resumed_state.calculation_results = []
    resumed_state.treatment_recommendations = []
    resumed_state.protocol_recommendations = []
    resumed_state.assessment_bundle = {}
    resumed_state.calculation_bundle = {}
    resumed_state.trial_retrieval_bundle = {}
    resumed_state.treatment_bundle = {}
    resumed_state.reporter_result = {}
    resumed_state.review_report = {}
    resumed_state.clinical_answer = []
    resumed_state.final_output = {}

    for step in resumed_state.plan:
        step.status = "pending"
        step.result = None

    return resumed_state


class FileSnapshotStore:
    def __init__(self, root_dir: str | Path, *, thread_id: str) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.thread_id = _sanitize_thread_id(thread_id)
        self.thread_dir = self.root_dir / self.thread_id
        self.thread_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.thread_dir / "manifest.json"
        self.latest_path = self.thread_dir / "latest.json"
        self._sequence = self._load_next_sequence()

    def _load_next_sequence(self) -> int:
        if not self.manifest_path.exists():
            return 0

        try:
            manifest = _read_json_file(self.manifest_path)
        except Exception:
            return 0

        sequences: list[int] = []
        for item in list(manifest.get("entries") or []):
            if not isinstance(item, dict):
                continue
            raw_sequence = item.get("sequence")
            try:
                sequences.append(int(raw_sequence))
            except (TypeError, ValueError):
                continue
        return (max(sequences) + 1) if sequences else 0

    def record(
        self,
        state: GraphState | dict[str, Any],
        *,
        phase: SnapshotPhase,
        agent_name: str | None = None,
    ) -> Path:
        graph_state = ensure_state(state)
        state_payload = state_to_dict(graph_state)
        filename = self._build_filename(phase=phase, agent_name=agent_name)
        snapshot_path = self.thread_dir / filename

        envelope = {
            "snapshot_version": _SNAPSHOT_SCHEMA_VERSION,
            "thread_id": self.thread_id,
            "sequence": self._sequence,
            "phase": phase,
            "agent_name": str(agent_name or "").strip() or None,
            "timestamp": _utc_now_iso(),
            "state": state_payload,
        }
        snapshot_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        self.latest_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_manifest_entry(envelope=envelope, filename=filename)
        self._sequence += 1
        return snapshot_path

    def _build_filename(self, *, phase: SnapshotPhase, agent_name: str | None) -> str:
        parts = [f"{self._sequence:04d}", phase]
        normalized_agent_name = str(agent_name or "").strip()
        if normalized_agent_name:
            parts.append(_sanitize_thread_id(normalized_agent_name))
        return "_".join(parts) + ".json"

    def _write_manifest_entry(self, *, envelope: dict[str, Any], filename: str) -> None:
        manifest: dict[str, Any]
        if self.manifest_path.exists():
            try:
                manifest = _read_json_file(self.manifest_path)
            except Exception:
                manifest = {}
        else:
            manifest = {}

        entries = [dict(item) for item in list(manifest.get("entries") or []) if isinstance(item, dict)]
        entries.append(
            {
                "sequence": envelope.get("sequence"),
                "phase": envelope.get("phase"),
                "agent_name": envelope.get("agent_name"),
                "timestamp": envelope.get("timestamp"),
                "path": filename,
            }
        )
        manifest_payload = {
            "snapshot_version": _SNAPSHOT_SCHEMA_VERSION,
            "thread_id": self.thread_id,
            "latest": filename,
            "entries": entries,
        }
        self.manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
