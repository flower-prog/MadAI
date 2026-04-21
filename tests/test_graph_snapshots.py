from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph.builder import SimpleAgentGraph
from agent.graph.snapshots import load_state_snapshot


class GraphSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"graph-snapshots-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_simple_graph_records_input_node_and_final_snapshots(self) -> None:
        graph = SimpleAgentGraph()

        def _single_step_node(state):
            state.status = "completed"
            state.next_agent = "FINISH"
            return state

        graph.node_map = {"orchestrator": _single_step_node}
        result = graph.invoke(
            {"request": "snapshot demo"},
            config={
                "configurable": {
                    "thread_id": "snapshot-thread-001",
                    "snapshot_dir": str(self.temp_root),
                }
            },
        )

        self.assertEqual(result["status"], "completed")

        thread_dir = self.temp_root / "snapshot-thread-001"
        manifest = json.loads((thread_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["phase"] for entry in manifest["entries"]], ["input", "node", "final"])
        self.assertEqual(manifest["latest"], "0002_final.json")

        node_snapshot = json.loads((thread_dir / "0001_node_orchestrator.json").read_text(encoding="utf-8"))
        self.assertEqual(node_snapshot["agent_name"], "orchestrator")
        self.assertEqual(node_snapshot["state"]["status"], "completed")

        latest_snapshot = json.loads((thread_dir / "latest.json").read_text(encoding="utf-8"))
        self.assertEqual(latest_snapshot["phase"], "final")

    def test_load_state_snapshot_strips_serialized_tool_registry_placeholder(self) -> None:
        snapshot_path = self.temp_root / "single_snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "snapshot_version": 1,
                    "thread_id": "snapshot-thread-002",
                    "phase": "input",
                    "state": {
                        "request": "恢复状态",
                        "tool_registry": {"registered_tools": ["demo_tool"]},
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        loaded_snapshot = load_state_snapshot(snapshot_path)

        self.assertEqual(loaded_snapshot.thread_id, "snapshot-thread-002")
        self.assertEqual(loaded_snapshot.state.request, "恢复状态")
        self.assertEqual(loaded_snapshot.state.tool_registry, {})


if __name__ == "__main__":
    unittest.main()
