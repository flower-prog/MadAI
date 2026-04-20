from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.retrieval.qdrant import default_trial_qdrant_storage_path
from agent.retrieval.qdrant.trial_chunk_sync import (
    build_and_sync_trial_chunk_kb_to_qdrant,
    main,
    sync_trial_chunk_kb_to_qdrant,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_sample_trial_kb(output_dir: Path) -> None:
    _write_jsonl(
        output_dir / "trial_record.jsonl",
        [
            {
                "nct_id": "NCTSYNC001",
                "display_title": "Sunitinib Trial",
                "brief_title": "Sunitinib Trial",
                "overall_status": "Recruiting",
                "phase": "Phase 2",
                "study_type": "Interventional",
                "primary_purpose": "Treatment",
                "conditions": ["Renal Cell Carcinoma"],
                "interventions": ["Sunitinib"],
            }
        ],
    )
    _write_jsonl(
        output_dir / "trial_chunk.jsonl",
        [
            {
                "chunk_id": "NCTSYNC001::overview::0",
                "nct_id": "NCTSYNC001",
                "chunk_type": "overview",
                "sequence": 0,
                "text": "Sunitinib trial for renal cell carcinoma",
                "embedding_text": "Sunitinib trial for renal cell carcinoma",
                "source_fields": ["conditions", "interventions"],
                "token_estimate": 8,
                "rank_weight": 1.0,
            }
        ],
    )


class _FakeManager:
    last_init: dict[str, object] | None = None

    def __init__(
        self,
        catalog,
        *,
        collection_name: str,
        client=None,
        models=None,
        url=None,
        api_key=None,
        path=None,
        embedding_function=None,
    ) -> None:
        del client, models, embedding_function
        self.catalog = catalog
        self.collection_name = str(collection_name)
        self.url = url
        self.api_key = api_key
        self.path = path
        type(self).last_init = {
            "catalog_size": len(self.catalog.documents()),
            "collection_name": self.collection_name,
            "url": self.url,
            "path": self.path,
            "has_api_key": bool(self.api_key),
        }

    def sync_catalog(self, *, recreate: bool = False, batch_size: int = 64) -> dict[str, object]:
        return {
            "collection_name": self.collection_name,
            "point_count": len(self.catalog.documents()),
            "chunk_count": len(self.catalog.documents()),
            "batch_size": int(batch_size),
            "recreated": bool(recreate),
        }


class TrialQdrantSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = PROJECT_ROOT / ".tmp_test_artifacts" / f"trial-qdrant-sync-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_sync_trial_chunk_kb_to_qdrant_uses_existing_jsonl_catalog(self) -> None:
        kb_root = self.temp_root / "kb"
        _write_sample_trial_kb(kb_root)

        with patch("agent.retrieval.qdrant.trial_chunk_sync.QdrantTrialChunkIndexManager", _FakeManager):
            bundle = sync_trial_chunk_kb_to_qdrant(
                output_dir=kb_root,
                batch_size=11,
            )

        self.assertEqual(bundle["output_dir"], str(kb_root.resolve()))
        self.assertEqual(bundle["qdrant_sync"]["point_count"], 1)
        self.assertEqual(bundle["qdrant_sync"]["batch_size"], 11)
        self.assertEqual(
            bundle["qdrant_config"]["path"],
            str(default_trial_qdrant_storage_path().resolve()),
        )
        self.assertEqual(_FakeManager.last_init["catalog_size"], 1)
        self.assertEqual(_FakeManager.last_init["collection_name"], bundle["qdrant_sync"]["collection_name"])

    def test_build_and_sync_trial_chunk_kb_to_qdrant_runs_build_first(self) -> None:
        build_root = self.temp_root / "built-kb"

        def _fake_build_trial_vector_kb(**kwargs):
            requested_output_dir = Path(kwargs["output_dir"]).expanduser().resolve()
            _write_sample_trial_kb(requested_output_dir)
            return {
                "output_dir": str(requested_output_dir),
                "trial_record_count": 1,
                "trial_chunk_count": 1,
            }

        with (
            patch("agent.retrieval.qdrant.trial_chunk_sync.build_trial_vector_kb", side_effect=_fake_build_trial_vector_kb),
            patch("agent.retrieval.qdrant.trial_chunk_sync.QdrantTrialChunkIndexManager", _FakeManager),
        ):
            bundle = build_and_sync_trial_chunk_kb_to_qdrant(
                input_paths=[self.temp_root],
                output_dir=build_root,
                recreate=True,
                batch_size=7,
            )

        self.assertEqual(bundle["build_manifest"]["trial_record_count"], 1)
        self.assertEqual(bundle["qdrant_sync"]["point_count"], 1)
        self.assertEqual(bundle["qdrant_sync"]["batch_size"], 7)
        self.assertTrue(bundle["qdrant_sync"]["recreated"])
        self.assertEqual(bundle["output_dir"], str(build_root.resolve()))

    def test_cli_main_supports_skip_build_mode(self) -> None:
        kb_root = self.temp_root / "kb-cli"
        _write_sample_trial_kb(kb_root)
        stdout = io.StringIO()

        with (
            patch("agent.retrieval.qdrant.trial_chunk_sync.QdrantTrialChunkIndexManager", _FakeManager),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--skip-build",
                    "--output-dir",
                    str(kb_root),
                    "--batch-size",
                    "5",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["output_dir"], str(kb_root.resolve()))
        self.assertEqual(payload["qdrant_sync"]["point_count"], 1)
        self.assertEqual(payload["qdrant_sync"]["batch_size"], 5)


if __name__ == "__main__":
    unittest.main()
