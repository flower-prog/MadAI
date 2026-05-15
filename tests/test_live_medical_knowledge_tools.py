from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.tools.medical_knowledge_tools import create_live_medical_knowledge_retriever
from agent.workflow import _with_optional_live_medical_knowledge


class LiveMedicalKnowledgeToolTests(unittest.TestCase):
    def test_workflow_registry_helper_defaults_to_disabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            registry = _with_optional_live_medical_knowledge({"trial_retriever": object()})

        self.assertNotIn("medical_knowledge_retriever", registry)

    def test_workflow_registry_helper_can_enable_retriever(self) -> None:
        with patch("agent.workflow.create_live_medical_knowledge_retriever") as factory:
            factory.return_value = "retriever"
            registry = _with_optional_live_medical_knowledge({}, enabled=True)

        self.assertEqual(registry["medical_knowledge_retriever"], "retriever")

    def test_create_live_retriever_respects_source_limit(self) -> None:
        retriever = create_live_medical_knowledge_retriever(max_sources=1)

        self.assertEqual(len(retriever.sources), 1)


if __name__ == "__main__":
    unittest.main()
