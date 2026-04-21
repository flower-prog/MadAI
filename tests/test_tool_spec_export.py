from __future__ import annotations

import unittest

from agent.graph.nodes import _default_tool_specs


class ToolSpecExportTests(unittest.TestCase):
    def test_default_tool_specs_include_decorated_clinical_tools(self) -> None:
        specs = _default_tool_specs()
        spec_by_name = {spec.name: spec for spec in specs}

        self.assertIn("structured_bm25_retriever", spec_by_name)
        self.assertIn("structured_vector_retriever", spec_by_name)
        self.assertIn("trial_coarse_retriever", spec_by_name)
        self.assertIn("trial_candidate_retriever", spec_by_name)
        self.assertIn("riskcalc_coarse_retriever", spec_by_name)
        self.assertIn("riskcalc_computation_retriever", spec_by_name)
        self.assertIn("riskcalc_executor", spec_by_name)
        self.assertEqual(
            spec_by_name["structured_bm25_retriever"].input_schema["structured_case"]["raw_text"],
            "str",
        )
        self.assertEqual(
            spec_by_name["structured_bm25_retriever"].input_schema["candidate_ids"],
            "list[str] | set[str] | None",
        )
        self.assertEqual(
            spec_by_name["structured_vector_retriever"].input_schema["candidate_ids"],
            "list[str] | set[str] | None",
        )
        self.assertEqual(
            spec_by_name["trial_coarse_retriever"].input_schema["candidate_ids"],
            "list[str] | set[str] | None",
        )
        self.assertEqual(
            spec_by_name["trial_candidate_retriever"].input_schema["coarse_top_k"],
            "int",
        )
        self.assertEqual(
            spec_by_name["riskcalc_coarse_retriever"].input_schema["structured_case"]["raw_text"],
            "str",
        )
        self.assertEqual(
            spec_by_name["riskcalc_coarse_retriever"].input_schema["candidate_pmids"],
            "set[str] | None",
        )
        self.assertEqual(
            spec_by_name["riskcalc_computation_retriever"].input_schema["candidate_pmids"],
            "list[str] | set[str]",
        )
        self.assertEqual(
            spec_by_name["riskcalc_executor"].input_schema["clinical_text"],
            "str",
        )
        self.assertEqual(
            spec_by_name["riskcalc_executor"].input_schema["mode"],
            "str",
        )


if __name__ == "__main__":
    unittest.main()
