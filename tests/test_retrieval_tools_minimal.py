from __future__ import annotations

import contextlib
import json
import unittest
from pathlib import Path

import agent.tools as tools
import agent.tools.retrieval_tools as retrieval_tools
from agent.graph.types import ClinicalToolJob
from agent.tools import collect_tools, create_computation_retrieval_tool, create_retrieval_tool
from agent.tools.retrieval_tools import RiskCalcParameterRetrievalTool


class _FakeDocument:
    def __init__(
        self,
        pmid: str,
        title: str,
        purpose: str,
        eligibility: str,
        retrieval_text: str,
        *,
        example: str = "",
    ) -> None:
        self.pmid = pmid
        self.title = title
        self.purpose = purpose
        self.eligibility = eligibility
        self.retrieval_text = retrieval_text
        self.abstract = ""
        self.example = example

    def to_brief(self) -> dict[str, object]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "purpose": self.purpose,
            "eligibility": self.eligibility,
            "taxonomy": {},
        }


class _FakeCatalog:
    def __init__(self, *, department_payload_root: Path | None = None) -> None:
        self.runtime_cache_key = "fake-catalog"
        self.department_payload_root = department_payload_root
        self._documents = {
            "1": _FakeDocument(
                "1",
                "CHADS2 Stroke Risk Calculator",
                "Estimate stroke risk in atrial fibrillation.",
                "AF patients.",
                "atrial fibrillation hypertension diabetes stroke history heart failure",
                example="Example: AF patient with HTN, DM, prior stroke, and CHF.",
            ),
            "2": _FakeDocument(
                "2",
                "FIB-4 Index for Liver Fibrosis",
                "Estimate liver fibrosis risk.",
                "Liver disease patients.",
                "fibrosis liver disease ast alt platelet age",
                example="Example: chronic liver disease with AST ALT and platelets.",
            ),
        }

    def documents(self):
        return list(self._documents.values())

    def get(self, pmid: str):
        return self._documents[str(pmid)]

    def dense_index_cache_paths(self, *, query_model_name: str, doc_model_name: str):
        return None


class _FakeCatalogWithSourceFiles(_FakeCatalog):
    def __init__(self, riskcalcs_path: Path, *, department_payload_root: Path | None = None) -> None:
        super().__init__(department_payload_root=department_payload_root)
        self.runtime_cache_key = f"fake-catalog-with-source-files:{riskcalcs_path.resolve()}"
        self.source_files = (str(riskcalcs_path),)


class _FakeVectorRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        rows = [
            {
                "pmid": "2",
                "title": "FIB-4 Index for Liver Fibrosis",
                "purpose": "Estimate liver fibrosis risk.",
                "score": 0.92,
            },
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
                "purpose": "Estimate stroke risk in atrial fibrillation.",
                "score": 0.41,
            },
        ]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("pmid")) in candidate_pmids]
        return rows[:top_k]


class _ManyFakeCatalog:
    def __init__(self, count: int = 11) -> None:
        self.runtime_cache_key = f"many-fake-catalog-{count}"
        self._documents = {
            str(index): _FakeDocument(
                str(index),
                f"Calculator {index:03d}",
                f"Estimate risk with calculator {index}.",
                "Eligible patients.",
                f"condition calculator {index}",
                example=f"Example for calculator {index}.",
            )
            for index in range(1, count + 1)
        }

    def documents(self):
        return list(self._documents.values())

    def get(self, pmid: str):
        return self._documents[str(pmid)]

    def dense_index_cache_paths(self, *, query_model_name: str, doc_model_name: str):
        return None


class _TopFiveVectorRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        rows = [
            {
                "pmid": str(index),
                "title": f"Calculator {index:03d}",
                "purpose": f"Estimate risk with calculator {index}.",
                "score": float(20 - index),
            }
            for index in range(6, 12)
        ]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("pmid")) in candidate_pmids]
        return rows[:top_k]


class _RangeVectorRetriever:
    def __init__(self, start: int, stop: int) -> None:
        self.start = int(start)
        self.stop = int(stop)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        del query
        rows = [
            {
                "pmid": str(index),
                "title": f"Calculator {index:03d}",
                "purpose": f"Estimate risk with calculator {index}.",
                "score": float((self.stop + 1) - index),
            }
            for index in range(self.start, self.stop + 1)
        ]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("pmid")) in candidate_pmids]
        return rows[:top_k]


def _af_parameter_payload() -> dict[str, object]:
    return {
        "1": {
            "title": "CHADS2 Stroke Risk Calculator",
            "purpose": "Estimate stroke risk in atrial fibrillation.",
            "eligibility": "AF patients.",
            "parameter_names": [
                "congestive_heart_failure",
                "hypertension",
                "age",
                "diabetes",
                "stroke_history",
            ],
            "parameter_aliases": {
                "congestive_heart_failure": ["congestive heart failure", "heart failure", "chf"],
                "hypertension": ["hypertension", "htn", "high blood pressure"],
                "age": ["age"],
                "diabetes": ["diabetes", "dm"],
                "stroke_history": ["stroke history", "tia", "prior stroke"],
            },
            "parameter_text": (
                "congestive heart failure chf hypertension htn age diabetes dm "
                "stroke history tia prior stroke"
            ),
        },
        "2": {
            "title": "FIB-4 Index for Liver Fibrosis",
            "purpose": "Estimate liver fibrosis risk.",
            "eligibility": "Liver disease patients.",
            "parameter_names": ["age", "ast", "plt", "alt"],
            "parameter_aliases": {
                "age": ["age"],
                "ast": ["ast"],
                "plt": ["plt", "platelet"],
                "alt": ["alt"],
            },
            "parameter_text": "age ast alt platelet plt liver fibrosis",
        },
    }


def _af_riskcalcs_payload() -> dict[str, object]:
    return {
        "1": {
            "title": "CHADS2 Stroke Risk Calculator",
            "purpose": "Estimate stroke risk in atrial fibrillation.",
            "eligibility": "AF patients.",
            "computation": (
                "```python\n"
                "def compute_chads2(hypertension, diabetes, stroke_history):\n"
                "    return int(bool(hypertension)) + int(bool(diabetes)) + int(bool(stroke_history))\n"
                "```"
            ),
        },
        "2": {
            "title": "FIB-4 Index for Liver Fibrosis",
            "purpose": "Estimate liver fibrosis risk.",
            "eligibility": "Liver disease patients.",
            "computation": (
                "```python\n"
                "def compute_fib4(age, ast, alt, platelet):\n"
                "    return age * ast / max(alt * platelet, 1)\n"
                "```"
            ),
        },
    }


class _FakeComputationVectorRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        del query
        rows = [
            {
                "pmid": "2",
                "title": "FIB-4 Index for Liver Fibrosis",
                "score": 0.88,
            },
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
                "score": 0.52,
            },
        ]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("pmid")) in candidate_pmids]
        return rows[:top_k]


class _FiveWayComputationVectorRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_pmids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        del query
        rows = [
            {
                "pmid": str(index),
                "title": f"Calculator {index}",
                "score": float(6 - index),
            }
            for index in range(1, 6)
        ]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("pmid")) in candidate_pmids]
        return rows[:top_k]


class _NoOpLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeEncodedInput(dict):
    def to(self, device: str):
        del device
        return self


class _FakeTensorLike:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        del key
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _FakeModelOutput:
    def __init__(self, value):
        self.last_hidden_state = _FakeTensorLike(value)


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        del texts, kwargs
        return _FakeEncodedInput()


class _FakeQueryEncoder:
    def __call__(self, **kwargs):
        del kwargs
        return _FakeModelOutput([[1.0, 0.0]])


class _FakeTorchModule:
    @staticmethod
    def no_grad():
        return _NoOpLock()


class _FakeDenseIndex:
    def __init__(self) -> None:
        self._scores_by_index = {
            0: 0.99,
            1: 0.98,
            2: 0.70,
            3: 0.60,
            4: 0.50,
        }
        self._vectors_by_index = {
            0: [0.99, 0.0],
            1: [0.98, 0.0],
            2: [0.70, 0.0],
            3: [0.60, 0.0],
            4: [0.50, 0.0],
        }

    def search(self, query_embedding, k: int):
        del query_embedding
        ordered_indices = list(self._scores_by_index.keys())[:k]
        ordered_scores = [self._scores_by_index[index] for index in ordered_indices]
        return [ordered_scores], [ordered_indices]

    def reconstruct(self, index: int):
        return list(self._vectors_by_index[int(index)])


class RetrievalToolsMinimalTests(unittest.TestCase):
    def _build_temp_parameter_path(self, name: str = "riskcalcs_parameter.test.json") -> Path:
        temp_root = Path(__file__).resolve().parent / ".tmp_test_artifacts" / "retrieval_tools"
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root / name

    def _build_temp_riskcalcs_path(self, name: str = "riskcalcs.test.json") -> Path:
        temp_root = Path(__file__).resolve().parent / ".tmp_test_artifacts" / "retrieval_tools"
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root / name

    def _write_parameter_payload(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _write_riskcalcs_payload(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _build_temp_department_root(self, name: str = "department_payloads") -> Path:
        temp_root = Path(__file__).resolve().parent / ".tmp_test_artifacts" / "retrieval_tools" / name
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root

    def _write_department_payload(self, root: Path, department: str, payload: dict[str, object]) -> None:
        department_dir = root / department
        department_dir.mkdir(parents=True, exist_ok=True)
        (department_dir / "riskcalcs.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def _many_parameter_payload(self, count: int = 6) -> dict[str, object]:
        return {
            str(index): {
                "title": f"Calculator {index:03d}",
                "purpose": f"Estimate risk with calculator {index}.",
                "eligibility": "Eligible patients.",
                "parameter_names": [f"feature_{index}"],
                "parameter_aliases": {f"feature_{index}": [f"feature_{index}"]},
                "parameter_text": "condition",
            }
            for index in range(1, count + 1)
        }

    def test_extract_parameter_names_from_prose_only_computation(self) -> None:
        computation = (
            "The risk score is computed based on twelve characteristics. "
            "Each characteristic is assigned a score of 1 if the condition is met, and 0 otherwise. "
            "The characteristics are:\n\n"
            "1. Age > 40\n"
            "2. Male\n"
            "3. Non-white ethnicity\n"
            "4. Oxygen saturations < 93%\n"
            "5. Radiological severity score > 3\n"
            "6. Neutrophil count > 8.0 x 10^9/L\n"
            "7. CRP > 40 mg/L\n"
            "8. Albumin < 34 g/L\n"
            "9. Creatinine > 100 umol/L\n"
            "10. Diabetes mellitus\n"
            "11. Hypertension\n"
            "12. Chronic lung disease\n"
        )

        parameter_names = tools.extract_parameter_names_from_computation(computation)

        self.assertIn("age", parameter_names)
        self.assertIn("sex", parameter_names)
        self.assertIn("non_white_ethnicity", parameter_names)
        self.assertIn("oxygen_saturation", parameter_names)
        self.assertIn("radiological_severity_score", parameter_names)
        self.assertIn("crp", parameter_names)
        self.assertIn("albumin", parameter_names)
        self.assertIn("creatinine", parameter_names)
        self.assertIn("diabetes", parameter_names)
        self.assertIn("hypertension", parameter_names)
        self.assertIn("chronic_lung_disease", parameter_names)

    def test_extract_parameter_names_merges_signature_and_prose_candidates(self) -> None:
        computation = (
            "The score uses four factors:\n\n"
            "1. Age >= 65 years\n"
            "2. Albumin < 34 g/L\n"
            "3. Creatinine > 100 umol/L\n"
            "4. CRP > 40 mg/L\n\n"
            "```python\n"
            "def compute_score(age, creatinine):\n"
            "    return int(age >= 65) + int(creatinine > 100)\n"
            "```\n"
        )

        parameter_names = tools.extract_parameter_names_from_computation(computation)

        self.assertIn("age", parameter_names)
        self.assertIn("creatinine", parameter_names)
        self.assertIn("albumin", parameter_names)
        self.assertIn("crp", parameter_names)

    def test_build_parameter_document_payload_backfills_stale_parameter_names(self) -> None:
        payload = {
            "title": "Example Risk Score",
            "purpose": "Estimate risk for a clinical cohort.",
            "eligibility": "Eligible patients.",
            "parameter_names": ["score"],
            "computation": (
                "The score is computed from the following factors:\n"
                "1. Age >= 65 years\n"
                "2. Albumin < 34 g/L\n"
                "3. Hypertension\n"
            ),
        }

        document = tools.build_parameter_document_payload("example", payload)

        self.assertNotEqual(document["parameter_names"], ["score"])
        self.assertIn("age", document["parameter_names"])
        self.assertIn("albumin", document["parameter_names"])
        self.assertIn("hypertension", document["parameter_names"])

    def test_create_retrieval_tool_no_longer_exposes_legacy_free_text_api(self) -> None:
        catalog = _FakeCatalog()
        tool = create_retrieval_tool(catalog)

        self.assertFalse(hasattr(tool, "retrieve"))
        self.assertTrue(hasattr(tool, "retrieve_from_structured_case"))
        self.assertTrue(hasattr(tool, "structured_tools"))

    def test_agent_tools_facade_exports_consolidated_retrieval_stack(self) -> None:
        expected_exports = {
            "RiskCalcCatalog",
            "RiskCalcComputationRetrievalTool",
            "RiskCalcDocument",
            "KeywordToolRetriever",
            "MedCPTRetriever",
            "HybridRetriever",
            "StructuredRetrievalTool",
            "create_computation_retrieval_tool",
            "create_retriever",
            "RiskCalcRetrievalTool",
            "create_retrieval_tool",
            "create_structured_retrieval_tool",
            "build_case_summary",
            "build_structured_query_text",
            "build_patient_note_queries",
            "generate_risk_hints",
        }
        self.assertTrue(expected_exports.issubset(set(tools.__all__)))
        for export_name in expected_exports:
            self.assertTrue(hasattr(tools, export_name))
        self.assertFalse(hasattr(tools, "normalize_retriever_backend"))
        self.assertFalse(hasattr(tools, "public_retriever_backend_name"))

    def test_build_case_summary_handles_chinese_problem_list(self) -> None:
        summary = tools.build_case_summary(problem_list=["房颤", "高血压", "糖尿病"])

        self.assertTrue(summary)
        self.assertIn("房颤", summary)

    def test_build_case_query_text_keeps_short_raw_text_and_appends_case_summary(self) -> None:
        query_text = tools.build_case_query_text(
            raw_text="68 year old with atrial fibrillation and hypertension.",
            case_summary="房颤合并高血压，需做卒中风险评估。",
            problem_list=["atrial fibrillation"],
            known_facts=["hypertension"],
        )

        self.assertTrue(query_text.startswith("68 year old with atrial fibrillation and hypertension."))
        self.assertIn("case_summary: 房颤合并高血压，需做卒中风险评估。", query_text)
        self.assertIn("problem_list: atrial fibrillation", query_text)

    def test_build_case_query_text_uses_case_summary_when_raw_text_is_too_long(self) -> None:
        long_raw_text = "very long noisy note " * 100
        query_text = tools.build_case_query_text(
            raw_text=long_raw_text,
            case_summary="房颤伴既往卒中史和高血压。",
            problem_list=["atrial fibrillation"],
            known_facts=["prior stroke", "hypertension"],
        )

        self.assertTrue(query_text.startswith("房颤伴既往卒中史和高血压。"))
        self.assertNotIn(long_raw_text.strip(), query_text)
        self.assertNotIn("case_summary: 房颤伴既往卒中史和高血压。", query_text)
        self.assertIn("known_facts: prior stroke ; hypertension", query_text)

    def test_build_patient_note_queries_keeps_one_problem_segment_per_anchor(self) -> None:
        queries = tools.build_patient_note_queries(
            case_summary="房颤合并卒中高危因素",
            risk_hints=[],
            problem_list=[
                "房颤合并既往TIA病史",
                "既往高血压和糖尿病",
                "当前未使用华法林",
            ],
            risk_count=0,
        )

        anchor_queries = [query for query in queries if query.intent == "problem_anchor"]
        self.assertEqual(
            [query.text for query in anchor_queries],
            ["房颤合并既往TIA病史", "既往高血压和糖尿病", "当前未使用华法林"],
        )

    def test_create_retrieval_tool_returns_context_bundle_for_structured_case(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path()
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )

            payload = tool.retrieve_from_structured_case(
                {
                    "raw_text": (
                        "68 year old with atrial fibrillation, hypertension, diabetes, "
                        "prior stroke, and congestive heart failure."
                    ),
                    "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                    "known_facts": ["prior stroke", "heart failure"],
                },
                top_k=2,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertIn("retrieved_tools", payload)
        self.assertIn("candidate_ranking", payload)
        self.assertIn("bm25_top5", payload)
        self.assertIn("vector_top5", payload)
        self.assertIn("bm25_raw_top5", payload)
        self.assertIn("vector_raw_top5", payload)
        self.assertIn("model_context", payload)
        self.assertEqual(payload["retrieved_tools"][0]["pmid"], "1")
        self.assertIn("hypertension", payload["retrieved_tools"][0]["matched_parameter_names"])
        self.assertIn("stroke_history", payload["retrieved_tools"][0]["matched_parameter_names"])
        self.assertNotIn("score", payload["retrieved_tools"][0])
        self.assertNotIn("rank", payload["retrieved_tools"][0])
        self.assertEqual(payload["retrieved_tools"][0]["recommended"], ["bm25"])
        self.assertEqual(payload["candidate_ranking"], payload["retrieved_tools"])
        self.assertEqual(payload["bm25_top5"]["1"]["title"], "CHADS2 Stroke Risk Calculator")
        self.assertEqual(payload["vector_top5"]["2"]["purpose"], "Estimate liver fibrosis risk.")
        self.assertIn("example", payload["model_context"]["bm25_top5"]["1"])
        self.assertEqual(payload["bm25_raw_top5"][0]["channel"], "bm25")
        self.assertEqual(payload["bm25_raw_top5"][0]["calculator_payload"]["pmid"], "1")
        self.assertNotIn("score", payload["bm25_raw_top5"][0])
        self.assertIn("title", payload["vector_raw_top5"][0]["calculator_payload"])
        self.assertNotIn("score", payload["vector_raw_top5"][0])
        self.assertEqual(payload["recommended_pmids"], ["1", "2"])
        self.assertIn("query_text", payload)

    def test_create_retrieval_tool_can_return_scoreful_bundle_for_internal_ranking(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.internal.json")
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )

            payload = tool.retrieve_from_structured_case(
                {
                    "raw_text": (
                        "68 year old with atrial fibrillation, hypertension, diabetes, "
                        "prior stroke, and congestive heart failure."
                    ),
                    "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                    "known_facts": ["prior stroke", "heart failure"],
                },
                top_k=2,
                include_scores=True,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertIn("score", payload["candidate_ranking"][0])
        self.assertIn("score", payload["bm25_raw_top5"][0])
        self.assertIn("vector_score", payload["vector_raw_top5"][0])

    def test_retrieval_query_text_falls_back_to_case_summary_for_long_raw_text(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.case-summary.json")
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )

            payload = tool.retrieve_from_structured_case(
                {
                    "raw_text": "very long noisy note " * 100,
                    "case_summary": "68 year old with atrial fibrillation, hypertension, and prior stroke.",
                    "problem_list": ["atrial fibrillation", "hypertension"],
                    "known_facts": ["prior stroke"],
                },
                top_k=2,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertTrue(
            payload["query_text"].startswith(
                "68 year old with atrial fibrillation, hypertension, and prior stroke."
            )
        )
        self.assertNotIn("very long noisy note", payload["query_text"])

    def test_decorated_retrieval_tool_can_be_collected_and_invoked_with_state(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path()
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool_provider = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )
            registered_tools = collect_tools(tool_provider)
            retrieval_tool = registered_tools["riskcalc_parameter_retriever"]

            rows_from_kwargs = retrieval_tool.invoke(
                structured_case={
                    "raw_text": (
                        "68 year old with atrial fibrillation, hypertension, diabetes, "
                        "prior stroke, and congestive heart failure."
                    ),
                    "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                    "known_facts": ["prior stroke", "heart failure"],
                },
                top_k=2,
            )
            rows_from_state = retrieval_tool.invoke(
                ClinicalToolJob(
                    mode="patient_note",
                    text=(
                        "68 year old with atrial fibrillation, hypertension, diabetes, "
                        "prior stroke, and congestive heart failure."
                    ),
                    structured_case={
                        "raw_text": (
                            "68 year old with atrial fibrillation, hypertension, diabetes, "
                            "prior stroke, and congestive heart failure."
                        ),
                        "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                        "known_facts": ["prior stroke", "heart failure"],
                    },
                    top_k=2,
                )
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertIn("riskcalc_parameter_retriever", registered_tools)
        self.assertEqual(rows_from_kwargs["candidate_ranking"][0]["pmid"], "1")
        self.assertEqual(rows_from_state["candidate_ranking"][0]["pmid"], "1")
        self.assertIn("2", rows_from_state["vector_top5"])

    def test_generic_structured_retrieval_tools_expose_bm25_and_vector_modes(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.generic-structured.json")
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool_provider = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )
            registered_tools = collect_tools(tool_provider.structured_tools)

            bm25_payload = registered_tools["structured_bm25_retriever"].invoke(
                structured_case={
                    "raw_text": (
                        "68 year old with atrial fibrillation, hypertension, diabetes, "
                        "prior stroke, and congestive heart failure."
                    ),
                    "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                    "known_facts": ["prior stroke", "heart failure"],
                },
                top_k=2,
                include_scores=True,
            )
            vector_payload = registered_tools["structured_vector_retriever"].invoke(
                structured_case={
                    "raw_text": "68 year old with atrial fibrillation and prior stroke.",
                    "problem_list": ["atrial fibrillation"],
                    "known_facts": ["prior stroke"],
                },
                top_k=2,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertIn("structured_bm25_retriever", registered_tools)
        self.assertIn("structured_vector_retriever", registered_tools)
        self.assertEqual(bm25_payload["backend_used"], "bm25")
        self.assertEqual(bm25_payload["candidate_ranking"][0]["document_id"], "1")
        self.assertEqual(bm25_payload["candidate_ranking"][0]["pmid"], "1")
        self.assertIn("score", bm25_payload["candidate_ranking"][0])
        self.assertEqual(vector_payload["backend_used"], "vector")
        self.assertEqual(vector_payload["candidate_ranking"][0]["document_id"], "2")
        self.assertEqual(vector_payload["retrieved_ids"], ["2", "1"])

    def test_coarse_retrieval_tool_returns_minimal_pmid_title_bundle(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.coarse.json")
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool_provider = create_retrieval_tool(
                catalog,
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )
            registered_tools = collect_tools(tool_provider)
            coarse_tool = registered_tools["riskcalc_coarse_retriever"]

            payload = coarse_tool.invoke(
                ClinicalToolJob(
                    mode="patient_note",
                    text="68 year old with atrial fibrillation and prior stroke.",
                    structured_case={
                        "raw_text": "68 year old with atrial fibrillation and prior stroke.",
                        "problem_list": ["atrial fibrillation"],
                        "known_facts": ["prior stroke"],
                    },
                )
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertEqual(payload["candidate_ranking"][0], {"pmid": "1", "title": "CHADS2 Stroke Risk Calculator"})
        self.assertNotIn("purpose", payload["candidate_ranking"][0])
        self.assertEqual(payload["candidate_pmids"][0], "1")

    def test_default_parameter_path_resolves_repo_data_data_layout(self) -> None:
        resolved = RiskCalcParameterRetrievalTool._resolve_parameter_path(None)

        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.name, "riskcalcs_parameter.json")
        self.assertIn("/data/data/", str(resolved))

    def test_coarse_retrieval_tool_falls_back_to_catalog_keyword_when_parameter_payload_missing(self) -> None:
        catalog = _FakeCatalog()
        tool_provider = create_retrieval_tool(
            catalog,
            backend="keyword",
            vector_retriever=None,
        )

        payload = tool_provider.retrieve_coarse_from_case_fields(
            raw_text="68 year old with atrial fibrillation, hypertension, diabetes, and prior stroke.",
            problem_list=["atrial fibrillation", "hypertension", "diabetes", "prior stroke"],
            known_facts=["heart failure"],
            top_k=2,
            backend="keyword",
        )

        self.assertEqual(payload["candidate_ranking"][0], {"pmid": "1", "title": "CHADS2 Stroke Risk Calculator"})
        self.assertEqual(payload["candidate_pmids"][0], "1")

    def test_computation_retrieval_tool_returns_union_with_full_calculator_payloads(self) -> None:
        riskcalcs_path = self._build_temp_riskcalcs_path("riskcalcs.computation.json")
        try:
            self._write_riskcalcs_payload(riskcalcs_path, _af_riskcalcs_payload())
            tool_provider = create_computation_retrieval_tool(_FakeCatalogWithSourceFiles(riskcalcs_path))
            tool_provider.vector_retriever = _FakeComputationVectorRetriever()
            payload = tool_provider.retrieve_from_structured_case(
                {
                    "structured_inputs": {
                        "hypertension": True,
                        "diabetes": True,
                        "stroke_history": True,
                    },
                    "problem_list": ["atrial fibrillation"],
                    "known_facts": ["prior TIA"],
                },
                candidate_pmids=["1", "2"],
            )
        finally:
            riskcalcs_path.unlink(missing_ok=True)

        returned_pmids = [str(row["pmid"]) for row in payload["candidate_ranking"]]
        self.assertEqual(returned_pmids, ["1", "2"])
        self.assertEqual(payload["bm25_raw_top3"][0]["pmid"], "1")
        self.assertEqual(payload["vector_raw_top3"][0]["pmid"], "2")
        self.assertEqual(payload["candidate_ranking"][0]["calculator_payload"]["pmid"], "1")
        self.assertIn("computation", payload["candidate_ranking"][1]["calculator_payload"])

    def test_computation_retrieval_reranks_full_coarse_pool_instead_of_filtering_to_top_k(self) -> None:
        riskcalcs_path = self._build_temp_riskcalcs_path("riskcalcs.computation.full-pool.json")
        payload = {
            str(index): {
                "title": f"Calculator {index}",
                "purpose": f"Purpose for calculator {index}.",
                "eligibility": "Eligible patients.",
                "computation": (
                    "```python\n"
                    f"def compute_calc_{index}(age, marker_{index}):\n"
                    f"    return age + marker_{index}\n"
                    "```"
                ),
            }
            for index in range(1, 6)
        }
        try:
            self._write_riskcalcs_payload(riskcalcs_path, payload)
            tool_provider = create_computation_retrieval_tool(_FakeCatalogWithSourceFiles(riskcalcs_path))
            tool_provider.vector_retriever = _FiveWayComputationVectorRetriever()
            rows = tool_provider.retrieve_from_structured_case(
                {
                    "structured_inputs": {
                        "age": 68,
                    },
                    "problem_list": ["atrial fibrillation"],
                    "known_facts": ["prior stroke"],
                },
                candidate_pmids=["1", "2", "3", "4", "5"],
                top_k_per_channel=3,
            )
        finally:
            riskcalcs_path.unlink(missing_ok=True)

        self.assertEqual([str(row["pmid"]) for row in rows["bm25_raw_top3"]], ["1", "2", "3"])
        self.assertEqual([str(row["pmid"]) for row in rows["vector_raw_top3"]], ["1", "2", "3"])
        self.assertEqual(
            [str(row["pmid"]) for row in rows["candidate_ranking"]],
            ["1", "2", "3", "4", "5"],
        )
        self.assertEqual(len(rows["candidate_ranking"]), 5)

    def test_medcpt_retriever_scores_candidate_subset_instead_of_filtering_global_top_k(self) -> None:
        retriever = tools.MedCPTRetriever.__new__(tools.MedCPTRetriever)
        retriever.catalog = _ManyFakeCatalog(count=5)
        retriever._torch = _FakeTorchModule()
        retriever._device = "cpu"
        retriever._tokenizer = _FakeTokenizer()
        retriever._query_encoder = _FakeQueryEncoder()
        retriever._pmids = ["1", "2", "3", "4", "5"]
        retriever._pmid_to_index = {pmid: index for index, pmid in enumerate(retriever._pmids)}
        retriever._inference_lock = _NoOpLock()
        retriever._index = _FakeDenseIndex()

        rows = retriever.retrieve("condition", top_k=2, candidate_pmids={"4", "5"})

        self.assertEqual([str(row["pmid"]) for row in rows], ["4", "5"])
        self.assertEqual([float(row["score"]) for row in rows], [0.6, 0.5])

    def test_inline_medcpt_retriever_scores_candidate_subset_instead_of_filtering_global_top_k(self) -> None:
        retriever = retrieval_tools._InlineMedCPTRetriever.__new__(retrieval_tools._InlineMedCPTRetriever)
        retriever.catalog = _ManyFakeCatalog(count=5)
        retriever._torch = _FakeTorchModule()
        retriever._device = "cpu"
        retriever._tokenizer = _FakeTokenizer()
        retriever._query_encoder = _FakeQueryEncoder()
        retriever._pmids = ["1", "2", "3", "4", "5"]
        retriever._pmid_to_index = {pmid: index for index, pmid in enumerate(retriever._pmids)}
        retriever._inference_lock = _NoOpLock()
        retriever._index = _FakeDenseIndex()

        rows = retriever.retrieve("condition", top_k=2, candidate_pmids={"4", "5"})

        self.assertEqual([str(row["pmid"]) for row in rows], ["4", "5"])
        self.assertEqual([float(row["score"]) for row in rows], [0.6, 0.5])

    def test_retrieval_defaults_to_department_tag_candidate_pool(self) -> None:
        department_root = self._build_temp_department_root("department_filter")
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.department.json")
        self._write_department_payload(
            department_root,
            "内科",
            {
                "1": {"title": "CHADS2 Stroke Risk Calculator"},
            },
        )
        self._write_department_payload(
            department_root,
            "肿瘤科",
            {
                "2": {"title": "FIB-4 Index for Liver Fibrosis"},
            },
        )
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool_provider = create_retrieval_tool(
                _FakeCatalog(department_payload_root=department_root),
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )
            registered_tools = collect_tools(tool_provider)
            retrieval_tool = registered_tools["riskcalc_parameter_retriever"]

            rows_from_state = retrieval_tool.invoke(
                ClinicalToolJob(
                    mode="patient_note",
                    text="68 year old with atrial fibrillation and prior stroke.",
                    structured_case={
                        "raw_text": "68 year old with atrial fibrillation and prior stroke.",
                        "problem_list": ["atrial fibrillation"],
                        "known_facts": ["prior stroke"],
                        "department_tags": ["内科"],
                    },
                    top_k=5,
                ),
                include_scores=True,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        returned_pmids = {str(row["pmid"]) for row in rows_from_state["candidate_ranking"]}
        self.assertEqual(returned_pmids, {"1"})
        self.assertEqual(rows_from_state["candidate_ranking"][0]["pmid"], "1")
        self.assertNotIn("2", returned_pmids)
        self.assertEqual(rows_from_state["department_tags"], ["内科"])
        self.assertFalse(rows_from_state["fallback_to_full_catalog"])

    def test_retrieval_falls_back_to_full_catalog_when_department_pool_returns_nothing(self) -> None:
        department_root = self._build_temp_department_root("department_filter_fallback")
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.department.fallback.json")
        self._write_department_payload(
            department_root,
            "内科",
            {},
        )
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool_provider = create_retrieval_tool(
                _FakeCatalog(department_payload_root=department_root),
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )

            payload = tool_provider.retrieve_from_structured_case(
                {
                    "raw_text": (
                        "68 year old with atrial fibrillation, hypertension, diabetes, "
                        "prior stroke, and congestive heart failure."
                    ),
                    "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                    "known_facts": ["prior stroke", "heart failure"],
                    "department_tags": ["内科", "传染科"],
                },
                top_k=2,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        returned_pmids = [str(row["pmid"]) for row in payload["candidate_ranking"]]
        self.assertEqual(returned_pmids, ["1", "2"])
        self.assertEqual(payload["department_tags"], ["内科", "传染科"])
        self.assertTrue(payload["fallback_to_full_catalog"])

    def test_retrieval_resolves_repo_level_department_directory_when_data_sibling_missing(self) -> None:
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.repo_level_department.json")
        temp_root = parameter_path.parent
        data_dir = temp_root / "data"
        repo_level_department_root = temp_root / "数据" / "计算器科室"
        data_dir.mkdir(parents=True, exist_ok=True)
        riskcalcs_path = data_dir / "riskcalcs.json"
        riskcalcs_path.write_text("{}", encoding="utf-8")
        self._write_department_payload(
            repo_level_department_root,
            "内科",
            {
                "1": {"title": "CHADS2 Stroke Risk Calculator"},
            },
        )
        self._write_department_payload(
            repo_level_department_root,
            "肿瘤科",
            {
                "2": {"title": "FIB-4 Index for Liver Fibrosis"},
            },
        )
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            tool_provider = create_retrieval_tool(
                _FakeCatalogWithSourceFiles(riskcalcs_path),
                parameter_path=parameter_path,
                vector_retriever=_FakeVectorRetriever(),
            )

            payload = tool_provider.retrieve_from_structured_case(
                {
                    "raw_text": "68 year old with atrial fibrillation and prior stroke.",
                    "problem_list": ["atrial fibrillation"],
                    "known_facts": ["prior stroke"],
                    "department_tags": ["内科"],
                },
                top_k=5,
                include_scores=True,
            )
        finally:
            parameter_path.unlink(missing_ok=True)
            riskcalcs_path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                (repo_level_department_root / "内科" / "riskcalcs.json").unlink()
            with contextlib.suppress(OSError):
                (repo_level_department_root / "肿瘤科" / "riskcalcs.json").unlink()
            with contextlib.suppress(OSError):
                (repo_level_department_root / "内科").rmdir()
            with contextlib.suppress(OSError):
                (repo_level_department_root / "肿瘤科").rmdir()
            with contextlib.suppress(OSError):
                repo_level_department_root.rmdir()
            with contextlib.suppress(OSError):
                (temp_root / "数据").rmdir()
            with contextlib.suppress(OSError):
                data_dir.rmdir()

        returned_pmids = {str(row["pmid"]) for row in payload["candidate_ranking"]}
        self.assertEqual(returned_pmids, {"1"})
        self.assertEqual(payload["department_tags"], ["内科"])
        self.assertFalse(payload["fallback_to_full_catalog"])

    def test_hybrid_candidate_ranking_uses_half_top_k_per_branch(self) -> None:
        catalog = _ManyFakeCatalog()
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.hybrid.union.json")
        try:
            self._write_parameter_payload(parameter_path, self._many_parameter_payload())
            tool = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_TopFiveVectorRetriever(),
            )

            payload = tool.retrieve_from_structured_case(
                {
                    "raw_text": "condition",
                    "problem_list": ["condition"],
                },
                top_k=10,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        returned_pmids = {str(row["pmid"]) for row in payload["candidate_ranking"]}
        self.assertEqual(len(payload["candidate_ranking"]), 10)
        self.assertEqual(returned_pmids, {str(index) for index in range(1, 11)})
        self.assertNotIn("11", returned_pmids)
        self.assertEqual(payload["hybrid_branch_top_k"], 5)

    def test_hybrid_candidate_ranking_uses_half_of_large_top_k_per_branch(self) -> None:
        catalog = _ManyFakeCatalog(count=60)
        parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.hybrid.large_union.json")
        try:
            self._write_parameter_payload(parameter_path, self._many_parameter_payload(count=60))
            tool = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=_RangeVectorRetriever(26, 60),
            )

            payload = tool.retrieve_from_structured_case(
                {
                    "raw_text": "condition",
                    "problem_list": ["condition"],
                },
                top_k=50,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        returned_pmids = {str(row["pmid"]) for row in payload["candidate_ranking"]}
        self.assertEqual(len(payload["candidate_ranking"]), 50)
        self.assertEqual(returned_pmids, {str(index) for index in range(1, 51)})
        self.assertNotIn("51", returned_pmids)
        self.assertEqual(payload["hybrid_branch_top_k"], 25)

    def test_create_retrieval_tool_reuses_cached_instance_for_same_configuration(self) -> None:
        catalog = _FakeCatalog()
        parameter_path = self._build_temp_parameter_path()
        vector_retriever = _FakeVectorRetriever()
        try:
            self._write_parameter_payload(parameter_path, _af_parameter_payload())
            first = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=vector_retriever,
            )
            second = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=parameter_path,
                vector_retriever=vector_retriever,
            )
        finally:
            parameter_path.unlink(missing_ok=True)

        self.assertIs(first, second)
        self.assertIs(first.parameter_retriever, second.parameter_retriever)
        self.assertIs(first.vector_retriever, vector_retriever)

    def test_create_retrieval_tool_does_not_reuse_cache_across_parameter_files(self) -> None:
        catalog = _FakeCatalog()
        first_parameter_path = self._build_temp_parameter_path()
        second_parameter_path = self._build_temp_parameter_path("riskcalcs_parameter.test.alt.json")
        vector_retriever = _FakeVectorRetriever()
        try:
            self._write_parameter_payload(
                first_parameter_path,
                {
                    "1": {
                        "title": "CHADS2 Stroke Risk Calculator",
                        "purpose": "Estimate stroke risk in atrial fibrillation.",
                        "eligibility": "AF patients.",
                        "parameter_names": ["hypertension"],
                        "parameter_aliases": {"hypertension": ["hypertension"]},
                        "parameter_text": "hypertension",
                    }
                },
            )
            self._write_parameter_payload(
                second_parameter_path,
                {
                    "2": {
                        "title": "FIB-4 Index for Liver Fibrosis",
                        "purpose": "Estimate liver fibrosis risk.",
                        "eligibility": "Liver disease patients.",
                        "parameter_names": ["ast"],
                        "parameter_aliases": {"ast": ["ast"]},
                        "parameter_text": "ast",
                    }
                },
            )
            first = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=first_parameter_path,
                vector_retriever=vector_retriever,
            )
            second = create_retrieval_tool(
                catalog,
                backend="hybrid",
                parameter_path=second_parameter_path,
                vector_retriever=vector_retriever,
            )
        finally:
            first_parameter_path.unlink(missing_ok=True)
            second_parameter_path.unlink(missing_ok=True)

        self.assertIsNot(first, second)
        self.assertIsNot(first.parameter_retriever, second.parameter_retriever)


if __name__ == "__main__":
    unittest.main()
