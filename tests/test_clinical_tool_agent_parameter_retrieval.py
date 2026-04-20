from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from agent.clinical_tool_agent import ClinicalToolAgent
from agent.graph.types import ClinicalToolJob, RetrievalQuery
from agent.tools import RiskCalcExecutionTool, RiskCalcRegistration, RiskCalcRegistry, tool


class _FakeDocument:
    def __init__(self, pmid: str, title: str, purpose: str, eligibility: str) -> None:
        self.pmid = pmid
        self.title = title
        self.purpose = purpose
        self.eligibility = eligibility
        self.utility = ""
        self.abstract = ""
        self.specialty = ""

    def to_brief(self) -> dict[str, str]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "purpose": self.purpose,
            "eligibility": self.eligibility,
            "taxonomy": {},
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "purpose": self.purpose,
            "eligibility": self.eligibility,
            "utility": self.utility,
            "abstract": self.abstract,
            "specialty": self.specialty,
        }


class _FakeCatalog:
    def __init__(self) -> None:
        self._documents = {
            "1": _FakeDocument(
                "1",
                "CHADS2 Stroke Risk Calculator",
                "Estimate stroke risk in atrial fibrillation.",
                "Atrial fibrillation patients.",
            ),
            "2": _FakeDocument(
                "2",
                "FIB-4 Index for Liver Fibrosis",
                "Estimate liver fibrosis risk.",
                "Chronic liver disease patients.",
            ),
        }

    def build_candidate_pool(self, clinical_text: str, limit: int):
        del clinical_text, limit
        return {"1", "2"}

    def get(self, pmid: str):
        return self._documents[str(pmid)]


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int = 10, *, candidate_pmids=None, backend=None):
        del query, top_k, candidate_pmids, backend
        return []

    @tool(
        name="riskcalc_coarse_retriever",
        description="Retrieve coarse calculator matches from structured_case.",
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
        },
        state_fields={
            "structured_case": ("structured_case", "clinical_tool_job.structured_case"),
            "top_k": ("top_k", "clinical_tool_job.top_k"),
        },
    )
    def retrieve_coarse_from_structured_case(self, structured_case, *, top_k: int = 10, candidate_pmids=None):
        del structured_case
        rows = [
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
            }
        ]
        if candidate_pmids is None:
            return {
                "query_text": "atrial fibrillation hypertension diabetes TIA",
                "candidate_ranking": rows[:top_k],
                "retrieved_tools": rows[:top_k],
            }
        filtered = [row for row in rows if str(row.get("pmid")) in candidate_pmids][:top_k]
        return {
            "query_text": "atrial fibrillation hypertension diabetes TIA",
            "candidate_ranking": filtered,
            "retrieved_tools": filtered,
        }


class _FakeComputationRetriever:
    @tool(
        name="riskcalc_computation_retriever",
        description="Refine coarse calculator matches with computation-derived parameters.",
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "candidate_pmids": "list[str] | set[str]",
            "top_k_per_channel": "int",
        },
        state_fields={
            "structured_case": ("structured_case", "clinical_tool_job.structured_case"),
        },
    )
    def retrieve_from_structured_case(self, structured_case, *, candidate_pmids, top_k_per_channel: int = 3):
        del structured_case, top_k_per_channel
        rows = [
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
                "purpose": "Estimate stroke risk in atrial fibrillation.",
                "eligibility": "Atrial fibrillation patients.",
                "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                "query_text": "atrial fibrillation hypertension diabetes TIA",
                "source_channels": ["bm25", "vector"],
                "calculator_payload": {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "purpose": "Estimate stroke risk in atrial fibrillation.",
                    "eligibility": "Atrial fibrillation patients.",
                    "computation": "def compute_chads2(hypertension, diabetes, stroke_history): return 3",
                },
            }
        ]
        filtered = [row for row in rows if str(row.get("pmid")) in set(candidate_pmids)]
        return {
            "query_text": "hypertension ; diabetes ; stroke history",
            "bm25_raw_top3": [
                {
                    "rank": 1,
                    "channel": "bm25",
                    "pmid": "1",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                    "calculator_payload": dict(filtered[0]["calculator_payload"]) if filtered else {},
                }
            ]
            if filtered
            else [],
            "vector_raw_top3": [
                {
                    "rank": 1,
                    "channel": "vector",
                    "pmid": "1",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                    "calculator_payload": dict(filtered[0]["calculator_payload"]) if filtered else {},
                }
            ]
            if filtered
            else [],
            "candidate_ranking": filtered,
            "retrieved_tools": filtered,
        }


class _QuestionOrderCoarseRetriever:
    @tool(
        name="riskcalc_coarse_retriever",
        description="Retrieve coarse calculator matches from structured_case.",
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
        },
        state_fields={
            "structured_case": ("structured_case", "clinical_tool_job.structured_case"),
            "top_k": ("top_k", "clinical_tool_job.top_k"),
        },
    )
    def retrieve_coarse_from_structured_case(self, structured_case, *, top_k: int = 10, candidate_pmids=None):
        del structured_case
        rows = [
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
            },
            {
                "pmid": "2",
                "title": "FIB-4 Index for Liver Fibrosis",
            },
        ]
        if candidate_pmids is not None:
            rows = [row for row in rows if str(row.get("pmid")) in set(candidate_pmids)]
        rows = rows[:top_k]
        return {
            "query_text": "atrial fibrillation hypertension diabetes prior TIA",
            "candidate_ranking": rows,
            "retrieved_tools": rows,
        }


class _QuestionOrderComputationRetriever:
    @tool(
        name="riskcalc_computation_retriever",
        description="Refine coarse calculator matches with computation-derived parameters.",
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "candidate_pmids": "list[str] | set[str]",
            "top_k_per_channel": "int",
        },
        state_fields={
            "structured_case": ("structured_case", "clinical_tool_job.structured_case"),
        },
    )
    def retrieve_from_structured_case(self, structured_case, *, candidate_pmids, top_k_per_channel: int = 3):
        del structured_case, top_k_per_channel
        rows = [
            {
                "pmid": "2",
                "title": "FIB-4 Index for Liver Fibrosis",
                "purpose": "Estimate liver fibrosis risk.",
                "eligibility": "Chronic liver disease patients.",
                "parameter_names": ["age", "ast", "alt", "platelet_count"],
                "query_text": "age ; ast ; alt ; platelet count",
                "source_channels": ["bm25"],
                "calculator_payload": {
                    "pmid": "2",
                    "title": "FIB-4 Index for Liver Fibrosis",
                    "purpose": "Estimate liver fibrosis risk.",
                    "eligibility": "Chronic liver disease patients.",
                },
            },
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
                "purpose": "Estimate stroke risk in atrial fibrillation.",
                "eligibility": "Atrial fibrillation patients.",
                "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                "query_text": "hypertension ; diabetes ; stroke history",
                "source_channels": ["bm25", "vector"],
                "calculator_payload": {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "purpose": "Estimate stroke risk in atrial fibrillation.",
                    "eligibility": "Atrial fibrillation patients.",
                },
            },
        ]
        filtered = [row for row in rows if str(row.get("pmid")) in set(candidate_pmids)]
        return {
            "query_text": "age ; ast ; alt ; platelet count ; hypertension ; diabetes ; stroke history",
            "bm25_raw_top3": [
                {
                    "rank": 1,
                    "channel": "bm25",
                    "pmid": "2",
                    "parameter_names": ["age", "ast", "alt", "platelet_count"],
                    "calculator_payload": {
                        "pmid": "2",
                        "title": "FIB-4 Index for Liver Fibrosis",
                        "purpose": "Estimate liver fibrosis risk.",
                        "eligibility": "Chronic liver disease patients.",
                    },
                },
                {
                    "rank": 2,
                    "channel": "bm25",
                    "pmid": "1",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                    "calculator_payload": {
                        "pmid": "1",
                        "title": "CHADS2 Stroke Risk Calculator",
                        "purpose": "Estimate stroke risk in atrial fibrillation.",
                        "eligibility": "Atrial fibrillation patients.",
                    },
                },
            ],
            "vector_raw_top3": [
                {
                    "rank": 1,
                    "channel": "vector",
                    "pmid": "1",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                    "calculator_payload": {
                        "pmid": "1",
                        "title": "CHADS2 Stroke Risk Calculator",
                        "purpose": "Estimate stroke risk in atrial fibrillation.",
                        "eligibility": "Atrial fibrillation patients.",
                    },
                }
            ],
            "candidate_ranking": filtered,
            "retrieved_tools": filtered,
        }


class _FakeChatClient:
    def __init__(self, response: str = "", responses: list[str] | None = None) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.last_messages = None

    def complete(self, messages, model=None, temperature=0.0):
        del model, temperature
        self.last_messages = list(messages)
        if self.responses:
            return self.responses.pop(0)
        return self.response


class _FakeExecutionTool:
    def __init__(
        self,
        registration: RiskCalcRegistration,
        *,
        extracted_inputs: dict[str, object] | None,
        execution_result: dict[str, object],
        summary_text: str = "",
    ) -> None:
        self.registration = registration
        self._extracted_inputs = extracted_inputs
        self._execution_result = dict(execution_result)
        self._summary_text = summary_text
        self.last_extract_kwargs = None

    def has_registration(self, calculator: str) -> bool:
        return str(calculator) == self.registration.calc_id

    def get_registration(self, calculator: str) -> RiskCalcRegistration:
        if not self.has_registration(calculator):
            raise KeyError(calculator)
        return self.registration

    def extract_inputs(
        self,
        *,
        calculator: str,
        clinical_text: str,
        structured_case=None,
        calculator_payload=None,
        retrieval_query_text=None,
        llm_model=None,
        temperature=0.0,
    ):
        self.last_extract_kwargs = {
            "calculator": calculator,
            "clinical_text": clinical_text,
            "structured_case": structured_case,
            "calculator_payload": calculator_payload,
            "retrieval_query_text": retrieval_query_text,
            "llm_model": llm_model,
            "temperature": temperature,
        }
        return {
            "inputs": dict(self._extracted_inputs) if isinstance(self._extracted_inputs, dict) else self._extracted_inputs,
            "raw_response": "{}",
        }

    def execute_registered(self, *, calculator: str, inputs: dict[str, object]):
        del calculator, inputs
        return dict(self._execution_result)

    def summarize_result(
        self,
        *,
        calculator: str,
        clinical_text: str,
        mode: str,
        execution: dict[str, object],
        llm_model=None,
        temperature=0.0,
    ):
        del calculator, clinical_text, mode, execution, llm_model, temperature
        return {"final_text": self._summary_text, "raw_response": "{}"}


def _build_registration() -> RiskCalcRegistration:
    return RiskCalcRegistration(
        calc_id="1",
        title="CHADS2 Stroke Risk Calculator",
        function_name="compute_chads2",
        parameter_names=["hypertension", "diabetes", "stroke_history"],
        code="def compute_chads2(hypertension, diabetes, stroke_history):\n    return 3\n",
        purpose="Estimate stroke risk in atrial fibrillation.",
        eligibility="Atrial fibrillation patients.",
        interpretation="Higher score means higher stroke risk.",
    )


class ClinicalToolAgentParameterRetrievalTests(unittest.TestCase):
    def _build_agent(self, *, chat_client: _FakeChatClient | None = None) -> ClinicalToolAgent:
        return ClinicalToolAgent(
            catalog=_FakeCatalog(),
            retriever=_FakeRetriever(),
            computation_retriever=_FakeComputationRetriever(),
            chat_client=chat_client or _FakeChatClient(),
        )

    def test_agent_registers_retrieval_tool_in_tools(self) -> None:
        agent = self._build_agent()

        self.assertIn("riskcalc_coarse_retriever", agent.tools)
        self.assertIn("riskcalc_computation_retriever", agent.tools)
        self.assertTrue(hasattr(agent.tools["riskcalc_coarse_retriever"], "invoke"))
        self.assertTrue(hasattr(agent.tools["riskcalc_computation_retriever"], "invoke"))

    def test_retrieve_and_collect_candidates_includes_structured_case_parameter_batch(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="patient_note",
            text="Atrial fibrillation with hypertension, diabetes, and prior TIA.",
            case_summary="Atrial fibrillation with major stroke-risk factors.",
            structured_case={
                "raw_text": "Atrial fibrillation with hypertension, diabetes, and prior TIA.",
                "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                "known_facts": ["prior TIA"],
            },
            retrieval_queries=[
                RetrievalQuery(
                    stage="case_summary_dense",
                    text="Atrial fibrillation with major stroke-risk factors.",
                    intent="case_summary_dense",
                    priority=1,
                )
            ],
            top_k=5,
        )

        result = agent._retrieve_and_rank_candidates(job)

        self.assertGreaterEqual(len(result["retrieved_tools"]), 1)
        self.assertEqual(result["retrieved_tools"][0]["pmid"], "1")
        self.assertTrue(
            any(batch.get("channel") == "computation_parameter_match" for batch in result["retrieval_batches"])
        )

    def test_question_mode_preserves_coarse_order_when_computation_reranks_candidates(self) -> None:
        agent = ClinicalToolAgent(
            catalog=_FakeCatalog(),
            retriever=_QuestionOrderCoarseRetriever(),
            computation_retriever=_QuestionOrderComputationRetriever(),
            chat_client=_FakeChatClient(),
        )
        job = ClinicalToolJob(
            mode="question",
            text="Atrial fibrillation patient with hypertension, diabetes, and prior TIA. Which score applies?",
            structured_case={
                "raw_text": "Atrial fibrillation patient with hypertension, diabetes, and prior TIA.",
                "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                "known_facts": ["prior TIA"],
            },
            top_k=5,
        )

        result = agent._retrieve_and_rank_candidates(job)

        self.assertEqual(
            [candidate["pmid"] for candidate in result["retrieval_batches"][1]["tools"][:2]],
            ["2", "1"],
        )
        self.assertEqual(
            [candidate["pmid"] for candidate in result["question_selection_candidates"][:2]],
            ["1", "2"],
        )
        self.assertEqual(
            [candidate["pmid"] for candidate in result["retrieved_tools"][:2]],
            ["1", "2"],
        )
        self.assertEqual(
            [candidate["pmid"] for candidate in result["candidate_ranking"][:2]],
            ["1", "2"],
        )
        self.assertEqual(
            result["retrieved_tools"][0]["parameter_names"],
            ["hypertension", "diabetes", "stroke_history"],
        )

    def test_select_tool_for_question_returns_parameter_names(self) -> None:
        agent = self._build_agent(
            chat_client=_FakeChatClient(
                response=(
                    '{"selected_tool_id": "1", "parameter_names": ["hypertension", "diabetes", "stroke_history"], '
                    '"reason": "AF stroke-risk calculator matches the question."}'
                )
            )
        )
        job = ClinicalToolJob(
            mode="question",
            text="Atrial fibrillation patient with hypertension, diabetes, and prior TIA. Which score applies?",
        )
        retrieved = [
            {
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
                "purpose": "Estimate stroke risk in atrial fibrillation.",
                "parameter_names": ["hypertension", "diabetes", "stroke_history"],
            }
        ]

        selection = agent._select_tool_for_question(job, retrieved)

        self.assertEqual(selection["pmid"], "1")
        self.assertEqual(
            selection["parameter_names"],
            ["hypertension", "diabetes", "stroke_history"],
        )
        self.assertEqual(
            selection["model_parameter_names"],
            ["hypertension", "diabetes", "stroke_history"],
        )
        self.assertEqual(
            selection["reason"],
            "AF stroke-risk calculator matches the question.",
        )

    def test_execution_tool_extract_inputs_uses_selected_calculator_payload_and_structured_case(self) -> None:
        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        chat_client = _FakeChatClient(
            response='{"inputs":{"hypertension":true,"diabetes":true,"stroke_history":true}}'
        )
        execution_tool = RiskCalcExecutionTool(registry, chat_client=chat_client)

        structured_case = {
            "raw_text": "Atrial fibrillation with hypertension, diabetes, and prior TIA.",
            "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
            "known_facts": ["prior TIA"],
        }
        calculator_payload = {
            "pmid": "1",
            "title": "CHADS2 Stroke Risk Calculator",
            "purpose": "Estimate stroke risk in atrial fibrillation.",
            "eligibility": "Atrial fibrillation patients.",
            "specialty": "cardiology",
            "computation": "def compute_chads2(hypertension, diabetes, stroke_history): return 3",
        }

        extracted = execution_tool.extract_inputs(
            calculator="1",
            clinical_text="Atrial fibrillation with hypertension, diabetes, and prior TIA.",
            structured_case=structured_case,
            calculator_payload=calculator_payload,
            retrieval_query_text="atrial fibrillation hypertension diabetes TIA",
        )

        self.assertEqual(
            extracted["inputs"],
            {"hypertension": True, "diabetes": True, "stroke_history": True},
        )
        self.assertEqual(extracted["structured_case"], structured_case)
        self.assertEqual(
            extracted["retrieval_query_text"],
            "atrial fibrillation hypertension diabetes TIA",
        )
        self.assertEqual(extracted["calculator_payload"]["specialty"], "cardiology")
        self.assertIsNotNone(chat_client.last_messages)
        self.assertEqual(chat_client.last_messages[0]["role"], "system")
        self.assertIn("calculator child agent", chat_client.last_messages[0]["content"].lower())
        self.assertIn("Structured Case JSON", chat_client.last_messages[1]["content"])
        self.assertIn("Dispatch Retrieval Query Text", chat_client.last_messages[1]["content"])
        self.assertIn("Tool Input Schema", chat_client.last_messages[1]["content"])
        self.assertIn('"hypertension": "<value>"', chat_client.last_messages[1]["content"])
        self.assertIn('"specialty": "cardiology"', chat_client.last_messages[1]["content"])
        self.assertIn('"problem_list"', chat_client.last_messages[1]["content"])

    def test_probe_chat_client_accepts_current_input_extraction_prompt(self) -> None:
        probe_path = Path(__file__).resolve().parents[1] / "scripts" / "probe_agent_handoff.py"
        spec = importlib.util.spec_from_file_location("probe_agent_handoff", str(probe_path))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        chat_client = module.ProbeChatClient(registry)
        prompt = (
            "Extract inputs for the selected calculator.\n\n"
            "Selected Calculator PMID: 1\n"
            "Selected Calculator Title: CHADS2 Stroke Risk Calculator\n"
            "Selected Calculator Function: compute_chads2\n"
            "Selected Calculator Required Parameters: hypertension, diabetes, stroke_history\n\n"
            "Tool Input Schema (exact JSON shape; keep the parameter keys exactly as written and replace the values only):\n"
            '{"inputs": {"hypertension": "<value>", "diabetes": "<value>", "stroke_history": "<value>"}}\n\n'
            "Selected Calculator Repository Payload (resolved by PMID):\n"
            '{"pmid":"1"}\n\n'
            "Dispatch Retrieval Query Text (the upstream coarse retrieval query; if raw_text was too long upstream, this may already use case_summary):\n"
            "atrial fibrillation hypertension diabetes prior TIA\n\n"
            "Structured Case JSON:\n"
            '{"known_facts":["prior TIA"]}\n\n'
            "Raw Clinical Text (supporting evidence only):\n"
            "AF patient.\n\n"
            "Task:\n"
            "1. Read only the selected calculator payload above.\n"
        )

        answer = chat_client.complete([{"role": "user", "content": prompt}])
        payload = json.loads(answer)

        self.assertEqual(
            payload["inputs"],
            {"hypertension": True, "diabetes": True, "stroke_history": True},
        )

    def test_extract_registered_inputs_passes_selected_calculator_payload_and_structured_case(self) -> None:
        agent = self._build_agent()
        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        execution_tool = _FakeExecutionTool(
            registration,
            extracted_inputs={
                "hypertension": True,
                "diabetes": True,
                "stroke_history": True,
            },
            execution_result={"status": "completed", "inputs": {}, "defaults_used": {}, "result": 3},
        )
        agent.registry = registry
        agent.execution_tool = execution_tool
        agent.tools["riskcalc_executor"] = execution_tool
        job = ClinicalToolJob(
            mode="question",
            text="AF patient with hypertension, diabetes, and prior TIA.",
            case_summary="AF with hypertension, diabetes, and prior TIA.",
            structured_case={
                "raw_text": "AF patient with hypertension, diabetes, and prior TIA.",
                "case_summary": "AF with hypertension, diabetes, and prior TIA.",
                "problem_list": ["atrial fibrillation", "hypertension", "diabetes"],
                "known_facts": ["prior TIA"],
            },
        )

        inputs = agent._extract_registered_inputs(job, registration)

        self.assertEqual(
            inputs,
            {"hypertension": True, "diabetes": True, "stroke_history": True},
        )
        self.assertIsNotNone(execution_tool.last_extract_kwargs)
        self.assertEqual(
            execution_tool.last_extract_kwargs["structured_case"],
            job.structured_case,
        )
        self.assertEqual(
            execution_tool.last_extract_kwargs["calculator_payload"]["pmid"],
            "1",
        )
        self.assertEqual(
            execution_tool.last_extract_kwargs["calculator_payload"]["title"],
            "CHADS2 Stroke Risk Calculator",
        )
        self.assertEqual(
            execution_tool.last_extract_kwargs["retrieval_query_text"],
            "AF with hypertension, diabetes, and prior TIA.",
        )
        self.assertIn(
            "Estimate stroke risk in atrial fibrillation.",
            str(execution_tool.last_extract_kwargs["calculator_payload"].get("purpose") or ""),
        )

    def test_extract_registered_inputs_prefers_saved_coarse_query_text_when_available(self) -> None:
        agent = self._build_agent()
        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        execution_tool = _FakeExecutionTool(
            registration,
            extracted_inputs={
                "hypertension": True,
                "diabetes": True,
                "stroke_history": True,
            },
            execution_result={"status": "completed", "inputs": {}, "defaults_used": {}, "result": 3},
        )
        agent.registry = registry
        agent.execution_tool = execution_tool
        agent.tools["riskcalc_executor"] = execution_tool
        agent._latest_coarse_retrieval_bundle = {
            "query_text": "atrial fibrillation hypertension diabetes prior TIA",
            "candidate_ranking": [{"pmid": "1", "title": "CHADS2 Stroke Risk Calculator"}],
        }
        job = ClinicalToolJob(
            mode="question",
            text="AF patient with hypertension, diabetes, and prior TIA.",
            case_summary="This fallback summary should not be used when coarse query text exists.",
            structured_case={
                "raw_text": "AF patient with hypertension, diabetes, and prior TIA.",
                "case_summary": "This fallback summary should not be used when coarse query text exists.",
            },
        )

        inputs = agent._extract_registered_inputs(job, registration)

        self.assertEqual(
            inputs,
            {"hypertension": True, "diabetes": True, "stroke_history": True},
        )
        self.assertEqual(
            execution_tool.last_extract_kwargs["retrieval_query_text"],
            "atrial fibrillation hypertension diabetes prior TIA",
        )

    def test_extract_registered_inputs_prefers_parent_dispatch_query_text_when_available(self) -> None:
        agent = self._build_agent()
        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        execution_tool = _FakeExecutionTool(
            registration,
            extracted_inputs={
                "hypertension": True,
                "diabetes": True,
                "stroke_history": True,
            },
            execution_result={"status": "completed", "inputs": {}, "defaults_used": {}, "result": 3},
        )
        agent.registry = registry
        agent.execution_tool = execution_tool
        agent.tools["riskcalc_executor"] = execution_tool
        agent._latest_coarse_retrieval_bundle = {
            "query_text": "this coarse query should be ignored when the parent already dispatched one",
        }
        job = ClinicalToolJob(
            mode="question",
            text="AF patient with hypertension, diabetes, and prior TIA.",
            case_summary="Fallback summary that should not win.",
            structured_case={
                "raw_text": "AF patient with hypertension, diabetes, and prior TIA.",
                "case_summary": "Fallback summary that should not win.",
            },
            dispatch_query_text="parent dispatched atrial fibrillation stroke risk query",
            selection_context={
                "dispatch_query_text": "selection context copy of the parent query",
            },
        )

        inputs = agent._extract_registered_inputs(job, registration)

        self.assertEqual(
            inputs,
            {"hypertension": True, "diabetes": True, "stroke_history": True},
        )
        self.assertEqual(
            execution_tool.last_extract_kwargs["retrieval_query_text"],
            "parent dispatched atrial fibrillation stroke risk query",
        )

    def test_assess_candidate_execution_gate_marks_calculator_passed_when_score_computed(self) -> None:
        agent = self._build_agent()
        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        execution_tool = _FakeExecutionTool(
            registration,
            extracted_inputs={
                "hypertension": True,
                "diabetes": True,
                "stroke_history": True,
            },
            execution_result={
                "status": "completed",
                "inputs": {
                    "hypertension": True,
                    "diabetes": True,
                    "stroke_history": True,
                },
                "defaults_used": {},
                "result": 3,
            },
            summary_text="CHADS2 score is 3.",
        )
        agent.registry = registry
        agent.execution_tool = execution_tool
        agent.tools["riskcalc_executor"] = execution_tool

        decision, execution = agent._assess_candidate_execution_gate(
            ClinicalToolJob(mode="question", text="AF patient with hypertension, diabetes, and prior TIA."),
            {"pmid": "1", "title": registration.title},
        )

        self.assertEqual(decision["gate_status"], "passed")
        self.assertEqual(decision["execution_status"], "completed")
        self.assertEqual(decision["result"], 3)
        self.assertIsNotNone(execution)
        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["result"], 3)
        self.assertIn("CHADS2 score is 3", execution["final_text"])

    def test_assess_candidate_execution_gate_marks_missing_inputs_as_failed_gate(self) -> None:
        agent = self._build_agent()
        registration = _build_registration()
        registry = RiskCalcRegistry({registration.calc_id: registration})
        execution_tool = _FakeExecutionTool(
            registration,
            extracted_inputs={"hypertension": True},
            execution_result={
                "status": "missing_inputs",
                "inputs": {"hypertension": True},
                "defaults_used": {},
                "missing_inputs": ["diabetes", "stroke_history"],
            },
        )
        agent.registry = registry
        agent.execution_tool = execution_tool
        agent.tools["riskcalc_executor"] = execution_tool

        decision, execution = agent._assess_candidate_execution_gate(
            ClinicalToolJob(mode="question", text="AF patient with hypertension only."),
            {"pmid": "1", "title": registration.title},
        )

        self.assertEqual(decision["gate_status"], "failed_missing_inputs")
        self.assertEqual(decision["execution_status"], "missing_inputs")
        self.assertEqual(decision["patient_eligible"], "no")
        self.assertEqual(decision["missing_all_parameters"], "no")
        self.assertEqual(decision["missing_inputs"], ["diabetes", "stroke_history"])
        self.assertIsNone(execution)

    def test_run_question_executes_only_selected_candidate_after_selection(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="question",
            text="Atrial fibrillation patient with hypertension, diabetes, and prior TIA. Which stroke score applies?",
        )
        agent._retrieve_and_rank_candidates = lambda _: {
            "retrieved_tools": [
                {
                    "pmid": "2",
                    "title": "FIB-4 Index for Liver Fibrosis",
                    "purpose": "Estimate liver fibrosis risk.",
                    "parameter_names": ["age", "ast", "alt", "platelet_count"],
                },
                {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "purpose": "Estimate stroke risk in atrial fibrillation.",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                },
            ],
            "question_selection_candidates": [
                {
                    "pmid": "2",
                    "title": "FIB-4 Index for Liver Fibrosis",
                    "purpose": "Estimate liver fibrosis risk.",
                    "parameter_names": ["age", "ast", "alt", "platelet_count"],
                },
                {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "purpose": "Estimate stroke risk in atrial fibrillation.",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                },
            ],
            "retrieval_batches": [],
            "risk_hints": [],
            "bm25_raw_top5": [],
            "vector_raw_top5": [],
        }
        agent._assess_candidate_execution_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection should not pre-run candidate execution gates")
        )
        agent._select_tool_for_question = lambda current_job, retrieved, **kwargs: {
            "pmid": "1",
            "title": "CHADS2 Stroke Risk Calculator",
            "purpose": "Estimate stroke risk in atrial fibrillation.",
            "parameter_names": ["hypertension", "diabetes", "stroke_history"],
            "reason": "AF stroke-risk calculator matches the question.",
            "raw_response": "",
        }
        executed_pmids: list[str] = []
        agent._execute_calculator = lambda current_job, pmid, **kwargs: executed_pmids.append(pmid) or {
            "pmid": pmid,
            "title": "CHADS2 Stroke Risk Calculator",
            "status": "completed",
            "result": 3,
            "final_text": "CHADS2 score is 3.",
        }

        result = agent._run_question(job)

        self.assertEqual(executed_pmids, ["1"])
        self.assertEqual(result["selected_tool"]["pmid"], "1")
        self.assertEqual(result["execution"]["pmid"], "1")
        self.assertEqual(result["execution"]["result"], 3)
        self.assertEqual([item["pmid"] for item in result["executions"]], ["1"])
        self.assertEqual(result["selection_decisions"], [])
        self.assertEqual(result["eligible_tools"], [])

    def test_run_question_uses_parent_selected_tool_without_retrieval_or_reselection(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="question",
            text="Atrial fibrillation patient with hypertension, diabetes, and prior TIA. Which stroke score applies?",
            selected_tool_pmid="1",
            selected_tool={
                "pmid": "1",
                "title": "CHADS2 Stroke Risk Calculator",
                "purpose": "Estimate stroke risk in atrial fibrillation.",
                "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                "reason": "Calculator PMID selected by clinical_assisstment.",
            },
            dispatch_query_text="parent dispatched atrial fibrillation stroke risk query",
            selection_context={
                "retrieval_batches": [{"channel": "coarse", "result_count": 1}],
                "retrieved_tools": [
                    {
                        "pmid": "1",
                        "title": "CHADS2 Stroke Risk Calculator",
                        "purpose": "Estimate stroke risk in atrial fibrillation.",
                        "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                    }
                ],
                "candidate_ranking": [
                    {
                        "pmid": "1",
                        "title": "CHADS2 Stroke Risk Calculator",
                        "purpose": "Estimate stroke risk in atrial fibrillation.",
                        "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                    }
                ],
                "selection_candidates": [
                    {
                        "pmid": "1",
                        "title": "CHADS2 Stroke Risk Calculator",
                        "purpose": "Estimate stroke risk in atrial fibrillation.",
                        "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                        "query_text": "atrial fibrillation hypertension diabetes TIA",
                    }
                ],
                "selection_mode": "parent_selected",
                "dispatch_query_text": "parent dispatched atrial fibrillation stroke risk query",
            },
        )
        agent._retrieve_and_rank_candidates = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preselected execution should not trigger retrieval")
        )
        agent._select_tool_for_question = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preselected execution should not trigger question reselection")
        )
        executed: list[tuple[str, dict[str, object] | None]] = []
        agent._execute_calculator = (
            lambda current_job, pmid, **kwargs: executed.append((pmid, kwargs.get("selected_candidate"))) or {
                "pmid": pmid,
                "title": "CHADS2 Stroke Risk Calculator",
                "status": "completed",
                "result": 3,
                "final_text": "CHADS2 score is 3.",
            }
        )

        result = agent._run_question(job)

        self.assertEqual([item[0] for item in executed], ["1"])
        self.assertEqual(executed[0][1]["pmid"], "1")
        self.assertEqual(result["selected_tool"]["pmid"], "1")
        self.assertEqual(result["execution"]["pmid"], "1")
        self.assertEqual(result["dispatch_query_text"], "parent dispatched atrial fibrillation stroke risk query")
        self.assertEqual(result["selection_mode"], "parent_selected")
        self.assertEqual(result["candidate_ranking"][0]["pmid"], "1")

    def test_run_question_reports_skipped_when_no_candidate_is_selected(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="question",
            text="Atrial fibrillation patient asking about stroke risk.",
        )
        agent._retrieve_and_rank_candidates = lambda _: {
            "retrieved_tools": [
                {
                    "pmid": "2",
                    "title": "FIB-4 Index for Liver Fibrosis",
                    "purpose": "Estimate liver fibrosis risk.",
                    "parameter_names": ["age", "ast", "alt", "platelet_count"],
                }
            ],
            "question_selection_candidates": [
                {
                    "pmid": "2",
                    "title": "FIB-4 Index for Liver Fibrosis",
                    "purpose": "Estimate liver fibrosis risk.",
                    "parameter_names": ["age", "ast", "alt", "platelet_count"],
                }
            ],
            "retrieval_batches": [],
            "risk_hints": [],
            "bm25_raw_top5": [],
            "vector_raw_top5": [],
        }
        agent._assess_candidate_execution_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection should not pre-run candidate execution gates")
        )
        agent._select_tool_for_question = lambda current_job, retrieved, **kwargs: {
            "pmid": "",
            "title": "",
            "purpose": "",
            "parameter_names": [],
            "model_parameter_names": [],
            "model_selected_tool_id": None,
            "fallback_used": False,
            "reason": "No valid calculator was selected from the second-stage candidate pool.",
            "raw_response": "",
        }
        agent._execute_calculator = lambda current_job, pmid, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"question mode should not execute calculator {pmid} when no tool is selected")
        )

        result = agent._run_question(job)

        self.assertEqual(result["selection_decisions"], [])
        self.assertEqual(result["selected_tool"]["pmid"], "")
        self.assertEqual(result["execution"]["status"], "skipped")
        self.assertIn("second-stage candidate pool", result["execution"]["final_text"])

    def test_run_patient_note_executes_only_selected_candidate_after_selection(self) -> None:
        agent = self._build_agent()
        job = ClinicalToolJob(
            mode="patient_note",
            text="Atrial fibrillation patient with hypertension, diabetes, and prior TIA.",
        )
        agent._retrieve_and_rank_candidates = lambda _: {
            "retrieved_tools": [
                {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "purpose": "Estimate stroke risk in atrial fibrillation.",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                },
                {
                    "pmid": "2",
                    "title": "Auxiliary Risk Calculator",
                    "purpose": "Estimate another risk.",
                    "parameter_names": ["age"],
                },
            ],
            "question_selection_candidates": [
                {
                    "pmid": "1",
                    "title": "CHADS2 Stroke Risk Calculator",
                    "purpose": "Estimate stroke risk in atrial fibrillation.",
                    "parameter_names": ["hypertension", "diabetes", "stroke_history"],
                },
                {
                    "pmid": "2",
                    "title": "Auxiliary Risk Calculator",
                    "purpose": "Estimate another risk.",
                    "parameter_names": ["age"],
                },
            ],
            "retrieval_batches": [],
            "risk_hints": ["stroke risk"],
            "bm25_raw_top5": [],
            "vector_raw_top5": [],
        }
        agent._assess_candidate_execution_gate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection should not pre-run candidate execution gates")
        )
        agent._select_tool_for_question = lambda current_job, retrieved, **kwargs: {
            "pmid": "1",
            "title": "CHADS2 Stroke Risk Calculator",
            "purpose": "Estimate stroke risk in atrial fibrillation.",
            "parameter_names": ["hypertension", "diabetes", "stroke_history"],
            "reason": "Most relevant stroke-risk score.",
            "raw_response": "",
        }
        executed_pmids: list[str] = []
        agent._execute_calculator = lambda current_job, pmid, **kwargs: executed_pmids.append(pmid) or {
            "pmid": pmid,
            "title": "CHADS2 Stroke Risk Calculator",
            "status": "completed",
            "result": 3,
            "final_text": "CHADS2 score is 3.",
        }

        result = agent._run_patient_note(job)

        self.assertEqual(executed_pmids, ["1"])
        self.assertEqual(result["selected_tool"]["pmid"], "1")
        self.assertEqual([item["pmid"] for item in result["executions"]], ["1"])
        self.assertEqual(result["execution"]["pmid"], "1")
        self.assertEqual(result["execution"]["result"], 3)


if __name__ == "__main__":
    unittest.main()
