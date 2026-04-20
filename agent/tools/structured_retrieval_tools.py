from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.retrieval import (
    StructuredRetrievalDocument,
    StructuredRetriever,
    build_structured_query_text,
    create_structured_retriever,
)

from .execution_tools import tool


class StructuredRetrievalTool(StructuredRetriever):
    @tool(
        name="structured_bm25_retriever",
        description=(
            "Recall document identifiers from structured_case with BM25. "
            "Use this when the downstream agent wants a deterministic sparse retrieval pass."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
            "candidate_ids": "list[str] | set[str] | None",
            "include_scores": "bool",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
            "top_k": (
                "top_k",
                "clinical_tool_job.top_k",
            ),
        },
    )
    def retrieve_with_bm25(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        return self.retrieve_from_structured_case(
            structured_case,
            top_k=top_k,
            candidate_ids=candidate_ids,
            backend="bm25",
            include_scores=include_scores,
        )

    @tool(
        name="structured_vector_retriever",
        description=(
            "Recall document identifiers from structured_case with vector retrieval. "
            "Use this when semantic matching is needed before downstream parameter alignment."
        ),
        input_schema={
            "structured_case": {
                "raw_text": "str",
                "case_summary": "str",
                "problem_list": "list[str]",
                "known_facts": "list[str]",
            },
            "top_k": "int",
            "candidate_ids": "list[str] | set[str] | None",
            "include_scores": "bool",
        },
        state_fields={
            "structured_case": (
                "structured_case",
                "structured_case_json",
                "clinical_tool_job.structured_case",
            ),
            "top_k": (
                "top_k",
                "clinical_tool_job.top_k",
            ),
        },
    )
    def retrieve_with_vector(
        self,
        structured_case: Mapping[str, Any],
        *,
        top_k: int = 10,
        candidate_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        return self.retrieve_from_structured_case(
            structured_case,
            top_k=top_k,
            candidate_ids=candidate_ids,
            backend="vector",
            include_scores=include_scores,
        )


def create_structured_retrieval_tool(
    catalog: Any,
    *,
    bm25_retriever: Any | None,
    vector_retriever: Any | None = None,
    query_builder: Any | None = None,
    default_backend: str = "hybrid",
    id_field: str = "document_id",
) -> StructuredRetrievalTool:
    return StructuredRetrievalTool(
        catalog,
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        query_builder=query_builder,
        default_backend=default_backend,
        id_field=id_field,
    )


__all__ = [
    "StructuredRetrievalDocument",
    "StructuredRetrievalTool",
    "build_structured_query_text",
    "create_structured_retrieval_tool",
    "create_structured_retriever",
]
