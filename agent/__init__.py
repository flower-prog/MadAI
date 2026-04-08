from typing import TYPE_CHECKING

__all__ = [
    "ClinicalToolAgent",
    "RiskCalcCatalog",
    "RiskCalcDocument",
    "RiskCalcExecutionTool",
    "RiskCalcExecutor",
    "RiskCalcRegistration",
    "RiskCalcRegistry",
    "RiskCalcRetrievalTool",
    "create_execution_tool",
    "create_retrieval_tool",
    "prewarm_clinical_tool_job",
    "run_agent_workflow",
    "run_clinical_tool_workflow",
    "run_workflow",
]

if TYPE_CHECKING:
    from .clinical_tool_agent import ClinicalToolAgent, prewarm_clinical_tool_job
    from .tools import (
        RiskCalcCatalog,
        RiskCalcDocument,
        RiskCalcExecutionTool,
        RiskCalcExecutor,
        RiskCalcRegistration,
        RiskCalcRegistry,
        RiskCalcRetrievalTool,
        create_execution_tool,
        create_retrieval_tool,
    )
    from .workflow import run_agent_workflow, run_clinical_tool_workflow, run_workflow


def __getattr__(name: str):
    if name in {"ClinicalToolAgent", "prewarm_clinical_tool_job"}:
        from . import clinical_tool_agent as _clinical_tool_agent

        return getattr(_clinical_tool_agent, name)

    if name in {
        "RiskCalcCatalog",
        "RiskCalcDocument",
        "RiskCalcExecutionTool",
        "RiskCalcExecutor",
        "RiskCalcRegistration",
        "RiskCalcRegistry",
        "RiskCalcRetrievalTool",
        "create_execution_tool",
        "create_retrieval_tool",
    }:
        from . import tools as _tools

        return getattr(_tools, name)

    if name in {"run_agent_workflow", "run_clinical_tool_workflow", "run_workflow"}:
        from . import workflow as _workflow

        return getattr(_workflow, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
