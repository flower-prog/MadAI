from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["research_agent", "coder_agent", "browser_agent", "expert_agent"]

# Avoid importing src.agents.agents at package import time.
# Several tools import `src.agents.llm`, which would otherwise trigger
# `src.agents.__init__` -> `src.agents.agents` -> `src.tools` and create a
# circular import. We lazily load the agents only when the attributes are
# accessed (e.g. `from src.agents import research_agent`).
if TYPE_CHECKING:
    from .agents import browser_agent, coder_agent, expert_agent, research_agent


def __getattr__(name: str):
    if name in __all__:
        from . import agents as _agents

        return getattr(_agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
