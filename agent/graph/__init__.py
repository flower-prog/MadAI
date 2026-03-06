from typing import TYPE_CHECKING

__all__ = ["build_graph", "build_graph_with_memory"]

# Avoid importing builder at package import time to prevent circular imports.
if TYPE_CHECKING:
    from .builder import build_graph, build_graph_with_memory


def __getattr__(name: str):
    if name in __all__:
        from . import builder as _builder
        return getattr(_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
