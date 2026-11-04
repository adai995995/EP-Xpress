"""EP-Xpress: inference-oriented, host-staged expert parallelism.

Importing the package does not initialize CUDA or compile extensions.
"""

from .config import Config

__version__ = "0.1.0"
__all__ = ["Config", "Context", "DispatchBatch", "Pending", "RouteHandle",
           "NCCLCommHandle", "get_nccl_comm_handle"]


def __getattr__(name):
    if name in {"NCCLCommHandle", "get_nccl_comm_handle"}:
        from . import comm
        return getattr(comm, name)
    if name in {"Context", "DispatchBatch", "Pending", "RouteHandle"}:
        from . import context
        return getattr(context, name)
    raise AttributeError(name)
