"""Explicit synchronous CG/PCG policies for device-resident systems."""

from .api import prepare, prepare_host, solve, solve_into
from .prepared import PreparedSolver
from .workspace import Workspace

__all__ = [
    "PreparedSolver",
    "Workspace",
    "prepare",
    "prepare_host",
    "solve",
    "solve_into",
]
