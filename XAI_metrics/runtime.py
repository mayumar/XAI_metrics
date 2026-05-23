"""Runtime helpers for optional dependencies."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_optional_dependency(module_name: str) -> ModuleType:
    """Import an optional dependency only when it is actually needed."""
    return import_module(module_name)
