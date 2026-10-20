"""Shared exception types for v4sup."""
from __future__ import annotations


class SupervisorError(Exception):
    """Base class for all v4sup errors."""


class RegistryError(SupervisorError):
    """The phase registry JSON is malformed or internally inconsistent."""


class WorkspaceEscapeError(SupervisorError):
    """A path resolves outside the declared workspace root (T44)."""
