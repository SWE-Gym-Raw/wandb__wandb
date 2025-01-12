"""Scopes in which a W&B Automation can be triggered."""

from __future__ import annotations

from typing import Literal

from ._base import Base, Base64Id, Typename
from ._generated import TriggerScopeType

# NOTE: Shorter name for readability, defined in a public module for easier access
ScopeType = TriggerScopeType
"""The type of scope that triggers an automation."""


class _ScopeInfo(Base):
    typename__: Typename[str]
    id: Base64Id
    name: str | None = None


class ArtifactCollectionScope(_ScopeInfo):
    """The ID and name of the ArtifactCollection scope of an automation."""

    typename__: Typename[
        Literal["ArtifactSequence", "ArtifactPortfolio", "ArtifactCollection"]
    ] = "ArtifactCollection"


class ProjectScope(_ScopeInfo):
    """The ID and name of the Project scope of an automation."""

    typename__: Typename[Literal["Project"]] = "Project"
