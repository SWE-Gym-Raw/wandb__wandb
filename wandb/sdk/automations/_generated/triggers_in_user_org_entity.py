# Generated by ariadne-codegen
# Source: tools/pydantic_codegen/queries-automations.graphql

from __future__ import annotations

from pydantic import Field

from .base_model import BaseModel
from .fragments import PaginatedProjectTriggers


class TriggersInUserOrgEntity(BaseModel):
    viewer: TriggersInUserOrgEntityViewer | None


class TriggersInUserOrgEntityViewer(BaseModel):
    organizations: list[TriggersInUserOrgEntityViewerOrganizations]


class TriggersInUserOrgEntityViewerOrganizations(BaseModel):
    org_entity: TriggersInUserOrgEntityViewerOrganizationsOrgEntity | None = Field(
        alias="orgEntity"
    )


class TriggersInUserOrgEntityViewerOrganizationsOrgEntity(BaseModel):
    projects: TriggersInUserOrgEntityViewerOrganizationsOrgEntityProjects | None


class TriggersInUserOrgEntityViewerOrganizationsOrgEntityProjects(
    PaginatedProjectTriggers
):
    pass


TriggersInUserOrgEntity.model_rebuild()
TriggersInUserOrgEntityViewer.model_rebuild()
TriggersInUserOrgEntityViewerOrganizations.model_rebuild()
TriggersInUserOrgEntityViewerOrganizationsOrgEntity.model_rebuild()
