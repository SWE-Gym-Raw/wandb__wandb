# Generated by ariadne-codegen
# Source: tools/graphql_codegen/automations/

from __future__ import annotations

from pydantic import Field

from .base import GQLBase
from .fragments import CreateFilterTriggerResult


class CreateFilterTrigger(GQLBase):
    create_filter_trigger: CreateFilterTriggerResult | None = Field(
        alias="createFilterTrigger"
    )


CreateFilterTrigger.model_rebuild()
