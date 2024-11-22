from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from ._generated import (
    CreateFilterTriggerInput,
    GenericWebhookActionInput,
    NotificationActionInput,
    TriggeredActionConfig,
)
from .actions import ActionType
from .scopes import ScopeType, _ScopeInfo

if TYPE_CHECKING:
    from .automations import NewAutomation, PreparedAutomation, _ActionInputT
    from .events import EventType

T = TypeVar("T")


_SCOPE_TYPE_MAP: dict[str, ScopeType] = {
    "Project": ScopeType.PROJECT,
    "ArtifactCollection": ScopeType.ARTIFACT_COLLECTION,
    "ArtifactPortfolio": ScopeType.ARTIFACT_COLLECTION,
    "ArtifactSequence": ScopeType.ARTIFACT_COLLECTION,
}
"""Mapping of `__typename`s to automation scope types."""


def get_scope_type(obj: Any) -> ScopeType:
    """Discriminator callable to get the scope type from an object."""
    from wandb.apis import public

    # Accept and handle "public API" types that users may already be familiar with
    if isinstance(obj, (public.ArtifactCollection, public.Project)):
        return _SCOPE_TYPE_MAP[type(obj).__name__]

    # ... or decoded JSON dicts
    if isinstance(obj, Mapping) and (typename := obj.get("__typename")):
        return _SCOPE_TYPE_MAP[typename]

    # ... or Pydantic models with a `typename__` attribute
    if isinstance(obj, _ScopeInfo):
        return _SCOPE_TYPE_MAP[obj.typename__]

    # ... as a last resort, infer from the prefix of the base64-encoded ID
    if isinstance(obj, Mapping) and (id_ := obj.get("id")):
        decoded_id = base64.b64decode(id_).decode("utf-8")
        type_name, *_ = decoded_id.split(":")
        return _SCOPE_TYPE_MAP[type_name]

    raise ValueError(f"Cannot determine scope type of {type(obj)!r} object")


def get_event_type(obj: Any) -> EventType:
    from .events import (
        EventType,
        OnAddArtifactAlias,
        OnCreateArtifact,
        OnLinkArtifact,
        OnRunMetric,
    )

    if isinstance(obj, OnCreateArtifact):
        return EventType.CREATE_ARTIFACT
    if isinstance(obj, OnLinkArtifact):
        return EventType.LINK_MODEL
    if isinstance(obj, OnAddArtifactAlias):
        return EventType.ADD_ARTIFACT_ALIAS
    if isinstance(obj, OnRunMetric):
        return EventType.RUN_METRIC
    raise ValueError(f"Cannot determine event type of {type(obj)!r} object")


ACTION_TYPE_MAP: dict[str, ActionType] = {
    "NotificationAction": ActionType.NOTIFICATION,
    "GenericWebhookAction": ActionType.GENERIC_WEBHOOK,
    "QueueJobAction": ActionType.QUEUE_JOB,
}
"""Mapping of GraphQL `__typename`s to automation action types."""


def get_action_type(obj: Any) -> ActionType:
    """Return the `ActionType` associated with the automation ActionInput."""
    if isinstance(obj, NotificationActionInput):
        return ActionType.NOTIFICATION
    if isinstance(obj, GenericWebhookActionInput):
        return ActionType.GENERIC_WEBHOOK
    raise ValueError(f"Cannot determine action type of {type(obj)!r} object")


def prepare_action_config(obj: _ActionInputT) -> TriggeredActionConfig:
    """Return a `TriggeredActionConfig` as required in the input schema of CreateFilterTriggerInput."""
    if isinstance(obj, NotificationActionInput):
        return TriggeredActionConfig(notification_action_input=obj)
    if isinstance(obj, GenericWebhookActionInput):
        return TriggeredActionConfig(generic_webhook_action_input=obj)
    raise ValueError(f"Cannot prepare action config from {type(obj)!r} object")


def prepare_create_trigger_input(
    obj: PreparedAutomation | NewAutomation,
    **updates: Any,
) -> CreateFilterTriggerInput:
    from .automations import PreparedAutomation

    # Apply any updates to the properties of the automation
    prepared = PreparedAutomation.model_validate({**obj.model_dump(), **updates})

    # Prepare the input as required for the GraphQL request
    return CreateFilterTriggerInput(
        name=prepared.name,
        description=prepared.description,
        enabled=prepared.enabled,
        client_mutation_id=prepared.client_mutation_id,
        scope_type=get_scope_type(prepared.scope),
        scope_id=prepared.scope.id,
        triggering_event_type=get_event_type(prepared.event),
        event_filter=prepared.event.filter,
        triggered_action_type=get_action_type(prepared.action),
        triggered_action_config=prepare_action_config(prepared.action),
    )
