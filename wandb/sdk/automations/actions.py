"""Actions that are triggered by W&B Automations."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

from pydantic import Field, JsonValue

from ._base import Base64Id, SerializedToJson, Typename
from ._generated import (
    AlertSeverity,
    GenericWebhookActionInput,
    NotificationActionFields,
    NotificationActionInput,
    QueueJobActionFields,
    SlackIntegrationFields,
    TriggeredActionType,
    WebhookActionFields,
    WebhookIntegrationFields,
)

if TYPE_CHECKING:
    from wandb.sdk import AlertLevel

if sys.version_info >= (3, 12):
    from typing import Self
else:
    from typing_extensions import Self


# NOTE: Shorter name for readability, defined in a public module for easier access
ActionType = TriggeredActionType
"""The type of action triggered by an automation."""


class LaunchJobAction(QueueJobActionFields):
    typename__: Typename[Literal["QueueJobTriggeredAction"]]
    # ...note: other fields inherited as-is from generated class


class NotificationAction(NotificationActionFields):
    typename__: Typename[Literal["NotificationTriggeredAction"]]
    integration: SlackIntegrationFields  # type: ignore[assignment]
    # ...note: other fields inherited as-is from generated class


class WebhookAction(WebhookActionFields):
    typename__: Typename[Literal["GenericWebhookTriggeredAction"]]
    integration: WebhookIntegrationFields  # type: ignore[assignment]
    # ...note: other fields inherited as-is from generated class


# ------------------------------------------------------------------------------
class DoNotification(NotificationActionInput):
    """Schema for defining a triggered notification action."""

    integration_id: Base64Id = Field(alias="integrationID")
    title: str = ""

    # Validation aliases allow arg names from previous `wandb.alert()` API
    message: str = Field(default="", validation_alias="text")
    severity: AlertSeverity = Field(
        default=AlertSeverity.INFO, validation_alias="level"
    )

    @classmethod
    def from_integration(
        cls,
        integration: SlackIntegrationFields,
        *,
        title: str = "",
        text: str = "",
        level: AlertSeverity | AlertLevel | str = AlertSeverity.INFO,
    ) -> Self:
        """Define a notification action that sends to the given (Slack) integration."""
        return cls(integration_id=integration.id, title=title, text=text, level=level)

    @classmethod
    def for_team(
        cls,
        entity: str,
        *,
        title: str = "",
        text: str = "",
        level: AlertSeverity | AlertLevel | str = AlertSeverity.INFO,
    ) -> Self:
        """Define a notification action that sends to the team's existing (Slack) integration."""
        from wandb.apis.public.api import Api

        integration = Api().slack_integration(entity)
        return cls(
            integration_id=integration.id,
            title=title,
            text=text,
            level=level,
        )


class DoWebhook(GenericWebhookActionInput):
    """Schema for defining a triggered webhook action."""

    integration_id: Base64Id = Field(alias="integrationID")
    request_payload: SerializedToJson[JsonValue]

    @classmethod
    def from_integration(
        cls,
        integration: WebhookIntegrationFields,
        *,
        request_payload: JsonValue,
    ) -> Self:
        """Define a webhook action that sends to the given (webhook) integration."""
        return cls(integration_id=integration.id, request_payload=request_payload)


# NOTE: `QueueJobActionInput` for defining a Launch job is deprecated,
# so we deliberately don't currently expose it in the API for creating automations.
