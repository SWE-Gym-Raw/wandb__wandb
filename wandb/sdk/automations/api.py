from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from itertools import chain, product
from typing import Literal, Union

from more_itertools import always_iterable
from pydantic import Field, TypeAdapter
from tqdm.auto import tqdm
from typing_extensions import Annotated

from wandb import util
from wandb.sdk.automations._typing import Base64Id, TypenameField
from wandb.sdk.automations._utils import (
    _get_api,
    get_orgs_info,
    iter_entity_project_pairs,
)
from wandb.sdk.automations.actions import AnyAction
from wandb.sdk.automations.base import Base
from wandb.sdk.automations.events import AnyEvent

reset_path = util.vendor_setup()

from wandb_gql import gql  # noqa: E402

_FETCH_ORG_TRIGGERS = gql(
    """
    query TriggersInViewerOrgs ($entityName: String) {
        viewer(entityName: $entityName) {
            # __typename
            # id
            # username
            organizations {
                # __typename
                # id
                # name
                # orgType
                orgEntity {
                    # __typename
                    # name
                    # id
                    projects {
                        edges {
                            node {
                                # __typename
                                # id
                                # name
                                triggers {
                                    id
                                    createdAt
                                    createdBy {id username}
                                    updatedAt
                                    name
                                    description
                                    enabled
                                    triggeringCondition {
                                        __typename
                                        ... on FilterEventTriggeringCondition {
                                            eventType
                                            filter
                                        }
                                    }
                                    triggeredAction {
                                        __typename
                                        ... on QueueJobTriggeredAction {
                                            template
                                            queue {
                                                __typename
                                                id
                                                name
                                            }
                                        }
                                        ... on NotificationTriggeredAction {
                                            title
                                            message
                                            severity
                                            integration {
                                                __typename
                                                ... on GenericWebhookIntegration {
                                                    id
                                                    name
                                                    urlEndpoint
                                                    secretRef
                                                    accessTokenRef
                                                    createdAt
                                                }
                                                ... on GitHubOAuthIntegration {
                                                    id
                                                }
                                                ... on SlackIntegration {
                                                    id
                                                    teamName
                                                    channelName
                                                }
                                            }
                                        }
                                        ... on GenericWebhookTriggeredAction {
                                            requestPayload
                                            integration {
                                                __typename
                                                ... on GenericWebhookIntegration {
                                                    id
                                                    name
                                                    urlEndpoint
                                                    secretRef
                                                    accessTokenRef
                                                    createdAt
                                                }
                                                ... on GitHubOAuthIntegration {
                                                    id
                                                }
                                                ... on SlackIntegration {
                                                    id
                                                    teamName
                                                    channelName
                                                }
                                            }
                                        }
                                    }
                                    scope {
                                        __typename
                                        ... on ArtifactPortfolio {id name}
                                        ... on ArtifactSequence {id name}
                                        ... on Project {id name}
                                    }
                                }
                            }
                        }
                    }
                }
                teams {
                    # __typename
                    # id
                    # name
                    projects {
                        edges {
                            node {
                                # __typename
                                # id
                                # name
                                triggers {
                                    id
                                    createdAt
                                    createdBy {id username}
                                    updatedAt
                                    name
                                    description
                                    enabled
                                    scope {
                                        __typename
                                        ... on ArtifactPortfolio {id name}
                                        ... on ArtifactSequence {id name}
                                        ... on Project {id name}
                                    }
                                    triggeringCondition {
                                        __typename
                                        ... on FilterEventTriggeringCondition {
                                            eventType
                                            filter
                                        }
                                    }
                                    triggeredAction {
                                        __typename
                                        ... on QueueJobTriggeredAction {
                                            template
                                            queue {
                                                __typename
                                                id
                                                name
                                            }
                                        }
                                        ... on NotificationTriggeredAction {
                                            title
                                            message
                                            severity
                                            integration {
                                                __typename
                                                ... on GenericWebhookIntegration {
                                                    id
                                                    urlEndpoint
                                                    name
                                                    secretRef
                                                    accessTokenRef
                                                    createdAt
                                                }
                                                ... on GitHubOAuthIntegration {
                                                    id
                                                }
                                                ... on SlackIntegration {
                                                    id
                                                    teamName
                                                    channelName
                                                }
                                            }
                                        }
                                        ... on GenericWebhookTriggeredAction {
                                            requestPayload
                                            integration {
                                                __typename
                                                ... on GenericWebhookIntegration {
                                                    id
                                                    urlEndpoint
                                                    name
                                                    secretRef
                                                    accessTokenRef
                                                    createdAt
                                                }
                                                ... on GitHubOAuthIntegration {
                                                    id
                                                }
                                                ... on SlackIntegration {
                                                    id
                                                    teamName
                                                    channelName
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """
)

_FETCH_PROJECT_TRIGGERS = gql(
    """
    query FetchProjectTriggers(
        $projectName: String!,
        $entityName: String!,
    ) {
        project(
            name: $projectName,
            entityName: $entityName,
        ) {
            triggers {
                id
                name
                enabled
                createdAt
                createdBy {id username}
                description
                scope {
                    __typename
                    ... on Project {
                        id
                        name
                    }
                    ... on ArtifactSequence {
                        id
                        name
                    }
                    ... on ArtifactPortfolio {
                        id
                        name
                    }
                }
                triggeringCondition {
                    __typename
                    ... on FilterEventTriggeringCondition {
                        eventType
                        filter
                    }
                }
                triggeredAction {
                    __typename
                    ... on QueueJobTriggeredAction {
                        queue {
                            __typename
                            id
                            name
                        }
                        template
                    }
                    ... on NotificationTriggeredAction {
                        title
                        message
                        severity
                        integration {
                            __typename
                            id
                            ... on GenericWebhookIntegration {
                                id
                                urlEndpoint
                                name
                                secretRef
                                accessTokenRef
                                createdAt
                            }
                            ... on GitHubOAuthIntegration {
                                id
                            }
                            ... on SlackIntegration {
                                id
                                teamName
                                channelName
                            }
                        }
                    }
                    ... on GenericWebhookTriggeredAction {
                        integration {
                            __typename
                            ... on GenericWebhookIntegration {
                                id
                                name
                                urlEndpoint
                                accessTokenRef
                                secretRef
                                createdAt
                            }
                        }
                        requestPayload
                    }
                }
            }
        }
    }
    """
)

# load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


# ------------------------------------------------------------------------------
class User(Base):
    id: Base64Id
    username: str


# Scopes
class ArtifactPortfolioScope(Base):
    typename__: TypenameField[Literal["ArtifactPortfolio"]]
    id: Base64Id
    name: str


class ArtifactSequenceScope(Base):
    typename__: TypenameField[Literal["ArtifactSequence"]]
    id: Base64Id
    name: str


class ProjectScope(Base):
    typename__: TypenameField[Literal["Project"]]
    id: Base64Id
    name: str


class EntityScope(Base):
    typename__: TypenameField[Literal["Entity"]]
    id: Base64Id
    name: str


AnyScope = Annotated[
    Union[ArtifactPortfolioScope, ArtifactSequenceScope, ProjectScope, EntityScope],
    Field(discriminator="typename__"),
]


class Automation(Base):
    """A defined W&B automation."""

    id: Base64Id

    name: str
    description: str | None

    created_by: User
    created_at: datetime
    updated_at: datetime | None

    scope: AnyScope

    event: AnyEvent
    action: AnyAction

    enabled: bool


class NewAutomation(Base):
    """A newly defined automation, to be prepared and sent by the client to the server."""

    name: str
    description: str | None

    scope: AnyScope

    event: AnyEvent
    action: AnyAction

    enabled: bool


AutomationsAdapter = TypeAdapter(list[Automation])


def fetch_automations() -> Iterator[Automation]:
    api = _get_api()
    data = api.client.execute(_FETCH_ORG_TRIGGERS, variable_values={"entityName": None})
    organizations = data["viewer"]["organizations"]
    entities = chain.from_iterable(
        [org["orgEntity"], *org["teams"]] for org in organizations
    )
    edges = chain.from_iterable(entity["projects"]["edges"] for entity in entities)
    projects = (edge["node"] for edge in edges)
    for proj in projects:
        yield from AutomationsAdapter.validate_python(proj["triggers"])
    # triggers = chain.from_iterable(proj["triggers"] for proj in projects)
    # return list(islice(triggers, 5))
    # # return projects


def get_automations(
    entities: Iterable[str] | str | None = "wandb_Y72QKAKNEFI3G",
    projects: Iterable[str] | str | None = "wandb-registry-model",
    # entities: Iterable[str] | str | None = None,
    # projects: Iterable[str] | str | None = None,
) -> Iterator[Automation]:
    api = _get_api()

    if (entities is None) and (projects is None):
        all_orgs = get_orgs_info()
        entity_project_pairs = iter_entity_project_pairs(all_orgs)
    elif (entities is not None) and (projects is not None):
        entity_project_pairs = product(
            always_iterable(entities), always_iterable(projects)
        )
    else:
        raise NotImplementedError(
            "Filtering on specific entity or project names not yet implemented"
        )

    for entity, project in tqdm(
        entity_project_pairs,
        desc="Fetching automations from entity-project pairs",
    ):
        params = {"entityName": entity, "projectName": project}
        data = api.client.execute(_FETCH_PROJECT_TRIGGERS, variable_values=params)
        yield from AutomationsAdapter.validate_python(data["project"]["triggers"])


# TODO: WIP
def new_automation(
    event_and_action: tuple[AnyEvent, AnyAction] | None = None,
    /,
    *,
    name: str,
    description: str | None = None,
    event: AnyEvent | None = None,
    action: AnyAction | None = None,
    enabled: bool = True,
) -> NewAutomation:
    if event_and_action is not None:
        event, action = event_and_action

    return NewAutomation(
        name=name,
        description=description,
        event=event,
        action=action,
        enabled=enabled,
    )
