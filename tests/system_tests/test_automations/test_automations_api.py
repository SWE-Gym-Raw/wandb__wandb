from __future__ import annotations

from collections import deque
from typing import Iterator

import wandb
from pytest import MonkeyPatch, fixture, raises
from wandb import Artifact
from wandb.apis import public
from wandb.apis.public.integrations import WebhookIntegration
from wandb.sdk.automations import Automation
from wandb.sdk.automations.actions import DoWebhook
from wandb.sdk.automations.events import (
    MetricFilter,
    OnCreateArtifact,
    OnRunMetric,
    RunEvent,
    RunFilter,
    RunMetricFilter,
)


@fixture(scope="session")
def user(session_mocker, backend_fixture_factory) -> Iterator[str]:
    """A session-scoped user that overrides the default `user` fixture from the root-level `conftest.py`."""
    username = backend_fixture_factory.make_user()
    with MonkeyPatch.context() as session_monkeypatch:
        session_monkeypatch.setenv("WANDB_API_KEY", username)
        session_monkeypatch.setenv("WANDB_ENTITY", username)
        session_monkeypatch.setenv("WANDB_USERNAME", username)
        yield username


@fixture(scope="session")
def api(user) -> wandb.Api:
    """A redefined, session-scoped `Api` fixture for tests in this module.

    Note that this overrides the default `api` fixture from the root-level
    `conftest.py`.  This is necessary for any tests in these subfolders,
    since the default `api` fixture is function-scoped, meaning it does not
    play well with other session-scoped fixtures.
    """
    return wandb.Api()


@fixture(scope="session")
def project(user, api) -> public.Project:
    """A wandb Project for tests in this module."""
    # Create the project first if it doesn't exist yet
    name = "test-project"
    api.create_project(name=name, entity=user)
    return api.project(name=name, entity=user)


@fixture(scope="session")
def artifact(user, project) -> Artifact:
    with wandb.init(entity=user, project=project.name) as run:
        artifact = Artifact("test-artifact", "dataset")
        logged_artifact = run.log_artifact(artifact)
        return logged_artifact.wait()


@fixture(scope="session")
def artifact_collection(artifact, api) -> public.ArtifactCollection:
    """A test ArtifactCollection for tests in this module."""
    return api.artifact(name=artifact.qualified_name, type=artifact.type).collection


@fixture(scope="session")
def webhook_integration(user, api) -> WebhookIntegration:
    from wandb_gql import gql

    # HACK: Set up a placeholder webhook integration and return it
    # At the time of testing/implementation, this is the action with
    # the lowest setup overhead and, if needed, probably least difficult
    # to patch/mock/stub/spy/intercept
    gql_mutation = gql(
        """
        mutation CreateGenericWebhookIntegration(
            $entityName: String!,
            $url: String!,
            $name: String!,
        ) {
            createGenericWebhookIntegration(
                input: {
                    entityName: $entityName,
                    urlEndpoint: $url,
                    name: $name,
                }
            ) {
                integration {
                    __typename
                    ... on GenericWebhookIntegration {
                        id
                        name
                        urlEndpoint
                        createdAt
                    }
                }
            }
        }
        """
    )
    data = api.client.execute(
        gql_mutation,
        variable_values={
            "entityName": api.default_entity,
            "url": "test-url",
            "name": "my-webhook",
        },
    )
    integration_data = data["createGenericWebhookIntegration"]["integration"]
    integration = WebhookIntegration.model_validate(integration_data)

    # Consistency check: the integration should be there now
    assert len(list(api.integrations(kind="webhook"))) == 1

    return integration


# ------------------------------------------------------------------------------
def test_no_initial_automations(api: wandb.Api):
    """No automations should be fetched by the API prior to creating any."""
    assert list(api.automations()) == []


def test_no_initial_integrations(user, api: wandb.Api):
    """No automations should be fetched by the API prior to creating any."""
    assert list(api.integrations(kind="slack")) == []
    assert list(api.integrations(kind="webhook")) == []


def test_no_initial_slack_integration(user, api: wandb.Api):
    with raises(ValueError, match="No Slack integration found"):
        _ = list(api.slack_integration(user))


def test_new_create_artifact_automation(
    request, artifact_collection, webhook_integration, api: wandb.Api
):
    # To ensure uniqueness, name the automation the fully qualified name of the current test
    automation_name = request.node.name

    event = OnCreateArtifact(
        scope=artifact_collection,
    )
    action = DoWebhook(
        integration_id=webhook_integration.id,
        request_payload={},
    )

    automation = api.create_automation(
        (event >> action),
        name=automation_name,
        description="longer description here",
    )

    # TODO: Go beyond smoke tests
    assert isinstance(automation, Automation)

    # We should be able to fetch the automation by name (optionally filtering by entity)
    entity_name = artifact_collection.entity
    assert len(list(api.automations(entity=entity_name, name=automation_name))) == 1
    assert len(list(api.automations(name=automation_name))) == 1

    # Delete the automation for good measure
    api.delete_automation(automation)
    assert len(list(api.automations(entity=entity_name, name=automation_name))) == 0
    assert len(list(api.automations(name=automation_name))) == 0


def test_new_run_metric_automation(request, project, webhook_integration, api):
    # To ensure uniqueness, name the automation the fully qualified name of the current test
    automation_name = request.node.name

    expected_filter = RunMetricFilter(
        run_filter=RunFilter(
            other=[{"display_name": {"$contains": "my-run"}}],
        ),
        metric_filter=MetricFilter(
            name="my-metric",
            window_size=5,
            agg_op="AVERAGE",
            cmp_op="$gt",
            threshold=0,
        ),
    )

    event = OnRunMetric(
        scope=project,
        filter=(RunEvent.metric("my-metric").mean(5) > 0)
        & (RunEvent.name.contains("my-run")),
    )
    action = DoWebhook(
        integration_id=webhook_integration.id,
        request_payload={},
    )

    automation = api.create_automation(
        (event >> action),
        name=automation_name,
        description="longer description here",
    )

    assert isinstance(automation, Automation)
    assert automation.event.filter == expected_filter

    # We should be able to fetch the automation by name (optionally filtering by entity)
    entity_name = project.entity
    assert len(list(api.automations(entity=entity_name, name=automation_name))) == 1
    assert len(list(api.automations(name=automation_name))) == 1

    # Delete the automation for good measure
    api.delete_automation(automation)
    assert len(list(api.automations(entity=entity_name, name=automation_name))) == 0
    assert len(list(api.automations(name=automation_name))) == 0


class TestPaginatedAutomations:
    @fixture(scope="class")
    def total_projects(self) -> int:
        return 10

    @fixture(scope="class")
    def page_size(self) -> int:
        return 1

    @fixture(scope="class")
    def setup_paginated_automations(
        self,
        user: str,
        api: wandb.Api,
        webhook_integration: WebhookIntegration,
        total_projects: int,
    ):
        # NOTE: For now, pagination is per project, NOT per automation, so
        # to test pagination, we'll create each automation in a separate project.
        #
        # UPDATE THIS in the future if we switch to per-automation pagination.
        automations = deque()
        for i in range(total_projects):
            # Create the placeholder project for the automation
            project_name = f"project-{i}"
            api.create_project(name=project_name, entity=user)
            project = api.project(name=project_name, entity=user)

            # Create the actual automation
            event = OnCreateArtifact(
                scope=project,
            )
            action = DoWebhook(
                integration_id=webhook_integration.id,
                request_payload={},
            )
            automation = api.create_automation(
                (event >> action),
                name=f"automation-{i}",
                description="longer description here",
            )

            # Retain for later cleanup
            automations.append(automation)

        yield

        # This particular fixture is deliberately class-scoped, but clean up the automations for good measure
        for automation in automations:
            api.delete_automation(automation)

    def test_paginated_automations(
        self,
        mocker,
        user,
        api: wandb.Api,
        setup_paginated_automations,
        total_projects,
        page_size,
    ):
        # Spy on the client method that makes the GQL request.  Not ideal, but it may have to do for now
        client_spy = mocker.spy(api.client, "execute")

        # Fetch the automations
        _ = list(api.automations(per_page=page_size))

        # Check that the number of GQL requests is what's expected from the pagination params
        expected_page_count = total_projects // page_size + 1
        assert client_spy.call_count == expected_page_count
