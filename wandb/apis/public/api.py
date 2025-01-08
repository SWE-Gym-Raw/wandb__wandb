"""Use the Public API to export or update data that you have saved to W&B.

Before using this API, you'll want to log data from your script — check the
[Quickstart](https://docs.wandb.ai/quickstart) for more details.

You might use the Public API to
 - update metadata or metrics for an experiment after it has been completed,
 - pull down your results as a dataframe for post-hoc analysis in a Jupyter notebook, or
 - check your saved model artifacts for those tagged as `ready-to-deploy`.

For more on using the Public API, check out [our guide](https://docs.wandb.com/guides/track/public-api-guide).
"""

import json
import logging
import os
import urllib
from typing import Any, Dict, List, Optional

import requests
from wandb_gql import Client, gql
from wandb_gql.client import RetryError

import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.apis.public.utils import PathType, parse_org_from_registry_path
from wandb.sdk.artifacts._validators import is_artifact_registry_project
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.launch.utils import LAUNCH_DEFAULT_PROJECT
from wandb.sdk.lib import retry, runid
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.sdk.lib.gql_request import GraphQLSession

logger = logging.getLogger(__name__)


class RetryingClient:
    INFO_QUERY = gql(
        """
        query ServerInfo{
            serverInfo {
                cliVersionInfo
                latestLocalVersionInfo {
                    outOfDate
                    latestVersionString
                    versionOnThisInstanceString
                }
            }
        }
        """
    )

    def __init__(self, client: Client):
        self._server_info = None
        self._client = client

    @property
    def app_url(self):
        return util.app_url(self._client.transport.url.replace("/graphql", "")) + "/"

    @retry.retriable(
        retry_timedelta=RETRY_TIMEDELTA,
        check_retry_fn=util.no_retry_auth,
        retryable_exceptions=(RetryError, requests.RequestException),
    )
    def execute(self, *args, **kwargs):  # noqa: D102  # User not encouraged to use this class directly
        try:
            return self._client.execute(*args, **kwargs)
        except requests.exceptions.ReadTimeout:
            if "timeout" not in kwargs:
                timeout = self._client.transport.default_timeout
                wandb.termwarn(
                    f"A graphql request initiated by the public wandb API timed out (timeout={timeout} sec). "
                    f"Create a new API with an integer timeout larger than {timeout}, e.g., `api = wandb.Api(timeout={timeout + 10})` "
                    f"to increase the graphql timeout."
                )
            raise

    @property
    def server_info(self):
        if self._server_info is None:
            self._server_info = self.execute(self.INFO_QUERY).get("serverInfo")
        return self._server_info

    def version_supported(self, min_version: str) -> bool:  # noqa: D102  # User not encouraged to use this class directly
        from wandb.util import parse_version

        return parse_version(min_version) <= parse_version(
            self.server_info["cliVersionInfo"]["max_cli_version"]
        )


class Api:
    """Used for querying the W&B server.

    Args:
        overrides Optional[Dict[str, Any]]: You can set `base_url` if you are
        using a W&B server other than `https://api.wandb.ai`. You can also set
        defaults for `entity`, `project`, and `run`.

    Examples:
    ```python
    import wandb

    wandb.Api()
    ```
    """

    _HTTP_TIMEOUT = env.get_http_timeout(19)
    DEFAULT_ENTITY_QUERY = gql(
        """
        query Viewer{
            viewer {
                id
                entity
            }
        }
        """
    )

    VIEWER_QUERY = gql(
        """
        query Viewer{
            viewer {
                id
                flags
                entity
                username
                email
                admin
                apiKeys {
                    edges {
                        node {
                            id
                            name
                            description
                        }
                    }
                }
                teams {
                    edges {
                        node {
                            name
                        }
                    }
                }
            }
        }
        """
    )
    USERS_QUERY = gql(
        """
        query SearchUsers($query: String) {
            users(query: $query) {
                edges {
                    node {
                        id
                        flags
                        entity
                        admin
                        email
                        deletedAt
                        username
                        apiKeys {
                            edges {
                                node {
                                    id
                                    name
                                    description
                                }
                            }
                        }
                        teams {
                            edges {
                                node {
                                    name
                                }
                            }
                        }
                    }
                }
            }
        }
        """
    )

    CREATE_PROJECT = gql(
        """
        mutation upsertModel(
            $description: String
            $entityName: String
            $id: String
            $name: String
            $framework: String
            $access: String
            $views: JSONString
        ) {
            upsertModel(
            input: {
                description: $description
                entityName: $entityName
                id: $id
                name: $name
                framework: $framework
                access: $access
                views: $views
            }
            ) {
            project {
                id
                name
                entityName
                description
                access
                views
            }
            model {
                id
                name
                entityName
                description
                access
                views
            }
            inserted
            }
        }
    """
    )

    def __init__(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.settings = InternalApi().settings()
        _overrides = overrides or {}
        self._api_key = api_key
        if self.api_key is None and _thread_local_api_settings.cookies is None:
            wandb.login(host=_overrides.get("base_url"))
        self.settings.update(_overrides)
        if "username" in _overrides and "entity" not in _overrides:
            wandb.termwarn(
                'Passing "username" to Api is deprecated. please use "entity" instead.'
            )
            self.settings["entity"] = _overrides["username"]
        self.settings["base_url"] = self.settings["base_url"].rstrip("/")

        self._viewer = None
        self._projects = {}
        self._runs = {}
        self._sweeps = {}
        self._reports = {}
        self._default_entity = None
        self._timeout = timeout if timeout is not None else self._HTTP_TIMEOUT
        auth = None
        if not _thread_local_api_settings.cookies:
            auth = ("api", self.api_key)
        proxies = self.settings.get("_proxies") or json.loads(
            os.environ.get("WANDB__PROXIES", "{}")
        )
        self._base_client = Client(
            transport=GraphQLSession(
                headers={
                    "User-Agent": self.user_agent,
                    "Use-Admin-Privileges": "true",
                    **(_thread_local_api_settings.headers or {}),
                },
                use_json=True,
                # this timeout won't apply when the DNS lookup fails. in that case, it will be 60s
                # https://bugs.python.org/issue22889
                timeout=self._timeout,
                auth=auth,
                url="{}/graphql".format(self.settings["base_url"]),
                cookies=_thread_local_api_settings.cookies,
                proxies=proxies,
            )
        )
        self._client = RetryingClient(self._base_client)

    def create_project(self, name: str, entity: str) -> None:
        """Create a new project.

        Args:
            name: The name of the new project.
            entity: The entity of the new project.
        """
        self.client.execute(self.CREATE_PROJECT, {"entityName": entity, "name": name})

    def create_run(
        self,
        *,
        run_id: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> "public.Run":
        """Create a new run.

        Args:
            run_id: The ID to assign to the run. If not specified, W&B
                creates random ID.
            project: The project where to log the run to. If no project is specified,
                log the run to a project called "Uncategorized".
            entity: The entity that owns the project. If no entity is
                specified, log the run to the default entity.

        Returns:
            The newly created `Run`.
        """
        if entity is None:
            entity = self.default_entity
        return public.Run.create(self, run_id=run_id, project=project, entity=entity)

    def create_run_queue(
        self,
        name: str,
        type: "public.RunQueueResourceType",
        entity: Optional[str] = None,
        prioritization_mode: Optional["public.RunQueuePrioritizationMode"] = None,
        config: Optional[dict] = None,
        template_variables: Optional[dict] = None,
    ) -> "public.RunQueue":
        """Create a new run queue in W&B Launch.

        Args:
            name: Name of the queue to create
            type: Type of resource to be used for the queue. One of
                "local-container", "local-process", "kubernetes","sagemaker",
                or "gcp-vertex".
            entity: Name of the entity to create the queue. If `None`, use
                the configured or default entity.
            prioritization_mode: Version of prioritization to use.
                Either "V0" or `None`.
            config: Default resource configuration to be used for the queue.
                Use handlebars (eg. `{{var}}`) to specify template variables.
            template_variables: A dictionary of template variable schemas to
                use with the config.

        Returns:
            The newly created `RunQueue`.

        Raises:
            `ValueError` if any of the parameters are invalid
            `wandb.Error` on wandb API errors
        """
        # TODO(np): Need to check server capabilities for this feature
        # 0. assert params are valid/normalized
        if entity is None:
            entity = self.settings["entity"] or self.default_entity
            if entity is None:
                raise ValueError(
                    "entity must be passed as a parameter, or set in settings"
                )

        if len(name) == 0:
            raise ValueError("name must be non-empty")
        if len(name) > 64:
            raise ValueError("name must be less than 64 characters")

        if type not in [
            "local-container",
            "local-process",
            "kubernetes",
            "sagemaker",
            "gcp-vertex",
        ]:
            raise ValueError(
                "resource_type must be one of 'local-container', 'local-process', 'kubernetes', 'sagemaker', or 'gcp-vertex'"
            )

        if prioritization_mode:
            prioritization_mode = prioritization_mode.upper()
            if prioritization_mode not in ["V0"]:
                raise ValueError("prioritization_mode must be 'V0' if present")

        if config is None:
            config = {}

        # 1. create required default launch project in the entity
        self.create_project(LAUNCH_DEFAULT_PROJECT, entity)

        api = InternalApi(
            default_settings={
                "entity": entity,
                "project": self.project(LAUNCH_DEFAULT_PROJECT),
            },
            retry_timedelta=RETRY_TIMEDELTA,
        )

        # 2. create default resource config, receive config id
        config_json = json.dumps({"resource_args": {type: config}})
        create_config_result = api.create_default_resource_config(
            entity, type, config_json, template_variables
        )
        if not create_config_result["success"]:
            raise wandb.Error("failed to create default resource config")
        config_id = create_config_result["defaultResourceConfigID"]

        # 3. create run queue
        create_queue_result = api.create_run_queue(
            entity,
            LAUNCH_DEFAULT_PROJECT,
            name,
            "PROJECT",
            prioritization_mode,
            config_id,
        )
        if not create_queue_result["success"]:
            raise wandb.Error("failed to create run queue")

        return public.RunQueue(
            client=self.client,
            name=name,
            entity=entity,
            prioritization_mode=prioritization_mode,
            _access="PROJECT",
            _default_resource_config_id=config_id,
            _default_resource_config=config,
        )

    def upsert_run_queue(
        self,
        name: str,
        resource_config: dict,
        resource_type: "public.RunQueueResourceType",
        entity: Optional[str] = None,
        template_variables: Optional[dict] = None,
        external_links: Optional[dict] = None,
        prioritization_mode: Optional["public.RunQueuePrioritizationMode"] = None,
    ):
        """Upsert a run queue in W&B Launch.

        Args:
            name: Name of the queue to create
            entity: Optional name of the entity to create the queue. If `None`,
                use the configured or default entity.
            resource_config: Optional default resource configuration to be used
                for the queue. Use handlebars (eg. `{{var}}`) to specify
                template variables.
            resource_type: Type of resource to be used for the queue. One of
                "local-container", "local-process", "kubernetes", "sagemaker",
                or "gcp-vertex".
            template_variables: A dictionary of template variable schemas to
                be used with the config.
            external_links: Optional dictionary of external links to be used
                with the queue.
            prioritization_mode: Optional version of prioritization to use.
                Either "V0" or None

        Returns:
            The upserted `RunQueue`.

        Raises:
            ValueError if any of the parameters are invalid
            wandb.Error on wandb API errors
        """
        if entity is None:
            entity = self.settings["entity"] or self.default_entity
            if entity is None:
                raise ValueError(
                    "entity must be passed as a parameter, or set in settings"
                )

        if len(name) == 0:
            raise ValueError("name must be non-empty")
        if len(name) > 64:
            raise ValueError("name must be less than 64 characters")

        prioritization_mode = prioritization_mode or "DISABLED"
        prioritization_mode = prioritization_mode.upper()
        if prioritization_mode not in ["V0", "DISABLED"]:
            raise ValueError(
                "prioritization_mode must be 'V0' or 'DISABLED' if present"
            )

        if resource_type not in [
            "local-container",
            "local-process",
            "kubernetes",
            "sagemaker",
            "gcp-vertex",
        ]:
            raise ValueError(
                "resource_type must be one of 'local-container', 'local-process', 'kubernetes', 'sagemaker', or 'gcp-vertex'"
            )

        self.create_project(LAUNCH_DEFAULT_PROJECT, entity)
        api = InternalApi(
            default_settings={
                "entity": entity,
                "project": self.project(LAUNCH_DEFAULT_PROJECT),
            },
            retry_timedelta=RETRY_TIMEDELTA,
        )
        # User provides external_links as a dict with name: url format
        # but backend stores it as a list of dicts with url and label keys.
        external_links = external_links or {}
        external_links = {
            "links": [
                {
                    "label": key,
                    "url": value,
                }
                for key, value in external_links.items()
            ]
        }
        upsert_run_queue_result = api.upsert_run_queue(
            name,
            entity,
            resource_type,
            {"resource_args": {resource_type: resource_config}},
            template_variables=template_variables,
            external_links=external_links,
            prioritization_mode=prioritization_mode,
        )
        if not upsert_run_queue_result["success"]:
            raise wandb.Error("failed to create run queue")
        schema_errors = (
            upsert_run_queue_result.get("configSchemaValidationErrors") or []
        )
        for error in schema_errors:
            wandb.termwarn(f"resource config validation: {error}")

        return public.RunQueue(
            client=self.client,
            name=name,
            entity=entity,
        )

    def create_user(self, email: str, admin: Optional[bool] = False):
        """Create a new user.

        Args:
            email: The email address of the user.
            admin: Set user as a global instance administrator.

        Returns:
            A `User` object.
        """
        return public.User.create(self, email, admin)

    def sync_tensorboard(self, root_dir, run_id=None, project=None, entity=None):
        """Sync a local directory containing tfevent files to wandb."""
        from wandb.sync import SyncManager  # TODO: circular import madness

        run_id = run_id or runid.generate_id()
        project = project or self.settings.get("project") or "uncategorized"
        entity = entity or self.default_entity
        # TODO: pipe through log_path to inform the user how to debug
        sm = SyncManager(
            project=project,
            entity=entity,
            run_id=run_id,
            mark_synced=False,
            app_url=self.client.app_url,
            view=False,
            verbose=False,
            sync_tensorboard=True,
        )
        sm.add(root_dir)
        sm.start()
        while not sm.is_done():
            _ = sm.poll()
        return self.run("/".join([entity, project, run_id]))

    @property
    def client(self) -> RetryingClient:
        """Returns the client object."""
        return self._client

    @property
    def user_agent(self) -> str:
        """Returns W&B public user agent."""
        return "W&B Public Client {}".format(wandb.__version__)

    @property
    def api_key(self) -> Optional[str]:
        """Returns W&B API key."""
        # just use thread local api key if it's set
        if _thread_local_api_settings.api_key:
            return _thread_local_api_settings.api_key
        if self._api_key is not None:
            return self._api_key
        auth = requests.utils.get_netrc_auth(self.settings["base_url"])
        key = None
        if auth:
            key = auth[-1]
        # Environment should take precedence
        if os.getenv("WANDB_API_KEY"):
            key = os.environ["WANDB_API_KEY"]
        self._api_key = key  # memoize key
        return key

    @property
    def default_entity(self) -> Optional[str]:
        """Returns the default W&B entity."""
        if self._default_entity is None:
            res = self._client.execute(self.DEFAULT_ENTITY_QUERY)
            self._default_entity = (res.get("viewer") or {}).get("entity")
        return self._default_entity

    @property
    def viewer(self) -> "public.User":
        """Returns the viewer object."""
        if self._viewer is None:
            self._viewer = public.User(
                self._client, self._client.execute(self.VIEWER_QUERY).get("viewer")
            )
            self._default_entity = self._viewer.entity
        return self._viewer

    def flush(self):
        """Flush the local cache.

        The api object keeps a local cache of runs, so if the state of the run
        may change while executing your script you must clear the local cache
        with `api.flush()` to get the latest values associated with the run.
        """
        self._runs = {}

    def from_path(self, path: str):
        """Return a run, sweep, project or report from a path.

        Args:
            path: The path to the project, run, sweep or report

        Returns:
            A `Project`, `Run`, `Sweep`, or `BetaReport` instance.

        Raises:
            `wandb.Error` if path is invalid or the object doesn't exist.

        Examples:

        In the proceeding code snippets "project", "team", "run_id", "sweep_id",
        and "report_name" are placeholders for the project, team, run ID,
        sweep ID, and the name of a specific report, respectively.

        ```python
        import wandb

        api = wandb.Api()

        project = api.from_path("project")
        team_project = api.from_path("team/project")
        run = api.from_path("team/project/runs/run_id")
        sweep = api.from_path("team/project/sweeps/sweep_id")
        report = api.from_path("team/project/reports/report_name")
        ```
        """
        parts = path.strip("/ ").split("/")
        if len(parts) == 1:
            return self.project(path)
        elif len(parts) == 2:
            return self.project(parts[1], parts[0])
        elif len(parts) == 3:
            return self.run(path)
        elif len(parts) == 4:
            if parts[2].startswith("run"):
                return self.run(path)
            elif parts[2].startswith("sweep"):
                return self.sweep(path)
            elif parts[2].startswith("report"):
                if "--" not in parts[-1]:
                    if "-" in parts[-1]:
                        raise wandb.Error(
                            "Invalid report path, should be team/project/reports/Name--XXXX"
                        )
                    else:
                        parts[-1] = "--" + parts[-1]
                name, id = parts[-1].split("--")
                return public.BetaReport(
                    self.client,
                    {
                        "display_name": urllib.parse.unquote(name.replace("-", " ")),
                        "id": id,
                        "spec": "{}",
                    },
                    parts[0],
                    parts[1],
                )
        raise wandb.Error(
            "Invalid path, should be TEAM/PROJECT/TYPE/ID where TYPE is runs, sweeps, or reports"
        )

    def _parse_project_path(self, path):
        """Return project and entity for project specified by path."""
        project = self.settings["project"] or "uncategorized"
        entity = self.settings["entity"] or self.default_entity
        if path is None:
            return entity, project
        parts = path.split("/", 1)
        if len(parts) == 1:
            return entity, path
        return parts

    def _parse_path(self, path):
        """Parse url, filepath, or docker paths.

        Allows paths in the following formats:
        - url: entity/project/runs/id
        - path: entity/project/id
        - docker: entity/project:id

        Entity is optional and will fall back to the current logged-in user.
        """
        project = self.settings["project"] or "uncategorized"
        entity = self.settings["entity"] or self.default_entity
        parts = (
            path.replace("/runs/", "/").replace("/sweeps/", "/").strip("/ ").split("/")
        )
        if ":" in parts[-1]:
            id = parts[-1].split(":")[-1]
            parts[-1] = parts[-1].split(":")[0]
        elif parts[-1]:
            id = parts[-1]
        if len(parts) == 1 and project != "uncategorized":
            pass
        elif len(parts) > 1:
            project = parts[1]
            if entity and id == project:
                project = parts[0]
            else:
                entity = parts[0]
            if len(parts) == 3:
                entity = parts[0]
        else:
            project = parts[0]
        return entity, project, id

    def _parse_artifact_path(self, path):
        """Return project, entity and artifact name for project specified by path."""
        project = self.settings["project"] or "uncategorized"
        entity = self.settings["entity"] or self.default_entity
        if path is None:
            return entity, project

        path, colon, alias = path.partition(":")
        full_alias = colon + alias

        parts = path.split("/")
        if len(parts) > 3:
            raise ValueError("Invalid artifact path: {}".format(path))
        elif len(parts) == 1:
            return entity, project, path + full_alias
        elif len(parts) == 2:
            return entity, parts[0], parts[1] + full_alias
        parts[-1] += full_alias
        return parts

    def projects(
        self, entity: Optional[str] = None, per_page: Optional[int] = 200
    ) -> "public.Projects":
        """Get projects for a given entity.

        Args:
            entity: Name of the entity requested.  If None, will fall back to
                the default entity passed to `Api`.  If no default entity,
                will raise a `ValueError`.
            per_page: Sets the page size for query pagination. If set to `None`,
                use the default size. Usually there is no reason to change this.

        Returns:
            A `Projects` object which is an iterable collection of `Project`objects.
        """
        if entity is None:
            entity = self.settings["entity"] or self.default_entity
            if entity is None:
                raise ValueError(
                    "entity must be passed as a parameter, or set in settings"
                )
        if entity not in self._projects:
            self._projects[entity] = public.Projects(
                self.client, entity, per_page=per_page
            )
        return self._projects[entity]

    def project(self, name: str, entity: Optional[str] = None) -> "public.Project":
        """Return the `Project` with the given name (and entity, if given).

        Args:
            name: The project name.
            entity: Name of the entity requested.  If None, will fall back to the
                default entity passed to `Api`.  If no default entity, will
                raise a `ValueError`.

        Returns:
            A `Project` object.
        """
        # For registry artifacts, capture potential org user inputted before resolving entity
        org = entity if is_artifact_registry_project(name) else ""

        if entity is None:
            entity = self.settings["entity"] or self.default_entity

        # For registry artifacts, resolve org-based entity
        if is_artifact_registry_project(name):
            settings_entity = self.settings["entity"] or self.default_entity
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=org
            )
        return public.Project(self.client, entity, name, {})

    def reports(
        self, path: str = "", name: Optional[str] = None, per_page: Optional[int] = 50
    ) -> "public.Reports":
        """Get reports for a given project path.

        Note: `wandb.Api.reports()` API is in beta and will likely change in
        future releases.

        Args:
            path: The path to project the report resides in. Specify the
                entity that created the project as a prefix followed by a
                forward slash.
            name: Name of the report requested.
            per_page: Sets the page size for query pagination. If set to
                `None`, use the default size. Usually there is no reason to
                change this.

        Returns:
            A `Reports` object which is an iterable collection of
                `BetaReport` objects.

        Examples:

        ```python
        import wandb

        wandb.Api.reports("entity/project")
        ```

        """
        entity, project, _ = self._parse_path(path + "/fake_run")

        if name:
            name = urllib.parse.unquote(name)
            key = "/".join([entity, project, str(name)])
        else:
            key = "/".join([entity, project])

        if key not in self._reports:
            self._reports[key] = public.Reports(
                self.client,
                public.Project(self.client, entity, project, {}),
                name=name,
                per_page=per_page,
            )
        return self._reports[key]

    def create_team(
        self, team: str, admin_username: Optional[str] = None
    ) -> "public.Team":
        """Create a new team.

        Args:
            team: The name of the team
            admin_username: Username of the admin user of the team.
                Defaults to the current user.

        Returns:
            A `Team` object.
        """
        return public.Team.create(self, team, admin_username)

    def team(self, team: str) -> "public.Team":
        """Return the matching `Team` with the given name.

        Args:
            team: The name of the team.

        Returns:
            A `Team` object.
        """
        return public.Team(self.client, team)

    def user(self, username_or_email: str) -> Optional["public.User"]:
        """Return a user from a username or email address.

        This function only works for local administrators. Use `api.viewer`
            to get your own user object.

        Args:
            username_or_email: The username or email address of the user.

        Returns:
            A `User` object or None if a user is not found.
        """
        res = self._client.execute(self.USERS_QUERY, {"query": username_or_email})
        if len(res["users"]["edges"]) == 0:
            return None
        elif len(res["users"]["edges"]) > 1:
            wandb.termwarn(
                "Found multiple users, returning the first user matching {}".format(
                    username_or_email
                )
            )
        return public.User(self._client, res["users"]["edges"][0]["node"])

    def users(self, username_or_email: str) -> List["public.User"]:
        """Return all users from a partial username or email address query.

        This function only works for local administrators. Use `api.viewer`
            to get your own user object.

        Args:
            username_or_email: The prefix or suffix of the user you want to find.

        Returns:
            An array of `User` objects.
        """
        res = self._client.execute(self.USERS_QUERY, {"query": username_or_email})
        return [
            public.User(self._client, edge["node"]) for edge in res["users"]["edges"]
        ]

    def runs(
        self,
        path: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order: str = "+created_at",
        per_page: int = 50,
        include_sweeps: bool = True,
    ):
        """Return a set of runs from a project that match the filters provided.

        You can filter by `config.*`, `summary_metrics.*`, `tags`, `state`,
        `entity`, `createdAt`, and so forth. You can also compose operations
        to make more complicated queries. For  more information,
        see Query and Project Operators MongoDb Reference
        at https://docs.mongodb.com/manual/reference/operator/query.

        Args:
            path: Path to project, should be in the form: "entity/project"
            filters: Queries for specific runs using the MongoDB query language.
            order: Order can be `created_at`, `heartbeat_at`, `config.*.value`,
                or `summary_metrics.*`. If you prepend order with a `+` order
                is ascending. If you prepend order with a `-` order is
                descending (default). The default order is `run.created_at`
                from oldest to newest.
            per_page: Sets the page size for query pagination.
            include_sweeps: Whether to include the sweep runs in the results.

        Returns:
            A `Runs` object, which is an iterable collection of `Run` objects.

        Examples:

        ```python
        # Find runs in project where config.experiment_name has been set to "foo"
        api.runs(path="my_entity/project", filters={"config.experiment_name": "foo"})
        ```

        ```python
        # Find runs in project where config.experiment_name has been set to "foo" or "bar"
        api.runs(
            path="my_entity/project",
            filters={
                "$or": [
                    {"config.experiment_name": "foo"},
                    {"config.experiment_name": "bar"},
                ]
            },
        )
        ```

        ```python
        # Find runs in project where config.experiment_name matches a regex
        # (anchors are not supported)
        api.runs(
            path="my_entity/project",
            filters={"config.experiment_name": {"$regex": "b.*"}},
        )
        ```

        ```python
        # Find runs in project where the run name matches a regex
        # (anchors are not supported)
        api.runs(
            path="my_entity/project", filters={"display_name": {"$regex": "^foo.*"}}
        )
        ```

        ```python
        # Find runs in project sorted by ascending loss
        api.runs(path="my_entity/project", order="+summary_metrics.loss")
        ```
        """
        entity, project = self._parse_project_path(path)
        filters = filters or {}
        key = (path or "") + str(filters) + str(order)
        if not self._runs.get(key):
            self._runs[key] = public.Runs(
                self.client,
                entity,
                project,
                filters=filters,
                order=order,
                per_page=per_page,
                include_sweeps=include_sweeps,
            )
        return self._runs[key]

    @normalize_exceptions
    def run(self, path=""):
        """Return a single run by parsing path in the form `entity/project/run_id`.

        Args:
            path: Path to run in the form `entity/project/run_id`.
                If `api.entity` is set, this can be in the form `project/run_id`
                and if `api.project` is set this can just be the run_id.

        Returns:
            A `Run` object.
        """
        entity, project, run_id = self._parse_path(path)
        if not self._runs.get(path):
            self._runs[path] = public.Run(self.client, entity, project, run_id)
        return self._runs[path]

    def queued_run(
        self,
        entity: str,
        project: str,
        queue_name: str,
        run_queue_item_id: str,
        project_queue=None,
        priority=None,
    ):
        """Return a single queued run based on the path.

        Parses paths of the form `entity/project/queue_id/run_queue_item_id`.
        """
        return public.QueuedRun(
            self.client,
            entity,
            project,
            queue_name,
            run_queue_item_id,
            project_queue=project_queue,
            priority=priority,
        )

    def run_queue(
        self,
        entity: str,
        name: str,
    ):
        """Return the named `RunQueue` for entity.

        See `Api.create_run_queue` for more information on how to create a run queue.
        """
        return public.RunQueue(
            self.client,
            name,
            entity,
        )

    @normalize_exceptions
    def sweep(self, path=""):
        """Return a sweep by parsing path in the form `entity/project/sweep_id`.

        Args:
            path: Path to sweep in the form entity/project/sweep_id.
                If `api.entity` is set, this can be in the form
                project/sweep_id and if `api.project` is set
                this can just be the sweep_id.

        Returns:
            A `Sweep` object.
        """
        entity, project, sweep_id = self._parse_path(path)
        if not self._sweeps.get(path):
            self._sweeps[path] = public.Sweep(self.client, entity, project, sweep_id)
        return self._sweeps[path]

    @normalize_exceptions
    def artifact_types(self, project: Optional[str] = None) -> "public.ArtifactTypes":
        """Returns a collection of matching artifact types.

        Args:
            project: The project name or path to filter on.

        Returns:
            An iterable `ArtifactTypes` object.
        """
        project_path = project
        entity, project = self._parse_project_path(project_path)
        # If its a Registry project, the entity is considered to be an org instead
        if is_artifact_registry_project(project):
            settings_entity = self.settings["entity"] or self.default_entity
            org = parse_org_from_registry_path(project_path, PathType.PROJECT)
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=org
            )
        return public.ArtifactTypes(self.client, entity, project)

    @normalize_exceptions
    def artifact_type(
        self, type_name: str, project: Optional[str] = None
    ) -> "public.ArtifactType":
        """Returns the matching `ArtifactType`.

        Args:
            type_name: The name of the artifact type to retrieve.
            project: If given, a project name or path to filter on.

        Returns:
            An `ArtifactType` object.
        """
        project_path = project
        entity, project = self._parse_project_path(project_path)
        # If its an Registry artifact, the entity is an org instead
        if is_artifact_registry_project(project):
            org = parse_org_from_registry_path(project_path, PathType.PROJECT)
            settings_entity = self.settings["entity"] or self.default_entity
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=org
            )
        return public.ArtifactType(self.client, entity, project, type_name)

    @normalize_exceptions
    def artifact_collections(
        self, project_name: str, type_name: str, per_page: Optional[int] = 50
    ) -> "public.ArtifactCollections":
        """Returns a collection of matching artifact collections.

        Args:
            project_name: The name of the project to filter on.
            type_name: The name of the artifact type to filter on.
            per_page: Sets the page size for query pagination.  None will use the default size.
                Usually there is no reason to change this.

        Returns:
            An iterable `ArtifactCollections` object.
        """
        entity, project = self._parse_project_path(project_name)
        # If iterating through Registry project, the entity is considered to be an org instead
        if is_artifact_registry_project(project):
            org = parse_org_from_registry_path(project_name, PathType.PROJECT)
            settings_entity = self.settings["entity"] or self.default_entity
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=org
            )
        return public.ArtifactCollections(
            self.client, entity, project, type_name, per_page=per_page
        )

    @normalize_exceptions
    def artifact_collection(
        self, type_name: str, name: str
    ) -> "public.ArtifactCollection":
        """Returns a single artifact collection by type.

        You can use the returned `ArtifactCollection` object to retrieve
        information about specific artifacts in that collection, and more.

        Args:
            type_name: The type of artifact collection to fetch.
            name: An artifact collection name. Optionally append the entity
                that logged the artifact as a prefix followed by a forward
                slash.

        Returns:
            An `ArtifactCollection` object.

        Examples:

        In the proceeding code snippet "type", "entity", "project", and
        "artifact_name" are placeholders for the collection type, your W&B
        entity, name of the project the artifact is in, and the name of
        the artifact, respectively.

        ```python
        import wandb

        collections = wandb.Api().artifact_collection(
            type_name="type", name="entity/project/artifact_name"
        )

        # Get the first artifact in the collection
        artifact_example = collections.artifacts()[0]

        # Download the contents of the artifact to the specified root directory.
        artifact_example.download()
        ```
        """
        entity, project, collection_name = self._parse_artifact_path(name)
        # If its an Registry artifact, the entity is considered to be an org instead
        if is_artifact_registry_project(project):
            org = parse_org_from_registry_path(name, PathType.ARTIFACT)
            settings_entity = self.settings["entity"] or self.default_entity
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=org
            )
        return public.ArtifactCollection(
            self.client, entity, project, collection_name, type_name
        )

    @normalize_exceptions
    def artifact_versions(self, type_name, name, per_page=50):
        """Deprecated. Use `Api.artifacts(type_name, name)` method instead."""
        deprecate(
            field_name=Deprecated.api__artifact_versions,
            warning_message=(
                "Api.artifact_versions(type_name, name) is deprecated, "
                "use Api.artifacts(type_name, name) instead."
            ),
        )
        return self.artifacts(type_name, name, per_page=per_page)

    @normalize_exceptions
    def artifacts(
        self,
        type_name: str,
        name: str,
        per_page: Optional[int] = 50,
        tags: Optional[List[str]] = None,
    ) -> "public.Artifacts":
        """Return an `Artifacts` collection.

        Args:
            type_name: The type of artifacts to fetch.
            name: The artifact's collection name. Optionally append the
                entity that logged the artifact as a prefix followed by
                a forward slash.
            per_page: Sets the page size for query pagination. If set to
                `None`, use the default size. Usually there is no reason
                to change this.
            tags: Only return artifacts with all of these tags.

        Returns:
            An iterable `Artifacts` object.

        Examples:

        In the proceeding code snippet, "type", "entity", "project", and
        "artifact_name" are placeholders for the artifact type, W&B entity,
        name of the project the artifact was logged to,
        and the name of the artifact, respectively.

        ```python
        import wandb

        wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
        ```
        """
        entity, project, collection_name = self._parse_artifact_path(name)
        # If its an Registry project, the entity is considered to be an org instead
        if is_artifact_registry_project(project):
            org = parse_org_from_registry_path(name, PathType.ARTIFACT)
            settings_entity = self.settings["entity"] or self.default_entity
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=org
            )
        return public.Artifacts(
            self.client,
            entity,
            project,
            collection_name,
            type_name,
            per_page=per_page,
            tags=tags,
        )

    @normalize_exceptions
    def _artifact(
        self, name: str, type: Optional[str] = None, enable_tracking: bool = False
    ):
        if name is None:
            raise ValueError("You must specify name= to fetch an artifact.")
        entity, project, artifact_name = self._parse_artifact_path(name)

        # If its an Registry artifact, the entity is an org instead
        if is_artifact_registry_project(project):
            organization = name.split("/")[0] if name.count("/") == 2 else ""
            # set entity to match the settings since in above code it was potentially set to an org
            settings_entity = self.settings["entity"] or self.default_entity
            # Registry artifacts are under the org entity. Because we offer a shorthand and alias for this path,
            # we need to fetch the org entity to for the user behind the scenes.
            entity = InternalApi()._resolve_org_entity_name(
                entity=settings_entity, organization=organization
            )
        artifact = wandb.Artifact._from_name(
            entity=entity,
            project=project,
            name=artifact_name,
            client=self.client,
            enable_tracking=enable_tracking,
        )
        if type is not None and artifact.type != type:
            raise ValueError(
                f"type {type} specified but this artifact is of type {artifact.type}"
            )
        return artifact

    @normalize_exceptions
    def artifact(self, name: str, type: Optional[str] = None):
        """Returns a single artifact.

        Args:
            name: The artifact's name. The name of an artifact resembles a
                filepath that consists, at a minimum, the name of the project
                the artifact was logged to, the name of the artifact, and the
                artifact's version or alias. Optionally append the entity that
                logged the artifact as a prefix followed by a forward slash.
                If no entity is specified in the name, the Run or API
                setting's entity is used.
            type: The type of artifact to fetch.

        Returns:
            An `Artifact` object.

        Raises:
            ValueError: If the artifact name is not specified.
            ValueError: If the artifact type is specified but does not
                match the type of the fetched artifact.

        Examples:

        In the proceeding code snippets "entity", "project", "artifact",
        "version", and "alias" are placeholders for your W&B entity, name
        of the project the artifact is in, the name of the artifact,
        and artifact's version, respectively.

        ```python
        import wandb

        # Specify the project, artifact's name, and the artifact's alias
        wandb.Api().artifact(name="project/artifact:alias")

        # Specify the project, artifact's name, and a specific artifact version
        wandb.Api().artifact(name="project/artifact:version")

        # Specify the entity, project, artifact's name, and the artifact's alias
        wandb.Api().artifact(name="entity/project/artifact:alias")

        # Specify the entity, project, artifact's name, and a specific artifact version
        wandb.Api().artifact(name="entity/project/artifact:version")
        ```

        Note:
        This method is intended for external use only. Do not call `api.artifact()` within the wandb repository code.
        """
        return self._artifact(name=name, type=type, enable_tracking=True)

    @normalize_exceptions
    def job(self, name: Optional[str], path: Optional[str] = None) -> "public.Job":
        """Return a `Job` object.

        Args:
            name: The name of the job.
            path: The root path to download the job artifact.

        Returns:
            A `Job` object.
        """
        if name is None:
            raise ValueError("You must specify name= to fetch a job.")
        elif name.count("/") != 2 or ":" not in name:
            raise ValueError(
                "Invalid job specification. A job must be of the form: <entity>/<project>/<job-name>:<alias-or-version>"
            )
        return public.Job(self, name, path)

    @normalize_exceptions
    def list_jobs(self, entity: str, project: str) -> List[Dict[str, Any]]:
        """Return a list of jobs, if any, for the given entity and project.

        Args:
            entity: The entity for the listed jobs.
            project: The project for the listed jobs.

        Returns:
            A list of matching jobs.
        """
        if entity is None:
            raise ValueError("Specify an entity when listing jobs")
        if project is None:
            raise ValueError("Specify a project when listing jobs")

        query = gql(
            """
        query ArtifactOfType(
            $entityName: String!,
            $projectName: String!,
            $artifactTypeName: String!,
        ) {
            project(name: $projectName, entityName: $entityName) {
                artifactType(name: $artifactTypeName) {
                    artifactCollections {
                        edges {
                            node {
                                artifacts {
                                    edges {
                                        node {
                                            id
                                            state
                                            aliases {
                                                alias
                                            }
                                            artifactSequence {
                                                name
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

        try:
            artifact_query = self._client.execute(
                query,
                {
                    "projectName": project,
                    "entityName": entity,
                    "artifactTypeName": "job",
                },
            )

            if not artifact_query or not artifact_query["project"]:
                wandb.termerror(
                    f"Project: '{project}' not found in entity: '{entity}' or access denied."
                )
                return []

            if artifact_query["project"]["artifactType"] is None:
                return []

            artifacts = artifact_query["project"]["artifactType"][
                "artifactCollections"
            ]["edges"]

            return [x["node"]["artifacts"] for x in artifacts]
        except requests.exceptions.HTTPError:
            return False

    @normalize_exceptions
    def artifact_exists(self, name: str, type: Optional[str] = None) -> bool:
        """Whether an artifact version exists within the specified project and entity.

        Args:
            name: The name of artifact. Add the artifact's entity and project
                as a prefix. Append the version or the alias of the artifact
                with a colon. If the entity or project is not specified,
                W&B uses override parameters if populated. Otherwise, the
                entity is pulled from the user settings and the project is
                set to "Uncategorized".
            type: The type of artifact.

        Returns:
            True if the artifact version exists, False otherwise.

        Examples:

        In the proceeding code snippets "entity", "project", "artifact",
        "version", and "alias" are placeholders for your W&B entity, name of
        the project the artifact is in, the name of the artifact, and
        artifact's version, respectively.

        ```python
        import wandb

        wandb.Api().artifact_exists("entity/project/artifact:version")
        wandb.Api().artifact_exists("entity/project/artifact:alias")
        ```

        """
        try:
            self._artifact(name, type)
            return True
        except wandb.errors.CommError:
            return False

    @normalize_exceptions
    def artifact_collection_exists(self, name: str, type: str) -> bool:
        """Whether an artifact collection exists within a specified project and entity.

        Args:
            name: An artifact collection name. Optionally append the
                entity that logged the artifact as a prefix followed by
                a forward slash. If entity or project is not specified,
                infer the collection from the override params if they exist.
                Otherwise, entity is pulled from the user settings and project
                will default to "uncategorized".
            type: The type of artifact collection.

        Returns:
            True if the artifact collection exists, False otherwise.

        Examples:

        In the proceeding code snippet "type", and "collection_name" refer to the type
        of the artifact collection and the name of the collection, respectively.

        ```python
        import wandb

        wandb.Api.artifact_collection_exists(type="type", name="collection_name")
        ```
        """
        try:
            self.artifact_collection(type, name)
            return True
        except wandb.errors.CommError:
            return False
