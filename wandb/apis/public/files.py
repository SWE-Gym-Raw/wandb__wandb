"""Public API: files."""

import io
import os
from typing import Optional

import requests
from wandb_gql import gql
from wandb_gql.client import RetryError

import wandb
from wandb import util
from wandb.apis.attrs import Attrs
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.apis.public import utils
from wandb.apis.public.api import Api
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.lib import retry

FILE_FRAGMENT = """fragment RunFilesFragment on Run {
    files(names: $fileNames, after: $fileCursor, first: $fileLimit) {
        edges {
            node {
                id
                name
                url(upload: $upload)
                directUrl
                sizeBytes
                mimetype
                updatedAt
                md5
            }
            cursor
        }
        pageInfo {
            endCursor
            hasNextPage
        }
    }
}"""


class Files(Paginator):
    """An iterable collection of `File` objects."""

    QUERY = gql(
        """
        query RunFiles($project: String!, $entity: String!, $name: String!, $fileCursor: String,
            $fileLimit: Int = 50, $fileNames: [String] = [], $upload: Boolean = false) {{
            project(name: $project, entityName: $entity) {{
                internalId
                run(name: $name) {{
                    fileCount
                    ...RunFilesFragment
                }}
            }}
        }}
        {}
        """.format(
            FILE_FRAGMENT
        )
    )

    def __init__(self, client, run, names=None, per_page=50, upload=False):
        self.run = run
        print(
            "self.run: {}, self.run._project_internal_id: {}".format(
                self.run, self.run._project_internal_id
            )
        )
        variables = {
            "project": run.project,
            "entity": run.entity,
            "name": run.id,
            "fileNames": names or [],
            "upload": upload,
        }
        super().__init__(client, variables, per_page)

    @property
    def length(self):
        if self.last_response:
            return self.last_response["project"]["run"]["fileCount"]
        else:
            return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response["project"]["run"]["files"]["pageInfo"][
                "hasNextPage"
            ]
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response["project"]["run"]["files"]["edges"][-1]["cursor"]
        else:
            return None

    def update_variables(self):
        self.variables.update({"fileLimit": self.per_page, "fileCursor": self.cursor})

    def convert_objects(self):
        return [
            File(self.client, r["node"], self.run)
            for r in self.last_response["project"]["run"]["files"]["edges"]
        ]

    def __repr__(self):
        # print("self.run._project_internal_id: ", self.run._internal_project_id) # auth error
        return "<Files {} ({})>".format("/".join(self.run.path), len(self))


class File(Attrs):
    """File is a class associated with a file saved by wandb.

    Attributes:
        name (string): filename
        url (string): path to file
        direct_url (string): path to file in the bucket
        md5 (string): md5 of file
        mimetype (string): mimetype of file
        updated_at (string): timestamp of last update
        size (int): size of file in bytes
        path_uri (str): path to file in the bucket, currently only available for files stored in S3
    """

    def __init__(self, client, attrs, run=None):
        self.client = client
        self._attrs = attrs
        self.run = run
        print(
            "File.__init__() — self.run._project_internal_id: ",
            self.run._project_internal_id,
        )
        super().__init__(dict(attrs))

    @property
    def size(self):
        size_bytes = self._attrs["sizeBytes"]
        if size_bytes is not None:
            return int(size_bytes)
        return 0

    @property
    def path_uri(self) -> str:
        """
        Returns the uri path to the file in the storage bucket.
        """
        path_uri = ""
        try:
            path_uri = utils.parse_s3_url_to_s3_uri(self._attrs["directUrl"])
        except ValueError:
            wandb.termwarn("path_uri is only available for files stored in S3")
        except LookupError:
            wandb.termwarn("Unable to find direct_url of file")
        return path_uri

    @normalize_exceptions
    @retry.retriable(
        retry_timedelta=RETRY_TIMEDELTA,
        check_retry_fn=util.no_retry_auth,
        retryable_exceptions=(RetryError, requests.RequestException),
    )
    def download(
        self,
        root: str = ".",
        replace: bool = False,
        exist_ok: bool = False,
        api: Optional[Api] = None,
    ) -> io.TextIOWrapper:
        """Downloads a file previously saved by a run from the wandb server.

        Args:
            replace (boolean): If `True`, download will overwrite a local file
                if it exists. Defaults to `False`.
            root (str): Local directory to save the file.  Defaults to ".".
            exist_ok (boolean): If `True`, will not raise ValueError if file already
                exists and will not re-download unless replace=True. Defaults to `False`.
            api (Api, optional): If given, the `Api` instance used to download the file.

        Raises:
            `ValueError` if file already exists, replace=False and exist_ok=False.
        """
        if api is None:
            api = wandb.Api()

        path = os.path.join(root, self.name)
        if os.path.exists(path) and not replace:
            if exist_ok:
                return open(path)
            else:
                raise ValueError(
                    "File already exists, pass replace=True to overwrite or exist_ok=True to leave it as is and don't error."
                )

        util.download_file_from_url(path, self.url, api.api_key)
        return open(path)

    @normalize_exceptions
    def delete(self):
        print(
            "self in delete: self — {} project_id — {} run — {}".format(
                self, self.run._internal_project_id, self.run
            )
        )
        mutation = gql(
            """
            mutation deleteFiles($files: [ID!]!, $projectId: Int) {
                deleteFiles(input: {
                    files: $files
                    projectId: $projectId
                }) {
                    success
                }
            }
            """
        )
        self.client.execute(
            mutation,
            variable_values={
                "files": [self.id],
                "projectId": self.run._project_internal_id,
            },
        )

    def __repr__(self):
        return "<File {} ({}) {}>".format(
            self.name,
            self.mimetype,
            util.to_human_size(self.size, units=util.POW_2_BYTES),
        )
