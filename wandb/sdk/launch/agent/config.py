"""Definition of the config object used by the Launch agent."""

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class RegistryConfig(BaseModel):
    """Configuration for registry block.

    Note that we don't forbid extra fields here because:
    - We want to allow all fields supported by each registry
    - We will perform validation on the registry object itself later
    - Registry block is being deprecated in favor of destination field in builder
    """

    type: str = Field(
        ..., description="The type of registry to use.", values=["ecr", "acr", "gcr"]
    )
    uri: str = Field(..., description="The URI of the registry.")


class EnvironmentConfig(BaseModel):
    """Configuration for the environment block."""

    type: str = Field(
        ...,
        description="The type of environment to use.",
        values=["azure", "aws", "gcp"],
    )
    region: Optional[str] = Field(..., description="The region to use.")

    class Config:
        extra = "forbid"


class BuilderConfig(BaseModel):
    type: Optional[str] = Field(
        ...,
        description="The type of builder to use.",
        values=["docker", "kaniko", "noop"],
    )
    destination: Optional[str] = Field(
        description="The destination to use for the built image. If not provided, "
        "the image will be pushed to the registry.",
    )

    @classmethod
    @validator("destination")
    def validate_destination(cls, destination) -> str:
        """Validate that the destination is a valid container registry URI."""
        return destination


class DockerBuilderConfig(BuilderConfig):
    platform: Optional[str] = Field(
        description="The platform to use for the built image. If not provided, "
        "the platform will be detected automatically.",
        values=["linux/amd64", "linux/arm64", "linux/arm/v7", "linux/arm/v6", "all"],
    )


class KanikoBuilderConfig(BuilderConfig):
    build_context_store: Optional[str] = Field(
        ...,
        description="The build context store to use. If not provided, "
        "the build context will be uploaded to the registry.",
        alias="build-context-store",
    )
    build_job_name: Optional[str] = Field(
        "wandb-launch-container-build",
        description="Name prefix of the build job.",
        alias="build-job-name",
    )
    secret_name: Optional[str] = Field(
        description="The name of the secret to use for the build job.",
        alias="secret-name",
    )
    secret_key: Optional[str] = Field(
        description="The key of the secret to use for the build job.",
        alias="secret-key",
    )
    kaniko_image: Optional[str] = Field(
        "gcr.io/kaniko-project/executor:latest",
        description="The image to use for the kaniko executor.",
        alias="kaniko-image",
    )

    class Config:
        extra = "forbid"

    @classmethod
    @validator("build_context_store")
    def validate_build_context_store(cls, build_context_store) -> str:
        """Validate that the build context store is a valid container registry URI."""
        return build_context_store


class AgentConfig(BaseModel):
    """Configuration for the Launch agent."""

    queues: List[str] = Field(
        ...,
        description="The queues to use for this agent.",
    )
    entity: Optional[str] = Field(
        description="The W&B entity to use for this agent.",
    )
    max_jobs: Optional[int] = Field(
        1,
        description="The maximum number of jobs to run concurrently.",
    )
    max_schedulers: Optional[int] = Field(
        1,
        description="The maximum number of sweep schedulers to run concurrently.",
    )
    secure_mode: Optional[bool] = Field(
        False,
        description="Whether to use secure mode for this agent. If True, the "
        "agent will reject runs that attempt to override the entrypoint or image.",
    )
    registry: Optional[RegistryConfig] = Field(
        description="The registry to use.",
    )
    environment: Optional[EnvironmentConfig] = Field(
        description="The environment to use.",
    )
    builder: Optional[BuilderConfig] = Field(
        description="The builder to use.",
    )

    class Config:
        extra = "forbid"
