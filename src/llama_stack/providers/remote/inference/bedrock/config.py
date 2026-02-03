# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.bedrock.config import BedrockBaseConfig
from llama_stack.providers.utils.inference.model_registry import NetworkConfig


class BedrockProviderDataValidator(BaseModel):
    """Validator for per-request provider data passed via x-llamastack-provider-data header."""

    aws_bearer_token_bedrock: str | None = Field(
        default=None,
        description="API Key (Bearer token) for Amazon Bedrock",
    )


class BedrockConfig(BedrockBaseConfig):
    """
    Configuration for AWS Bedrock inference provider.

    Authentication priority:
    1. Bearer token from provider data header (per-request override)
    2. Bearer token from config (api_key / AWS_BEARER_TOKEN_BEDROCK)
    3. AWS credential chain (IRSA/web identity, profile, IMDS, static creds)

    For enterprise/Kubernetes deployments, leave api_key unset and configure
    AWS credentials via environment variables or IAM roles.
    """

    # NOTE: BedrockBaseConfig is shared across multiple worktrees in this dev environment.
    # We explicitly declare core auth/network fields here so type checking and docs generation
    # remain correct and the config supports both bearer-token and SigV4 modes.

    auth_credential: SecretStr | None = Field(
        default=None,
        description="Authentication credential for the provider",
        alias="api_key",
    )
    network: NetworkConfig | None = Field(
        default=None,
        description="Network configuration including TLS, proxy, and timeout settings.",
    )

    aws_access_key_id: SecretStr | None = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("AWS_ACCESS_KEY_ID")) else None,
        description="The AWS access key to use. Default use environment variable: AWS_ACCESS_KEY_ID",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("AWS_SECRET_ACCESS_KEY")) else None,
        description="The AWS secret access key to use. Default use environment variable: AWS_SECRET_ACCESS_KEY",
    )
    aws_session_token: SecretStr | None = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("AWS_SESSION_TOKEN")) else None,
        description="The AWS session token to use. Default use environment variable: AWS_SESSION_TOKEN",
    )
    aws_role_arn: str | None = Field(
        default_factory=lambda: os.getenv("AWS_ROLE_ARN"),
        description="The AWS role ARN to assume. Default use environment variable: AWS_ROLE_ARN",
    )
    aws_web_identity_token_file: str | None = Field(
        default_factory=lambda: os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE"),
        description="The path to the web identity token file. Default use environment variable: AWS_WEB_IDENTITY_TOKEN_FILE",
    )
    aws_role_session_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_ROLE_SESSION_NAME"),
        description="The session name to use when assuming a role. Default use environment variable: AWS_ROLE_SESSION_NAME",
    )
    profile_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_PROFILE"),
        description="The profile name that contains credentials to use. Default use environment variable: AWS_PROFILE",
    )
    session_ttl: int | None = Field(
        default_factory=lambda: int(os.getenv("AWS_SESSION_TTL", "3600")),
        description="The time in seconds till a session expires. The default is 3600 seconds (1 hour).",
    )

    region_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        description="AWS Region for the Bedrock Runtime endpoint",
    )

    def has_bearer_token(self) -> bool:
        """Check if a bearer token is configured."""
        if self.auth_credential is None:
            return False
        token = self.auth_credential.get_secret_value()
        return bool(token and token.strip())

    @classmethod
    def sample_run_config(cls, **kwargs):
        return {
            "api_key": "${env.AWS_BEARER_TOKEN_BEDROCK:=}",
            "region_name": "${env.AWS_DEFAULT_REGION:=us-east-2}",
            "aws_role_arn": "${env.AWS_ROLE_ARN:=}",
            "aws_web_identity_token_file": "${env.AWS_WEB_IDENTITY_TOKEN_FILE:=}",
        }
