# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from pydantic import BaseModel, Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig


class BedrockProviderDataValidator(BaseModel):
    """Validator for per-request provider data passed via x-llamastack-provider-data header."""

    aws_bearer_token_bedrock: str | None = Field(
        default=None,
        description="API Key (Bearer token) for Amazon Bedrock",
    )


class BedrockConfig(RemoteInferenceProviderConfig):
    """
    Configuration for AWS Bedrock inference provider.

    Authentication priority:
    1. Bearer token from provider data header (per-request override)
    2. Bearer token from config (api_key / AWS_BEARER_TOKEN_BEDROCK)
    3. AWS credential chain (IRSA/web identity, profile, IMDS, static creds)

    For enterprise/Kubernetes deployments, leave api_key unset and configure
    AWS credentials via environment variables or IAM roles.
    """

    region_name: str = Field(
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
        }
