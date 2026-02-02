# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable
from typing import Any, NoReturn

import httpx
from openai import AuthenticationError
from pydantic import PrivateAttr

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

from .config import BedrockConfig

logger = get_logger(name=__name__, category="inference::bedrock")


class BedrockInferenceAdapter(OpenAIMixin):
    """
    Adapter for AWS Bedrock's OpenAI-compatible API endpoints.

    Supports Llama models across regions and GPT-OSS models (us-west-2 only).

    Authentication modes:
    1. Bearer token (legacy): Set AWS_BEARER_TOKEN_BEDROCK or api_key in config
    2. AWS credential chain (enterprise): Leave api_key unset, configure AWS creds
       - Web Identity Federation (IRSA, GitHub Actions OIDC)
       - IAM roles (EC2, ECS, Lambda)
       - AWS profiles
       - Static credentials

    When using AWS credential chain, requests are signed using SigV4 with the
    "bedrock" signing name (note: the endpoint hostname uses "bedrock-runtime",
    but SigV4 credential scope uses the signing name "bedrock").

    Web Identity Federation Examples:

    Kubernetes/OpenShift (IRSA):
        Set these environment variables in your pod spec:
        - AWS_ROLE_ARN=arn:aws:iam::123456789012:role/llama-stack-role
        - AWS_WEB_IDENTITY_TOKEN_FILE=<path-to-serviceaccount-token>
          Common paths:
          - EKS: /var/run/secrets/eks.amazonaws.com/serviceaccount/token
          - Generic K8s: /var/run/secrets/kubernetes.io/serviceaccount/token
        - AWS_DEFAULT_REGION=us-east-2

    GitHub Actions:
        Use aws-actions/configure-aws-credentials with OIDC:

        permissions:
          id-token: write  # Required for OIDC

        steps:
          - uses: aws-actions/configure-aws-credentials@v4
            with:
              role-to-assume: arn:aws:iam::123456789012:role/github-actions-role
              aws-region: us-east-2

    Credentials are automatically refreshed by boto3 when they expire.

    Note: Bedrock's OpenAI-compatible endpoint does not support /v1/models
    for dynamic model discovery. Models must be pre-registered in the config.
    """

    config: BedrockConfig
    provider_data_api_key_field: str = "aws_bearer_token_bedrock"

    # Cached SigV4 auth handler (reuses boto3 session across requests)
    _sigv4_auth: Any = PrivateAttr(default=None)
    # Cached httpx client for SigV4 mode (prevents socket leaks under load)
    _sigv4_http_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return f"https://bedrock-runtime.{self.config.region_name}.amazonaws.com/openai/v1"

    def _should_use_sigv4(self) -> bool:
        """
        Determine if SigV4 authentication should be used.

        Returns True if:
        - No bearer token is configured in config
        - No bearer token in provider data (checked at request time)

        Note: This check is performed per-request to support mixed auth modes.
        """
        # Check if bearer token is in config
        if self.config.has_bearer_token():
            return False

        # Check provider data at request time (strip whitespace for consistency with config)
        provider_data = self.get_request_provider_data()
        if provider_data:
            token = getattr(provider_data, "aws_bearer_token_bedrock", None)
            if token and token.strip():
                return False

        return True

    def _get_sigv4_auth(self) -> Any:
        """Get or create the SigV4 auth handler (cached for session reuse)."""
        if self._sigv4_auth is None:
            from llama_stack.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

            # Use "bedrock" as signing name (NOT "bedrock-runtime")
            # This is the SigV4 signing name from botocore service metadata
            self._sigv4_auth = BedrockSigV4Auth(
                region=self.config.region_name,
                service="bedrock",  # Signing name, not endpoint prefix
            )
        return self._sigv4_auth

    def get_api_key(self) -> str | None:
        """
        Get API key for authentication.

        In SigV4 mode, returns a non-empty placeholder to satisfy OpenAIMixin
        validation while the actual auth is handled by the SigV4 http_client.
        This follows the same pattern as the OCI provider.
        """
        if self._should_use_sigv4():
            # Return placeholder - SigV4 auth handles Authorization header via http_client
            # The OpenAI SDK will add "Authorization: Bearer <NOTUSED>" but our SigV4
            # auth handler replaces it with the proper SigV4 signature.
            return "<NOTUSED>"
        return super().get_api_key()

    def _get_sigv4_http_client(self) -> httpx.AsyncClient:
        """Get or create the cached httpx client for SigV4 mode."""
        if self._sigv4_http_client is None:
            self._sigv4_http_client = httpx.AsyncClient(auth=self._get_sigv4_auth())
        return self._sigv4_http_client

    def get_extra_client_params(self) -> dict[str, Any]:
        """
        Get extra parameters for the AsyncOpenAI client.

        In SigV4 mode, provides an http_client with SigV4 authentication.
        The OpenAIMixin.client property will merge network config into this client.
        """
        if self._should_use_sigv4():
            return {
                "http_client": self._get_sigv4_http_client(),
            }
        return {}

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        Bedrock's OpenAI-compatible endpoint does not support the /v1/models endpoint.
        Returns empty list since models must be pre-registered in the config.
        """
        return []

    async def check_model_availability(self, model: str) -> bool:
        """
        Bedrock doesn't support dynamic model listing via /v1/models.
        Always return True to accept all models registered in the config.
        """
        return True

    async def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        if self._sigv4_http_client is not None:
            await self._sigv4_http_client.aclose()
            self._sigv4_http_client = None

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Bedrock's OpenAI-compatible API does not support the /v1/embeddings endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/embeddings endpoint. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Bedrock's OpenAI-compatible API does not support the /v1/completions endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/completions endpoint. "
            "Only /v1/chat/completions is supported. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Override to handle authentication errors and null responses."""
        # Determine auth mode at request time (request-scoped, not instance state)
        use_sigv4 = self._should_use_sigv4()

        try:
            logger.debug(
                f"Calling Bedrock OpenAI API with model={params.model}, stream={params.stream}, sigv4={use_sigv4}"
            )
            result = await super().openai_chat_completion(params=params)
            logger.debug(f"Bedrock API returned: {type(result).__name__ if result is not None else 'None'}")

            if result is None:
                logger.error(f"Bedrock OpenAI client returned None for model={params.model}, stream={params.stream}")
                raise RuntimeError(
                    f"Bedrock API returned no response for model '{params.model}'. "
                    "This may indicate the model is not supported or a network/API issue occurred."
                )

            return result
        except AuthenticationError as e:
            error_msg = str(e)
            self._handle_auth_error(error_msg, e, use_sigv4=use_sigv4)
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock API: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _handle_auth_error(self, error_msg: str, original_error: Exception, *, use_sigv4: bool) -> NoReturn:
        """Handle authentication errors with appropriate messages."""
        if use_sigv4:
            # SigV4 auth failure
            logger.error(f"AWS Bedrock SigV4 authentication failed: {error_msg}")
            raise ValueError(
                f"AWS Bedrock SigV4 authentication failed: {error_msg}. "
                "Please verify your AWS credentials are correctly configured. "
                "Check AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, "
                "or IAM role configuration. For IRSA/web identity, verify "
                "AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE."
            ) from original_error

        # Bearer token auth failure
        if "expired" in error_msg.lower() or "Bearer Token has expired" in error_msg:
            logger.error(f"AWS Bedrock authentication token expired: {error_msg}")
            raise ValueError(
                "AWS Bedrock authentication failed: Bearer token has expired. "
                "The AWS_BEARER_TOKEN_BEDROCK environment variable contains an expired pre-signed URL. "
                "Please refresh your token by generating a new pre-signed URL with AWS credentials. "
                "Alternatively, remove the bearer token and configure AWS credentials directly "
                "to use SigV4 authentication (recommended for enterprise deployments)."
            ) from original_error
        else:
            logger.error(f"AWS Bedrock authentication failed: {error_msg}")
            raise ValueError(
                f"AWS Bedrock authentication failed: {error_msg}. "
                "Please verify your API key is correct in the provider config or x-llamastack-provider-data header. "
                "The API key should be a valid AWS pre-signed URL for Bedrock's OpenAI-compatible endpoint."
            ) from original_error
