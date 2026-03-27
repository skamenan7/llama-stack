# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
SigV4 authentication for AWS Bedrock OpenAI-compatible endpoint.

This module provides httpx.Auth implementation that signs requests using
AWS Signature Version 4, enabling IAM/STS authentication with the Bedrock
OpenAI-compatible API endpoint.

Supported credential sources (via boto3 credential chain):
- Static credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Web Identity Federation (AWS_ROLE_ARN, AWS_WEB_IDENTITY_TOKEN_FILE)
- IAM roles (IMDS for EC2, ECS task roles, Lambda execution roles)
- AWS profiles (~/.aws/credentials)

Web Identity Federation enables keyless authentication in:
- Kubernetes/OpenShift with IRSA (IAM Roles for Service Accounts)
- GitHub Actions with OIDC (aws-actions/configure-aws-credentials)
- Any OIDC-compatible identity provider

Environment variables for Web Identity:
    AWS_ROLE_ARN: ARN of the IAM role to assume
    AWS_WEB_IDENTITY_TOKEN_FILE: Path to the OIDC token file
        Common paths:
        - EKS: /var/run/secrets/eks.amazonaws.com/serviceaccount/token
        - Generic Kubernetes: /var/run/secrets/kubernetes.io/serviceaccount/token
        - GitHub Actions: Set automatically by aws-actions/configure-aws-credentials
    AWS_DEFAULT_REGION: AWS region for the Bedrock endpoint

Credentials are automatically refreshed by boto3 when they expire.

References:
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html
- https://github.com/meta-llama/llama-stack/issues/4730
- https://github.com/opendatahub-io/llama-stack-distribution/issues/112
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="providers")


class BedrockSigV4Auth(httpx.Auth):
    """
    httpx.Auth implementation for AWS SigV4 signing.

    This auth handler signs HTTP requests using AWS Signature Version 4,
    which is required for IAM-based authentication with AWS services.

    The implementation:
    - Uses boto3's credential chain for automatic credential resolution
    - Supports credential refresh for temporary credentials (STS, IRSA)
    - Only signs headers that won't be modified by httpx during transmission
    - Replaces any existing Authorization header with SigV4 signature
    """

    def __init__(
        self,
        region: str,
        service: str = "bedrock",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        aws_role_arn: str | None = None,
        aws_web_identity_token_file: str | None = None,
        aws_role_session_name: str | None = None,
        session_ttl: int | None = 3600,
    ):
        """
        Initialize SigV4 auth handler.

        Args:
            region: AWS region (e.g., "us-east-1")
            service: AWS service name for SigV4 signing. Use "bedrock" (the signing
                     name from botocore metadata), NOT "bedrock-runtime" (the endpoint
                     prefix). The signing name is used in the SigV4 credential scope.
            aws_role_arn: Optional IAM role ARN to assume.
            aws_web_identity_token_file: Optional path to web identity token file.
            aws_role_session_name: Optional session name for role assumption.
            session_ttl: Optional session TTL in seconds.
        """
        self._region = region
        self._service = service
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._profile_name = profile_name
        self._aws_role_arn = aws_role_arn
        self._aws_web_identity_token_file = aws_web_identity_token_file
        self._aws_role_session_name = aws_role_session_name
        self._session_ttl = session_ttl or 3600
        self._lock = threading.Lock()
        self._session: Any = None  # boto3.Session | None — Any because boto3 is an optional dep

    def _get_credentials(self) -> Any:
        """Get current AWS credentials from boto3 session."""
        from llama_stack.providers.utils.bedrock.refreshable_boto_session import (
            RefreshableBotoSession,
        )

        with self._lock:
            if self._session is None:
                if self._aws_role_arn:
                    self._session = RefreshableBotoSession(
                        region_name=self._region,
                        aws_access_key_id=self._aws_access_key_id,
                        aws_secret_access_key=self._aws_secret_access_key,
                        aws_session_token=self._aws_session_token,
                        profile_name=self._profile_name,
                        sts_arn=self._aws_role_arn,
                        web_identity_token_file=self._aws_web_identity_token_file,
                        session_name=self._aws_role_session_name,
                        session_ttl=self._session_ttl,
                    ).refreshable_session()
                else:
                    import boto3

                    session_args = {
                        "region_name": self._region,
                        "aws_access_key_id": self._aws_access_key_id,
                        "aws_secret_access_key": self._aws_secret_access_key,
                        "aws_session_token": self._aws_session_token,
                        "profile_name": self._profile_name,
                    }
                    session_args = {k: v for k, v in session_args.items() if v is not None}
                    self._session = boto3.Session(**session_args)

            credentials = self._session.get_credentials()  # type: ignore[attr-defined]
            if credentials is None:
                raise RuntimeError(
                    "Failed to load AWS credentials. Ensure AWS credentials are "
                    "configured via environment variables (AWS_ACCESS_KEY_ID, "
                    "AWS_SECRET_ACCESS_KEY), IAM role, or AWS profile."
                )
            return credentials.get_frozen_credentials()

    def _sign_request(self, request: httpx.Request) -> None:
        """
        Sign the request using SigV4 (modifies request in place).

        This is the core signing logic, extracted to be reusable by both
        sync and async auth flows.

        Note: Any existing Authorization header (e.g., from OpenAI SDK's
        "Bearer <NOTUSED>" placeholder) is explicitly removed and replaced
        with the SigV4 signature.
        """
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials = self._get_credentials()

        # Remove any existing Authorization header (e.g., Bearer placeholder from OpenAI SDK)
        # SigV4 will add the proper Authorization header after signing
        if "authorization" in request.headers:
            del request.headers["authorization"]

        # Prepare headers to sign - only include stable headers
        # that won't be modified by httpx during transmission.
        #
        # Use netloc (host:port) to handle non-default ports correctly.
        # If an explicit Host header exists, prefer that for consistency.
        host = request.headers.get("host") or str(request.url.netloc)
        headers_to_sign = {"host": host}

        # Only include content-type if present in the request.
        # Don't force a default to avoid signature mismatch if actual differs.
        if "content-type" in request.headers:
            headers_to_sign["content-type"] = request.headers["content-type"]

        # Add other headers that should be signed if present
        for header_name in ["x-amz-content-sha256", "x-amz-security-token"]:
            if header_name in request.headers:
                headers_to_sign[header_name] = request.headers[header_name]

        # Read request content, handling streaming requests
        try:
            content = request.content
        except httpx.RequestNotRead:
            # For streaming requests, read the content first
            content = request.read()

        # Create AWS request for signing
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=content,
            headers=headers_to_sign,
        )

        # Sign the request
        signer = SigV4Auth(credentials, self._service, self._region)
        signer.add_auth(aws_request)

        # Copy signed headers back to the original request
        # This includes Authorization, X-Amz-Date, and potentially X-Amz-Security-Token
        for key, value in aws_request.headers.items():
            request.headers[key] = value

        logger.debug(
            f"SigV4 signed request: method={request.method}, "
            f"path={request.url.path}, service={self._service}, region={self._region}"
        )

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """
        Sign the request using SigV4 (sync version).

        This method is called by httpx for sync clients.
        """
        self._sign_request(request)
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """
        Sign the request using SigV4 (async version).

        This method is called by httpx for async clients. It offloads the
        signing operation to a thread pool to avoid blocking the event loop
        during credential resolution (which may involve IMDS calls or file I/O).
        """
        await asyncio.to_thread(self._sign_request, request)
        yield request
