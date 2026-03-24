# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urlparse

from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.forward_headers import build_forwarded_headers
from llama_stack.providers.utils.tools.mcp import invoke_mcp_tool, list_mcp_tools
from llama_stack_api import (
    URL,
    Api,
    ListToolDefsResponse,
    ToolGroup,
    ToolGroupsProtocolPrivate,
    ToolInvocationResult,
    ToolRuntime,
)

from .config import MCPProviderConfig

logger = get_logger(__name__, category="tools")


class ModelContextProtocolToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    """Tool runtime for discovering and invoking tools via the Model Context Protocol."""

    def __init__(self, config: MCPProviderConfig, _deps: dict[Api, Any]):
        self.config = config

    async def initialize(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def list_runtime_tools(
        self,
        tool_group_id: str | None = None,
        mcp_endpoint: URL | None = None,
        authorization: str | None = None,
    ) -> ListToolDefsResponse:
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        forwarded_headers, forwarded_auth = self._get_forwarded_headers_and_auth()
        # legacy mcp_headers URI-keyed path (backward compat)
        legacy_headers = await self.get_headers_from_request(mcp_endpoint.uri)
        merged_headers = {**forwarded_headers, **legacy_headers}
        # explicit authorization= param from caller wins over forwarded
        effective_auth = authorization or forwarded_auth

        return await list_mcp_tools(endpoint=mcp_endpoint.uri, headers=merged_headers, authorization=effective_auth)

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint: str = tool.metadata["endpoint"]
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        forwarded_headers, forwarded_auth = self._get_forwarded_headers_and_auth()
        # legacy mcp_headers URI-keyed path (backward compat)
        legacy_headers = await self.get_headers_from_request(endpoint)
        merged_headers = {**forwarded_headers, **legacy_headers}
        # explicit authorization= param from caller wins over forwarded
        effective_auth = authorization or forwarded_auth

        return await invoke_mcp_tool(
            endpoint=endpoint,
            tool_name=tool_name,
            kwargs=kwargs,
            headers=merged_headers,
            authorization=effective_auth,
        )

    def _get_forwarded_headers_and_auth(self) -> tuple[dict[str, str], str | None]:
        """Extract forwarded headers from provider data per the admin-configured allowlist.

        Splits the output of build_forwarded_headers() into non-Authorization headers
        and an auth token. Authorization-mapped values must be bare tokens (no 'Bearer '
        prefix) per the forward_headers field description — prepare_mcp_headers() adds
        the prefix when passing via the authorization= param.

        Returns:
            (non_auth_headers, auth_token) where auth_token is None if not configured.
        """
        provider_data = self.get_request_provider_data()
        all_headers = build_forwarded_headers(provider_data, self.config.forward_headers)

        if not all_headers:
            if self.config.forward_headers:
                logger.warning(
                    "forward_headers is configured but no headers were forwarded — "
                    "outbound request may be unauthenticated"
                )
            return {}, None

        # Pull out Authorization (case-insensitive) so it goes via the authorization=
        # param — prepare_mcp_headers() rejects Authorization in the headers= dict.
        auth_token: str | None = None
        non_auth: dict[str, str] = {}
        for name, value in all_headers.items():
            if name.lower() == "authorization":
                auth_token = value
            else:
                non_auth[name] = value

        return non_auth, auth_token

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> dict[str, str]:
        """Extract headers from the legacy mcp_headers URI-keyed provider data.

        Kept for backward compatibility. New deployments should use forward_headers
        in the provider config instead.

        Raises:
            ValueError: If Authorization header is found in mcp_headers (must use
                the dedicated authorization parameter instead).
        """

        def canonicalize_uri(uri: str) -> str:
            return f"{urlparse(uri).netloc or ''}/{urlparse(uri).path or ''}"

        headers = {}

        provider_data = self.get_request_provider_data()
        if provider_data and hasattr(provider_data, "mcp_headers") and provider_data.mcp_headers:
            for uri, values in provider_data.mcp_headers.items():
                if canonicalize_uri(uri) != canonicalize_uri(mcp_endpoint_uri):
                    continue

                # Reject Authorization in mcp_headers - must use authorization parameter
                for key in values.keys():
                    if key.lower() == "authorization":
                        raise ValueError(
                            "Authorization cannot be provided via mcp_headers in provider_data. "
                            "Please use the dedicated 'authorization' parameter instead. "
                            "Example: tool_runtime.invoke_tool(..., authorization='your-token')"
                        )
                    headers[key] = values[key]

        return headers
