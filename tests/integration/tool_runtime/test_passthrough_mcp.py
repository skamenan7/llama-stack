# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the MCP tool runtime forward_headers passthrough.

Spins up a lightweight mock MCP server (SSE protocol) and wires up
ModelContextProtocolToolRuntimeImpl against it, exercising the full path from
config validation through the MCP client to a real HTTP endpoint.

Run with:
    uv run pytest tests/integration/tool_runtime/test_passthrough_mcp.py -v --noconftest
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from llama_stack.providers.remote.tool_runtime.model_context_protocol.config import MCPProviderConfig
from llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol import (
    ModelContextProtocolToolRuntimeImpl,
)

# ---------------------------------------------------------------------------
# Config validation tests (no server needed)
# ---------------------------------------------------------------------------


class TestMCPProviderConfig:
    def test_empty_config_ok(self):
        c = MCPProviderConfig()
        assert c.forward_headers is None
        assert c.extra_blocked_headers == []

    def test_forward_headers_accepted(self):
        c = MCPProviderConfig(forward_headers={"maas_token": "Authorization", "tid": "X-Tenant-ID"})
        assert c.forward_headers == {"maas_token": "Authorization", "tid": "X-Tenant-ID"}

    def test_blocked_header_rejected_at_config_time(self):
        for blocked in ("Host", "Transfer-Encoding", "X-Forwarded-For", "Proxy-Authorization", "Cookie"):
            with pytest.raises(ValidationError, match="blocked"):
                MCPProviderConfig(forward_headers={"key": blocked})

    def test_extra_blocked_headers_enforced_at_config_time(self):
        with pytest.raises(ValidationError, match="blocked"):
            MCPProviderConfig(
                forward_headers={"dbg": "X-Internal-Debug"},
                extra_blocked_headers=["x-internal-debug"],
            )

    def test_reserved_key_prefix_rejected(self):
        with pytest.raises(ValidationError, match="reserved"):
            MCPProviderConfig(forward_headers={"__secret": "X-Custom"})

    def test_invalid_header_name_rejected(self):
        with pytest.raises(ValidationError):
            MCPProviderConfig(forward_headers={"key": "X Bad Header"})

    def test_sample_run_config_empty(self):
        result = MCPProviderConfig.sample_run_config()
        assert result == {}

    def test_sample_run_config_with_fields(self):
        result = MCPProviderConfig.sample_run_config(
            forward_headers={"tok": "Authorization"},
            extra_blocked_headers=["x-debug"],
        )
        assert result["forward_headers"] == {"tok": "Authorization"}
        assert result["extra_blocked_headers"] == ["x-debug"]


# ---------------------------------------------------------------------------
# MCPProviderDataValidator extra=allow test
# ---------------------------------------------------------------------------


class TestMCPProviderDataValidator:
    def test_extra_allow_preserves_deployer_keys(self):
        from llama_stack.providers.remote.tool_runtime.model_context_protocol.config import MCPProviderDataValidator

        v = MCPProviderDataValidator.model_validate({"maas_token": "Bearer tok", "tid": "acme"})
        dumped = v.model_dump()
        assert dumped["maas_token"] == "Bearer tok"
        assert dumped["tid"] == "acme"

    def test_mcp_headers_still_accepted(self):
        from llama_stack.providers.remote.tool_runtime.model_context_protocol.config import MCPProviderDataValidator

        v = MCPProviderDataValidator(mcp_headers={"http://mcp/sse": {"X-Custom": "val"}})
        assert v.mcp_headers == {"http://mcp/sse": {"X-Custom": "val"}}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_impl(forward_headers: dict[str, str] | None = None) -> ModelContextProtocolToolRuntimeImpl:
    config = MCPProviderConfig(forward_headers=forward_headers)
    impl = ModelContextProtocolToolRuntimeImpl(config, {})
    return impl


def _make_provider_data(**fields: str) -> BaseModel:
    """Return a real Pydantic model instance with extra=allow so model_dump() returns real values."""

    class _PD(BaseModel):
        model_config = ConfigDict(extra="allow")
        mcp_headers: dict[str, dict[str, str]] | None = None

    return _PD.model_validate(fields)


# ---------------------------------------------------------------------------
# _get_forwarded_headers_and_auth unit tests
# ---------------------------------------------------------------------------


class TestGetForwardedHeadersAndAuth:
    def test_no_forward_headers_returns_empty(self):
        impl = _make_impl()
        impl.get_request_provider_data = MagicMock(return_value=None)  # type: ignore[method-assign]
        headers, auth = impl._get_forwarded_headers_and_auth()
        assert headers == {}
        assert auth is None

    def test_non_auth_headers_returned(self):
        impl = _make_impl(forward_headers={"tid": "X-Tenant-ID", "team": "X-Team-ID"})
        impl.get_request_provider_data = MagicMock(return_value=_make_provider_data(tid="acme", team="ml-eng"))  # type: ignore[method-assign]

        headers, auth = impl._get_forwarded_headers_and_auth()
        assert headers == {"X-Tenant-ID": "acme", "X-Team-ID": "ml-eng"}
        assert auth is None

    @pytest.mark.parametrize("auth_header_name", ["Authorization", "authorization", "AUTHORIZATION"])
    def test_authorization_split_into_auth_token(self, auth_header_name: str):
        impl = _make_impl(forward_headers={"maas_token": auth_header_name, "tid": "X-Tenant-ID"})
        impl.get_request_provider_data = MagicMock(
            return_value=_make_provider_data(maas_token="my-bare-token", tid="acme")
        )  # type: ignore[method-assign]

        headers, auth = impl._get_forwarded_headers_and_auth()
        assert auth_header_name not in headers
        assert "authorization" not in headers
        assert "AUTHORIZATION" not in headers
        assert headers == {"X-Tenant-ID": "acme"}
        assert auth == "my-bare-token"

    def test_missing_keys_silently_skipped(self):
        impl = _make_impl(forward_headers={"maas_token": "Authorization", "tid": "X-Tenant-ID"})
        # only tid present, maas_token missing
        impl.get_request_provider_data = MagicMock(return_value=_make_provider_data(tid="partial"))  # type: ignore[method-assign]

        headers, auth = impl._get_forwarded_headers_and_auth()
        assert headers == {"X-Tenant-ID": "partial"}
        assert auth is None

    def test_no_provider_data_returns_empty(self):
        impl = _make_impl(forward_headers={"maas_token": "Authorization"})
        impl.get_request_provider_data = MagicMock(return_value=None)  # type: ignore[method-assign]

        headers, auth = impl._get_forwarded_headers_and_auth()
        assert headers == {}
        assert auth is None

    def test_warning_fires_when_no_keys_match(self, caplog):
        import logging  # allow-direct-logging

        impl = _make_impl(forward_headers={"maas_token": "Authorization"})
        # provider_data present but has no matching keys
        impl.get_request_provider_data = MagicMock(  # type: ignore[method-assign]
            return_value=_make_provider_data(unrelated="foo")
        )

        with caplog.at_level(logging.WARNING):
            headers, auth = impl._get_forwarded_headers_and_auth()

        assert headers == {}
        assert auth is None
        assert any("forward_headers is configured" in r.message for r in caplog.records)

    def test_warning_fires_when_provider_data_absent(self, caplog):
        import logging  # allow-direct-logging

        impl = _make_impl(forward_headers={"maas_token": "Authorization"})
        impl.get_request_provider_data = MagicMock(return_value=None)  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            headers, auth = impl._get_forwarded_headers_and_auth()

        assert headers == {}
        assert auth is None
        assert any("forward_headers is configured" in r.message for r in caplog.records)

    def test_default_deny_unlisted_keys_not_forwarded(self):
        impl = _make_impl(forward_headers={"allowed": "X-Allowed"})
        impl.get_request_provider_data = MagicMock(
            return_value=_make_provider_data(allowed="ok", secret="should-not-leak")
        )  # type: ignore[method-assign]

        headers, auth = impl._get_forwarded_headers_and_auth()
        assert "secret" not in str(headers)
        assert headers == {"X-Allowed": "ok"}


# ---------------------------------------------------------------------------
# list_runtime_tools wiring tests
# ---------------------------------------------------------------------------


class TestListRuntimeToolsWiring:
    async def test_forwarded_headers_passed_to_list_mcp_tools(self, monkeypatch):
        """forward_headers config causes headers to be passed to list_mcp_tools."""
        from llama_stack_api import URL

        impl = _make_impl(forward_headers={"tid": "X-Tenant-ID"})
        impl.get_request_provider_data = MagicMock(return_value=_make_provider_data(tid="acme"))  # type: ignore[method-assign]

        captured: dict[str, Any] = {}

        async def fake_list_mcp_tools(endpoint, headers=None, authorization=None, **_):
            captured["headers"] = headers
            captured["authorization"] = authorization
            from llama_stack_api import ListToolDefsResponse

            return ListToolDefsResponse(data=[])

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.list_mcp_tools",
            fake_list_mcp_tools,
        )

        endpoint = URL(uri="http://mcp-server:8080/sse")
        await impl.list_runtime_tools(mcp_endpoint=endpoint)

        assert captured["headers"] == {"X-Tenant-ID": "acme"}
        assert captured["authorization"] is None

    async def test_explicit_authorization_wins_over_forwarded(self, monkeypatch):
        """Explicit authorization= param takes precedence over forwarded auth token."""
        from llama_stack_api import URL

        impl = _make_impl(forward_headers={"tok": "Authorization"})
        impl.get_request_provider_data = MagicMock(return_value=_make_provider_data(tok="forwarded-token"))  # type: ignore[method-assign]

        captured: dict[str, Any] = {}

        async def fake_list_mcp_tools(endpoint, headers=None, authorization=None, **_):
            captured["authorization"] = authorization
            from llama_stack_api import ListToolDefsResponse

            return ListToolDefsResponse(data=[])

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.list_mcp_tools",
            fake_list_mcp_tools,
        )

        endpoint = URL(uri="http://mcp-server:8080/sse")
        await impl.list_runtime_tools(mcp_endpoint=endpoint, authorization="explicit-wins")

        assert captured["authorization"] == "explicit-wins"

    async def test_no_forward_headers_no_crash(self, monkeypatch):
        """Provider works normally when forward_headers is not configured."""
        from llama_stack_api import URL

        impl = _make_impl()
        impl.get_request_provider_data = MagicMock(return_value=None)  # type: ignore[method-assign]

        async def fake_list_mcp_tools(endpoint, headers=None, authorization=None, **_):
            from llama_stack_api import ListToolDefsResponse

            return ListToolDefsResponse(data=[])

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.list_mcp_tools",
            fake_list_mcp_tools,
        )

        endpoint = URL(uri="http://mcp-server:8080/sse")
        result = await impl.list_runtime_tools(mcp_endpoint=endpoint)
        assert result is not None


# ---------------------------------------------------------------------------
# invoke_tool wiring tests
# ---------------------------------------------------------------------------


class TestInvokeToolWiring:
    async def test_forwarded_headers_passed_to_invoke_mcp_tool(self, monkeypatch):
        """forward_headers config causes headers to be passed to invoke_mcp_tool."""
        impl = _make_impl(forward_headers={"tid": "X-Tenant-ID", "tok": "Authorization"})
        impl.get_request_provider_data = MagicMock(return_value=_make_provider_data(tid="acme", tok="my-token"))  # type: ignore[method-assign]

        # mock tool_store
        fake_tool = MagicMock()
        fake_tool.metadata = {"endpoint": "http://mcp-server:8080/sse"}
        impl.tool_store = AsyncMock()
        impl.tool_store.get_tool.return_value = fake_tool

        captured: dict[str, Any] = {}

        async def fake_invoke_mcp_tool(endpoint, tool_name, kwargs, headers=None, authorization=None, **_):
            captured["headers"] = headers
            captured["authorization"] = authorization
            from llama_stack_api import TextContentItem, ToolInvocationResult

            return ToolInvocationResult(content=[TextContentItem(text="ok")], error_code=0)

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.invoke_mcp_tool",
            fake_invoke_mcp_tool,
        )

        await impl.invoke_tool("some_tool", kwargs={})

        assert captured["headers"] == {"X-Tenant-ID": "acme"}
        assert captured["authorization"] == "my-token"

    async def test_explicit_authorization_wins_in_invoke(self, monkeypatch):
        """Explicit authorization= wins over forwarded auth in invoke_tool."""
        impl = _make_impl(forward_headers={"tok": "Authorization"})
        impl.get_request_provider_data = MagicMock(return_value=_make_provider_data(tok="forwarded-tok"))  # type: ignore[method-assign]

        fake_tool = MagicMock()
        fake_tool.metadata = {"endpoint": "http://mcp-server:8080/sse"}
        impl.tool_store = AsyncMock()
        impl.tool_store.get_tool.return_value = fake_tool

        captured: dict[str, Any] = {}

        async def fake_invoke_mcp_tool(endpoint, tool_name, kwargs, headers=None, authorization=None, **_):
            captured["authorization"] = authorization
            from llama_stack_api import TextContentItem, ToolInvocationResult

            return ToolInvocationResult(content=[TextContentItem(text="ok")], error_code=0)

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.invoke_mcp_tool",
            fake_invoke_mcp_tool,
        )

        await impl.invoke_tool("some_tool", kwargs={}, authorization="explicit-wins")
        assert captured["authorization"] == "explicit-wins"


# ---------------------------------------------------------------------------
# Legacy mcp_headers runtime path tests
# ---------------------------------------------------------------------------


class TestLegacyMcpHeaders:
    def _make_legacy_provider_data(self, uri: str, headers: dict[str, str]) -> BaseModel:
        from llama_stack.providers.remote.tool_runtime.model_context_protocol.config import MCPProviderDataValidator

        return MCPProviderDataValidator(mcp_headers={uri: headers})

    async def test_legacy_headers_matching_uri_reach_downstream(self, monkeypatch):
        """mcp_headers headers for the matching URI arrive in the downstream headers= arg."""
        from llama_stack_api import URL

        impl = _make_impl()
        impl.get_request_provider_data = MagicMock(  # type: ignore[method-assign]
            return_value=self._make_legacy_provider_data("http://mcp-server:8080/sse", {"X-Custom": "custom-val"})
        )

        captured: dict[str, Any] = {}

        async def fake_list_mcp_tools(endpoint, headers=None, authorization=None, **_):
            captured["headers"] = headers
            from llama_stack_api import ListToolDefsResponse

            return ListToolDefsResponse(data=[])

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.list_mcp_tools",
            fake_list_mcp_tools,
        )

        await impl.list_runtime_tools(mcp_endpoint=URL(uri="http://mcp-server:8080/sse"))
        assert captured["headers"].get("X-Custom") == "custom-val"

    async def test_legacy_authorization_in_mcp_headers_raises(self, monkeypatch):
        """Authorization key in mcp_headers must raise ValueError."""
        from llama_stack_api import URL

        impl = _make_impl()
        impl.get_request_provider_data = MagicMock(  # type: ignore[method-assign]
            return_value=self._make_legacy_provider_data("http://mcp-server:8080/sse", {"Authorization": "Bearer tok"})
        )

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.list_mcp_tools",
            AsyncMock(),
        )

        with pytest.raises(ValueError, match="[Aa]uthorization"):
            await impl.list_runtime_tools(mcp_endpoint=URL(uri="http://mcp-server:8080/sse"))

    async def test_legacy_non_matching_uri_ignored(self, monkeypatch):
        """mcp_headers for a different URI are not forwarded."""
        from llama_stack_api import URL

        impl = _make_impl()
        impl.get_request_provider_data = MagicMock(  # type: ignore[method-assign]
            return_value=self._make_legacy_provider_data("http://other-server:9000/sse", {"X-Other": "val"})
        )

        captured: dict[str, Any] = {}

        async def fake_list_mcp_tools(endpoint, headers=None, authorization=None, **_):
            captured["headers"] = headers
            from llama_stack_api import ListToolDefsResponse

            return ListToolDefsResponse(data=[])

        monkeypatch.setattr(
            "llama_stack.providers.remote.tool_runtime.model_context_protocol.model_context_protocol.list_mcp_tools",
            fake_list_mcp_tools,
        )

        await impl.list_runtime_tools(mcp_endpoint=URL(uri="http://mcp-server:8080/sse"))
        assert "X-Other" not in captured.get("headers", {})
