# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llama_stack.providers.utils.forward_headers import validate_forward_headers_config


class MCPProviderDataValidator(BaseModel):
    """
    Validator for MCP provider-specific data passed via request headers.

    extra="allow" so deployer-defined forward_headers keys (e.g. "maas_api_token")
    survive Pydantic parsing — they can't be declared as typed fields because the
    key names are operator-configured at deploy time.

    The legacy mcp_headers URI-keyed path is kept for backward compatibility.
    """

    model_config = ConfigDict(extra="allow")

    mcp_headers: dict[str, dict[str, str]] | None = Field(
        default=None,
        description="Legacy URI-keyed headers dict for backward compatibility. New deployments should use forward_headers in the provider config instead.",
    )


class MCPProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forward_headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Mapping of X-LlamaStack-Provider-Data keys to outbound HTTP header names. "
            "Only listed keys are forwarded — all others are ignored (default-deny). "
            "When targeting 'Authorization', the provider-data value must be a bare "
            "Bearer token (e.g. 'my-jwt-token', not 'Bearer my-jwt-token') — the "
            "'Bearer ' prefix is added automatically by the MCP client. "
            "Header name values should use canonical HTTP casing (e.g. 'Authorization', 'X-Tenant-ID'). "
            "Keys with a __ prefix and core security-sensitive headers (for example Host, "
            "Content-Type, Transfer-Encoding, Cookie) are rejected at config parse time. "
            'Example: {"maas_api_token": "Authorization", "tenant_id": "X-Tenant-ID"}'
        ),
    )
    extra_blocked_headers: list[str] = Field(
        default_factory=list,
        description=(
            "Additional outbound header names to block in forward_headers. "
            "Names are matched case-insensitively and added to the core blocked list. "
            "This can tighten policy but cannot unblock core security-sensitive headers."
        ),
    )

    @model_validator(mode="after")
    def validate_forward_headers(self) -> "MCPProviderConfig":
        validate_forward_headers_config(self.forward_headers, self.extra_blocked_headers)
        return self

    @classmethod
    def sample_run_config(
        cls,
        forward_headers: dict[str, str] | None = None,
        extra_blocked_headers: list[str] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if forward_headers is not None:
            config["forward_headers"] = forward_headers
        if extra_blocked_headers is not None:
            config["extra_blocked_headers"] = extra_blocked_headers
        return config
