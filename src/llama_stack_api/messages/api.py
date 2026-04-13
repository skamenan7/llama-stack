# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .models import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
    Session,
    SessionMetadata,
)


@runtime_checkable
class Messages(Protocol):
    """Protocol for the Anthropic Messages API."""

    async def create_message(
        self,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]: ...

    async def count_message_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse: ...

    async def create_session(
        self,
        metadata: SessionMetadata | None = None,
    ) -> Session: ...

    async def get_session(
        self,
        session_id: str,
    ) -> Session | None: ...

    async def list_sessions(
        self,
        limit: int = 20,
        after: str | None = None,
        status: str | None = None,
    ) -> list[Session]: ...

    async def delete_session(
        self,
        session_id: str,
    ) -> bool: ...
