# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for session context persistence.

Tests cover context loading, deduplication, sliding window
truncation, and system prompt preservation.
"""

from unittest.mock import AsyncMock

import pytest

from llama_stack.providers.inline.messages.config import MessagesConfig, SessionStoreConfig
from llama_stack.providers.inline.messages.impl import BuiltinMessagesImpl
from llama_stack_api.messages.models import (
    AnthropicMessage,
    AnthropicTextBlock,
)


@pytest.fixture
def messages_impl():
    config = MessagesConfig(
        session_store=SessionStoreConfig(enabled=False, max_history_turns=10),
    )
    inference = AsyncMock()
    impl = BuiltinMessagesImpl(config, inference)
    return impl


class TestSlidingWindow:
    def test_no_truncation_under_limit(self, messages_impl):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text=f"msg {i}")],
            )
            for i in range(5)
        ]
        result = messages_impl._apply_sliding_window(msgs, max_turns=10)
        assert len(result) == 5

    def test_truncation_over_limit(self, messages_impl):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text=f"msg {i}")],
            )
            for i in range(20)
        ]
        result = messages_impl._apply_sliding_window(msgs, max_turns=10)
        assert len(result) == 10
        # Should keep the LAST 10
        assert "msg 10" in result[0].content[0].text
        assert "msg 19" in result[-1].content[0].text

    def test_exact_limit_no_truncation(self, messages_impl):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text=f"msg {i}")],
            )
            for i in range(10)
        ]
        result = messages_impl._apply_sliding_window(msgs, max_turns=10)
        assert len(result) == 10

    def test_empty_messages(self, messages_impl):
        result = messages_impl._apply_sliding_window([], max_turns=10)
        assert len(result) == 0


class TestFindOverlap:
    def test_no_overlap(self, messages_impl):
        history = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text="A")],
            ),
        ]
        incoming = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text="B")],
            ),
        ]
        assert messages_impl._find_overlap(history, incoming) == 0

    def test_full_overlap(self, messages_impl):
        msg = AnthropicMessage(
            role="user",
            content=[AnthropicTextBlock(type="text", text="A")],
        )
        history = [msg]
        incoming = [msg]
        assert messages_impl._find_overlap(history, incoming) == 1

    def test_partial_overlap(self, messages_impl):
        msgs = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text=f"msg {i}")],
            )
            for i in range(5)
        ]
        history = msgs[:3]
        incoming = msgs[1:5]
        assert messages_impl._find_overlap(history, incoming) == 2

    def test_empty_history(self, messages_impl):
        incoming = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text="A")],
            ),
        ]
        assert messages_impl._find_overlap([], incoming) == 0

    def test_empty_incoming(self, messages_impl):
        history = [
            AnthropicMessage(
                role="user",
                content=[AnthropicTextBlock(type="text", text="A")],
            ),
        ]
        assert messages_impl._find_overlap(history, []) == 0
