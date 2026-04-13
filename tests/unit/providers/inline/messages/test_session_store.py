# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Messages API session store.

Tests cover CRUD operations, message ordering, pagination,
session expiration, and corruption tolerance.
"""

import time

import pytest

from llama_stack.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from llama_stack.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from llama_stack.providers.inline.messages.session_store import SessionStore
from llama_stack_api.messages.models import (
    AnthropicTextBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    SessionMetadata,
)


@pytest.fixture
async def session_store(tmp_path):
    """Create a SQLite-backed session store for testing."""
    backend_name = "sql_test_sessions"
    db_path = tmp_path / "test_sessions.db"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})

    store = SessionStore(
        sql_store_config=SqlStoreReference(backend=backend_name, table_name="sessions"),
        policy=[],
    )
    await store.initialize()
    yield store
    await store.shutdown()


class TestSessionCreate:
    async def test_create_session_returns_valid_id(self, session_store):
        session = await session_store.create_session()
        assert session.id.startswith("sess_")
        assert len(session.id) == 29  # sess_ + 24 hex chars
        assert session.status == "active"
        assert session.message_count == 0

    async def test_create_session_with_metadata(self, session_store):
        meta = SessionMetadata(model="claude-sonnet-4-20250514")
        session = await session_store.create_session(metadata=meta)
        assert session.metadata.model == "claude-sonnet-4-20250514"

    async def test_create_multiple_sessions(self, session_store):
        s1 = await session_store.create_session()
        s2 = await session_store.create_session()
        assert s1.id != s2.id


class TestSessionRetrieve:
    async def test_get_existing_session(self, session_store):
        created = await session_store.create_session()
        fetched = await session_store.get_session(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_nonexistent_session_returns_none(self, session_store):
        result = await session_store.get_session("sess_nonexistent000000000")
        assert result is None

    async def test_list_sessions(self, session_store):
        await session_store.create_session()
        await session_store.create_session()
        sessions = await session_store.list_sessions()
        assert len(sessions) == 2

    async def test_list_sessions_with_status_filter(self, session_store):
        s1 = await session_store.create_session()
        await session_store.create_session()

        # Manually expire one
        await session_store.sql_store.update(
            "sessions",
            data={"status": "expired"},
            where={"id": s1.id},
        )

        active = await session_store.list_sessions(status="active")
        assert len(active) == 1

    async def test_delete_session(self, session_store):
        session = await session_store.create_session()
        deleted = await session_store.delete_session(session.id)
        assert deleted is True
        fetched = await session_store.get_session(session.id)
        assert fetched is None

    async def test_delete_nonexistent_returns_false(self, session_store):
        deleted = await session_store.delete_session("sess_nonexistent000000000")
        assert deleted is False


class TestMessageAppend:
    async def test_append_increments_count(self, session_store):
        session = await session_store.create_session()
        await session_store.append_message(
            session_id=session.id,
            role="user",
            content=[AnthropicTextBlock(type="text", text="Hello")],
        )
        updated = await session_store.get_session(session.id)
        assert updated.message_count == 1

    async def test_messages_ordered_by_sequence(self, session_store):
        session = await session_store.create_session()
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            await session_store.append_message(
                session_id=session.id,
                role=role,
                content=[AnthropicTextBlock(type="text", text=f"Msg {i}")],
            )
        messages = await session_store.get_messages(session.id)
        assert len(messages) == 5
        for i, msg in enumerate(messages):
            assert f"Msg {i}" in msg.content[0].text

    async def test_append_to_nonexistent_session_raises(self, session_store):
        with pytest.raises(ValueError, match="Failed to find session"):
            await session_store.append_message(
                session_id="sess_doesnotexist000000000",
                role="user",
                content=[AnthropicTextBlock(type="text", text="test")],
            )

    async def test_append_with_tool_use_content(self, session_store):
        session = await session_store.create_session()
        await session_store.append_message(
            session_id=session.id,
            role="assistant",
            content=[
                AnthropicTextBlock(type="text", text="Let me search for that."),
                AnthropicToolUseBlock(
                    type="tool_use",
                    id="toolu_abc123",
                    name="search",
                    input={"query": "test"},
                ),
            ],
        )
        messages = await session_store.get_messages(session.id)
        assert len(messages) == 1
        assert len(messages[0].content) == 2

    async def test_append_with_usage(self, session_store):
        session = await session_store.create_session()
        usage = AnthropicUsage(input_tokens=100, output_tokens=50)
        await session_store.append_message(
            session_id=session.id,
            role="assistant",
            content=[AnthropicTextBlock(type="text", text="response")],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=usage,
        )
        messages = await session_store.get_messages(session.id)
        assert messages[0].usage is not None
        assert messages[0].usage.input_tokens == 100
        assert messages[0].usage.output_tokens == 50
        assert messages[0].model == "claude-sonnet-4-20250514"
        assert messages[0].stop_reason == "end_turn"


class TestSessionExpiration:
    async def test_expire_old_sessions(self, session_store):
        session = await session_store.create_session()
        # Manually set updated_at to 2 days ago
        await session_store.sql_store.update(
            "sessions",
            data={"updated_at": int(time.time()) - 172800},
            where={"id": session.id},
        )
        expired = await session_store.expire_sessions(ttl_seconds=86400)
        assert expired == 1
        updated = await session_store.get_session(session.id)
        assert updated.status == "expired"

    async def test_active_sessions_not_expired(self, session_store):
        await session_store.create_session()
        expired = await session_store.expire_sessions(ttl_seconds=86400)
        assert expired == 0


class TestMessagePagination:
    async def test_get_messages_with_limit(self, session_store):
        session = await session_store.create_session()
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            await session_store.append_message(
                session_id=session.id,
                role=role,
                content=[AnthropicTextBlock(type="text", text=f"Msg {i}")],
            )
        messages = await session_store.get_messages(session.id, limit=5)
        assert len(messages) == 5
