# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Server-side session store for Anthropic Messages API conversations.

Provides persistent storage for multi-turn conversations, enabling
session recovery and context persistence for Claude Code / Agent SDK.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from llama_stack.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from llama_stack.core.storage.sqlstore.sqlstore import sqlstore_impl
from llama_stack.log import get_logger
from llama_stack_api.internal.sqlstore import ColumnDefinition, ColumnType
from llama_stack_api.messages.models import (
    AnthropicContentBlock,
    AnthropicUsage,
    Session,
    SessionMessage,
    SessionMetadata,
)

logger = get_logger(name=__name__, category="messages")


class SessionStore:
    """Persistent store for Anthropic Messages API sessions."""

    def __init__(self, sql_store_config, policy):
        self.config = sql_store_config
        self.policy = policy

    async def initialize(self):
        """Create tables if they do not exist."""
        base_store = sqlstore_impl(self.config)
        self.sql_store = AuthorizedSqlStore(base_store, self.policy)

        await self.sql_store.create_table(
            "sessions",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "updated_at": ColumnType.INTEGER,
                "metadata": ColumnType.JSON,
                "status": ColumnType.STRING,
                "message_count": ColumnType.INTEGER,
            },
        )

        await self.sql_store.create_table(
            "session_messages",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "session_id": ColumnType.STRING,
                "order_key": ColumnType.INTEGER,
                "role": ColumnType.STRING,
                "content": ColumnType.JSON,
                "created_at": ColumnType.INTEGER,
                "model": ColumnType.STRING,
                "stop_reason": ColumnType.STRING,
                "usage": ColumnType.JSON,
            },
        )

    async def shutdown(self) -> None:
        pass

    async def create_session(
        self,
        metadata: SessionMetadata | None = None,
    ) -> Session:
        """Create a new session and return it."""
        now = int(time.time())
        session = Session(
            id=f"sess_{uuid.uuid4().hex[:24]}",
            created_at=now,
            updated_at=now,
            metadata=metadata or SessionMetadata(),
            message_count=0,
            status="active",
        )
        await self.sql_store.insert(
            "sessions",
            {
                "id": session.id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "metadata": session.metadata.model_dump(),
                "status": session.status,
                "message_count": 0,
            },
        )
        logger.info("Session created", session_id=session.id)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID, or None if not found."""
        row = await self.sql_store.fetch_one(
            "sessions",
            where={"id": session_id},
        )
        if not row:
            return None
        return Session(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=SessionMetadata(**row["metadata"]),
            status=row["status"],
            message_count=row["message_count"],
        )

    async def list_sessions(
        self,
        limit: int = 20,
        after: str | None = None,
        status: str | None = None,
    ) -> list[Session]:
        """List sessions with optional filtering and pagination."""
        where = {}
        if status:
            where["status"] = status

        result = await self.sql_store.fetch_all(
            "sessions",
            where=where if where else None,
            order_by=[("updated_at", "desc")],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        return [
            Session(
                id=row["id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=SessionMetadata(**row["metadata"]),
                status=row["status"],
                message_count=row["message_count"],
            )
            for row in result.data
        ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages. Returns True if found."""
        session = await self.get_session(session_id)
        if session is None:
            return False
        await self.sql_store.delete("session_messages", where={"session_id": session_id})
        await self.sql_store.delete("sessions", where={"id": session_id})
        logger.info("Session deleted", session_id=session_id)
        return True

    async def append_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: list[AnthropicContentBlock],
        model: str | None = None,
        stop_reason: str | None = None,
        usage: AnthropicUsage | None = None,
    ) -> SessionMessage:
        """Append a message to a session's history.

        Uses UUID-based message IDs and timestamp-based ordering to avoid
        TOCTOU races when concurrent appends hit the same session.
        """
        now = int(time.time())
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Failed to find session {session_id}")

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        msg = SessionMessage(
            id=msg_id,
            role=role,
            content=content,
            created_at=now,
            model=model,
            stop_reason=stop_reason,
            usage=usage,
        )

        # Use created_at timestamp for ordering instead of a racy sequence counter.
        # Concurrent appends get unique IDs and slightly different timestamps.
        await self.sql_store.insert(
            "session_messages",
            {
                "id": msg_id,
                "session_id": session_id,
                "order_key": now,
                "role": msg.role,
                "content": [c.model_dump() for c in content],
                "created_at": now,
                "model": model,
                "stop_reason": stop_reason,
                "usage": usage.model_dump() if usage else None,
            },
        )

        # Increment message_count. In case of concurrent writes, the count
        # may drift slightly but messages are ordered by order_key (timestamp).
        await self.sql_store.update(
            "sessions",
            data={
                "updated_at": now,
                "message_count": session.message_count + 1,
            },
            where={"id": session_id},
        )

        logger.info(
            "Message appended",
            session_id=session_id,
            role=role,
            message_id=msg_id,
        )
        return msg

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        after: str | None = None,
    ) -> list[SessionMessage]:
        """Retrieve messages for a session, ordered by sequence."""
        result = await self.sql_store.fetch_all(
            "session_messages",
            where={"session_id": session_id},
            order_by=[("order_key", "asc")],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        messages = []
        for row in result.data:
            messages.append(
                SessionMessage.model_validate(
                    {
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "created_at": row["created_at"],
                        "model": row.get("model"),
                        "stop_reason": row.get("stop_reason"),
                        "usage": row.get("usage"),
                    }
                )
            )
        return messages

    async def expire_sessions(self, ttl_seconds: int) -> int:
        """Mark sessions older than TTL as expired. Returns count.

        Processes in batches of 500 to avoid loading unbounded rows.
        """
        cutoff = int(time.time()) - ttl_seconds
        result = await self.sql_store.fetch_all(
            "sessions",
            where={"status": "active"},
            order_by=[("updated_at", "asc")],
            limit=500,
        )
        expired = 0
        for row in result.data:
            if row["updated_at"] >= cutoff:
                break
            await self.sql_store.update(
                "sessions",
                data={"status": "expired"},
                where={"id": row["id"]},
            )
            expired += 1
        if expired:
            logger.info("Expired sessions", count=expired, ttl=ttl_seconds)
        return expired
