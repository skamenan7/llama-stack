# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field


class SessionStoreConfig(BaseModel):
    """Configuration for the session persistence store."""

    enabled: bool = Field(
        default=False,
        description="Enable server-side session persistence.",
    )
    session_ttl_seconds: int = Field(
        default=86400,
        description="Session time-to-live in seconds (default: 24 hours).",
    )
    max_history_turns: int = Field(
        default=100,
        description="Maximum number of message turns to store per session.",
    )


class MessagesConfig(BaseModel):
    """Configuration for the built-in Anthropic Messages API adapter."""

    session_store: SessionStoreConfig = Field(
        default_factory=SessionStoreConfig,
        description="Session persistence configuration.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "") -> dict[str, Any]:
        return {
            "session_store": {
                "enabled": False,
                "session_ttl_seconds": 86400,
                "max_history_turns": 100,
            }
        }
