# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx connect codex` CLI command."""

import argparse
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from ogx.cli.connect.codex import ConnectCodex


@pytest.fixture
def connect_codex() -> ConnectCodex:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return ConnectCodex(subparsers)


def _make_model(model_id: str, model_type: str = "llm") -> MagicMock:
    model = MagicMock()
    model.id = model_id
    model.model_extra = {"custom_metadata": {"model_type": model_type}}
    return model


def _make_mock_client(models: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.data = models
    client.models.list.return_value = response
    return client


class TestArguments:
    def test_defaults(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args([])
        assert args.port == 8321
        assert args.host == "localhost"
        assert args.model is None

    def test_port_override(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_host_override(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_model_override(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        assert args.model == "openai/gpt-4o"

    def test_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_PORT", "9999")
        subparsers = argparse.ArgumentParser().add_subparsers()
        instance = ConnectCodex(subparsers)
        args = instance.parser.parse_args([])
        assert args.port == 9999


class TestCodexDetection:
    def test_exits_when_codex_not_in_path(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args([])
        with patch("ogx.cli.connect.codex.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                connect_codex._run_connect_codex_cmd(args)

    def test_continues_when_codex_found(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit) as exc_info:
                connect_codex._run_connect_codex_cmd(args)
            assert exc_info.value.code == 0


class TestServerProbe:
    def test_uses_explicit_timeout(self, connect_codex: ConnectCodex) -> None:
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client) as mock_openai:
            connect_codex._fetch_models("http://localhost:8321/v1")

        assert mock_openai.call_args.kwargs["timeout"] == connect_codex.MODEL_DISCOVERY_TIMEOUT_SECONDS

    def test_exits_when_server_unreachable(self, connect_codex: ConnectCodex) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_codex._fetch_models("http://localhost:8321/v1")

    def test_exits_when_server_probe_times_out(self, connect_codex: ConnectCodex) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APITimeoutError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_codex._fetch_models("http://localhost:8321/v1")

    def test_exits_on_server_error(self, connect_codex: ConnectCodex) -> None:
        mock_client = MagicMock()
        mock_response = httpx.Response(500, request=httpx.Request("GET", "http://localhost"))
        mock_client.models.list.side_effect = APIStatusError("server error", response=mock_response, body=None)

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_codex._fetch_models("http://localhost:8321/v1")

    def test_returns_models_on_success(self, connect_codex: ConnectCodex) -> None:
        mock_client = _make_mock_client([_make_model("openai/gpt-4o"), _make_model("meta/llama-3.1-8b")])

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            models = connect_codex._fetch_models("http://localhost:8321/v1")
        assert models == ["openai/gpt-4o", "meta/llama-3.1-8b"]


class TestModelSelection:
    def test_uses_specified_model(self, connect_codex: ConnectCodex) -> None:
        result = connect_codex._select_default_model("openai/gpt-4o", ["openai/gpt-4o", "meta/llama-3.1-8b"])
        assert result == "openai/gpt-4o"

    def test_exits_when_specified_model_not_found(self, connect_codex: ConnectCodex) -> None:
        with pytest.raises(SystemExit):
            connect_codex._select_default_model("nonexistent", ["openai/gpt-4o", "meta/llama-3.1-8b"])

    def test_defaults_to_first_model(self, connect_codex: ConnectCodex) -> None:
        result = connect_codex._select_default_model(None, ["openai/gpt-4o", "meta/llama-3.1-8b"])
        assert result == "openai/gpt-4o"

    def test_filters_out_embedding_models(self, connect_codex: ConnectCodex) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("openai/gpt-4o"),
                _make_model("openai/text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            models = connect_codex._fetch_models("http://localhost:8321/v1")
        assert "openai/text-embedding-3-small" not in models
        assert "openai/gpt-4o" in models

    def test_exits_when_no_llm_models(self, connect_codex: ConnectCodex) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("openai/text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            models = connect_codex._fetch_models("http://localhost:8321/v1")
        assert models == []


class TestCommandGeneration:
    def test_builds_config_overrides_for_codex(self, connect_codex: ConnectCodex) -> None:
        command = connect_codex._build_codex_command("openai/gpt-4o", "http://localhost:8321/v1")

        assert command == [
            "codex",
            "--config",
            'model="openai/gpt-4o"',
            "--config",
            'model_provider="ogx"',
            "--config",
            'model_providers.ogx.name="OpenAI"',
            "--config",
            'model_providers.ogx.base_url="http://localhost:8321/v1"',
            "--config",
            'model_providers.ogx.wire_api="responses"',
            "--config",
            "model_providers.ogx.supports_websockets=false",
        ]


class TestConnect:
    def test_launches_codex_with_temporary_config(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_codex._run_connect_codex_cmd(args)

            mock_run.assert_called_once_with(
                [
                    "codex",
                    "--config",
                    'model="openai/gpt-4o"',
                    "--config",
                    'model_provider="ogx"',
                    "--config",
                    'model_providers.ogx.name="OpenAI"',
                    "--config",
                    'model_providers.ogx.base_url="http://localhost:8321/v1"',
                    "--config",
                    'model_providers.ogx.wire_api="responses"',
                    "--config",
                    "model_providers.ogx.supports_websockets=false",
                ]
            )

    def test_propagates_exit_code(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=42)
            with pytest.raises(SystemExit) as exc_info:
                connect_codex._run_connect_codex_cmd(args)
            assert exc_info.value.code == 42
