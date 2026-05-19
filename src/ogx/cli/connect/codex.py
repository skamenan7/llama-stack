# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import os
import shutil
import subprocess
import sys

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")


class ConnectCodex(Subcommand):
    """Connect Codex to the running OGX server."""

    MODEL_DISCOVERY_TIMEOUT_SECONDS = 20.0

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "codex",
            prog="ogx connect codex",
            description="Launch Codex connected to the running OGX server.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_connect_codex_cmd)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Default model ID. If omitted, the first available model is used.",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            help="OGX server port.",
            default=int(os.getenv("OGX_PORT", 8321)),
        )
        self.parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="OGX server host.",
        )

    def _run_connect_codex_cmd(self, args: argparse.Namespace) -> None:
        if not shutil.which("codex"):
            cprint(
                "Failed to find 'codex' in PATH. Install it from https://github.com/openai/codex",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = f"http://{args.host}:{args.port}/v1"

        models = self._fetch_models(base_url)
        if not models:
            cprint("Failed to find any LLM models on the OGX server.", color="red", file=sys.stderr)
            sys.exit(1)

        default_model = self._select_default_model(args.model, models)
        command = self._build_codex_command(default_model, base_url)

        logger.info("Connecting to Codex", default_model=default_model, models=len(models), base_url=base_url)

        result = subprocess.run(command)
        sys.exit(result.returncode)

    def _fetch_models(self, base_url: str) -> list[str]:
        client = OpenAI(
            base_url=base_url,
            api_key="unused",
            timeout=self.MODEL_DISCOVERY_TIMEOUT_SECONDS,
        )
        try:
            response = client.models.list()
        except APITimeoutError:
            cprint(
                f"Failed to connect to OGX server at {base_url}\n"
                f"Timed out while querying available models after {self.MODEL_DISCOVERY_TIMEOUT_SECONDS} seconds.",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        except APIConnectionError:
            cprint(
                f"Failed to connect to OGX server at {base_url}\nStart the server first with: ogx stack run <config>",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        except APIStatusError as e:
            cprint(
                f"Failed to query models from OGX server at {base_url} (HTTP {e.status_code})",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        models: list[str] = []
        for model in response.data:
            metadata = (model.model_extra or {}).get("custom_metadata") or {}
            if metadata.get("model_type") != "embedding":
                models.append(model.id)
        return models

    def _select_default_model(self, requested_model: str | None, available_models: list[str]) -> str:
        if requested_model:
            if requested_model not in available_models:
                cprint(
                    f"Failed to find model '{requested_model}' on the OGX server.\n"
                    f"Available models: {', '.join(available_models)}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            return requested_model

        return available_models[0]

    def _build_codex_command(self, default_model: str, base_url: str) -> list[str]:
        return [
            "codex",
            "--config",
            self._string_config_override("model", default_model),
            "--config",
            self._string_config_override("model_provider", "ogx"),
            "--config",
            self._string_config_override("model_providers.ogx.name", "OpenAI"),
            "--config",
            self._string_config_override("model_providers.ogx.base_url", base_url),
            "--config",
            self._string_config_override("model_providers.ogx.wire_api", "responses"),
            "--config",
            "model_providers.ogx.supports_websockets=false",
        ]

    def _string_config_override(self, key: str, value: str) -> str:
        return f"{key}={json.dumps(value)}"
