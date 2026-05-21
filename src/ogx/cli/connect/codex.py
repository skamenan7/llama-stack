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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")


@dataclass(frozen=True)
class DiscoveredCodexModel:
    """Model entry returned by OGX and adapted into the generated Codex catalog."""

    model_id: str
    custom_metadata: dict[str, Any]


class ConnectCodex(Subcommand):
    """Connect Codex to the running OGX server."""

    MODEL_DISCOVERY_TIMEOUT_SECONDS = 20.0
    DEFAULT_BASE_URL = "http://localhost:8321/v1"
    DEFAULT_CONTEXT_WINDOW = 128000
    DEFAULT_BASE_INSTRUCTIONS = (
        "You are Codex, a coding agent. You and the user share the same workspace and "
        "collaborate to achieve the user's goals."
    )

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
            "--base-url",
            type=str,
            default=os.getenv("OGX_BASE_URL", self.DEFAULT_BASE_URL),
            help="OGX OpenAI-compatible base URL. If no path is provided, /v1 is appended.",
        )

    def _run_connect_codex_cmd(self, args: argparse.Namespace) -> None:
        if not shutil.which("codex"):
            cprint(
                "Failed to find 'codex' in PATH. Install it from https://github.com/openai/codex",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = self._normalize_base_url(args.base_url)

        models = self._fetch_models(base_url)
        if not models:
            cprint("Failed to find any LLM models on the OGX server.", color="red", file=sys.stderr)
            sys.exit(1)

        default_model = self._select_default_model(args.model, models)

        logger.info("Connecting to Codex", default_model=default_model.model_id, models=len(models), base_url=base_url)

        with tempfile.TemporaryDirectory(prefix="ogx-codex-") as codex_home:
            codex_home_path = Path(codex_home)
            self._write_codex_session_files(codex_home_path, base_url, models, default_model.model_id)
            command = self._build_codex_command()
            env = {**os.environ, "CODEX_HOME": str(codex_home_path)}
            result = subprocess.run(command, env=env)
            sys.exit(result.returncode)

    def _normalize_base_url(self, raw_base_url: str) -> str:
        parsed = urlsplit(raw_base_url.strip())
        if not parsed.scheme or not parsed.netloc:
            cprint(
                f"Failed to parse OGX base URL '{raw_base_url}'.\n"
                "Provide a full URL such as http://localhost:8321/v1 or https://ogx.example.com/v1.",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        path = parsed.path.rstrip("/")
        normalized_path = "/v1" if not path else path
        return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, parsed.query, ""))

    def _fetch_models(self, base_url: str) -> list[DiscoveredCodexModel]:
        default_headers = self._build_request_headers()
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("OGX_API_KEY", "").strip() or "unused",
            timeout=self.MODEL_DISCOVERY_TIMEOUT_SECONDS,
            default_headers=default_headers or None,
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

        models: list[DiscoveredCodexModel] = []
        for model in response.data:
            metadata = self._extract_custom_metadata(model)
            if metadata.get("model_type") != "embedding":
                models.append(DiscoveredCodexModel(model_id=model.id, custom_metadata=metadata))
        return models

    def _extract_custom_metadata(self, model: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        model_extra = getattr(model, "model_extra", None)
        if isinstance(model_extra, dict):
            extra_custom_metadata = model_extra.get("custom_metadata")
            if isinstance(extra_custom_metadata, dict):
                metadata.update(extra_custom_metadata)

        direct_custom_metadata = getattr(model, "custom_metadata", None)
        if isinstance(direct_custom_metadata, dict):
            metadata.update(direct_custom_metadata)

        return metadata

    def _build_request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        provider_data = os.getenv("OGX_PROVIDER_DATA", "").strip()
        if provider_data:
            headers["X-OGX-Provider-Data"] = provider_data
        return headers

    def _select_default_model(
        self, requested_model: str | None, available_models: list[DiscoveredCodexModel]
    ) -> DiscoveredCodexModel:
        available_model_ids = [model.model_id for model in available_models]
        if requested_model:
            if requested_model not in available_model_ids:
                cprint(
                    f"Failed to find model '{requested_model}' on the OGX server.\n"
                    f"Available models: {', '.join(available_model_ids)}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            return next(model for model in available_models if model.model_id == requested_model)

        return available_models[0]

    def _build_codex_command(self) -> list[str]:
        return ["codex", "-p", "ogx"]

    def _write_codex_session_files(
        self,
        codex_home: Path,
        base_url: str,
        available_models: list[DiscoveredCodexModel],
        default_model: str,
    ) -> None:
        model_catalog_path = codex_home / "ogx-model-catalog.json"
        config_path = codex_home / "config.toml"
        model_catalog_path.write_text(json.dumps(self._build_model_catalog(available_models, default_model), indent=2))
        config_path.write_text(self._build_codex_config(base_url, model_catalog_path, default_model))

    def _build_codex_config(self, base_url: str, model_catalog_path: Path, default_model: str) -> str:
        env_http_headers = '{ "X-OGX-Provider-Data" = "OGX_PROVIDER_DATA" }'
        config_lines = [
            "[model_providers.ogx]",
            'name = "OGX"',
            f"base_url = {json.dumps(base_url)}",
            'wire_api = "responses"',
            "supports_websockets = false",
        ]
        if os.getenv("OGX_API_KEY", "").strip():
            config_lines.extend(
                [
                    'env_key = "OGX_API_KEY"',
                    'env_key_instructions = "Set OGX_API_KEY when your OGX deployment requires bearer authentication."',
                ]
            )
        config_lines.extend(
            [
                f"env_http_headers = {env_http_headers}",
                "",
                "[profiles.ogx]",
                f"model = {json.dumps(default_model)}",
                'model_provider = "ogx"',
                f"model_catalog_json = {json.dumps(str(model_catalog_path))}",
                "",
            ]
        )
        return "\n".join(config_lines)

    def _build_model_catalog(
        self, available_models: list[DiscoveredCodexModel], default_model: str
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "models": [
                self._build_model_catalog_entry(model, index=index, is_default=model.model_id == default_model)
                for index, model in enumerate(available_models)
            ]
        }

    def _build_model_catalog_entry(
        self, model: DiscoveredCodexModel, *, index: int, is_default: bool
    ) -> dict[str, Any]:
        metadata = model.custom_metadata
        context_window = self._coerce_int(
            metadata.get("context_window") or metadata.get("context_length"),
            self.DEFAULT_CONTEXT_WINDOW,
        )
        entry: dict[str, Any] = {
            "slug": model.model_id,
            "display_name": self._coerce_str(
                metadata.get("display_name") or metadata.get("provider_model_id"),
                model.model_id,
            ),
            "description": self._coerce_str(
                metadata.get("description"),
                f"Model exposed by the running OGX server as {model.model_id}.",
            ),
            "default_reasoning_level": None,
            "context_window": context_window,
            "max_context_window": context_window,
            "auto_compact_token_limit": self._coerce_optional_int(metadata.get("auto_compact_token_limit")),
            "shell_type": "default",
            "additional_speed_tiers": [],
            "service_tiers": [],
            "default_service_tier": None,
            "availability_nux": None,
            "upgrade": None,
            "base_instructions": self.DEFAULT_BASE_INSTRUCTIONS,
            "model_messages": None,
            "supports_reasoning_summaries": False,
            "default_reasoning_summary": "auto",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_parallel_tool_calls": False,
            "supports_image_detail_original": False,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": self._coerce_string_list(metadata.get("input_modalities"), fallback=["text"]),
            "supported_reasoning_levels": [],
            "used_fallback_model_metadata": False,
            "supports_search_tool": False,
            "visibility": "list",
            "priority": 0 if is_default else index + 1,
            "supported_in_api": True,
        }

        supported_reasoning_levels = self._build_reasoning_levels(metadata)
        if supported_reasoning_levels:
            entry["supported_reasoning_levels"] = supported_reasoning_levels
            entry["default_reasoning_level"] = self._coerce_str(
                metadata.get("default_reasoning_level") or metadata.get("defaultReasoningEffort"),
                supported_reasoning_levels[0]["effort"],
            )

        return entry

    def _build_reasoning_levels(self, metadata: dict[str, Any]) -> list[dict[str, str]]:
        raw_levels = metadata.get("supported_reasoning_levels") or metadata.get("supportedReasoningEfforts") or []
        if not isinstance(raw_levels, list):
            return []

        levels: list[dict[str, str]] = []
        for item in raw_levels:
            if not isinstance(item, dict):
                continue
            effort = item.get("effort") or item.get("reasoningEffort")
            description = item.get("description")
            if isinstance(effort, str) and isinstance(description, str):
                levels.append({"effort": effort, "description": description})
        return levels

    def _coerce_int(self, value: Any, fallback: int) -> int:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return fallback

    def _coerce_optional_int(self, value: Any) -> int | None:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _coerce_str(self, value: Any, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return value
        return fallback

    def _coerce_string_list(self, value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            items = [item for item in value if isinstance(item, str) and item]
            if items:
                return items
        return [
            *fallback,
        ]
