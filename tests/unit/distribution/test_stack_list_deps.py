# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import tomllib
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from llama_stack.cli.stack._list_deps import (
    format_output_deps_only,
    run_stack_list_deps_command,
)


def _package_names(dependencies: list[str]) -> set[str]:
    return {
        dependency.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0] for dependency in dependencies
    }


def _output_deps(output: str) -> set[str]:
    return {dependency.strip("'") for dependency in output.split()}


def test_base_dependencies_do_not_include_oci():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    base_dependencies = _package_names(pyproject["project"]["dependencies"])
    oci_extra = _package_names(pyproject["project"]["optional-dependencies"]["oci"])

    assert "oci" not in base_dependencies
    assert "oracledb" not in base_dependencies
    assert "oci" in oci_extra
    assert "oracledb" in oci_extra


def test_stack_list_deps_basic():
    args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="inference=remote::ollama",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(args)
        output = mock_stdout.getvalue()

        # deps-only format should NOT include "uv pip install" or "Dependencies for"
        assert "uv pip install" not in output
        assert "Dependencies for" not in output

        # Check that expected dependencies are present
        assert "ollama" in output
        assert "aiohttp" in output
        assert "fastapi" in output


def test_stack_list_deps_with_distro_uv():
    args = argparse.Namespace(
        config="starter",
        env_name=None,
        providers=None,
        format="uv",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(args)
        output = mock_stdout.getvalue()

        assert "uv pip install" in output


def test_starter_distro_list_deps_does_not_include_oci():
    args = argparse.Namespace(
        config="starter",
        env_name=None,
        providers=None,
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(args)
        output = mock_stdout.getvalue()

    deps = _output_deps(output)
    assert "oci" not in deps
    assert "oracledb" not in deps


def test_explicit_oci_provider_still_lists_oci_dependency():
    args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="inference=remote::oci",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(args)
        output = mock_stdout.getvalue()

    assert "oci" in _output_deps(output)


def test_list_deps_formatting_quotes_only_for_uv():
    deps_only = format_output_deps_only(["mcp>=1.23.0"], [], [], uv=False)
    assert deps_only.strip() == "mcp>=1.23.0"

    uv_format = format_output_deps_only(["mcp>=1.23.0"], [], [], uv=True)
    assert uv_format.strip() == "uv pip install 'mcp>=1.23.0'"
