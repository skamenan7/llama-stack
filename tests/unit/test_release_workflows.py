# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Exercise release workflow shell decisions without publishing artifacts."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _step(workflow: str, job: str, step_id: str) -> dict[str, Any]:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return next(step for step in document["jobs"][job]["steps"] if step.get("id", step.get("name")) == step_id)


def _run(
    step: dict[str, Any], tmp_path: Path, values: dict[str, str], prefix: str = ""
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        return values.get(expression, "")

    output = tmp_path / "output"
    output.write_text("")
    env = {**os.environ, "GITHUB_OUTPUT": str(output), "GITHUB_REPOSITORY": "ogx-ai/ogx"}
    env.update({key: re.sub(r"\$\{\{(.*?)\}\}", replace, str(value)) for key, value in step.get("env", {}).items()})
    script = re.sub(r"\$\{\{(.*?)\}\}", replace, step["run"])
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", prefix + "\n" + script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    return result, outputs


@pytest.mark.parametrize("step_id", ["version-check", "npm-version-check"])
@pytest.mark.parametrize("version,expected", [("1.0.3", "1.0.3"), ("1.3.2.dev20260909", "1.3.2.dev20260909")])
def test_client_build_keeps_requested_version_when_registry_has_it(
    tmp_path: Path, step_id: str, version: str, expected: str
) -> None:
    result, outputs = _run(
        _step("pypi.yml", "build-package", step_id),
        tmp_path,
        {"needs.compute-version.outputs.version": version, "matrix.package": "ogx-client-python"},
        "curl() { return 0; }",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["version"] == (expected.replace(".dev", "-dev.") if step_id == "npm-version-check" else expected)


@pytest.mark.parametrize("job,step_id", [("build-package", "should-build"), ("publish-packages", "should-publish")])
@pytest.mark.parametrize(
    "event,selection,kind,skip",
    [
        ("release", "all", "external", "true"),
        ("release", "all", "openapi-sdk", "true"),
        ("release", "all", "local", "false"),
        ("workflow_dispatch", "ogx-only", "openapi-sdk", "true"),
        ("workflow_dispatch", "ogx-only", "local", "false"),
        ("workflow_dispatch", "clients-only", "openapi-sdk", "false"),
        ("workflow_dispatch", "clients-only", "external", "false"),
        ("workflow_dispatch", "clients-only", "local", "true"),
        ("push", "all", "external", "false"),
    ],
)
def test_package_selection(
    tmp_path: Path, job: str, step_id: str, event: str, selection: str, kind: str, skip: str
) -> None:
    result, outputs = _run(
        _step("pypi.yml", job, step_id),
        tmp_path,
        {
            "github.event_name": event,
            "inputs.packages || 'all'": selection,
            "matrix.type": kind,
            "matrix.package": "test-package",
        },
    )
    assert result.returncode == 0, result.stderr
    assert outputs["skip"] == skip


@pytest.mark.parametrize(
    "event,version,latest,skip,latest_tag",
    [
        ("release", "1.0.3", "v1.3.1", "false", False),
        ("release", "1.3.1", "v1.3.1", "false", True),
        ("release", "1.10.0", "v1.9.9", "false", True),
        ("release", "1.3.2rc1", "v1.3.1", "false", False),
        ("workflow_dispatch", "1.0.3", "v1.3.1", "true", False),
        ("workflow_dispatch", "1.3.2", "v1.3.1", "false", True),
    ],
)
def test_docker_latest_tag(tmp_path: Path, event: str, version: str, latest: str, skip: str, latest_tag: bool) -> None:
    result, outputs = _run(
        _step("pypi.yml", "publish-docker-images", "meta"),
        tmp_path,
        {
            "github.event_name": event,
            "inputs.dry_run": "off",
            "inputs.skip_latest": skip,
            "inputs.package_name || 'ogx'": "ogx",
            "matrix.distro": "starter",
            "needs.compute-version.outputs.version": version,
        },
        f"gh() {{ printf '%s\\n' '{latest}'; }}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["tags"] == "ogxai/distribution-starter:" + version + (
        ",ogxai/distribution-starter:latest" if latest_tag else ""
    )


@pytest.mark.parametrize("command", ["return 1", "printf '%s\\n' invalid"])
def test_docker_fails_closed_when_latest_release_is_unavailable(tmp_path: Path, command: str) -> None:
    result, outputs = _run(
        _step("pypi.yml", "publish-docker-images", "meta"),
        tmp_path,
        {
            "github.event_name": "release",
            "inputs.skip_latest": "false",
            "matrix.distro": "starter",
            "inputs.package_name || 'ogx'": "ogx",
            "needs.compute-version.outputs.version": "1.0.3",
        },
        f"gh() {{ {command}; }}",
    )
    assert result.returncode != 0
    assert "tags" not in outputs


@pytest.mark.parametrize("skip,expected", [("true", "release-1.0"), ("false", "latest")])
def test_npm_backfill_uses_release_stream_dist_tag(tmp_path: Path, skip: str, expected: str) -> None:
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "client.tgz").touch()
    result, _ = _run(
        _step("pypi.yml", "publish-packages", "Publish to npm"),
        tmp_path,
        {
            "github.event_name": "workflow_dispatch",
            "inputs.dry_run": "off",
            "inputs.skip_latest": skip,
            "needs.compute-version.outputs.version": "1.0.3",
            "secrets.NPM_TOKEN": "test-token",
        },
        "npm() { printf '%s\\n' \"$@\" > npm-arguments; }",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = (tmp_path / "npm-arguments").read_text().splitlines()
    assert arguments[arguments.index("--tag") + 1] == expected


@pytest.mark.parametrize(
    "version,current,update",
    [
        ("1.0.3", "1.3.1.dev0", "false"),
        ("1.3.1", "1.3.1.dev0", "true"),
        ("1.10.0", "1.9.9.dev0", "true"),
        ("1.3.2rc1", "1.3.1.dev0", "false"),
        ("1.0.3rc1", "1.3.1.dev0", "false"),
    ],
)
def test_old_release_does_not_change_main_version(tmp_path: Path, version: str, current: str, update: str) -> None:
    result, outputs = _run(
        _step("post-release.yml", "post-release", "main-version"),
        tmp_path,
        {
            "steps.parse.outputs.version": version,
        },
        f"git() {{ printf '%s\\n' 'fallback_version = \"{current}\"'; }}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["update_main"] == update


@pytest.mark.parametrize("version", ["1.4.3+rhaiv.0", "1.4.3.1+rhaiv.2"])
@pytest.mark.parametrize("branch", ["release-3.5", "release-3.6"])
def test_prepare_release_accepts_package_version_independent_of_product_stream(
    tmp_path: Path, version: str, branch: str
) -> None:
    result, _ = _run(
        _step("prepare-release.yml", "prepare-release", "Validate inputs"),
        tmp_path,
        {
            "inputs.version": version,
            "inputs.release_branch": branch,
            "github.ref_name": branch,
            "github.ref_type": "branch",
        },
        "git() { return 0; }",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "version", ["1.4.3", "v1.4.3+rhaiv.0", "1.4+rhaiv.0", "1.4.3+rhaiv.x", "1.4.3+rhaiv.0\ntouch injected"]
)
def test_prepare_release_rejects_invalid_version(tmp_path: Path, version: str) -> None:
    result, _ = _run(
        _step("prepare-release.yml", "prepare-release", "Validate inputs"),
        tmp_path,
        {
            "inputs.version": version,
            "inputs.release_branch": "release-3.6",
            "github.ref_name": "release-3.6",
            "github.ref_type": "branch",
        },
        "git() { return 0; }",
    )
    assert result.returncode != 0
    assert "Failed to validate" in result.stdout
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize(
    "branch,ref,ref_type",
    [
        ("main", "main", "branch"),
        ("release-3.7", "release-3.7", "branch"),
        ("release-3.6", "main", "branch"),
        ("release-3.6", "release-3.6", "tag"),
    ],
)
def test_prepare_release_rejects_unsupported_or_mismatched_ref(
    tmp_path: Path, branch: str, ref: str, ref_type: str
) -> None:
    result, _ = _run(
        _step("prepare-release.yml", "prepare-release", "Validate inputs"),
        tmp_path,
        {
            "inputs.version": "1.4.3+rhaiv.0",
            "inputs.release_branch": branch,
            "github.ref_name": ref,
            "github.ref_type": ref_type,
        },
        "git() { return 0; }",
    )
    assert result.returncode != 0
    assert "Failed to validate" in result.stdout


@pytest.mark.parametrize("status,success", [(0, False), (2, True), (1, False), (128, False)])
def test_prepare_release_tag_check_fails_closed(tmp_path: Path, status: int, success: bool) -> None:
    result, _ = _run(
        _step("prepare-release.yml", "prepare-release", "Check if tag already exists"),
        tmp_path,
        {"inputs.version": "1.4.3+rhaiv.0"},
        f"git() {{ return {status}; }}",
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr


@pytest.mark.parametrize("dynamic", [False, True])
def test_prepare_release_updates_both_projects_and_lock_without_changing_client(tmp_path: Path, dynamic: bool) -> None:
    api = tmp_path / "src" / "ogx_api"
    api.mkdir(parents=True)
    version = 'dynamic = ["version"]' if dynamic else 'version = "1.4.2+rhaiv.0"'
    original = f'[project]\nname = "ogx"\n{version}\ndependencies = ["ogx-client==1.4.2"]\n\n[tool.setuptools_scm]\nfallback_version = "1.4.2"\n'
    (tmp_path / "pyproject.toml").write_text(original)
    (api / "pyproject.toml").write_text(original.replace('name = "ogx"', 'name = "ogx-api"'))
    result, _ = _run(
        _step("prepare-release.yml", "prepare-release", "Update version files and lock dependencies"),
        tmp_path,
        {"inputs.version": "1.4.3+rhaiv.0"},
        'uv() { printf "%s\\n" "$*" >> uv-arguments; }',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for project in [tmp_path / "pyproject.toml", api / "pyproject.toml"]:
        content = project.read_text()
        assert 'version = "1.4.3+rhaiv.0"' in content
        assert 'dynamic = ["version"]' not in content
        assert '"ogx-client==1.4.2"' in content
    assert (tmp_path / "uv-arguments").read_text().splitlines() == ["lock", "lock --check"]


def test_prepare_release_validates_before_privileged_checkout() -> None:
    workflow = yaml.safe_load((WORKFLOWS / "prepare-release.yml").read_text())
    job = workflow["jobs"]["prepare-release"]
    steps = job["steps"]
    names = [step["name"] for step in steps]
    assert names.index("Validate inputs") < names.index("Checkout repository")
    assert workflow["permissions"] == {"contents": "read"}
    assert "github.repository == 'opendatahub-io/ogx'" in job["if"]


@pytest.mark.parametrize("cross_repository", [[], [True], [False], [True, False], [False, True]])
def test_prepare_release_pushes_only_working_branch_and_opens_pr(tmp_path: Path, cross_repository: list[bool]) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "release-3.6"], cwd=tmp_path, check=True, capture_output=True)
    for key, value in [("user.name", "Test"), ("user.email", "test@example.com")]:
        subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    (tmp_path / "src" / "ogx_api").mkdir(parents=True)
    for path in ["pyproject.toml", "src/ogx_api/pyproject.toml", "uv.lock"]:
        (tmp_path / path).write_text("before\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial version"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=tmp_path, check=True)
    subprocess.run(["git", "push", "origin", "release-3.6"], cwd=tmp_path, check=True, capture_output=True)
    before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    for path in ["pyproject.toml", "src/ogx_api/pyproject.toml", "uv.lock"]:
        (tmp_path / path).write_text("after\n")
    existing_prs = [
        {"url": f"https://github.com/opendatahub-io/ogx/pull/{index}", "isCrossRepository": cross}
        for index, cross in enumerate(cross_repository, start=1)
    ]
    already_open = False in cross_repository
    result, _ = _run(
        _step("prepare-release.yml", "prepare-release", "Open preparation PR"),
        tmp_path,
        {"inputs.version": "1.4.3+rhaiv.0", "inputs.release_branch": "release-3.6"},
        f"""gh() {{
          printf '%s\\0' "$@" >> gh-arguments
          printf '\\0' >> gh-arguments
          if [[ "$1 $2" == "pr list" ]]; then
            printf '%s\\n' '{json.dumps(existing_prs)}'
          fi
        }}""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    base = subprocess.check_output(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/release-3.6"], text=True
    ).strip()
    assert base == before
    assert subprocess.check_output(["git", "--git-dir", str(remote), "tag"], text=True) == ""
    records = (tmp_path / "gh-arguments").read_bytes().split(b"\0\0")
    assert records[-1] == b""
    calls = [[argument.decode() for argument in record.split(b"\0")] for record in records[:-1]]
    assert len(calls) == (1 if already_open else 2)
    assert calls[0] == [
        "pr",
        "list",
        "--repo",
        "ogx-ai/ogx",
        "--base",
        "release-3.6",
        "--head",
        "prepare-release-3.6-v1.4.3+rhaiv.0",
        "--state",
        "open",
        "--json",
        "url,isCrossRepository",
    ]
    if already_open:
        genuine_pr = next(pr for pr in existing_prs if pr["isCrossRepository"] is False)
        assert f"Preparation PR already exists: {genuine_pr['url']}" in result.stdout
    else:
        arguments = calls[1]
        assert arguments[:2] == ["pr", "create"]
        assert arguments[arguments.index("--base") + 1] == "release-3.6"
        head = arguments[arguments.index("--head") + 1]
        prepared = subprocess.check_output(
            ["git", "--git-dir", str(remote), "show", f"refs/heads/{head}:uv.lock"], text=True
        )
        assert prepared == "after\n"
