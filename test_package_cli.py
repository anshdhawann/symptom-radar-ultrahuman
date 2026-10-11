"""Credential and public CLI boundaries, tested without a real env file."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from symptomradar import config, client


@pytest.fixture
def checkout(tmp_path):
    source = Path(__file__).parent
    shutil.copytree(source / "symptomradar", tmp_path / "symptomradar")
    shutil.copyfile(source / "symptom_radar.py", tmp_path / "symptom_radar.py")
    (tmp_path / ".env").write_text("UNRELATED=value\n")
    return tmp_path


def command(checkout, *args):
    env = {key: value for key, value in os.environ.items() if key != "ULTRAHUMAN_TOKEN"}
    env["SYMPTOM_RADAR_DB"] = str(checkout / "offline.db")
    return subprocess.run([sys.executable, str(checkout / "symptom_radar.py"), *args],
                          cwd=checkout, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("args", [("--help",), ("--labels",),
    ("--label", "fine", "--label-date", "2024-01-01"),
    ("--confounders", "travel", "--label-date", "2024-01-01")])
def test_offline_cli_without_token(checkout, args):
    result = command(checkout, *args)
    assert result.returncode == 0, result.stderr
    assert result.stdout


@pytest.mark.parametrize("args", [(), ("--backfill", "2")])
def test_api_cli_missing_token_one_message(checkout, args):
    result = command(checkout, *args)
    assert result.returncode == 1
    assert result.stderr == "ULTRAHUMAN_TOKEN is not set; API commands require a token.\n"
    assert result.stdout == ""


def test_import_is_lazy_with_unrelated_env(checkout):
    env = {key: value for key, value in os.environ.items() if key != "ULTRAHUMAN_TOKEN"}
    result = subprocess.run([sys.executable, "-c",
        "import symptomradar.config, symptomradar.client, symptomradar.cli; print('imported')"],
        cwd=checkout, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "imported\n"


def test_token_file_and_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("ULTRAHUMAN_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("UNRELATED=value\n")
    with pytest.raises(config.MissingTokenError, match="ULTRAHUMAN_TOKEN is not set"):
        config.require_token(env_file)
    env_file.write_text('ULTRAHUMAN_TOKEN="fixture-token"\n')
    assert config.require_token(env_file) == "fixture-token"
    monkeypatch.setenv("ULTRAHUMAN_TOKEN", "env-token")
    assert config.require_token(env_file) == "env-token"
    assert config.require_token(tmp_path / "absent") == "env-token"


def test_offline_mcp_tools_without_token(checkout):
    env = {key: value for key, value in os.environ.items() if key != "ULTRAHUMAN_TOKEN"}
    env["SYMPTOM_RADAR_DB"] = str(checkout / "offline.db")
    requests = [{"jsonrpc": "2.0", "id": i, "method": "tools/call",
                 "params": {"name": tool}} for i, tool in enumerate(
                 ["symptom_radar_history", "symptom_radar_strain", "symptom_radar_labels"])]
    result = subprocess.run([sys.executable, "symptom_radar.py", "--mcp"], cwd=checkout,
        env=env, input="\n".join(map(json.dumps, requests)) + "\n", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 3
    assert all("result" in response for response in responses)
