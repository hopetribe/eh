# -*- coding: utf-8 -*-
"""v6 shadow 运维CLI的稳定输出与退出码测试。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.test_shadow_operations import _bars, _write_adjusted_inputs

_ROOT = Path(__file__).resolve().parents[1]
_SPEC_PATH = (
    _ROOT / "gcn" / "backtest" / "shadow_specs"
    / "v6-profit-arm20-keep50-20260905.json"
)


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gcn.backtest.shadow_cli", *arguments],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_argument_errors_are_one_json_line_on_stderr():
    completed = _run_cli("status")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.count("\n") == 1
    payload = json.loads(completed.stderr)
    assert payload == {
        "code": "USAGE_ERROR",
        "command": "status",
        "details": {},
        "experiment_id": None,
        "experiment_state": None,
        "ledger": None,
        "message": "the following arguments are required: --state-root",
        "ok": False,
        "schema_version": "gcn-shadow-cli-v1",
        "spec_hash": None,
    }


def test_cli_status_uninitialized_is_successful_canonical_json():
    with TemporaryDirectory() as directory:
        state_root = Path(directory) / "state"

        completed = _run_cli(
            "--state-root", str(state_root.resolve()), "status",
        )

        assert completed.returncode == 0
        assert completed.stderr == ""
        assert completed.stdout.count("\n") == 1
        assert completed.stdout == json.dumps(
            json.loads(completed.stdout),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == "gcn-shadow-cli-v1"
        assert payload["command"] == "status"
        assert payload["ok"] is True
        assert payload["code"] == "UNINITIALIZED"
        assert payload["ledger"] is None
        assert payload["details"] == {
            "cache_health": None,
            "sequence": None,
        }
        assert not state_root.exists()


def test_cli_status_maps_state_root_file_to_stable_state_blocked():
    with TemporaryDirectory() as directory:
        state_root = Path(directory) / "state"
        state_root.write_text("not-a-directory\n", encoding="utf-8")
        state_root.chmod(0o600)

        completed = _run_cli(
            "--state-root", str(state_root.resolve()), "status",
        )

        assert completed.returncode == 4
        assert completed.stdout == ""
        assert completed.stderr.count("\n") == 1
        payload = json.loads(completed.stderr)
        assert payload["code"] == "STATE_BLOCKED"
        assert "目录" in payload["message"]


def test_cli_write_commands_require_the_fixed_python_path():
    with TemporaryDirectory() as directory:
        state_root = Path(directory) / "state"

        completed = _run_cli(
            "--state-root", str(state_root.resolve()), "initialize",
        )

        assert completed.returncode == 2
        assert completed.stdout == ""
        payload = json.loads(completed.stderr)
        assert payload["code"] == "USAGE_ERROR"
        assert payload["message"] == (
            "initialize要求显式提供--expected-python"
        )
        assert not state_root.exists()


def test_cli_tampered_frozen_spec_is_a_configuration_error_without_traceback():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        state_root = root / "state"
        spec_path = root / _SPEC_PATH.name
        spec_path.write_text(
            _SPEC_PATH.read_text(encoding="utf-8").replace(
                '"frozen_on": "2026-09-05"',
                '"frozen_on": "2026-09-06"',
            ),
            encoding="utf-8",
        )
        spec_path.with_suffix(".sha256").write_bytes(
            _SPEC_PATH.with_suffix(".sha256").read_bytes()
        )

        completed = _run_cli(
            "--spec", str(spec_path),
            "--state-root", str(state_root.resolve()),
            "status",
        )

        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "Traceback" not in completed.stderr
        payload = json.loads(completed.stderr)
        assert payload["code"] == "CONFIG_ERROR"
        assert "哈希不匹配" in payload["message"]
        assert not state_root.exists()


def test_cli_initialize_and_status_expose_only_the_public_ledger():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(dates)
            for symbol in spec["universe"]["core_symbols"]
        })

        initialized = _run_cli(
            "--data-dir", str(data_dir),
            "--state-root", str(state_root.resolve()),
            "--expected-python", sys.executable,
            "initialize",
        )
        status = _run_cli(
            "--state-root", str(state_root.resolve()), "status",
        )

        assert initialized.returncode == status.returncode == 0
        assert initialized.stderr == status.stderr == ""
        initialized_payload = json.loads(initialized.stdout)
        status_payload = json.loads(status.stdout)
        assert initialized_payload["code"] == "INITIALIZED"
        assert status_payload["code"] == "STATUS"
        assert status_payload["ledger"] == initialized_payload["ledger"]
        serialized = initialized.stdout + status.stdout
        assert "protocol_state" not in serialized
        assert "evaluation_result" not in serialized
        assert all(
            field not in status_payload["ledger"]
            for field in spec["decision"]["pre_ready_forbidden_metrics"]
        )


def test_cli_preflight_waiting_is_success_and_keeps_state_absent():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        _write_adjusted_inputs(data_dir, {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        })

        completed = _run_cli(
            "--data-dir", str(data_dir),
            "--state-root", str(state_root.resolve()),
            "preflight",
        )

        assert completed.returncode == 0
        assert completed.stderr == ""
        payload = json.loads(completed.stdout)
        assert payload["code"] == "WAITING_FOR_FIRST_COMMON_SESSION"
        assert payload["details"]["latest_common_session"] is None
        assert len(payload["details"]["snapshot_sha256"]) == 64
        assert not state_root.exists()
