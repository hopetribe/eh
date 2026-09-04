# -*- coding: utf-8 -*-
"""受支持的v6前向shadow运维命令。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from gcn.backtest.shadow_operations import (
    ShadowConfigurationError,
    ShadowDataBlockedError,
    ShadowIntegrityError,
    ShadowLifecycleError,
    initialize_shadow,
    preflight_shadow,
    read_shadow_status,
    update_shadow,
)

_SCHEMA_VERSION = "gcn-shadow-cli-v1"
_COMMANDS = ("preflight", "initialize", "update", "status")
_DEFAULT_SPEC = (
    Path(__file__).resolve().parent
    / "shadow_specs"
    / "v6-profit-arm20-keep50-20260905.json"
)
_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"


class _UsageError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python3 -m gcn.backtest.shadow_cli",
        description="GCN v6冻结预注册的只读检查、显式初始化与增量更新",
    )
    parser.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--expected-python", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "preflight", help="只读校验行情与实验就绪状态",
    )
    subparsers.add_parser("initialize", help="首个共同交易日后一次性建账")
    subparsers.add_parser("update", help="只接续已经初始化的实验")
    subparsers.add_parser("status", help="只读重放权威状态且不修复缓存")
    return parser


def _envelope(
    *, command: str | None, ok: bool, code: str, message: str,
    experiment_id: str | None = None, spec_hash: str | None = None,
    experiment_state: str | None = None, ledger: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "command": command,
        "details": details or {},
        "experiment_id": experiment_id,
        "experiment_state": experiment_state,
        "ledger": ledger,
        "message": message,
        "ok": ok,
        "schema_version": _SCHEMA_VERSION,
        "spec_hash": spec_hash,
    }


def _emit(payload: dict[str, Any], stream: TextIO) -> None:
    stream.write(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n")


def _command_hint(arguments: Sequence[str]) -> str | None:
    return next((value for value in arguments if value in _COMMANDS), None)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = _command_hint(arguments)
    try:
        parsed = build_parser().parse_args(arguments)
    except _UsageError as error:
        _emit(_envelope(
            command=command,
            ok=False,
            code="USAGE_ERROR",
            message=str(error),
        ), sys.stderr)
        return 2
    if not parsed.state_root.is_absolute():
        _emit(_envelope(
            command=parsed.command,
            ok=False,
            code="USAGE_ERROR",
            message="--state-root必须是绝对路径",
        ), sys.stderr)
        return 2
    if parsed.command in {"initialize", "update"}:
        if parsed.expected_python is None:
            _emit(_envelope(
                command=parsed.command,
                ok=False,
                code="USAGE_ERROR",
                message=(
                    f"{parsed.command}要求显式提供--expected-python"
                ),
            ), sys.stderr)
            return 2
        expected_python = parsed.expected_python.resolve()
        if expected_python != Path(sys.executable).resolve():
            _emit(_envelope(
                command=parsed.command,
                ok=False,
                code="STATE_BLOCKED",
                message=(
                    "当前解释器偏离--expected-python: "
                    f"expected={expected_python}, actual={Path(sys.executable).resolve()}"
                ),
            ), sys.stderr)
            return 4
    try:
        if parsed.command == "status":
            result = read_shadow_status(parsed.spec, parsed.state_root)
        elif parsed.command == "preflight":
            result = preflight_shadow(
                parsed.spec, parsed.data_dir, parsed.state_root,
            )
        elif parsed.command == "initialize":
            result = initialize_shadow(
                parsed.spec, parsed.data_dir, parsed.state_root,
            )
        else:
            result = update_shadow(
                parsed.spec, parsed.data_dir, parsed.state_root,
            )
    except ShadowConfigurationError as error:
        _emit(_envelope(
            command=parsed.command,
            ok=False,
            code="CONFIG_ERROR",
            message=str(error),
        ), sys.stderr)
        return 2
    except ShadowDataBlockedError as error:
        _emit(_envelope(
            command=parsed.command,
            ok=False,
            code="DATA_BLOCKED",
            message=str(error),
        ), sys.stderr)
        return 3
    except (ShadowIntegrityError, ShadowLifecycleError) as error:
        _emit(_envelope(
            command=parsed.command,
            ok=False,
            code="STATE_BLOCKED",
            message=str(error),
        ), sys.stderr)
        return 4
    except Exception as error:  # noqa: BLE001 - CLI must not leak traceback
        _emit(_envelope(
            command=parsed.command,
            ok=False,
            code="INTERNAL_ERROR",
            message=str(error) or type(error).__name__,
        ), sys.stderr)
        return 1

    known = {
        "cache_health", "code", "experiment_id", "experiment_state",
        "ledger", "spec_hash",
    }
    details = {
        key: value for key, value in result.items() if key not in known
    }
    _emit(_envelope(
        command=parsed.command,
        ok=True,
        code=result["code"],
        message={
            "UNINITIALIZED": "影子实验尚未初始化",
            "WAITING_FOR_FIRST_COMMON_SESSION": "等待首个cutoff后共同交易日",
            "READY_FOR_INITIALIZE": "行情已满足显式初始化门禁",
            "INITIALIZED": "影子实验已初始化",
            "UPDATED": "影子实验已提交新共同交易日",
            "NO_CHANGE": "没有新的共同交易日",
            "STATUS": "权威状态与派生缓存一致",
            "REPAIR_REQUIRED": "权威状态有效，但派生缓存需要写入口修复",
        }.get(result["code"], result["code"]),
        experiment_id=result.get("experiment_id"),
        spec_hash=result.get("spec_hash"),
        experiment_state=result.get("experiment_state") or (
            result.get("ledger") or {}
        ).get("state"),
        ledger=result.get("ledger"),
        details={
            "cache_health": result.get("cache_health"),
            **details,
        },
    ), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
