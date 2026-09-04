# -*- coding: utf-8 -*-
"""v6 shadow 的严格输入快照与受支持运维生命周期。"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import stat
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from gcn.backtest import shadow_runner
from gcn.backtest.shadow_runner import (
    derive_shadow_boundaries,
    run_shadow_snapshot,
)
from gcn.backtest.shadow_validation import (
    canonical_spec_hash,
    load_spec,
    merge_accepted_bars,
)


class ShadowOperationError(ValueError):
    """所有可预期 shadow 运维拒绝的基类。"""


class ShadowConfigurationError(ShadowOperationError):
    """冻结spec或显式运维配置无效。"""


class ShadowDataBlockedError(ShadowOperationError):
    """行情来源、摘要或交易日不满足失败关闭门禁。"""


class ShadowLifecycleError(ShadowOperationError):
    """命令与实验当前生命周期不匹配。"""


class ShadowIntegrityError(ShadowOperationError):
    """已存在实验的权威链、运行时或权限不可信。"""


@dataclass(frozen=True)
class ShadowInputSnapshot:
    """同一锁窗口内只读取一次的十股行情及其来源收据。"""

    frames: dict[str, pd.DataFrame]
    receipts: dict[str, dict[str, Any]]
    snapshot_sha256: str


_DATA_THREAD_LOCKS_GUARD = threading.Lock()
_DATA_THREAD_LOCKS: dict[str, threading.Lock] = {}
_UMASK_GUARD = threading.Lock()
_EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_ALLOWED_SOURCES = frozenset({"futu", "yahoo"})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_frozen_spec(path: Path) -> dict[str, Any]:
    try:
        return load_spec(Path(path))
    except (OSError, ValueError) as error:
        raise ShadowConfigurationError(str(error)) from error


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _DATA_THREAD_LOCKS_GUARD:
        return _DATA_THREAD_LOCKS.setdefault(key, threading.Lock())


def _validate_state_root_contract(state_root: Path) -> Path:
    """状态根必须是仓库外的绝对私有目录，缺失目录保持只读。"""
    state_root = Path(state_root)
    if not state_root.is_absolute():
        raise ShadowConfigurationError("影子状态根必须是绝对路径")
    try:
        if state_root.is_symlink():
            raise ShadowIntegrityError("影子状态根不得是符号链接")
        resolved = state_root.resolve(strict=False)
        if (resolved == _REPOSITORY_ROOT
                or _REPOSITORY_ROOT in resolved.parents):
            raise ShadowConfigurationError("影子状态根必须位于仓库外")
        if state_root.exists():
            metadata = state_root.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise ShadowIntegrityError("影子状态根必须是目录")
            if metadata.st_uid != os.geteuid():
                raise ShadowIntegrityError("影子状态根所有者不是当前用户")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ShadowIntegrityError("影子状态根权限不私有")
    except ShadowOperationError:
        raise
    except OSError as error:
        raise ShadowIntegrityError("影子状态根不可验证") from error
    return resolved


def _validate_private_state_tree(state_root: Path) -> None:
    """拒绝符号链接或任何group/world权限；不自动chmod隐藏证据。"""
    if not state_root.exists():
        return
    try:
        paths = [state_root, *state_root.rglob("*")]
        for path in paths:
            if path.is_symlink():
                raise ShadowIntegrityError(
                    f"影子状态不得包含符号链接: {path.name}"
                )
            metadata = path.lstat()
            if metadata.st_uid != os.geteuid():
                raise ShadowIntegrityError(
                    f"影子状态所有者不是当前用户: {path.name}"
                )
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ShadowIntegrityError(
                    f"影子状态权限不私有: {path.name}"
                )
    except OSError as error:
        raise ShadowIntegrityError("影子状态权限不可验证") from error


@contextmanager
def _private_state_umask() -> Iterator[None]:
    """让官方写路径创建的目录/文件默认分别不宽于0700/0600。"""
    with _UMASK_GUARD:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


@contextmanager
def _locked_input_files(
    data_dir: Path, symbols: tuple[str, ...],
) -> Iterator[None]:
    """按稳定顺序同时持有全部相邻行情锁；缺锁时绝不创建文件。"""
    lock_paths = [data_dir / f"{symbol}_1d.lock" for symbol in sorted(symbols)]
    with ExitStack() as stack:
        for lock_path in lock_paths:
            if not lock_path.is_file():
                raise ShadowDataBlockedError(
                    f"缺少行情事务锁: {lock_path.name}"
                )
            local_lock = _thread_lock(lock_path)
            local_lock.acquire()
            stack.callback(local_lock.release)
            try:
                handle = stack.enter_context(lock_path.open("rb"))
            except OSError as error:
                raise ShadowDataBlockedError(
                    f"行情事务锁不可读: {lock_path.name}"
                ) from error
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as error:
                raise ShadowDataBlockedError(
                    f"行情事务锁不可获取: {lock_path.name}"
                ) from error
            stack.callback(
                _unlock_input_file, handle.fileno(), lock_path.name,
            )
        yield


def _unlock_input_file(file_descriptor: int, lock_name: str) -> None:
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    except OSError as error:
        raise ShadowDataBlockedError(
            f"行情事务锁不可释放: {lock_name}"
        ) from error


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"重复字段: {key}")
        value[key] = item
    return value


def _parse_sidecar(symbol: str, payload: bytes) -> dict[str, str]:
    try:
        meta = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"非有限常量: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ShadowDataBlockedError(
            f"{symbol} sidecar不是有效JSON"
        ) from error
    if not isinstance(meta, dict) or set(meta) != {
        "source", "adjustment", "sha256",
    }:
        raise ShadowDataBlockedError(f"{symbol} sidecar字段不匹配")
    source = meta.get("source")
    adjustment = meta.get("adjustment")
    digest = meta.get("sha256")
    if source not in _ALLOWED_SOURCES:
        raise ShadowDataBlockedError(f"{symbol} sidecar来源不受支持")
    if adjustment != "adjusted":
        raise ShadowDataBlockedError(f"{symbol} 行情不是adjusted口径")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ShadowDataBlockedError(f"{symbol} sidecar SHA-256无效")
    return {
        "source": source,
        "adjustment": adjustment,
        "sha256": digest,
    }


def _parse_csv_bytes(symbol: str, payload: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except Exception as error:  # pandas parser errors vary by supported version
        raise ShadowDataBlockedError(f"{symbol} 行情CSV不可解析") from error
    if list(raw.columns) != _EXPECTED_COLUMNS:
        raise ShadowDataBlockedError(
            f"{symbol} 行情字段必须严格为 {', '.join(_EXPECTED_COLUMNS)}"
        )
    dates = pd.to_datetime(raw.pop("date"), errors="coerce")
    if dates.isna().any():
        raise ShadowDataBlockedError(f"{symbol} 行情包含无效日期")
    raw.index = pd.DatetimeIndex(dates)
    raw.index.name = "date"
    try:
        return merge_accepted_bars(None, raw)
    except ValueError as error:
        raise ShadowDataBlockedError(f"{symbol} 行情无效: {error}") from error


def capture_adjusted_snapshot(
    spec: dict[str, Any], data_dir: Path,
) -> ShadowInputSnapshot:
    """捕获并验证一个不会跨数据服务事务代际的内存快照。"""
    data_dir = Path(data_dir)
    symbols = tuple(spec["universe"]["core_symbols"])
    frames: dict[str, pd.DataFrame] = {}
    receipts: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    with _locked_input_files(data_dir, symbols):
        for symbol in symbols:
            csv_path = data_dir / f"{symbol}_1d.csv"
            meta_path = data_dir / f"{symbol}_1d.csv.meta.json"
            try:
                csv_payload = csv_path.read_bytes()
                meta_payload = meta_path.read_bytes()
            except OSError as error:
                failures.append(f"{symbol} 行情CSV或sidecar缺失/不可读")
                continue
            try:
                meta = _parse_sidecar(symbol, meta_payload)
                csv_digest = hashlib.sha256(csv_payload).hexdigest()
                if meta["sha256"] != csv_digest:
                    raise ShadowDataBlockedError(
                        f"{symbol} sidecar与CSV的SHA-256不匹配"
                    )
                frame = _parse_csv_bytes(symbol, csv_payload)
            except ShadowDataBlockedError as error:
                failures.append(str(error))
                continue
            frames[symbol] = frame
            receipts[symbol] = {
                "adjustment": meta["adjustment"],
                "csv_sha256": csv_digest,
                "first_date": frame.index[0].date().isoformat(),
                "last_date": frame.index[-1].date().isoformat(),
                "meta_sha256": hashlib.sha256(meta_payload).hexdigest(),
                "row_count": len(frame),
                "source": meta["source"],
            }
    if failures:
        raise ShadowDataBlockedError("; ".join(failures))
    receipt_payload = json.dumps(
        receipts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return ShadowInputSnapshot(
        frames=frames,
        receipts=receipts,
        snapshot_sha256=hashlib.sha256(receipt_payload).hexdigest(),
    )


def preflight_shadow(
    spec_path: Path, data_dir: Path, state_root: Path,
) -> dict[str, Any]:
    """只读检查严格行情输入；不创建行情锁或实验状态。"""
    spec = _load_frozen_spec(Path(spec_path))
    state_root = _validate_state_root_contract(Path(state_root))
    snapshot = capture_adjusted_snapshot(spec, Path(data_dir))
    try:
        boundaries = derive_shadow_boundaries(spec, snapshot.frames)
    except ValueError as error:
        raise ShadowDataBlockedError(str(error)) from error
    existing = _read_shadow_status_loaded(spec, state_root)
    if existing["code"] != "UNINITIALIZED":
        return {
            "cache_health": existing["cache_health"],
            "code": "READY_FOR_UPDATE",
            "experiment_id": spec["experiment_id"],
            "experiment_state": existing["experiment_state"],
            "latest_common_session": boundaries["latest_common_session"],
            "sequence": existing["sequence"],
            "snapshot_sha256": snapshot.snapshot_sha256,
            "spec_hash": canonical_spec_hash(spec),
        }
    ready = boundaries["elapsed_common_sessions"] > 0
    return {
        "code": (
            "READY_FOR_INITIALIZE"
            if ready else "WAITING_FOR_FIRST_COMMON_SESSION"
        ),
        "experiment_id": spec["experiment_id"],
        "latest_common_session": boundaries["latest_common_session"],
        "snapshot_sha256": snapshot.snapshot_sha256,
        "spec_hash": canonical_spec_hash(spec),
    }


def initialize_shadow(
    spec_path: Path, data_dir: Path, state_root: Path,
) -> dict[str, Any]:
    """显式一次性建账；首个真实共同交易日之前保持零状态写入。"""
    spec = _load_frozen_spec(Path(spec_path))
    state_root = _validate_state_root_contract(Path(state_root))
    existing = _read_shadow_status_loaded(spec, state_root)
    if existing["code"] != "UNINITIALIZED":
        raise ShadowLifecycleError("影子实验已经初始化")
    snapshot = capture_adjusted_snapshot(spec, Path(data_dir))
    try:
        boundaries = derive_shadow_boundaries(spec, snapshot.frames)
    except ValueError as error:
        raise ShadowDataBlockedError(str(error)) from error
    if boundaries["elapsed_common_sessions"] == 0:
        return {
            "code": "WAITING_FOR_FIRST_COMMON_SESSION",
            "experiment_id": spec["experiment_id"],
            "latest_common_session": None,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "spec_hash": canonical_spec_hash(spec),
        }
    _validate_private_state_tree(state_root)
    try:
        with _private_state_umask():
            ledger = run_shadow_snapshot(
                spec, snapshot.frames, state_root,
                operation="initialize",
            )
    except shadow_runner.ShadowRunnerDataBlockedError as error:
        raise ShadowDataBlockedError(str(error)) from error
    except shadow_runner.ShadowRunnerLifecycleError as error:
        raise ShadowLifecycleError(str(error)) from error
    except OSError as error:
        raise ShadowIntegrityError("影子状态不可写或不可锁定") from error
    except ValueError as error:
        raise ShadowIntegrityError(str(error)) from error
    spec_hash = canonical_spec_hash(spec)
    experiment_dir = state_root / spec["experiment_id"] / spec_hash
    try:
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowIntegrityError("初始化后CURRENT不可读") from error
    return {
        "code": "INITIALIZED",
        "experiment_id": spec["experiment_id"],
        "ledger": ledger,
        "sequence": current["sequence"],
        "snapshot_sha256": snapshot.snapshot_sha256,
        "spec_hash": spec_hash,
    }


def update_shadow(
    spec_path: Path, data_dir: Path, state_root: Path,
) -> dict[str, Any]:
    """只接续既有权威实验；此入口永不隐式创建registration。"""
    spec = _load_frozen_spec(Path(spec_path))
    spec_hash = canonical_spec_hash(spec)
    state_root = _validate_state_root_contract(Path(state_root))
    snapshot: ShadowInputSnapshot | None = None
    try:
        with shadow_runner._experiment_lock(
            state_root, spec["experiment_id"], spec_hash, create=False,
        ):
            before_status = _read_shadow_status_locked(spec, state_root)
            if before_status["code"] == "UNINITIALIZED":
                raise ShadowLifecycleError("影子实验尚未初始化")
            before_sequence = before_status["sequence"]
            with _private_state_umask():
                if (before_status["experiment_state"]
                        in shadow_runner._TERMINAL_STATES):
                    ledger = shadow_runner._run_shadow_update_locked(
                        spec, None, state_root, operation="repair",
                    )
                else:
                    snapshot = capture_adjusted_snapshot(
                        spec, Path(data_dir),
                    )
                    ledger = shadow_runner._run_shadow_update_locked(
                        spec, None, state_root,
                        captured_frames=snapshot.frames,
                        operation="update",
                    )
            after_status = _read_shadow_status_locked(spec, state_root)
            after_sequence = after_status["sequence"]
    except ShadowOperationError:
        raise
    except shadow_runner.ShadowRunnerDataBlockedError as error:
        raise ShadowDataBlockedError(str(error)) from error
    except shadow_runner.ShadowRunnerLifecycleError as error:
        raise ShadowLifecycleError(str(error)) from error
    except (OSError, ValueError) as error:
        raise ShadowIntegrityError(str(error)) from error
    return {
        "code": (
            "UPDATED" if after_sequence > before_sequence else "NO_CHANGE"
        ),
        "experiment_id": spec["experiment_id"],
        "ledger": ledger,
        "sequence": after_sequence,
        "snapshot_sha256": (
            snapshot.snapshot_sha256 if snapshot is not None else None
        ),
        "spec_hash": spec_hash,
    }


def _uninitialized_status(
    spec: dict[str, Any], spec_hash: str,
) -> dict[str, Any]:
    return {
        "cache_health": None,
        "code": "UNINITIALIZED",
        "experiment_id": spec["experiment_id"],
        "ledger": None,
        "sequence": None,
        "spec_hash": spec_hash,
    }


def _read_shadow_status_locked(
    spec: dict[str, Any], state_root: Path,
) -> dict[str, Any]:
    """在已持有实验锁时从权威链重放只读状态。"""
    spec_hash = canonical_spec_hash(spec)
    experiment_dir = Path(state_root) / spec["experiment_id"] / spec_hash
    if not experiment_dir.exists():
        return _uninitialized_status(spec, spec_hash)
    _validate_private_state_tree(Path(state_root))
    registration_path = experiment_dir / "registration.json"
    if not registration_path.is_file():
        raise ShadowIntegrityError("影子状态缺少registration.json")
    try:
        registration_payload = registration_path.read_bytes()
        registration = json.loads(registration_payload.decode("utf-8"))
        if registration_payload != shadow_runner._canonical_json_bytes(
            registration
        ):
            raise ValueError("registration不是canonical JSON")
        shadow_runner._validate_registration(
            registration,
            spec,
            spec_hash,
            shadow_runner.algorithm_source_hashes(),
            shadow_runner.runtime_environment_identity(),
        )
        current = shadow_runner._read_current(
            experiment_dir, repair_cache=False,
        )
        frames, ledger, protocol = shadow_runner._replay_committed_frames(
            experiment_dir, registration, current, spec,
        )
        recomputed_ledger, recomputed_protocol = (
            shadow_runner._build_pre_ready_snapshot(
                spec,
                frames,
                formal_evaluation_count=protocol[
                    "formal_evaluation_count"
                ],
                evaluation_result=protocol["evaluation_result"],
            )
        )
        if ledger != recomputed_ledger:
            raise ValueError("头部公开账本与权威K线重算结果不匹配")
        if protocol != recomputed_protocol:
            raise ValueError("头部协议状态与权威K线重算结果不匹配")

        expected_current = shadow_runner._canonical_json_bytes(current)
        current_path = experiment_dir / "CURRENT"
        if not current_path.exists():
            current_health = "missing"
        else:
            current_health = (
                "ok"
                if current_path.read_bytes() == expected_current else "stale"
            )
        expected_ledger = (json.dumps(
            ledger,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n").encode("utf-8")
        ledger_path = experiment_dir / "ledger.json"
        if not ledger_path.exists():
            ledger_health = "missing"
        else:
            ledger_health = (
                "ok"
                if ledger_path.read_bytes() == expected_ledger else "stale"
            )
        accepted_health = "ok"
        for symbol, frame in frames.items():
            try:
                cached = shadow_runner._read_bar_csv(
                    experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
                )
            except ValueError:
                accepted_health = "stale"
                break
            if shadow_runner.canonical_bar_hash(cached) != (
                shadow_runner.canonical_bar_hash(frame)
            ):
                accepted_health = "stale"
                break
        evaluation_path = experiment_dir / "evaluation.json"
        evaluation_result = protocol.get("evaluation_result")
        if evaluation_result is None:
            evaluation_health = (
                "stale" if evaluation_path.exists() else "ok"
            )
        elif not evaluation_path.exists():
            evaluation_health = "missing"
        else:
            expected_evaluation = (json.dumps(
                evaluation_result,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n").encode("utf-8")
            evaluation_health = (
                "ok"
                if evaluation_path.read_bytes() == expected_evaluation
                else "stale"
            )
        cache_health = {
            "CURRENT": current_health,
            "accepted_bars": accepted_health,
            "evaluation.json": evaluation_health,
            "ledger.json": ledger_health,
        }
    except ShadowIntegrityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            KeyError, TypeError, ValueError) as error:
        raise ShadowIntegrityError(str(error)) from error

    repair_required = any(value != "ok" for value in cache_health.values())
    return {
        "cache_health": cache_health,
        "code": "REPAIR_REQUIRED" if repair_required else "STATUS",
        "experiment_id": spec["experiment_id"],
        "experiment_state": ledger["state"],
        "ledger": ledger,
        "sequence": current["sequence"],
        "spec_hash": spec_hash,
    }


def _read_shadow_status_loaded(
    spec: dict[str, Any], state_root: Path,
) -> dict[str, Any]:
    """从权威链只读检查状态；派生缓存异常也不得在此修复。"""
    spec_hash = canonical_spec_hash(spec)
    experiment_dir = Path(state_root) / spec["experiment_id"] / spec_hash
    if not experiment_dir.exists():
        return _uninitialized_status(spec, spec_hash)
    try:
        with shadow_runner._experiment_lock(
            Path(state_root), spec["experiment_id"], spec_hash,
            create=False, shared=True,
        ):
            return _read_shadow_status_locked(spec, Path(state_root))
    except ShadowIntegrityError:
        raise
    except shadow_runner.ShadowRunnerLifecycleError as error:
        raise ShadowIntegrityError(str(error)) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            KeyError, TypeError, ValueError) as error:
        raise ShadowIntegrityError(str(error)) from error


def read_shadow_status(
    spec_path: Path, state_root: Path,
) -> dict[str, Any]:
    """加载冻结spec后从权威链只读重放公开状态。"""
    spec = _load_frozen_spec(Path(spec_path))
    state_root = _validate_state_root_contract(Path(state_root))
    return _read_shadow_status_loaded(spec, state_root)
