# -*- coding: utf-8 -*-
"""v6 shadow 的严格行情门禁与官方生命周期测试。"""
from __future__ import annotations

import hashlib
import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from gcn.backtest import shadow_operations, shadow_runner
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


_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "gcn"
    / "backtest"
    / "shadow_specs"
    / "v6-profit-arm20-keep50-20260905.json"
)


def _bars(dates: list[str] | None = None) -> pd.DataFrame:
    index = pd.to_datetime(dates or [
        "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
    ])
    size = len(index)
    frame = pd.DataFrame({
        "open": [100.0 + value for value in range(size)],
        "high": [102.0 + value for value in range(size)],
        "low": [99.0 + value for value in range(size)],
        "close": [101.0 + value for value in range(size)],
        "volume": [1000.0 + value for value in range(size)],
    }, index=index)
    frame.index.name = "date"
    return frame


def _write_adjusted_inputs(
    data_dir: Path, frames: dict[str, pd.DataFrame],
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for symbol, frame in frames.items():
        csv_path = data_dir / f"{symbol}_1d.csv"
        frame.to_csv(csv_path)
        csv_bytes = csv_path.read_bytes()
        (data_dir / f"{symbol}_1d.lock").touch()
        (data_dir / f"{symbol}_1d.csv.meta.json").write_text(
            json.dumps({
                "source": "yahoo",
                "adjustment": "adjusted",
                "sha256": hashlib.sha256(csv_bytes).hexdigest(),
            }),
            encoding="utf-8",
        )


def _tree_fingerprint(root: Path) -> list[tuple[str, int, str | None]]:
    if not root.exists():
        return []
    result = []
    for path in sorted([root, *root.rglob("*")]):
        relative = "." if path == root else path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.stat().st_mode)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result.append((relative, mode, digest))
    return result


def test_preflight_blocks_sidecar_csv_mismatch_without_creating_state():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        frames = {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        }
        _write_adjusted_inputs(data_dir, frames)
        symbol = spec["universe"]["core_symbols"][-1]
        with (data_dir / f"{symbol}_1d.csv").open("ab") as output:
            output.write(b"\n")

        try:
            preflight_shadow(_SPEC_PATH, data_dir, state_root)
        except ShadowDataBlockedError as error:
            assert "SHA-256" in str(error)
            assert symbol in str(error)
        else:
            raise AssertionError("sidecar/CSV mismatch must block preflight")

        assert not state_root.exists()


def test_initialize_waits_for_first_forward_session_without_state_writes():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "custom-market-data"
        state_root = root / "state"
        frames = {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        }
        _write_adjusted_inputs(data_dir, frames)

        result = initialize_shadow(_SPEC_PATH, data_dir, state_root)

        assert result["code"] == "WAITING_FOR_FIRST_COMMON_SESSION"
        assert result["latest_common_session"] is None
        assert not state_root.exists()


def test_initialize_maps_state_lock_failure_to_integrity_error():
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
        original_run = shadow_operations.run_shadow_snapshot
        shadow_operations.run_shadow_snapshot = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("blocked")
            )
        )
        try:
            try:
                initialize_shadow(_SPEC_PATH, data_dir, state_root)
            except ShadowIntegrityError as error:
                assert "状态" in str(error)
            else:
                raise AssertionError("state lock errors must fail closed")
        finally:
            shadow_operations.run_shadow_snapshot = original_run

        assert not state_root.exists()


def test_initialize_publishes_first_forward_session_once_with_private_state():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        frames = {
            symbol: _bars(dates)
            for symbol in spec["universe"]["core_symbols"]
        }
        _write_adjusted_inputs(data_dir, frames)

        result = initialize_shadow(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root
            / spec["experiment_id"]
            / result["spec_hash"]
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        assert result["code"] == "INITIALIZED"
        assert result["sequence"] == current["sequence"] == 1
        assert result["ledger"]["elapsed_common_sessions"] == 1
        assert len(list((experiment_dir / "commits").glob("*.commit"))) == 2
        assert stat.S_IMODE(state_root.stat().st_mode) & 0o077 == 0
        assert all(
            stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
            for path in experiment_dir.rglob("*")
        )


def test_update_never_implicitly_initializes_an_empty_state_root():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        frames = {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        }
        _write_adjusted_inputs(data_dir, frames)

        try:
            update_shadow(_SPEC_PATH, data_dir, state_root)
        except ShadowLifecycleError as error:
            assert "尚未初始化" in str(error)
        else:
            raise AssertionError("update must reject an uninitialized experiment")

        assert not state_root.exists()


def test_status_reports_uninitialized_without_creating_state_or_lock():
    with TemporaryDirectory() as directory:
        state_root = Path(directory) / "state"

        result = read_shadow_status(_SPEC_PATH, state_root)

        assert result["code"] == "UNINITIALIZED"
        assert result["ledger"] is None
        assert result["sequence"] is None
        assert not state_root.exists()


def test_status_rejects_regular_file_as_state_root():
    with TemporaryDirectory() as directory:
        state_root = Path(directory) / "state"
        state_root.write_text("not-a-directory\n", encoding="utf-8")
        state_root.chmod(0o600)
        before = state_root.read_bytes()

        try:
            read_shadow_status(_SPEC_PATH, state_root)
        except ShadowIntegrityError as error:
            assert "目录" in str(error)
        else:
            raise AssertionError("a state root file must fail closed")

        assert state_root.read_bytes() == before


def test_status_rejects_state_root_inside_repository_without_writing():
    state_root = (
        Path(__file__).resolve().parents[1]
        / ".shadow-state-contract-test"
    )
    assert not state_root.exists()

    try:
        read_shadow_status(_SPEC_PATH, state_root)
    except ShadowConfigurationError as error:
        assert "仓库外" in str(error)
    else:
        raise AssertionError("repository-local shadow state must be rejected")

    assert not state_root.exists()


def test_status_replays_authority_and_reports_stale_caches_without_writing():
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
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        experiment_dir = (
            state_root / spec["experiment_id"] / initialized["spec_hash"]
        )
        (experiment_dir / "CURRENT").unlink()
        (experiment_dir / "ledger.json").write_text(
            '{"state":"forged-cache"}\n', encoding="utf-8"
        )
        before = _tree_fingerprint(state_root)

        status_result = read_shadow_status(_SPEC_PATH, state_root)

        assert status_result["code"] == "REPAIR_REQUIRED"
        assert status_result["sequence"] == 1
        assert status_result["ledger"] == initialized["ledger"]
        assert status_result["cache_health"]["CURRENT"] == "missing"
        assert status_result["cache_health"]["ledger.json"] == "stale"
        assert _tree_fingerprint(state_root) == before


def test_status_treats_ahead_current_as_stale_cache_not_authority():
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
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        experiment_dir = (
            state_root / spec["experiment_id"] / initialized["spec_hash"]
        )
        (experiment_dir / "CURRENT").write_bytes(
            shadow_runner._canonical_json_bytes({
                "generation_hash": "a" * 64,
                "sequence": 999,
            })
        )
        before = _tree_fingerprint(state_root)

        result = read_shadow_status(_SPEC_PATH, state_root)

        assert result["code"] == "REPAIR_REQUIRED"
        assert result["sequence"] == 1
        assert result["ledger"] == initialized["ledger"]
        assert result["cache_health"]["CURRENT"] == "stale"
        assert _tree_fingerprint(state_root) == before


def test_status_fails_closed_when_state_permissions_are_not_private():
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
        initialize_shadow(_SPEC_PATH, data_dir, state_root)
        state_root.chmod(0o755)
        before = _tree_fingerprint(state_root)

        try:
            read_shadow_status(_SPEC_PATH, state_root)
        except ShadowIntegrityError as error:
            assert "权限" in str(error)
        else:
            raise AssertionError("group/world-readable state must be blocked")

        assert _tree_fingerprint(state_root) == before


def test_preflight_distinguishes_existing_state_from_initialization_ready():
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
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)

        result = preflight_shadow(_SPEC_PATH, data_dir, state_root)

        assert result["code"] == "READY_FOR_UPDATE"
        assert result["sequence"] == initialized["sequence"]
        assert result["experiment_state"] == "INITIAL_EMBARGO"


def test_update_appends_only_new_common_sessions_and_then_is_idempotent():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        initial_dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(initial_dates)
            for symbol in spec["universe"]["core_symbols"]
        })
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        updated_dates = [*initial_dates, "2026-09-09"]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(updated_dates)
            for symbol in spec["universe"]["core_symbols"]
        })

        updated = update_shadow(_SPEC_PATH, data_dir, state_root)
        unchanged = update_shadow(_SPEC_PATH, data_dir, state_root)

        assert updated["code"] == "UPDATED"
        assert updated["sequence"] == initialized["sequence"] + 1
        assert updated["ledger"]["elapsed_common_sessions"] == 2
        assert unchanged["code"] == "NO_CHANGE"
        assert unchanged["sequence"] == updated["sequence"]
        assert unchanged["ledger"] == updated["ledger"]


def test_initialize_is_one_shot_and_second_attempt_preserves_state_bytes():
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
        initialize_shadow(_SPEC_PATH, data_dir, state_root)
        before = _tree_fingerprint(state_root)

        try:
            initialize_shadow(_SPEC_PATH, data_dir, state_root)
        except ShadowLifecycleError as error:
            assert "已经初始化" in str(error)
        else:
            raise AssertionError("initialize must be one-shot")

        assert _tree_fingerprint(state_root) == before


def test_initialize_consumes_captured_frames_without_rereading_csv_paths():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        captured_dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(captured_dates)
            for symbol in spec["universe"]["core_symbols"]
        })
        original_capture = shadow_operations.capture_adjusted_snapshot
        original_path_reader = shadow_runner._read_bar_csv

        def capture_then_replace_paths(frozen_spec, frozen_data_dir):
            snapshot = original_capture(frozen_spec, frozen_data_dir)
            replaced_dates = [*captured_dates, "2026-09-09"]
            _write_adjusted_inputs(data_dir, {
                symbol: _bars(replaced_dates)
                for symbol in spec["universe"]["core_symbols"]
            })
            return snapshot

        shadow_operations.capture_adjusted_snapshot = capture_then_replace_paths
        shadow_runner._read_bar_csv = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("official initialize must not reread CSV paths")
        )
        try:
            initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        finally:
            shadow_operations.capture_adjusted_snapshot = original_capture
            shadow_runner._read_bar_csv = original_path_reader

        assert initialized["sequence"] == 1
        assert initialized["ledger"]["elapsed_common_sessions"] == 1


def test_update_uses_authority_and_repairs_missing_current_cache():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        initial_dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(initial_dates)
            for symbol in spec["universe"]["core_symbols"]
        })
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        experiment_dir = (
            state_root / spec["experiment_id"] / initialized["spec_hash"]
        )
        (experiment_dir / "CURRENT").unlink()
        updated_dates = [*initial_dates, "2026-09-09"]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(updated_dates)
            for symbol in spec["universe"]["core_symbols"]
        })

        result = update_shadow(_SPEC_PATH, data_dir, state_root)

        assert result["code"] == "UPDATED"
        assert result["sequence"] == 2
        assert (experiment_dir / "CURRENT").is_file()


def test_preflight_rejects_missing_lock_and_untrusted_sidecar_variants():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    symbol = spec["universe"]["core_symbols"][0]
    cases = {
        "missing-lock": lambda data_dir: (
            data_dir / f"{symbol}_1d.lock"
        ).unlink(),
        "missing-meta": lambda data_dir: (
            data_dir / f"{symbol}_1d.csv.meta.json"
        ).unlink(),
        "wrong-adjustment": lambda data_dir: (
            data_dir / f"{symbol}_1d.csv.meta.json"
        ).write_text(json.dumps({
            "source": "yahoo", "adjustment": "raw", "sha256": "a" * 64,
        }), encoding="utf-8"),
        "unsupported-source": lambda data_dir: (
            data_dir / f"{symbol}_1d.csv.meta.json"
        ).write_text(json.dumps({
            "source": "cache", "adjustment": "adjusted", "sha256": "a" * 64,
        }), encoding="utf-8"),
        "duplicate-field": lambda data_dir: (
            data_dir / f"{symbol}_1d.csv.meta.json"
        ).write_text(
            '{"source":"yahoo","source":"futu",'
            '"adjustment":"adjusted","sha256":"' + "a" * 64 + '"}',
            encoding="utf-8",
        ),
    }
    for name, mutate in cases.items():
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            state_root = root / "state"
            _write_adjusted_inputs(data_dir, {
                item: _bars()
                for item in spec["universe"]["core_symbols"]
            })
            mutate(data_dir)

            try:
                preflight_shadow(_SPEC_PATH, data_dir, state_root)
            except ShadowDataBlockedError:
                pass
            else:
                raise AssertionError(f"{name} must be DATA_BLOCKED")

            assert not state_root.exists()
            if name == "missing-lock":
                assert not (data_dir / f"{symbol}_1d.lock").exists()


def test_preflight_maps_market_lock_failure_to_data_blocked():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        _write_adjusted_inputs(data_dir, {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        })
        original_flock = shadow_operations.fcntl.flock
        shadow_operations.fcntl.flock = lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(PermissionError("blocked"))
        )
        try:
            try:
                preflight_shadow(_SPEC_PATH, data_dir, state_root)
            except ShadowDataBlockedError as error:
                assert "事务锁" in str(error)
            else:
                raise AssertionError("market lock errors must be DATA_BLOCKED")
        finally:
            shadow_operations.fcntl.flock = original_flock

        assert not state_root.exists()


def test_snapshot_waits_for_csv_meta_transaction_and_sees_only_final_pair():
    import fcntl

    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    symbol = spec["universe"]["core_symbols"][0]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        _write_adjusted_inputs(data_dir, {
            item: _bars()
            for item in spec["universe"]["core_symbols"]
        })
        final_frame = _bars([
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ])
        csv_path = data_dir / f"{symbol}_1d.csv"
        meta_path = data_dir / f"{symbol}_1d.csv.meta.json"
        lock_path = data_dir / f"{symbol}_1d.lock"
        csv_written = threading.Event()
        finish_transaction = threading.Event()

        def write_transaction():
            with lock_path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    final_frame.to_csv(csv_path)
                    csv_written.set()
                    assert finish_transaction.wait(timeout=5)
                    csv_payload = csv_path.read_bytes()
                    meta_path.write_text(json.dumps({
                        "source": "yahoo",
                        "adjustment": "adjusted",
                        "sha256": hashlib.sha256(csv_payload).hexdigest(),
                    }), encoding="utf-8")
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        with ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(write_transaction)
            assert csv_written.wait(timeout=5)
            reader = executor.submit(
                shadow_operations.capture_adjusted_snapshot, spec, data_dir,
            )
            try:
                reader.result(timeout=0.05)
            except TimeoutError:
                pass
            else:
                raise AssertionError("snapshot read an in-flight CSV/meta pair")
            finally:
                finish_transaction.set()
            writer.result(timeout=5)
            snapshot = reader.result(timeout=5)

        assert len(snapshot.frames[symbol]) == len(final_frame)
        assert snapshot.receipts[symbol]["csv_sha256"] == hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest()


def test_preflight_reports_every_blocked_symbol_in_one_attempt():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    blocked = spec["universe"]["core_symbols"][-2:]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        _write_adjusted_inputs(data_dir, {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        })
        for symbol in blocked:
            meta_path = data_dir / f"{symbol}_1d.csv.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["sha256"] = "0" * 64
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

        try:
            preflight_shadow(_SPEC_PATH, data_dir, state_root)
        except ShadowDataBlockedError as error:
            message = str(error)
            assert all(symbol in message for symbol in blocked)
        else:
            raise AssertionError("all invalid symbols must be reported together")

        assert not state_root.exists()


def test_update_classifies_divergent_forward_calendar_as_data_blocked():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        initial_dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        frames = {
            symbol: _bars(initial_dates)
            for symbol in spec["universe"]["core_symbols"]
        }
        _write_adjusted_inputs(data_dir, frames)
        initialize_shadow(_SPEC_PATH, data_dir, state_root)
        symbol = spec["universe"]["core_symbols"][0]
        frames[symbol] = _bars([*initial_dates, "2026-09-09"])
        _write_adjusted_inputs(data_dir, frames)
        before = _tree_fingerprint(state_root)

        try:
            update_shadow(_SPEC_PATH, data_dir, state_root)
        except ShadowDataBlockedError as error:
            assert "前向交易日不一致" in str(error)
        else:
            raise AssertionError("calendar divergence must be DATA_BLOCKED")

        assert _tree_fingerprint(state_root) == before


def test_status_flags_unexpected_evaluation_cache_without_disclosing_it():
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
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        experiment_dir = (
            state_root / spec["experiment_id"] / initialized["spec_hash"]
        )
        evaluation_path = experiment_dir / "evaluation.json"
        evaluation_path.write_text('{"forged":true}\n', encoding="utf-8")
        evaluation_path.chmod(0o600)
        before = _tree_fingerprint(state_root)

        result = read_shadow_status(_SPEC_PATH, state_root)

        assert result["code"] == "REPAIR_REQUIRED"
        assert result["cache_health"]["evaluation.json"] == "stale"
        assert "evaluation" not in result
        assert _tree_fingerprint(state_root) == before


def test_terminal_update_does_not_read_external_market_data():
    original_lock = shadow_runner._experiment_lock
    original_status = shadow_operations._read_shadow_status_locked
    original_capture = shadow_operations.capture_adjusted_snapshot
    original_run = shadow_runner._run_shadow_update_locked
    repair_calls = []
    terminal_ledger = {
        "state": "REJECTED_KEEP_V5",
        "elapsed_common_sessions": 999,
    }
    shadow_runner._experiment_lock = lambda *_args, **_kwargs: nullcontext()
    shadow_operations._read_shadow_status_locked = lambda *_args, **_kwargs: {
        "code": "STATUS",
        "experiment_id": "v6-profit-arm20-keep50-20260905",
        "experiment_state": "REJECTED_KEEP_V5",
        "ledger": terminal_ledger,
        "sequence": 999,
        "spec_hash": "c" * 64,
    }
    shadow_operations.capture_adjusted_snapshot = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal update must not read market data")
        )
    )
    shadow_runner._run_shadow_update_locked = lambda *_args, **_kwargs: (
        repair_calls.append(True) or terminal_ledger
    )
    try:
        result = update_shadow(
            _SPEC_PATH, Path("/missing-data"), Path("/missing-state"),
        )
    finally:
        shadow_runner._experiment_lock = original_lock
        shadow_operations._read_shadow_status_locked = original_status
        shadow_operations.capture_adjusted_snapshot = original_capture
        shadow_runner._run_shadow_update_locked = original_run

    assert result["code"] == "NO_CHANGE"
    assert result["sequence"] == 999
    assert result["ledger"] == terminal_ledger
    assert result["snapshot_sha256"] is None
    assert repair_calls == [True]


def test_initialize_rejects_existing_state_even_when_incoming_is_stale():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        forward_dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(forward_dates)
            for symbol in spec["universe"]["core_symbols"]
        })
        initialize_shadow(_SPEC_PATH, data_dir, state_root)
        _write_adjusted_inputs(data_dir, {
            symbol: _bars()
            for symbol in spec["universe"]["core_symbols"]
        })
        before = _tree_fingerprint(state_root)

        try:
            initialize_shadow(_SPEC_PATH, data_dir, state_root)
        except ShadowLifecycleError as error:
            assert "已经初始化" in str(error)
        else:
            raise AssertionError("initialize must reject every existing state")

        assert _tree_fingerprint(state_root) == before


def test_active_update_removes_an_unexpected_evaluation_cache():
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
        initialized = initialize_shadow(_SPEC_PATH, data_dir, state_root)
        experiment_dir = (
            state_root / spec["experiment_id"] / initialized["spec_hash"]
        )
        evaluation_path = experiment_dir / "evaluation.json"
        evaluation_path.write_text('{"forged":true}\n', encoding="utf-8")
        evaluation_path.chmod(0o600)
        assert read_shadow_status(_SPEC_PATH, state_root)["code"] == (
            "REPAIR_REQUIRED"
        )

        result = update_shadow(_SPEC_PATH, data_dir, state_root)

        assert result["code"] == "NO_CHANGE"
        assert not evaluation_path.exists()
        assert read_shadow_status(_SPEC_PATH, state_root)["code"] == "STATUS"


def test_state_only_cache_repair_is_rejected_before_active_data_access():
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
        initialize_shadow(_SPEC_PATH, data_dir, state_root)
        original_reader = shadow_runner._read_bar_csv
        shadow_runner._read_bar_csv = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("state-only repair must never read external data")
        )
        try:
            try:
                shadow_runner.repair_shadow_caches(spec, state_root)
            except shadow_runner.ShadowRunnerLifecycleError as error:
                assert "非终态" in str(error)
            else:
                raise AssertionError("active state-only repair must be rejected")
        finally:
            shadow_runner._read_bar_csv = original_reader


def test_concurrent_updates_report_exactly_one_committed_transition():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "data"
        state_root = root / "state"
        initial_dates = [
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04",
            "2026-09-08",
        ]
        _write_adjusted_inputs(data_dir, {
            symbol: _bars(initial_dates)
            for symbol in spec["universe"]["core_symbols"]
        })
        initialize_shadow(_SPEC_PATH, data_dir, state_root)
        _write_adjusted_inputs(data_dir, {
            symbol: _bars([*initial_dates, "2026-09-09"])
            for symbol in spec["universe"]["core_symbols"]
        })
        original_status = shadow_operations._read_shadow_status_loaded
        both_prepared = threading.Barrier(2)
        status_reads = 0
        status_reads_lock = threading.Lock()

        def synchronized_status(*args, **kwargs):
            nonlocal status_reads
            result = original_status(*args, **kwargs)
            with status_reads_lock:
                status_reads += 1
                should_wait = status_reads <= 2
            if should_wait:
                both_prepared.wait(timeout=5)
            return result

        shadow_operations._read_shadow_status_loaded = synchronized_status
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    lambda _value: update_shadow(
                        _SPEC_PATH, data_dir, state_root,
                    ),
                    range(2),
                ))
        finally:
            shadow_operations._read_shadow_status_loaded = original_status

        assert sorted(result["code"] for result in results) == [
            "NO_CHANGE", "UPDATED",
        ]
        assert {result["sequence"] for result in results} == {2}
