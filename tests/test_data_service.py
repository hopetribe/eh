# -*- coding: utf-8 -*-
"""数据服务测试: 代码映射 / CSV 解析 / 缓存落盘。"""
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import pandas as pd

from gcn.data.service import (_cache_is_fresh, _load_cache, _save_cache,
                              parse_csv_text, to_futu_symbol, to_yahoo_symbol,
                              _rows_from_df)


def test_symbol_mapping():
    assert to_futu_symbol("TQQQ") == "US.TQQQ"
    assert to_futu_symbol("00700") == "HK.00700"
    assert to_futu_symbol("600519") == "SH.600519"
    assert to_futu_symbol("000001") == "SZ.000001"
    assert to_futu_symbol("HK.00700") == "HK.00700"
    assert to_futu_symbol("0700.HK") == "HK.00700"
    assert to_futu_symbol("600519.SS") == "SH.600519"
    assert to_yahoo_symbol("US.TQQQ") == "TQQQ"
    assert to_yahoo_symbol("00700") == "0700.HK"
    assert to_yahoo_symbol("600519") == "600519.SS"
    assert to_yahoo_symbol("000001") == "000001.SZ"


def test_cache_key_rejects_paths_and_normalizes_aliases():
    import gcn.data.service as svc
    assert svc._cache_path("00700", "1d") == svc._cache_path("0700.HK", "1d")
    assert svc._cache_path("600519", "1d") == svc._cache_path("SH.600519", "1d")
    assert svc._cache_path("BRK.B", "1d") != svc._cache_path("BRK_B", "1d")
    assert svc._cache_lock_path("00700", "1d") == svc._cache_lock_path("HK.00700", "1d")
    try:
        svc._cache_path("/tmp/escape", "1d")
    except ValueError:
        pass
    else:
        raise AssertionError("绝对路径型股票代码必须拒绝")


def test_parse_csv_cn_and_en(tmp_path=None):
    csv_cn = "时间,开盘价,最高价,最低价,收盘价,成交量\n2026-01-02,10,10.5,9.9,10.4,100\n"
    d1 = parse_csv_text(csv_cn)
    assert len(d1) == 1 and abs(d1["close"].iloc[0] - 10.4) < 1e-9
    csv_en = "date,open,high,low,close,volume\n2026-01-02,10,10.5,9.9,10.4,100\n"
    assert len(parse_csv_text(csv_en)) == 1


def test_parse_csv_mixed_dates_epoch_seconds_and_bad_row():
    text = ("timestamp,open,high,low,close,volume\n"
            "2026-01-01,1,1,1,1,1\n"
            "2026-01-02 12:30,2,2,2,2,2\n"
            "1767398400,3,3,3,3,3\n"
            "not-a-date,4,4,4,4,4\n")
    out = parse_csv_text(text)
    assert len(out) == 3
    assert list(out["close"]) == [1, 2, 3]
    assert set(out.index.year) == {2026}


def test_df_from_rows_drops_only_bad_dates():
    import gcn.data.service as svc
    rows = [["2026-01-01", 1, 1, 1, 1, 1],
            ["bad", 2, 2, 2, 2, 2],
            ["2026-01-03", 3, 3, 3, 3, 3]]
    out = svc.df_from_rows(rows)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out["close"]) == [1.0, 3.0]


def test_cache_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path as P
    tmp = P(tempfile.mkdtemp())
    import gcn.data.service as svc
    old = svc.DATA_DIR
    svc.DATA_DIR = tmp
    df = pd.DataFrame({"open": [10.0], "high": [11.0], "low": [9.0],
                       "close": [10.5], "volume": [1000.0]},
                      index=pd.to_datetime(["2026-08-28"]))
    df.attrs.update(source="yahoo", adjustment="adjusted")
    svc._save_cache("TEST.SYM", "1d", df)
    back = svc._load_cache("TEST.SYM", "1d")
    assert back is not None and abs(back["close"].iloc[0] - 10.5) < 1e-9
    assert back.attrs == {"source": "yahoo", "adjustment": "adjusted"}
    plain = df.copy()
    plain.attrs.clear()
    svc._save_cache("TEST.SYM", "1d", plain)
    assert svc._load_cache("TEST.SYM", "1d").attrs == {}
    svc.DATA_DIR = old


def test_daily_cache_freshness_uses_last_completed_session_not_mtime():
    import tempfile
    import gcn.data.service as svc
    path = Path(tempfile.mkdtemp()) / "600519_1d.csv"
    path.write_text("touched today", encoding="utf-8")
    frame = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-08-28"]))
    assert not _cache_is_fresh(
        frame, "1d", path, symbol="600519", now=pd.Timestamp("2026-08-31 18:00"))
    assert _cache_is_fresh(
        frame, "1d", path, symbol="600519", now=pd.Timestamp("2026-08-31 10:00"))
    assert not _cache_is_fresh(frame.iloc[:0], "1d", path, symbol="600519")


def test_symbol_lock_identity():
    from gcn.data.service import _symbol_lock
    assert _symbol_lock("AAPL") is _symbol_lock("AAPL")   # 同标的复用同一把锁
    assert _symbol_lock("AAPL") is not _symbol_lock("MSFT")


def test_fetch_quote_refreshes_fresh_but_short_cache():
    import tempfile
    import gcn.data.service as svc
    old_dir, old_fresh, old_open, old_yahoo = (
        svc.DATA_DIR, svc._cache_is_fresh, svc._opend_reachable, svc._fetch_yahoo)
    calls = {"n": 0}
    try:
        svc.DATA_DIR = Path(tempfile.mkdtemp())
        idx = pd.bdate_range("2026-01-01", periods=100)
        df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1.0}, index=idx)
        svc._save_cache("AAPL", "1d", df)
        svc._cache_is_fresh = lambda *a, **k: True
        svc._opend_reachable = lambda *a, **k: False
        def fake_yahoo(symbol, interval, count):
            calls["n"] += 1
            dates = pd.bdate_range("2025-01-01", periods=count)
            return [[d.strftime("%Y-%m-%d"), 1, 1, 1, 1, 1] for d in dates]
        svc._fetch_yahoo = fake_yahoo
        result = svc.fetch_quote("AAPL", "1d", count=200)
        assert calls["n"] == 1 and len(result["rows"]) == 200
        assert svc._load_cache("AAPL", "1d").attrs == {
            "source": "yahoo", "adjustment": "adjusted"}
    finally:
        svc.DATA_DIR, svc._cache_is_fresh, svc._opend_reachable, svc._fetch_yahoo = (
            old_dir, old_fresh, old_open, old_yahoo)


def test_fetch_quote_force_bypasses_fresh_full_cache():
    import tempfile
    import gcn.data.service as svc
    old_dir, old_fresh, old_open, old_yahoo = (
        svc.DATA_DIR, svc._cache_is_fresh, svc._opend_reachable, svc._fetch_yahoo)
    calls = {"n": 0}
    try:
        svc.DATA_DIR = Path(tempfile.mkdtemp())
        idx = pd.bdate_range("2025-01-01", periods=100)
        frame = pd.DataFrame({"open": 1, "high": 1, "low": 1,
                              "close": 1, "volume": 1}, index=idx)
        svc._save_cache("AAPL", "1d", frame)
        svc._cache_is_fresh = lambda *a, **k: True
        svc._opend_reachable = lambda *a, **k: False
        def online(*args):
            calls["n"] += 1
            return [[d.strftime("%Y-%m-%d"), 1, 1, 1, 1, 1] for d in idx]
        svc._fetch_yahoo = online
        result = svc.fetch_quote("AAPL", "1d", count=100, force=True)
        assert calls["n"] == 1 and result["refresh_failed"] is False
    finally:
        svc.DATA_DIR, svc._cache_is_fresh, svc._opend_reachable, svc._fetch_yahoo = (
            old_dir, old_fresh, old_open, old_yahoo)


def test_futu_history_uses_supported_paginated_contract():
    import gcn.data.service as svc
    calls = []
    pages = [
        (pd.DataFrame({"time_key": ["2026-01-01", "2026-01-02"],
                       "open": [1, 2], "high": [1, 2], "low": [1, 2],
                       "close": [1, 2], "volume": [1, 2]}), b"next"),
        (pd.DataFrame({"time_key": ["2026-01-03", "2026-01-04"],
                       "open": [3, 4], "high": [3, 4], "low": [3, 4],
                       "close": [3, 4], "volume": [3, 4]}), None),
    ]
    class FakeContext:
        def __init__(self, **kwargs): pass
        def request_history_kline(self, code, **kwargs):
            calls.append((code, kwargs))
            data, key = pages[len(calls) - 1]
            return 0, data, key
        def close(self): pass
    fake = SimpleNamespace(AuType=SimpleNamespace(QFQ="qfq"),
                           KLType=SimpleNamespace(K_DAY="day"),
                           OpenQuoteContext=FakeContext)
    old = sys.modules.get("futu")
    try:
        sys.modules["futu"] = fake
        out = svc._fetch_futu("600519.SS", "1d", 4)
    finally:
        if old is None:
            sys.modules.pop("futu", None)
        else:
            sys.modules["futu"] = old
    assert len(out) == 4 and calls[0][0] == "SH.600519"
    assert out.attrs == {"source": "futu", "adjustment": "adjusted"}
    assert calls[0][1]["page_req_key"] is None
    assert calls[1][1]["page_req_key"] == b"next"
    assert calls[0][1]["start"] and calls[0][1]["end"]
    assert "page_req_kcount" not in calls[0][1]


def test_adjusted_incremental_merge_preserves_history_and_replaces_legacy_basis():
    import gcn.data.service as svc
    def frame(day, value, adjustment=None):
        out = pd.DataFrame({"open": [value], "high": [value], "low": [value],
                            "close": [value], "volume": [1]},
                           index=pd.to_datetime([day]))
        if adjustment:
            out.attrs.update(source="futu", adjustment=adjustment)
        return out
    old = frame("2026-01-01", 1, "adjusted")
    fresh = frame("2026-01-02", 2, "adjusted")
    merged = svc._merge_market_data(old, fresh)
    assert len(merged) == 2 and list(merged["close"]) == [1, 2]
    legacy = frame("2020-01-01", 100)
    replaced = svc._merge_market_data(legacy, fresh)
    assert list(replaced.index) == list(fresh.index)


def test_yahoo_daily_ohlc_is_adjusted_consistently():
    import gcn.data.service as svc
    payload = {"chart": {"result": [{
        "timestamp": [1767225600], "meta": {"gmtoffset": 0},
        "indicators": {
            "quote": [{"open": [8], "high": [12], "low": [6],
                       "close": [10], "volume": [100]}],
            "adjclose": [{"adjclose": [5]}],
        }}]}}
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(payload).encode()
    old = svc.urllib.request.urlopen
    try:
        svc.urllib.request.urlopen = lambda *a, **k: Resp()
        row = svc._fetch_yahoo("AAPL", "1d", 10)[0]
    finally:
        svc.urllib.request.urlopen = old
    assert row[1:5] == [4.0, 6.0, 3.0, 5.0]
    assert row[5] == 100.0


def test_yahoo_intraday_timestamps_use_exchange_timezone_dst():
    import gcn.data.service as svc
    stamps = [int(pd.Timestamp("2026-01-02 14:30", tz="UTC").timestamp()),
              int(pd.Timestamp("2026-07-01 14:30", tz="UTC").timestamp())]
    payload = {"chart": {"result": [{
        "timestamp": stamps,
        "meta": {"gmtoffset": -18000, "exchangeTimezoneName": "America/New_York"},
        "indicators": {"quote": [{"open": [1, 1], "high": [1, 1],
                                    "low": [1, 1], "close": [1, 1],
                                    "volume": [1, 1]}]}}]}}
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(payload).encode()
    old = svc.urllib.request.urlopen
    try:
        svc.urllib.request.urlopen = lambda *a, **k: Resp()
        rows = svc._fetch_yahoo("AAPL", "5m", 10)
    finally:
        svc.urllib.request.urlopen = old
    assert [r[0] for r in rows] == ["2026-01-02 09:30", "2026-07-01 10:30"]


def test_online_failure_marks_cache_result_stale():
    import tempfile
    import gcn.data.service as svc
    old_dir, old_fresh, old_open, old_yahoo = (
        svc.DATA_DIR, svc._cache_is_fresh, svc._opend_reachable, svc._fetch_yahoo)
    try:
        svc.DATA_DIR = Path(tempfile.mkdtemp())
        frame = pd.DataFrame({"open": [1], "high": [1], "low": [1],
                              "close": [1], "volume": [1]},
                             index=pd.to_datetime(["2020-01-01"]))
        svc._save_cache("AAPL", "1d", frame)
        svc._cache_is_fresh = lambda *a, **k: False
        svc._opend_reachable = lambda *a, **k: False
        svc._fetch_yahoo = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        result = svc.fetch_quote("AAPL", "1d", count=100)
        assert result["source"] == "cache"
        assert result["stale"] is True and result["refresh_failed"] is True
    finally:
        svc.DATA_DIR, svc._cache_is_fresh, svc._opend_reachable, svc._fetch_yahoo = (
            old_dir, old_fresh, old_open, old_yahoo)


def test_fetch_quote_rejects_unknown_interval():
    import gcn.data.service as svc
    try:
        svc.fetch_quote("AAPL", "2h", count=100)
    except ValueError as exc:
        assert "周期" in str(exc)
    else:
        raise AssertionError("未知周期不应静默降级成日线")


def test_csv_and_rows_reject_non_finite_ohlcv():
    import gcn.data.service as svc
    csv = ("date,open,high,low,close,volume\n"
           "2026-01-01,1,2,0.5,1.5,100\n"
           "2026-01-02,1,2,0.5,inf,100\n")
    assert len(svc.parse_csv_text(csv)) == 1
    rows = [["2026-01-01", 1, 2, 0.5, 1.5, 100],
            ["2026-01-02", 1, 2, 0.5, float("-inf"), 100]]
    assert len(svc.df_from_rows(rows)) == 1


def test_source_ohlcv_sanitizer_drops_nonfinite_and_invalid_bars():
    import gcn.data.service as svc
    frame = pd.DataFrame({
        "open": [1, 1, 2, 1], "high": [2, 2, 1, 2],
        "low": [0.5, 0.5, 0.5, 0.5], "close": [1.5, float("nan"), 1.5, 1.5],
        "volume": [10, 10, 10, -1],
    }, index=pd.date_range("2026-01-01", periods=4))
    frame.attrs.update(source="futu", adjustment="adjusted")
    out = svc._sanitize_ohlcv(frame)
    assert len(out) == 1 and out.iloc[0]["close"] == 1.5
    assert out.attrs == frame.attrs
