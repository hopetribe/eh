# -*- coding: utf-8 -*-
"""机会雷达离线测试: 股票池 / 信号提取 / 扫描流程 / 缓存调度 (无网络)。

兼容 tests/run_all.py 的裸函数调用约定, 不依赖 pytest fixture。
"""
import tempfile
import time
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gcn.radar import engine
from gcn.radar.universe import RADAR_MARKETS, _static_universe


@contextmanager
def _patched(target, name, value):
    old = getattr(target, name)
    setattr(target, name, value)
    try:
        yield value
    finally:
        setattr(target, name, old)


# ---------------- 股票池 ----------------

def test_universe_lists_wellformed():
    for m, _ in RADAR_MARKETS:
        lst = _static_universe(m)
        assert len(lst) == 100, f"{m} 静态池应100只, 实际 {len(lst)}"
        codes = [c for c, _ in lst]
        assert len(set(codes)) == 100, f"{m} 代码重复"
        for c, n in lst:
            assert n, f"{m} {c} 缺名称"
            if m == "cn":
                assert len(c) == 6 and c.isdigit(), c
            elif m == "hk":
                assert len(c) == 5 and c.isdigit(), c
            else:
                assert c.replace("-", "").isalnum(), c


def test_threshold_cache_schema_tracks_inclusive_boundary():
    import gcn.radar.universe as universe
    assert universe.UNIVERSE_CACHE_SCHEMA == 3


def test_futu_universe_contract_and_cross_market_deduplication():
    import gcn.radar.universe as universe
    threshold = universe.MARKET_CAP_THRESHOLDS["cn"]
    seen_filters = []
    class SimpleFilter: pass
    class Context:
        def __init__(self, **kwargs): pass
        def get_stock_filter(self, market, **kwargs):
            seen_filters.append(kwargs["filter_list"][0])
            if market == "SH":
                rows = [SimpleNamespace(stock_code="SH.600519", stock_name="茅台", market_val=threshold + 100),
                        SimpleNamespace(stock_code="SZ.000001", stock_name="平安", market_val=threshold + 80),
                        SimpleNamespace(stock_code="SH.000000", stock_name="边界", market_val=threshold)]
            else:
                rows = [SimpleNamespace(stock_code="SH.600519", stock_name="茅台", market_val=threshold + 110),
                        SimpleNamespace(stock_code="SZ.000333", stock_name="美的", market_val=threshold + 90)]
            return 0, (True, len(rows), rows)
        def close(self): pass
    fake_futu = SimpleNamespace(OpenQuoteContext=Context, SimpleFilter=SimpleFilter,
                                SortDir=SimpleNamespace(DESCEND="desc"))
    fake_constant = SimpleNamespace(StockField=SimpleNamespace(MARKET_VAL="market_val"))
    old_futu, old_const = sys.modules.get("futu"), sys.modules.get("futu.common.constant")
    try:
        sys.modules["futu"] = fake_futu
        sys.modules["futu.common.constant"] = fake_constant
        out = universe._fetch_futu_top("cn", n=4)
    finally:
        if old_futu is None: sys.modules.pop("futu", None)
        else: sys.modules["futu"] = old_futu
        if old_const is None: sys.modules.pop("futu.common.constant", None)
        else: sys.modules["futu.common.constant"] = old_const
    assert out == [("600519", "茅台"), ("000333", "美的"), ("000001", "平安"),
                   ("000000", "边界")]
    assert seen_filters and all(f.filter_min == threshold and not f.is_no_filter
                                for f in seen_filters)


def test_threshold_universe_uses_stale_same_schema_cache_before_static():
    import json
    import gcn.radar.universe as universe
    with tempfile.TemporaryDirectory() as tmp, \
         _patched(universe, "DATA_DIR", Path(tmp)), \
         _patched(universe, "_opend_reachable", lambda: False), \
         _patched(universe, "_fetch_yahoo_threshold", lambda market: None):
        blob = {"schema": universe.UNIVERSE_CACHE_SCHEMA, "day": "2020-01-01",
                "threshold": universe.MARKET_CAP_THRESHOLDS["us"],
                "list": [["STALE", "旧阈值快照"]]}
        universe._universe_cache_path("us").write_text(
            json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        items, source = universe.get_universe("us")
    assert items == [("STALE", "旧阈值快照")]
    assert source == "dynamic-cache-stale"


def test_threshold_universe_keeps_previous_schema_as_failure_fallback():
    import json
    import gcn.radar.universe as universe
    with tempfile.TemporaryDirectory() as tmp, \
         _patched(universe, "DATA_DIR", Path(tmp)), \
         _patched(universe, "_opend_reachable", lambda: False), \
         _patched(universe, "_fetch_yahoo_threshold", lambda market: None):
        blob = {"schema": universe.UNIVERSE_CACHE_SCHEMA - 1,
                "day": time.strftime("%Y-%m-%d"),
                "threshold": universe.MARKET_CAP_THRESHOLDS["cn"],
                "list": [["600519", "旧全量快照"]]}
        universe._universe_cache_path("cn").write_text(
            json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        items, source = universe.get_universe("cn")
    assert items == [("600519", "旧全量快照")]
    assert source == "dynamic-cache-stale"


def test_yahoo_threshold_universe_paginates_and_normalizes_codes():
    import gcn.radar.universe as universe
    threshold = universe.MARKET_CAP_THRESHOLDS["hk"]
    offsets = []

    class EquityQuery:
        def __init__(self, op, args):
            self.op, self.args = op, args

    def screen(query, offset, size, **kwargs):
        offsets.append(offset)
        pages = {
            0: [
                {"symbol": "0700.HK", "shortName": "腾讯", "marketCap": threshold + 20,
                 "quoteType": "EQUITY"},
                {"symbol": "0005.HK", "shortName": "汇丰", "marketCap": threshold,
                 "quoteType": "EQUITY"},
            ],
            2: [{"symbol": "9988.HK", "longName": "阿里", "marketCap": threshold + 10,
                 "quoteType": "EQUITY"}],
        }
        return {"quotes": pages.get(offset, []), "total": 3}

    fake = SimpleNamespace(EquityQuery=EquityQuery, screen=screen)
    old = sys.modules.get("yfinance")
    try:
        sys.modules["yfinance"] = fake
        with _patched(universe, "YAHOO_PAGE_SIZE", 2):
            out = universe._fetch_yahoo_threshold("hk")
    finally:
        if old is None: sys.modules.pop("yfinance", None)
        else: sys.modules["yfinance"] = old
    assert offsets == [0, 2]
    assert out == [("00700", "腾讯"), ("09988", "阿里"), ("00005", "汇丰")]


def test_yahoo_threshold_discards_partial_pagination():
    import gcn.radar.universe as universe
    threshold = universe.MARKET_CAP_THRESHOLDS["us"]

    class EquityQuery:
        def __init__(self, op, args): pass

    def screen(query, offset, size, **kwargs):
        quotes = ([{"symbol": "AAPL", "shortName": "苹果",
                    "marketCap": threshold + 1, "quoteType": "EQUITY"},
                   {"symbol": "MSFT", "shortName": "微软",
                    "marketCap": threshold + 2, "quoteType": "EQUITY"}]
                  if offset == 0 else [])
        return {"quotes": quotes, "total": 4}

    fake = SimpleNamespace(EquityQuery=EquityQuery, screen=screen)
    old = sys.modules.get("yfinance")
    try:
        sys.modules["yfinance"] = fake
        with _patched(universe, "YAHOO_PAGE_SIZE", 2):
            out = universe._fetch_yahoo_threshold("us")
    finally:
        if old is None: sys.modules.pop("yfinance", None)
        else: sys.modules["yfinance"] = old
    assert out is None


def test_get_universe_uses_yahoo_when_opend_is_offline():
    import gcn.radar.universe as universe
    with tempfile.TemporaryDirectory() as tmp, \
         _patched(universe, "DATA_DIR", Path(tmp)), \
         _patched(universe, "_opend_reachable", lambda: False), \
         _patched(universe, "_fetch_yahoo_threshold", lambda market: [("YHOO", "动态池")]):
        items, source = universe.get_universe("us", use_cache=False)
        cached = universe._read_threshold_cache("us")
    assert items == [("YHOO", "动态池")] and source == "yahoo"
    assert cached["provider"] == "yahoo"


# ---------------- 信号提取 ----------------

def _make_res(n=30, b_at=(), jf_at=()):
    """构造带 B_SIGNAL/ICON_JUEFAN 布尔列的指标结果 (索引为交易日)。"""
    idx = pd.bdate_range("2025-08-01", periods=n)
    res = pd.DataFrame({"CLOSE": [10.0 + i * 0.1 for i in range(n)]}, index=idx)
    for col, marks in (("B_SIGNAL", b_at), ("ICON_JUEFAN", jf_at)):
        res[col] = False
        for days_ago in marks:
            res.loc[res.index[n - 1 - days_ago], col] = True
    return res


def test_extract_recent_windows():
    res = _make_res(30, b_at=(2, 20), jf_at=(7,))
    sigs = engine._extract_recent(res, max_days=15, as_of=res.index[-1])
    # days_ago=20 的 B买 超出窗口, 不应出现; 按新 -> 旧 (days_ago 升序)
    assert [(s["type"], s["days_ago"]) for s in sigs] == [("B买", 2), ("绝反", 7)]
    assert sigs[1]["date"] == str(res.index[22])[:10]
    assert sigs[1]["close"] == 12.2
    # 近一周 (5 根K线) 只剩 B买
    week = engine._extract_recent(res, max_days=5, as_of=res.index[-1])
    assert [(s["type"], s["days_ago"]) for s in week] == [("B买", 2)]


def test_extract_recent_empty():
    res = _make_res(30)
    assert engine._extract_recent(res, max_days=15) == []
    assert engine._extract_recent(res.iloc[:0]) == []


def test_extract_recent_age_is_relative_to_today_not_last_cached_bar():
    res = _make_res(5, b_at=(0,))
    sig = engine._extract_recent(res, as_of=pd.Timestamp("2026-08-31"))[0]
    assert sig["days_ago"] > 200


# ---------------- 单标的扫描 ----------------

def _fake_fetch(rows):
    return lambda code, interval, count: {"rows": rows[-count:], "source": "cache",
                                          "symbol": code, "interval": interval,
                                          "note": ""}


def _fake_compute(df, **kw):
    """绕开真实指标: 输出引擎所需的最小列集合。"""
    out = pd.DataFrame(index=df.index)
    out["CLOSE"] = df["close"]
    out["B_SIGNAL"] = False
    out["ICON_JUEFAN"] = False
    out.loc[out.index[-3], "ICON_JUEFAN"] = True  # 3根K线前绝反
    return out


def _synthetic_rows(n=30):
    today = pd.Timestamp.now().normalize().date()
    last_business_day = pd.Timestamp(np.busday_offset(today, 0, roll="backward"))
    idx = pd.bdate_range(end=last_business_day, periods=n)
    return [[t.strftime("%Y-%m-%d"), 10, 11, 9, 10 + i * 0.1, 1000]
            for i, t in enumerate(idx)]


def test_scan_symbol_hit():
    with _patched(engine, "fetch_quote", _fake_fetch(_synthetic_rows())), \
         _patched(engine, "compute_ehopt10", _fake_compute):
        r = engine.scan_symbol("00700", "hk", "腾讯控股")
    assert r["error"] is None
    assert r["code"] == "00700" and r["name"] == "腾讯控股" and r["market"] == "hk"
    assert [s["type"] for s in r["signals"]] == ["绝反"]
    signal_date = pd.Timestamp(r["signals"][0]["date"]).date()
    today = pd.Timestamp.now().normalize().date()
    assert r["signals"][0]["days_ago"] == max(
        2, int(np.busday_count(signal_date, today)),
    )
    assert r["close"] == 12.9 and r["chg_pct"] == 0.78  # 12.9/12.8-1


def test_scan_symbol_error_isolated():
    def boom(code, interval, count):
        raise RuntimeError("网络不可用")

    with _patched(engine, "fetch_quote", boom):
        r = engine.scan_symbol("AAPL", "us", "苹果")
    assert r["signals"] == [] and "RuntimeError" in r["error"]


# ---------------- 市场扫描与调度 ----------------

def test_scan_market_keeps_hits_sorted():
    def fake_scan(code, market, name="", count=300):
        if code == "A1":
            return {"code": code, "market": market, "name": name, "date": "d",
                    "close": 1, "chg_pct": 0, "error": None,
                    "signals": [{"type": "B买", "date": "d", "days_ago": 4, "close": 1}]}
        if code == "A2":
            return {"code": code, "market": market, "name": name, "date": "d",
                    "close": 1, "chg_pct": 0, "error": None,
                    "signals": [{"type": "绝反", "date": "d", "days_ago": 1, "close": 1}]}
        return {"code": code, "market": market, "name": name, "date": None,
                "close": None, "chg_pct": None, "signals": [], "error": "无数据"}

    with _patched(engine, "get_universe",
                  lambda m, n=100, use_cache=True:
                  ([("A1", "甲"), ("A2", "乙"), ("A3", "丙")], "static")), \
         _patched(engine, "scan_symbol", fake_scan):
        block = engine.scan_market("us")
    assert block["n_scanned"] == 3 and block["n_errors"] == 1
    assert block["universe_source"] == "static"
    assert [r["code"] for r in block["results"]] == ["A2", "A1"]  # 按最近信号新->旧
    assert block["n_hits"] == 2


def _wait_job(svc, market, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (svc.jobs.get(market) or {}).get("status") in ("done", "error"):
            return svc.jobs[market]
        time.sleep(0.05)
    raise AssertionError("后台扫描超时未结束")


def test_service_scan_cache_and_fresh():
    calls = {"n": 0}

    def fake_scan_market(market, progress=None, max_workers=6):
        calls["n"] += 1
        return {"market": market, "universe_source": "static", "n_scanned": 100,
                "universe_schema": engine.UNIVERSE_CACHE_SCHEMA,
                "n_errors": 0, "n_hits": 1,
                "results": [{"code": "X1", "market": market, "name": "x",
                             "date": "d", "close": 1, "chg_pct": 0,
                             "signals": [{"type": "B买", "date": "d",
                                          "days_ago": 0, "close": 1}]}],
                "generated_at": time.time()}

    with tempfile.TemporaryDirectory() as tmp, \
         _patched(engine, "DATA_DIR", Path(tmp)), \
         _patched(engine, "scan_market", fake_scan_market):
        svc = engine.RadarService()

        snap = svc.ensure_fresh(["us"])  # 无缓存 -> 自动开扫
        assert snap["markets"]["us"]["scanning"]
        job = _wait_job(svc, "us")
        assert job["status"] == "done" and calls["n"] == 1

        snap = svc.ensure_fresh(["us"])  # 缓存新鲜 -> 不再重扫
        assert not snap["markets"]["us"]["scanning"] and calls["n"] == 1
        assert snap["markets"]["us"]["cache"]["n_hits"] == 1

        started = svc.start_scan(["us", "xx"])  # 强制重扫; 非法市场被忽略
        assert started == ["us"]
        _wait_job(svc, "us")
        assert calls["n"] == 2

        # 结果确实落盘, 可被 load_cache 读回
        blob = engine.load_cache("us")
        assert blob["results"][0]["code"] == "X1"


def test_service_snapshot_error_kept():
    def fail_scan(market, progress=None, max_workers=6):
        raise RuntimeError("断网")

    with tempfile.TemporaryDirectory() as tmp, \
         _patched(engine, "DATA_DIR", Path(tmp)), \
         _patched(engine, "scan_market", fail_scan):
        svc = engine.RadarService()
        svc.start_scan(["hk"])
        job = _wait_job(svc, "hk")
        assert job["status"] == "error" and "断网" in (job.get("error") or "")
        snap = svc.snapshot(["hk"])
        assert snap["markets"]["hk"]["job"]["status"] == "error"


def test_service_marks_previous_universe_schema_cache_stale():
    old_block = {"market": "us", "n_scanned": 100, "n_errors": 0,
                 "n_hits": 0, "results": [], "generated_at": time.time()}
    with tempfile.TemporaryDirectory() as tmp, \
         _patched(engine, "DATA_DIR", Path(tmp)):
        engine.save_cache("us", old_block)
        view = engine.RadarService()._block_view("us")
    assert view["stale"] is True


def test_service_all_symbol_failures_preserve_previous_cache():
    def all_failed(market, progress=None, max_workers=6):
        return {"market": market, "universe_source": "static", "n_scanned": 3,
                "n_errors": 3, "n_hits": 0, "results": [],
                "generated_at": time.time()}
    old_block = {"market": "us", "n_scanned": 3, "n_errors": 0, "n_hits": 1,
                 "results": [{"code": "OLD"}], "generated_at": time.time() - 999}
    with tempfile.TemporaryDirectory() as tmp, \
         _patched(engine, "DATA_DIR", Path(tmp)), \
         _patched(engine, "scan_market", all_failed):
        engine.save_cache("us", old_block)
        svc = engine.RadarService()
        svc.start_scan(["us"])
        job = _wait_job(svc, "us")
        assert job["status"] == "error"
        assert engine.load_cache("us")["results"][0]["code"] == "OLD"


def test_service_partial_failures_keep_failed_symbols_previous_hits():
    def partial(market, progress=None, max_workers=6):
        return {"market": market, "universe_source": "static", "n_scanned": 2,
                "n_errors": 1, "n_hits": 1,
                "results": [{"code": "A", "signals": [{"days_ago": 0}]}],
                "failed_codes": ["B"], "generated_at": time.time()}
    old = {"market": "us", "n_scanned": 2, "n_errors": 0, "n_hits": 2,
           "results": [{"code": "A", "signals": [{"days_ago": 2}]},
                       {"code": "B", "signals": [{"days_ago": 3}]}],
           "generated_at": time.time() - 100}
    with tempfile.TemporaryDirectory() as tmp, \
         _patched(engine, "DATA_DIR", Path(tmp)), \
         _patched(engine, "scan_market", partial):
        engine.save_cache("us", old)
        svc = engine.RadarService()
        svc.start_scan(["us"])
        assert _wait_job(svc, "us")["status"] == "done"
        cached = engine.load_cache("us")
        assert [r["code"] for r in cached["results"]] == ["A", "B"]
        assert cached["results"][1]["stale"] is True


# ---------------- 日K预热守护 ----------------

def test_warm_market_only_refreshes_stale():
    """预热只请求陈旧/缺失标的; 新鲜的直接跳过。"""
    engine._warm_fails.clear()
    universe = [("FRESH", "甲"), ("STALE", "乙"), ("MISSING", "丙"), ("BAD", "丁")]
    fetched = []

    def fake_fetch(code, interval, count, force=False):
        fetched.append(code)
        if code == "BAD":
            raise RuntimeError("限流")

    with _patched(engine, "get_universe",
                  lambda m, n=100, use_cache=True: (universe, "static")), \
         _patched(engine, "_kline_cache_fresh", lambda c: c == "FRESH"), \
         _patched(engine, "fetch_quote", fake_fetch):
        r = engine.warm_market("us")
    assert r["n_total"] == 4 and r["n_targets"] == 3
    assert sorted(fetched) == ["BAD", "MISSING", "STALE"]
    assert r["n_ok"] == 2 and r["n_failed"] == 1 and r["failed"] == ["BAD"]


def test_warm_failure_backoff_and_force():
    """失败标的退避期内跳过, 退避到期/force 时重试。"""
    engine._warm_fails.clear()


def test_warm_stale_cache_fallback_counts_as_refresh_failure():
    engine._warm_fails.clear()


def test_kline_freshness_uses_symbol_market_calendar():
    seen = {}
    frame = pd.DataFrame({"close": [1]}, index=pd.to_datetime(["2026-01-01"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "600519_1d.csv"
        path.touch()
        with _patched(engine, "_cache_path", lambda *a: path), \
         _patched(engine, "_load_cache", lambda *a: frame), \
         _patched(engine, "_cache_is_fresh",
                  lambda *a, **k: seen.setdefault("symbol", k.get("symbol")) == "600519"):
            assert engine._kline_cache_fresh("600519")
    assert seen["symbol"] == "600519"
    universe = [("STALE", "旧缓存")]
    calls = {"n": 0}
    def stale_result(*args, **kwargs):
        calls["n"] += 1
        return {"source": "cache", "stale": True, "refresh_failed": True, "rows": [[1]]}
    with _patched(engine, "get_universe", lambda *a, **k: (universe, "static")), \
         _patched(engine, "_kline_cache_fresh", lambda c: False), \
         _patched(engine, "fetch_quote", stale_result):
        first = engine.warm_market("us")
        second = engine.warm_market("us")
    assert first["n_failed"] == 1 and second["n_targets"] == 0
    assert calls["n"] == 1
    engine._warm_fails.clear()
    universe = [("BAD", "丁")]
    calls = {"n": 0}

    def fake_fetch(code, interval, count, force=False):
        calls["n"] += 1
        raise RuntimeError("限流")

    with _patched(engine, "get_universe",
                  lambda m, n=100, use_cache=True: (universe, "static")), \
         _patched(engine, "_kline_cache_fresh", lambda c: False), \
         _patched(engine, "fetch_quote", fake_fetch):
        engine.warm_market("us")
        assert calls["n"] == 1
        engine.warm_market("us")            # 退避期内 -> 不再请求
        assert calls["n"] == 1
        engine._warm_fails.clear()          # 模拟退避到期
        engine.warm_market("us")
        assert calls["n"] == 2
        engine.warm_market("us", force=True)  # force 忽略退避
        assert calls["n"] == 3
    engine._warm_fails.clear()
