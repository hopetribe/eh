"""Radar configuration, chart parity and TOP-N regressions (offline)."""
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gcn.data.sample import make_sample_data
from gcn.data.service import _rows_from_df, DEFAULT_COUNT
from gcn.radar import engine, universe
from gcn.radar.config import normalize_config, signal_options
from gcn.server.app import build_payload
from gcn.recipes.gcn_main import compute_ehopt10


def test_radar_matches_main_chart_for_every_version_and_selected_signal():
    frame = make_sample_data(n=DEFAULT_COUNT)
    rows = _rows_from_df(frame)
    for version in ("v3", "v4", "v4-exp", "v5"):
        config = normalize_config({"version": version,
                                   "signals": [s["id"] for s in signal_options(version)]})
        expected = build_payload(frame, {}, version=version)
        with patch.object(engine, "fetch_quote", return_value={"rows": rows}), \
             patch.object(engine, "compute_ehopt10", wraps=compute_ehopt10) as compute:
            actual = engine.scan_symbol("TEST", "us", config=config)
        assert actual["error"] is None
        assert len(compute.call_args.args[0]) == DEFAULT_COUNT
        assert compute.call_args.kwargs["version"] == version
        pairs = set()
        for option in signal_options(version):
            for idx in expected[option["id"]]:
                if idx < len(frame) - engine.SIGNAL_HISTORY:
                    continue
                if option["id"] == "stageEntry" and idx in expected["bSignal"]:
                    continue
                pairs.add((option["label"], str(frame.index[idx])[:10]))
        assert {(s["type"], s["date"]) for s in actual["signals"]} == pairs


def test_signal_validation_and_nan_does_not_create_signal():
    for config in ({"version": "unknown"}, {"signals": []}, {"signals": "bSignal"},
                   {"version": "v4", "signals": ["stageSetup"]}):
        try:
            normalize_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError(config)
    result = pd.DataFrame({"CLOSE": [10, 11], "S_SIGNAL": [float("nan"), True],
                           "B_SIGNAL": [True, True]}, index=pd.bdate_range("2026-09-01", periods=2))
    signals = engine._extract_recent(result, config={"signals": ["sSignal"]})
    assert len(signals) == 1 and signals[0]["type"] == "S卖"


def test_persisted_config_controls_jobs_and_hides_incompatible_cache():
    selected = normalize_config({"version": "v3", "signals": ["sSignal"]})
    with tempfile.TemporaryDirectory() as tmp, patch.object(engine, "DATA_DIR", Path(tmp)), \
         patch.object(engine.threading, "Thread") as thread:
        svc = engine.RadarService()
        assert svc.start_scan(["us"], selected) == ["us"]
        assert thread.call_args.kwargs["args"] == ("us", selected)
        restarted = engine.RadarService()
        assert restarted.get_config() == selected
        restarted.start_scan(["hk"])
        assert thread.call_args.kwargs["args"] == ("hk", selected)
        engine.save_cache("us", {"config": selected, "universe_schema": universe.UNIVERSE_CACHE_SCHEMA,
                                 "generated_at": time.time(), "results": []})
        assert restarted.snapshot(["us"])["markets"]["us"]["cache"] is not None
        assert restarted.snapshot(["us"], {"version": "v5"})["markets"]["us"]["cache"] is None
        try:
            svc.start_scan(["cn"], {"version": "v5"})
        except ValueError as exc:
            assert "仍在运行" in str(exc)
        else:
            raise AssertionError("必须拒绝运行中的配置切换")
        assert svc.get_config() == selected


def test_partial_failure_never_merges_old_version_signals():
    selected = normalize_config({"version": "v3", "signals": ["sSignal"]})
    with tempfile.TemporaryDirectory() as tmp, patch.object(engine, "DATA_DIR", Path(tmp)):
        engine.save_cache("us", {"config": normalize_config(), "universe_schema": universe.UNIVERSE_CACHE_SCHEMA,
                                 "results": [{"code": "OLD", "signals": [{"type": "B买"}]}]})
        block = {"n_scanned": 2, "n_errors": 1, "failed_codes": ["OLD"],
                 "results": [], "generated_at": time.time()}
        with patch.object(engine, "scan_market", return_value=block):
            svc = engine.RadarService()
            svc._run("us", selected)
        assert engine.load_cache("us")["results"] == []


def test_yahoo_top_ranking_stops_after_enough_unique_equities():
    import yfinance as yf
    calls = []
    def screen(query, offset, size, **kwargs):
        calls.append(offset)
        assert kwargs == {"sortField": "intradaymarketcap", "sortAsc": False}
        return {"total": 10000, "quotes": [
            {"symbol": f"S{i}", "marketCap": 10000 - i, "quoteType": "EQUITY"}
            for i in range(offset, offset + size)]}
    with patch.object(yf, "screen", side_effect=screen):
        for market, limit in (("us", 300),):
            result = universe._fetch_yahoo_top(market)
            assert len(result) == limit and result[0][0] == "S0" and result[-1][0] == "S299"
    assert calls == [0, 250]


def test_futu_top_merges_both_exchanges_before_truncating():
    import sys
    from types import SimpleNamespace
    visited = []
    class Context:
        def __init__(self, **kwargs): pass
        def close(self): pass
        def get_stock_filter(self, market, begin, num, **kwargs):
            visited.append((market, begin))
            rows = [SimpleNamespace(stock_code=f"{market}.{i:06d}", stock_name=str(i),
                                    market_val=10000 - i)
                    for i in range(begin + (0 if market == "SZ" else 1000),
                                   begin + (0 if market == "SZ" else 1000) + num)]
            return 0, (False, 5000, rows)
    fake = SimpleNamespace(OpenQuoteContext=Context, SimpleFilter=SimpleNamespace,
                           SortDir=SimpleNamespace(DESCEND="desc"))
    const = SimpleNamespace(StockField=SimpleNamespace(MARKET_VAL="market_val"))
    with patch.dict(sys.modules, {"futu": fake, "futu.common.constant": const}):
        result = universe._fetch_futu_top("cn")
    assert len(result) == 300 and result[0][0] == "000000" and result[-1][0] == "000299"
    assert visited == [("SH", 0), ("SH", 200), ("SZ", 0), ("SZ", 200)]
