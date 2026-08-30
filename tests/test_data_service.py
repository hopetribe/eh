# -*- coding: utf-8 -*-
"""数据服务测试: 代码映射 / CSV 解析 / 缓存落盘。"""
from pathlib import Path

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
    assert to_yahoo_symbol("US.TQQQ") == "TQQQ"
    assert to_yahoo_symbol("00700") == "0700.HK"
    assert to_yahoo_symbol("600519") == "600519.SS"
    assert to_yahoo_symbol("000001") == "000001.SZ"


def test_parse_csv_cn_and_en(tmp_path=None):
    csv_cn = "时间,开盘价,最高价,最低价,收盘价,成交量\n2026-01-02,10,10.5,9.9,10.4,100\n"
    d1 = parse_csv_text(csv_cn)
    assert len(d1) == 1 and abs(d1["close"].iloc[0] - 10.4) < 1e-9
    csv_en = "date,open,high,low,close,volume\n2026-01-02,10,10.5,9.9,10.4,100\n"
    assert len(parse_csv_text(csv_en)) == 1


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
    svc._save_cache("TEST.SYM", "1d", df)
    back = svc._load_cache("TEST.SYM", "1d")
    assert back is not None and abs(back["close"].iloc[0] - 10.5) < 1e-9
    svc.DATA_DIR = old
