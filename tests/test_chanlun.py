from __future__ import annotations

import numpy as np
import pandas as pd

from gcn.chanlun.service import analyze_frame


def _wave_frame(rows: int = 240) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    phase = np.arange(rows, dtype=float)
    close = 100 + phase * 0.03 + np.sin(phase / 4.5) * 9
    opening = close + np.cos(phase / 3.2)
    high = np.maximum(opening, close) + 1.2
    low = np.minimum(opening, close) - 1.2
    return pd.DataFrame({"open": opening, "high": high, "low": low,
                         "close": close, "volume": 1_000 + phase}, index=dates)


def test_chanlun_analysis_serializes_structures_on_kk2_ohlcv():
    result = analyze_frame(_wave_frame(), symbol="BTC-USD", interval="1d")

    assert result["provider"] == "chanlun.py"
    assert result["revision"] == "2e4fa135b19eaa201fca7bfcc8ca4a86cbde7815"
    assert result["summary"]["bars"] == 240
    assert result["summary"]["strokes"] >= 6
    assert result["summary"]["segments"] >= 1
    assert result["buy_sell_points"] == []
    assert all(0 <= item["start_index"] < item["end_index"] < 240
               for item in result["strokes"])
    assert all(item["low"] <= item["high"] for item in result["centers"])
