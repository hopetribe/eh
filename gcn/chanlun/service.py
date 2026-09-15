# -*- coding: utf-8 -*-
"""Adapt the pinned chanlun.py core to KK2 OHLCV data frames.

The upstream project is intentionally kept as a pinned submodule so this
adapter can use its 笔、线段和中枢 implementation without embedding its web
application or TradingView assets in KK2.  See ``NOTICE.md`` in this package.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


UPSTREAM_REVISION = "2e4fa135b19eaa201fca7bfcc8ca4a86cbde7815"
_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_FILE = _ROOT / "vendor" / "chanlun.py" / "chan.py"
_INTERVAL_SECONDS = {
    "1d": 86_400,
    "1wk": 604_800,
    "60m": 3_600,
    "15m": 900,
    "5m": 300,
}


@lru_cache(maxsize=1)
def _upstream() -> ModuleType:
    """Load the checked-out upstream module once per process."""
    if not _UPSTREAM_FILE.is_file():
        raise RuntimeError("chanlun.py 子模块缺失，请执行 git submodule update --init --recursive")
    spec = importlib.util.spec_from_file_location("kk2_chanlun_upstream", _UPSTREAM_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 chanlun.py 核心模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _time_key(value) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return int(timestamp.value)


def _finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是有限数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 必须是有限数字")
    return number


def _normal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"K线数据缺少列: {missing}")
    if frame.index.has_duplicates:
        raise ValueError("缠论分析不接受重复时间戳")
    result = frame.sort_index().copy()
    if result.empty:
        return result
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.loc[:, required].isna().any().any():
        raise ValueError("K线数据包含无效数值")
    if not np.isfinite(result.loc[:, required].to_numpy(dtype=float)).all():
        raise ValueError("K线数据包含非有限数")
    return result


def _line_payload(items, time_indices: dict[int, int], kind: str) -> list[dict]:
    payload = []
    for item in items:
        start = item.文.中
        end = item.武.中
        start_index = time_indices.get(_time_key(start.时间戳))
        end_index = time_indices.get(_time_key(end.时间戳))
        if start_index is None or end_index is None or start_index >= end_index:
            continue
        payload.append({
            "kind": kind,
            "start_index": start_index,
            "start_price": round(_finite(start.分型特征值, "线起点"), 6),
            "end_index": end_index,
            "end_price": round(_finite(end.分型特征值, "线终点"), 6),
            "direction": str(item.方向),
        })
    return payload


def _center_payload(items, time_indices: dict[int, int], kind: str) -> list[dict]:
    payload = []
    for item in items:
        start_index = time_indices.get(_time_key(item.文.中.时间戳))
        end_index = time_indices.get(_time_key(item.武.中.时间戳))
        if start_index is None or end_index is None or start_index >= end_index:
            continue
        high = _finite(item.高, "中枢高点")
        low = _finite(item.低, "中枢低点")
        if low > high:
            raise ValueError("中枢高低值无效")
        payload.append({"kind": kind, "start_index": start_index, "end_index": end_index,
                        "high": round(high, 6), "low": round(low, 6)})
    return payload


def _buy_sell_payload(strokes: list[dict]) -> list[dict]:
    """Mark the endpoint of each completed stroke as a structural turn.

    The pinned upstream revision exposes buy/sell point data types but does not
    enable a concrete first/second/third-point rule.  A completed downward
    stroke therefore yields an explicitly labelled ``缠买`` endpoint and a
    completed upward stroke a ``缠卖`` endpoint.  These are confirmed structure
    observations, not executable trade signals.
    """
    points = []
    for stroke in strokes:
        is_buy = stroke["end_price"] < stroke["start_price"]
        points.append({
            "index": stroke["end_index"],
            "price": stroke["end_price"],
            "side": "buy" if is_buy else "sell",
            "label": "缠买" if is_buy else "缠卖",
            "rule": "completed_stroke_endpoint",
        })
    return points


def analyze_frame(frame: pd.DataFrame, symbol: str, interval: str) -> dict:
    """Return chanlun.py structure overlays for a validated KK2 OHLCV frame.

    This is an analysis-only endpoint. Its ``缠买`` / ``缠卖`` labels are
    completed-stroke endpoints, not the upstream project's disabled first,
    second, or third buy/sell rules and not a trading recommendation.
    """
    interval = str(interval).strip().lower()
    if interval not in _INTERVAL_SECONDS:
        raise ValueError(f"未知K线周期: {interval}")
    clean = _normal_frame(frame)
    time_indices = {_time_key(value): index for index, value in enumerate(clean.index)}
    result = {
        "provider": "chanlun.py",
        "revision": UPSTREAM_REVISION,
        "symbol": str(symbol).strip().upper(),
        "interval": interval,
        "summary": {"bars": len(clean), "chan_bars": 0, "strokes": 0,
                    "segments": 0, "centers": 0, "chan_buy": 0,
                    "chan_sell": 0},
        "strokes": [],
        "segments": [],
        "centers": [],
        "buy_sell_points": [],
        "note": "缠买/缠卖为已完成笔端点的结构标记，不是一二三类买卖点，也不构成交易建议。",
    }
    if len(clean) < 3:
        return result

    upstream = _upstream()
    config = upstream.缠论配置(
        图表展示=False,
        线段内部中枢图显=False,
        分析扩展线段=False,
    )
    observer = upstream.观察者(str(symbol).strip().upper(), _INTERVAL_SECONDS[interval], config)
    for timestamp, row in clean.iterrows():
        observer.投喂原始数据(
            pd.Timestamp(timestamp).to_pydatetime(),
            _finite(row.open, "open"), _finite(row.high, "high"),
            _finite(row.low, "low"), _finite(row.close, "close"),
            _finite(row.volume, "volume"),
        )

    strokes = _line_payload(observer.笔序列, time_indices, "stroke")
    segments = _line_payload(observer.线段序列, time_indices, "segment")
    centers = (_center_payload(observer.笔_中枢序列, time_indices, "stroke")
               + _center_payload(observer.中枢序列, time_indices, "segment"))
    buy_sell_points = _buy_sell_payload(strokes)
    result.update(strokes=strokes, segments=segments, centers=centers,
                  buy_sell_points=buy_sell_points)
    result["summary"].update(chan_bars=len(observer.缠论K线序列), strokes=len(strokes),
                             segments=len(segments), centers=len(centers),
                             chan_buy=sum(point["side"] == "buy" for point in buy_sell_points),
                             chan_sell=sum(point["side"] == "sell" for point in buy_sell_points))
    return result
