#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KK2 EHOPT10 指标前端 UI 服务

纯 Python 标准库 HTTP 服务 + 本地内置 ECharts (webui/echarts.min.js),
指标计算复用 kk2_ehopt10.compute_ehopt10 (与富途公式单一逻辑源)。

行情数据: 优先 Futu OpenAPI (需本机运行 FutuOpenD), 未安装/未连接时
自动回退 Yahoo Finance 免费接口。前端默认加载 TQQQ 日K。

用法:
    python3 kk2_ehopt10_ui.py [--port 8642] [--no-browser]

接口:
    GET  /              前端页面 (webui/index.html)
    GET  /echarts.min.js
    POST /api/fetch     {"symbol":"TQQQ","interval":"1d","count":1000}
                        -> {rows:[[date,o,h,l,c,v],...], source, symbol, note}
    POST /api/parse_csv {"csv":"..."} -> {rows:[...]}
    POST /api/compute   {"params":{SD,WIDTH,N,OFFSET}, "source":"sample"|"rows",
                         "seed":7, "rows":[...]} -> 指标全量数据 JSON
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from kk2_backtest import PRESETS, run_backtest
from kk2_ehopt10 import compute_ehopt10, make_sample_data

ROOT = Path(__file__).resolve().parent
WEBUI = ROOT / "webui"

MAX_BODY = 30 * 1024 * 1024  # 30MB, 上限足够大的 CSV

# 富途参数表: 参数名 (默认, 最小, 最大)
PARAM_LIMITS = {
    "SD": (20, 2, 120),
    "WIDTH": (2, 0, 100),
    "N": (4, 0, 1000),
    "OFFSET": (15, 0, 1000),
}

CSV_ALIASES = {
    "open": ["open", "o", "开盘价", "开盘"],
    "high": ["high", "h", "最高价", "最高"],
    "low": ["low", "l", "最低价", "最低"],
    "close": ["close", "c", "收盘价", "收盘"],
    "volume": ["volume", "vol", "v", "成交量"],
    "date": ["date", "time", "time_key", "datetime", "timestamp", "时间", "日期", "时间戳"],
}

# ==========================================================================
# 行情数据源: Futu OpenAPI (本机 FutuOpenD) 优先, 自动回退 Yahoo Finance
# ==========================================================================

# interval: (Futu KLType 名, Yahoo interval, Yahoo range)
INTERVALS = {
    "1d": ("K_DAY", "1d", "5y"),
    "1wk": ("K_WEEK", "1wk", "max"),
    "60m": ("K_60M", "60m", "2y"),
    "15m": ("K_15M", "15m", "60d"),
    "5m": ("K_5M", "5m", "60d"),
}
DEFAULT_COUNT = 1000

_opend_cache = {"ts": 0.0, "ok": False}


def _opend_reachable(host: str = "127.0.0.1", port: int = 11111, timeout: float = 0.5) -> bool:
    """探测 FutuOpenD 是否在运行 (结果缓存 30s, 避免频繁探测)。"""
    now = time.time()
    if now - _opend_cache["ts"] < 30:
        return _opend_cache["ok"]
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ok = True
    except OSError:
        ok = False
    _opend_cache.update(ts=now, ok=ok)
    return ok


def to_futu_symbol(s: str) -> str:
    """TQQQ->US.TQQQ, 00700->HK.00700, 600519->SH.600519, 已带市场前缀则原样。"""
    s = s.strip().upper()
    if "." in s:
        return s
    if s.isdigit():
        if len(s) == 5:
            return "HK." + s
        if len(s) == 6:
            return ("SH." if s[0] in "569" else "SZ.") + s
    return "US." + s


def to_yahoo_symbol(s: str) -> str:
    """US.TQQQ->TQQQ, 00700/HK.00700->0700.HK, 600519->600519.SS, 000001->000001.SZ。"""
    s = s.strip().upper()
    if "." in s:
        mkt, code = s.split(".", 1)
        if mkt == "US":
            return code
        if mkt == "HK" and code.isdigit():
            return f"{int(code):04d}.HK"
        if mkt in ("SH", "SS"):
            return code + ".SS"
        if mkt == "SZ":
            return code + ".SZ"
        return s
    if s.isdigit():
        if len(s) == 5:
            return f"{int(s):04d}.HK"
        if len(s) == 6:
            return s + (".SS" if s[0] in "569" else ".SZ")
    return s


def _rows_from_df(df: pd.DataFrame) -> list:
    """DataFrame -> [[date,open,high,low,close,volume], ...]"""
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        has_time = bool(((idx.hour != 0) | (idx.minute != 0)).any())
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        dates = [t.strftime(fmt) for t in idx]
    else:
        dates = [str(x) for x in idx]
    return [[d, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)]
            for d, r in zip(dates, df.itertuples(index=False))]


def _fetch_futu(symbol: str, interval: str, count: int) -> pd.DataFrame:
    from futu import AuType, KLType, OpenQuoteContext

    kl_name = INTERVALS[interval][0]
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        ret, data, err = ctx.request_history_kline(
            to_futu_symbol(symbol), ktype=getattr(KLType, kl_name),
            autype=AuType.QFQ, max_count=count, page_req_kcount=count)
        if ret != 0:
            raise RuntimeError(err or "FutuOpenD 请求失败")
        df = pd.DataFrame({
            "open": data["open"].astype(float), "high": data["high"].astype(float),
            "low": data["low"].astype(float), "close": data["close"].astype(float),
            "volume": data["volume"].astype(float),
        })
        df.index = pd.to_datetime(data["time_key"])
        return df
    finally:
        ctx.close()


def _fetch_yahoo(symbol: str, interval: str, count: int) -> list:
    _, yiv, yrange = INTERVALS[interval]
    ysym = to_yahoo_symbol(symbol)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ysym)}"
           f"?range={yrange}&interval={yiv}&includePrePost=false")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        err = (payload.get("chart") or {}).get("error") or {}
        raise RuntimeError(err.get("description") or f"Yahoo 无 {ysym} 数据")
    r0 = result[0]
    ts = r0.get("timestamp") or []
    q = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
    gmtoffset = int((r0.get("meta") or {}).get("gmtoffset") or 0)
    fmt = "%Y-%m-%d %H:%M" if interval.endswith("m") else "%Y-%m-%d"

    rows = []
    for i, t in enumerate(ts):
        try:
            o, h, l, c = (q[k][i] for k in ("open", "high", "low", "close"))
            v = q["volume"][i]
        except (KeyError, IndexError, TypeError):
            continue
        if o is None or h is None or l is None or c is None or v is None:
            continue  # 停牌/缺失 bar
        dt = datetime.fromtimestamp(int(t) + gmtoffset, tz=timezone.utc).replace(tzinfo=None)
        rows.append([dt.strftime(fmt), float(o), float(h), float(l), float(c), float(v)])
    if len(rows) > count:
        rows = rows[-count:]
    if not rows:
        raise RuntimeError(f"Yahoo 返回 {ysym} 无有效K线")
    return rows


def fetch_quote(symbol: str, interval: str = "1d", count: int = DEFAULT_COUNT) -> dict:
    """抓取K线: Futu 优先 (OpenD 在运行时), 失败自动回退 Yahoo。"""
    interval = interval if interval in INTERVALS else "1d"
    count = int(min(max(int(count or DEFAULT_COUNT), 100), 5000))
    symbol = (symbol or "TQQQ").strip().upper()
    if not symbol:
        raise ValueError("股票代码不能为空")

    notes = []
    try:
        import futu  # noqa: F401 - 检测 futu-api 是否安装
        futu_ready = _opend_reachable()
    except ImportError:
        futu_ready = False

    if futu_ready:
        try:
            df = _fetch_futu(symbol, interval, count)
            return {"rows": _rows_from_df(df), "source": "futu",
                    "symbol": to_futu_symbol(symbol), "interval": interval, "note": ""}
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Futu({exc})")

    try:
        rows = _fetch_yahoo(symbol, interval, count)
        return {"rows": rows, "source": "yahoo",
                "symbol": to_yahoo_symbol(symbol), "interval": interval,
                "note": "; ".join(notes)}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"获取 {symbol} 行情失败: {'; '.join(notes + [f'Yahoo({exc})'])}") from exc


def df_from_rows(rows) -> pd.DataFrame:
    """前端缓存的 rows -> 计算 DataFrame。"""
    if not rows:
        raise ValueError("K线数据为空")
    recs = []
    for r in rows:
        d, o, h, l, c, v = (list(r) + [None] * 6)[:6]
        if d is None or o is None or h is None or l is None or c is None:
            continue
        recs.append((str(d), float(o), float(h), float(l), float(c), float(v or 0.0)))
    if not recs:
        raise ValueError("K线数据无有效行")
    df = pd.DataFrame([r[1:] for r in recs],
                      columns=["open", "high", "low", "close", "volume"])
    try:
        idx = pd.to_datetime([r[0] for r in recs], format="mixed")
    except (ValueError, TypeError):
        idx = pd.RangeIndex(len(recs))
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
    if df.empty:
        raise ValueError("K线数据无有效行")
    return df


def clamp_params(raw: dict) -> dict:
    """按参数表范围收敛前端传入的参数。"""
    out = {}
    for key, (dflt, lo, hi) in PARAM_LIMITS.items():
        try:
            v = float(raw.get(key, dflt))
        except (TypeError, ValueError):
            v = float(dflt)
        if key in ("SD", "N", "OFFSET"):
            v = int(round(v))
        out[key] = min(max(v, lo), hi)
    return out


def parse_csv_text(text: str) -> pd.DataFrame:
    """解析上传的 CSV (兼容富途中/英文列名, 大小写不敏感)。"""
    df = pd.read_csv(io.StringIO(text.lstrip("\ufeff")))
    cols = {str(c).strip().lower(): c for c in df.columns}

    def find(key):
        for alias in CSV_ALIASES[key]:
            if alias in cols:
                return cols[alias]
        return None

    missing = [k for k in ("open", "high", "low", "close") if find(k) is None]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing} (支持列名: 开盘价/最高价/最低价/收盘价/成交量 或 open/high/low/close/volume)")

    out = pd.DataFrame({k: df[find(k)] for k in ("open", "high", "low", "close")})
    vcol = find("volume")
    out["volume"] = df[vcol] if vcol is not None else 0.0

    dcol = find("date")
    if dcol is not None:
        out.index = pd.to_datetime(df[dcol], errors="coerce")
        out = out[out.index.notna()]
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    if dcol is not None:
        out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("CSV 无有效数据行")
    return out


def fmt_index(idx) -> list:
    if isinstance(idx, pd.DatetimeIndex):
        has_time = bool(((idx.hour != 0) | (idx.minute != 0)).any())
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        return [t.strftime(fmt) for t in idx]
    return [str(x) for x in idx]


def jarr(values, nd: int = 6) -> list:
    """numpy 数组 -> JSON 数组, NaN/inf 记为 null。"""
    out = []
    for v in np.asarray(values, dtype=float):
        out.append(round(float(v), nd) if np.isfinite(v) else None)
    return out


def build_payload(df: pd.DataFrame, params: dict) -> dict:
    res = compute_ehopt10(df, **params)

    def flag_indices(col):
        return [int(i) for i in np.asarray(res[col], dtype=bool).nonzero()[0]]

    up_labels = [[int(i), int(v)] for i, v in enumerate(res["NINE2_UP_LABEL"].to_numpy()) if v > 0]
    down_labels = [[int(i), int(v)] for i, v in enumerate(res["NINE2_DOWN_LABEL"].to_numpy()) if v > 0]

    last = res.iloc[-1]

    def last_or_none(col):
        v = last[col]
        return round(float(v), 4) if np.isfinite(v) else None

    return {
        "params": params,
        "dates": fmt_index(res.index),
        "open": jarr(res["OPEN"]), "high": jarr(res["HIGH"]),
        "low": jarr(res["LOW"]), "close": jarr(res["CLOSE"]),
        "volume": jarr(res["VOLUME"], nd=2),
        "mid": jarr(res["MID"]), "upper": jarr(res["UPPER"]), "lower": jarr(res["LOWER"]),
        "profit": jarr(res["获利筹"]), "v3": jarr(res["V3"]),
        "dif": jarr(res["DIF"]), "dea": jarr(res["DEA"]), "macd": jarr(res["MACD"]),
        "rsi1": jarr(res["RSI1"]),
        "bScore": [int(v) for v in res["B_SCORE"].to_numpy()],
        "sScore": [int(v) for v in res["S_SCORE"].to_numpy()],
        "upCount": [int(v) for v in res["NINE2_UP_COUNT"].to_numpy()],
        "downCount": [int(v) for v in res["NINE2_DOWN_COUNT"].to_numpy()],
        "upLabel": up_labels, "upNine": flag_indices("NINE2_UP_9"),
        "downLabel": down_labels, "downNine": flag_indices("NINE2_DOWN_9"),
        "bSignal": flag_indices("B_SIGNAL"), "sSignal": flag_indices("S_SIGNAL"),
        "juefan": flag_indices("ICON_JUEFAN"),
        "buy": flag_indices("NINE2_BUY_SIGNAL"), "sell": flag_indices("NINE2_SELL_SIGNAL"),
        "summary": {
            "close": last_or_none("CLOSE"), "mid": last_or_none("MID"),
            "upper": last_or_none("UPPER"), "lower": last_or_none("LOWER"),
            "profit": last_or_none("获利筹"),
            "bScore": int(last["B_SCORE"]), "sScore": int(last["S_SCORE"]),
            "upCount": int(last["NINE2_UP_COUNT"]), "downCount": int(last["NINE2_DOWN_COUNT"]),
            "rows": len(res),
        },
    }


class UiHandler(BaseHTTPRequestHandler):
    server_version = "KK2-EHOPT10-UI/1.0"

    def log_message(self, *args):  # 静默访问日志
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send_file(self, path: Path, ctype: str):
        try:
            self._send(200, path.read_bytes(), ctype)
        except OSError:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(WEBUI / "index.html", "text/html; charset=utf-8")
        elif path == "/echarts.min.js":
            self._send_file(WEBUI / "echarts.min.js", "application/javascript; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/compute", "/api/fetch", "/api/parse_csv", "/api/backtest"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send_json({"error": f"请求体过大 ({length} 字节, 上限 {MAX_BODY})"}, 413)
                return
            req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")

            if path == "/api/fetch":
                payload = fetch_quote(str(req.get("symbol") or "TQQQ"),
                                      str(req.get("interval") or "1d"),
                                      int(req.get("count") or DEFAULT_COUNT))
                self._send_json(payload)
                return

            if path == "/api/parse_csv":
                df = parse_csv_text(req.get("csv") or "")
                self._send_json({"rows": _rows_from_df(df), "source": "csv",
                                 "note": f"已解析 {len(df)} 根K线"})
                return

            if path == "/api/backtest":
                params = clamp_params(req.get("params") or {})
                if req.get("rows"):
                    df = df_from_rows(req["rows"])
                elif req.get("source") == "csv":
                    df = parse_csv_text(req.get("csv") or "")
                elif req.get("source") == "sample":
                    df = make_sample_data(900, seed=int(req.get("seed") or 7))
                else:
                    raise ValueError("缺少K线数据")
                cost = min(max(float(req.get("cost") or 0.001), 0.0), 0.05)
                max_hold = req.get("max_hold")
                max_hold = int(max_hold) if max_hold else None
                res = compute_ehopt10(df, **params)
                report = run_backtest(res, cost=cost, max_hold=max_hold, presets=PRESETS)
                report["dates"] = fmt_index(res.index)
                report["config"] = {"params": params, "cost": cost, "max_hold": max_hold}
                self._send_json(report)
                return

            # /api/compute: 优先用前端缓存的 rows (改参数时不重复拉行情)
            params = clamp_params(req.get("params") or {})
            if req.get("rows"):
                df = df_from_rows(req["rows"])
            elif req.get("source") == "csv":
                df = parse_csv_text(req.get("csv") or "")
            else:
                df = make_sample_data(900, seed=int(req.get("seed") or 7))
            self._send_json(build_payload(df, params))
        except Exception as exc:  # noqa: BLE001 - 错误信息直接回显给前端
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main():
    ap = argparse.ArgumentParser(description="KK2 EHOPT10 指标前端 UI")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    if not (WEBUI / "index.html").exists():
        raise SystemExit(f"缺少前端文件: {WEBUI / 'index.html'}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), UiHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"KK2 EHOPT10 指标 UI 已启动: {url}  (Ctrl+C 退出)")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
