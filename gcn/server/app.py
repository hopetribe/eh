# -*- coding: utf-8 -*-
"""GCN Web 服务: 纯标准库 HTTP + 前端载荷构建。"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import DEFAULT_SYMBOLS, PRESETS, run_backtest, slice_years
from gcn.radar import engine as radar_engine
from gcn.screener.engine import run_screen
from gcn.screener.strategies import SAMPLE_UNIVERSE as SCREENER_UNIVERSE
from gcn.screener.strategies import STRATEGIES as SCREENER_STRATEGIES
from gcn.data.service import (_auto_refresh_loop, DATA_DIR, DEFAULT_COUNT,
                              df_from_rows, fetch_quote, parse_csv_text)
from gcn.data.sample import make_sample_data
from gcn.recipes.gcn_main import compute_ehopt10

PARAM_LIMITS = {
    "SD": (20, 2, 120),
    "WIDTH": (2, 0, 100),
    "N": (4, 0, 1000),
    "OFFSET": (15, 0, 1000),
}


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



ROOT = Path(__file__).resolve().parents[2]
WEBUI = ROOT / "webui"

MAX_BODY = 30 * 1024 * 1024  # 30MB, 上限足够大的 CSV

# 富途参数表: 参数名 (默认, 最小, 最大)
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


def _norm_version(raw) -> str:
    return "v3" if str(raw or "").strip().lower() == "v3" else "v4"


def build_payload(df: pd.DataFrame, params: dict, version: str = "v4") -> dict:
    res = compute_ehopt10(df, **params, version=version)

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
        "sCondition": flag_indices("S_CONDITION"),
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
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            self._send_file(WEBUI / "index.html", "text/html; charset=utf-8")
        elif path == "/echarts.min.js":
            self._send_file(WEBUI / "echarts.min.js", "application/javascript; charset=utf-8")
        elif path == "/api/screener/meta":
            self._send_json({
                "strategies": [{"id": k, "name": v["name"], "theme": v["theme"],
                                "min_mktcap_cny": v["min_mktcap_cny"],
                                "n_conditions": len(v["conditions"])}
                               for k, v in SCREENER_STRATEGIES.items()],
                "universes": SCREENER_UNIVERSE,
            })
        elif path == "/api/radar":
            req_markets = {v for k, v in urllib.parse.parse_qsl(query)
                           if k == "market" and v in radar_engine.MARKETS}
            markets = sorted(req_markets, key=radar_engine.MARKETS.index) \
                if req_markets else radar_engine.MARKETS
            # 过期/缺失的市场自动后台重扫, 先返回现有快照 (不阻塞前端)
            self._send_json(radar_engine.SERVICE.ensure_fresh(markets))
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/compute", "/api/fetch", "/api/parse_csv",
                        "/api/backtest", "/api/screener", "/api/radar/scan"):
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

            if path == "/api/radar/scan":
                market = str(req.get("market") or "all")
                markets = (radar_engine.MARKETS if market == "all"
                           else [market] if market in radar_engine.MARKETS
                           else [])
                if not markets:
                    raise ValueError(f"未知市场: {market}")
                started = radar_engine.SERVICE.start_scan(markets)
                snap = radar_engine.SERVICE.snapshot(markets)
                snap["started"] = started
                self._send_json(snap)
                return

            if path == "/api/screener":
                strategy = str(req.get("strategy") or "graham")
                if strategy not in SCREENER_STRATEGIES:
                    raise ValueError(f"未知策略: {strategy}")
                symbols = ([x.strip().upper() for x in (req.get("symbols") or "").split(",")
                            if x.strip()]
                           if req.get("symbols")
                           else SCREENER_UNIVERSE.get(str(req.get("market") or "us"), []))
                if not symbols:
                    raise ValueError("候选池为空")
                results = run_screen(symbols, strategy, log=False)
                self._send_json({"strategy": strategy, "results": results,
                                 "n_passed": sum(1 for r in results if r["passed"])})
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
                years = req.get("years")
                years = float(years) if years else None
                version = _norm_version(req.get("version"))
                res = compute_ehopt10(df, **params, version=version)
                res = slice_years(res, years)  # 指标全量计算后按年切片, 预热不丢失
                report = run_backtest(res, cost=cost, max_hold=max_hold, presets=PRESETS)
                report["dates"] = fmt_index(res.index)
                report["version"] = version
                report["config"] = {"params": params, "cost": cost,
                                    "max_hold": max_hold, "years": years}
                self._send_json(report)
                return

            # /api/compute: 优先用前端缓存的 rows (改参数时不重复拉行情)
            params = clamp_params(req.get("params") or {})
            version = _norm_version(req.get("version"))
            if req.get("rows"):
                df = df_from_rows(req["rows"])
            elif req.get("source") == "csv":
                df = parse_csv_text(req.get("csv") or "")
            else:
                df = make_sample_data(900, seed=int(req.get("seed") or 7))
            self._send_json(build_payload(df, params, version))
        except Exception as exc:  # noqa: BLE001 - 错误信息直接回显给前端
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main():
    ap = argparse.ArgumentParser(description="KK2 EHOPT10 指标前端 UI")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址；默认仅本机访问")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    if not (WEBUI / "index.html").exists():
        raise SystemExit(f"缺少前端文件: {WEBUI / 'index.html'}")

    server = ThreadingHTTPServer((args.host, args.port), UiHandler)
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}"
    print(f"KK2 EHOPT10 指标 UI 已启动: {url}  (Ctrl+C 退出)")
    print(f"K线本地缓存目录: {DATA_DIR}  (每日自动刷新已开启)")
    threading.Thread(target=_auto_refresh_loop, daemon=True).start()
    # 机会雷达: 三大市场市值前100标的日K缓存, 每小时巡检增量更新
    threading.Thread(target=radar_engine.warm_loop, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
