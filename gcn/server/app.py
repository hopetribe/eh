# -*- coding: utf-8 -*-
"""GCN Web 服务: 纯标准库 HTTP + 前端载荷构建。"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import socket
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import (
    PRESETS, TIMEFRAMES, run_backtest, slice_years,
)
from gcn.radar import engine as radar_engine
from gcn.radar import emailer as radar_emailer
from gcn.radar import scheduler as radar_scheduler
from gcn.screener.engine import run_screen
from gcn.screener.strategies import SAMPLE_UNIVERSE as SCREENER_UNIVERSE
from gcn.screener.strategies import STRATEGIES as SCREENER_STRATEGIES
from gcn.data.service import (_auto_refresh_loop, DATA_DIR, DEFAULT_COUNT,
                              _rows_from_df, df_from_rows, fetch_quote,
                              parse_csv_text)
from gcn.data.sample import make_sample_data
from gcn.recipes.gcn_main import compute_ehopt10

PARAM_LIMITS = {
    "SD": (20, 2, 120),
    "WIDTH": (2, 0, 100),
    "N": (4, 1, 1000),
    "OFFSET": (15, 0, 1000),
}


def clamp_params(raw: dict) -> dict:
    """按参数表范围收敛前端传入的参数。"""
    if not isinstance(raw, dict):
        raise ValueError("params 必须是 JSON 对象")
    out = {}
    for key, (dflt, lo, hi) in PARAM_LIMITS.items():
        try:
            v = float(raw.get(key, dflt))
        except (TypeError, ValueError):
            v = float(dflt)
        if not math.isfinite(v):
            v = float(dflt)
        if key in ("SD", "N", "OFFSET"):
            v = int(round(v))
        out[key] = min(max(v, lo), hi)
    return out


ROOT = Path(__file__).resolve().parents[2]
WEBUI = ROOT / "webui"

MAX_BODY = 30 * 1024 * 1024  # 30MB, 上限足够大的 CSV
MAX_ROWS = 100_000
MAX_SYMBOLS = 300
MAX_QUOTE_COUNT = 10_000
MAX_API_CONCURRENCY = 8
MAX_HTTP_THREADS = 32
SOCKET_TIMEOUT = 20

_API_SLOTS = threading.BoundedSemaphore(MAX_API_CONCURRENCY)


class RequestError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _dumps_json(obj) -> bytes:
    """严格 JSON 编码，禁止 Python 扩展的 NaN/Infinity 字面量。"""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _parse_json(raw: bytes) -> dict:
    def reject_constant(value):
        raise ValueError(f"JSON 不允许 {value}")

    value = json.loads(raw.decode("utf-8") or "{}", parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return value


def _finite_number(raw, name: str, default=None, lo=None, hi=None) -> float | None:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数")
    if ((lo is not None and value < lo)
            or (hi is not None and value > hi)):
        raise ValueError(f"{name} 超出允许范围")
    return value


def _positive_int(raw, name: str, default=None, maximum=None) -> int | None:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{name} 必须是正整数")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    try:
        exact = float(raw) == value
    except (TypeError, ValueError):
        exact = False
    if not exact or value <= 0 or maximum is not None and value > maximum:
        raise ValueError(f"{name} 必须是正整数且不超过 {maximum}" if maximum
                         else f"{name} 必须是正整数")
    return value


def _parse_markets(query: str) -> list[str]:
    requested = []
    for key, value in urllib.parse.parse_qsl(query):
        if key != "market":
            continue
        requested.extend(part.strip().lower() for part in value.split(",") if part.strip())
    if not requested:
        return list(radar_engine.MARKETS)
    unknown = set(requested) - set(radar_engine.MARKETS)
    if unknown:
        raise ValueError(f"未知市场: {', '.join(sorted(unknown))}")
    wanted = set(requested)
    return [market for market in radar_engine.MARKETS if market in wanted]


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("K线数据为空")
    if len(df) > MAX_ROWS:
        raise ValueError(f"K线数量超过上限 {MAX_ROWS}")
    columns = {str(col).lower(): col for col in df.columns}
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"K线数据缺少列: {missing}")
    values = {name: pd.to_numeric(df[columns[name]], errors="coerce").to_numpy(dtype=float)
              for name in required}
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("K线数据包含 NaN 或 Infinity")
    if any((values[name] <= 0).any() for name in ("open", "high", "low", "close")):
        raise ValueError("OHLC 价格必须大于 0")
    if (values["volume"] < 0).any():
        raise ValueError("成交量不能为负数")
    if ((values["high"] < np.maximum.reduce(
            [values["open"], values["close"], values["low"]])).any()
            or (values["low"] > np.minimum.reduce(
                [values["open"], values["close"], values["high"]])).any()):
        raise ValueError("K线 high/low 与 open/close 不一致")
    return df


def _is_loopback(host: str) -> bool:
    host = host.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
    version = str(raw or "v4").strip().lower()
    if version not in {"v3", "v4"}:
        raise ValueError(f"未知配方版本: {version}")
    return version


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

    def setup(self):
        super().setup()
        self.connection.settimeout(SOCKET_TIMEOUT)

    def log_message(self, *args):  # 静默访问日志
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200):
        self._send(code, _dumps_json(obj),
                   "application/json; charset=utf-8")

    def _request_guard(self):
        host_header = (self.headers.get("Host") or "").strip().lower()
        if not host_header:
            raise RequestError(400, "缺少 Host 请求头")
        request_host = urllib.parse.urlsplit(f"//{host_header}").hostname
        if not request_host:
            raise RequestError(400, "Host 请求头无效")
        bound_host = str(getattr(self.server, "bound_host", self.server.server_address[0]))
        allowed_hosts = getattr(self.server, "allowed_hosts", None)
        if allowed_hosts and request_host.lower() not in allowed_hosts:
            raise RequestError(421, "Host 不在允许列表")
        if _is_loopback(bound_host) and not _is_loopback(request_host):
            raise RequestError(421, "Host 与本地监听地址不匹配")
        if (not allowed_hosts and not _is_loopback(bound_host)
                and bound_host not in {"0.0.0.0", "::"}
                and request_host.lower() != bound_host.lower()):
            raise RequestError(421, "Host 与监听地址不匹配")

        if (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
            raise RequestError(403, "拒绝跨站请求")
        origin = self.headers.get("Origin")
        if origin:
            parsed = urllib.parse.urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != host_header:
                raise RequestError(403, "Origin 与 Host 不匹配")

    def _read_json_request(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(415, "不支持 Transfer-Encoding")
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise RequestError(415, "Content-Type 必须是 application/json")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise RequestError(400, "Content-Length 无效") from exc
        if length < 0:
            raise RequestError(400, "Content-Length 不能为负数")
        if length > MAX_BODY:
            raise RequestError(413, f"请求体过大 ({length} 字节, 上限 {MAX_BODY})")
        return _parse_json(self.rfile.read(length))

    def _send_exception(self, exc: Exception):
        if isinstance(exc, RequestError):
            self._send_json({"error": str(exc)}, exc.status)
        elif isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError)):
            self._send_json({"error": str(exc)}, 400)
        elif isinstance(exc, (ConnectionError, TimeoutError, socket.timeout, RuntimeError)):
            self._send_json({"error": "上游数据服务不可用"}, 502)
        else:
            self._send_json({"error": "内部服务错误"}, 500)

    def _send_file(self, path: Path, ctype: str):
        try:
            self._send(200, path.read_bytes(), ctype)
        except OSError:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_GET(self):
        try:
            self._request_guard()
            path, _, query = self.path.partition("?")
            if path in ("/", "/index.html"):
                self._send_file(WEBUI / "index.html", "text/html; charset=utf-8")
            elif path == "/styles.css":
                self._send_file(WEBUI / "styles.css", "text/css; charset=utf-8")
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
                self._send_json(radar_engine.SERVICE.snapshot(_parse_markets(query)))
            elif path == "/api/radar/email":
                self._send_json(radar_emailer.get_email_settings())
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
        except Exception as exc:  # noqa: BLE001 - 统一映射 HTTP 边界
            self._send_exception(exc)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/compute", "/api/fetch", "/api/parse_csv",
                        "/api/backtest", "/api/screener", "/api/radar/scan",
                        "/api/radar/email"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        if not _API_SLOTS.acquire(blocking=False):
            self._send_json({"error": "服务繁忙，请稍后重试"}, 503)
            return
        try:
            self._request_guard()
            req = self._read_json_request()

            if path == "/api/fetch":
                symbol = str(req.get("symbol") or "TQQQ").strip().upper()
                if not symbol or len(symbol) > 64:
                    raise ValueError("股票代码无效")
                interval = str(req.get("interval") or "1d").strip().lower()
                if interval not in TIMEFRAMES:
                    raise ValueError(f"未知K线周期: {interval}")
                count = _positive_int(req.get("count"), "count", DEFAULT_COUNT,
                                      MAX_QUOTE_COUNT)
                payload = fetch_quote(symbol, interval, count)
                self._send_json(payload)
                return

            if path == "/api/parse_csv":
                df = parse_csv_text(req.get("csv") or "")
                _validate_frame(df)
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

            if path == "/api/radar/email":
                settings = radar_emailer.update_recipient(
                    str(req.get("action") or ""), str(req.get("email") or ""))
                self._send_json(settings)
                return

            if path == "/api/screener":
                strategy = str(req.get("strategy") or "graham")
                if strategy not in SCREENER_STRATEGIES:
                    raise ValueError(f"未知策略: {strategy}")
                raw_symbols = req.get("symbols")
                if raw_symbols is not None and not isinstance(raw_symbols, str):
                    raise ValueError("symbols 必须是逗号分隔字符串")
                symbols = ([x.strip().upper() for x in (raw_symbols or "").split(",")
                            if x.strip()]
                           if raw_symbols
                           else SCREENER_UNIVERSE.get(str(req.get("market") or "us"), []))
                symbols = list(dict.fromkeys(symbols))
                if not symbols:
                    raise ValueError("候选池为空")
                if len(symbols) > MAX_SYMBOLS:
                    raise ValueError(f"候选池超过上限 {MAX_SYMBOLS}")
                results = run_screen(symbols, strategy, log=False)
                self._send_json({"strategy": strategy, "results": results,
                                 "n_passed": sum(1 for r in results if r["passed"])})
                return

            if path == "/api/backtest":
                params = clamp_params(req.get("params") or {})
                source = req.get("source")
                if source == "rows" or "rows" in req:
                    if source not in (None, "rows"):
                        raise ValueError("K线数据源与 rows 冲突")
                    df = df_from_rows(req.get("rows"))
                elif source == "csv":
                    df = parse_csv_text(req.get("csv") or "")
                elif source == "sample":
                    df = make_sample_data(900, seed=int(req.get("seed") or 7))
                else:
                    raise ValueError("缺少K线数据")
                _validate_frame(df)
                cost = _finite_number(req.get("cost"), "cost", 0.001, 0.0, 0.05)
                max_hold = _positive_int(req.get("max_hold"), "max_hold")
                years = _finite_number(req.get("years"), "years", None, 0.01, 100.0)
                interval = str(req.get("interval") or "1d").strip().lower()
                if interval not in TIMEFRAMES:
                    raise ValueError(f"未知K线周期: {interval}")
                version = _norm_version(req.get("version"))
                res = compute_ehopt10(df, **params, version=version)
                res = slice_years(res, years, interval)  # 指标全量计算后按年切片, 预热不丢失
                report = run_backtest(res, cost=cost, max_hold=max_hold,
                                      presets=PRESETS, interval=interval)
                report["dates"] = fmt_index(res.index)
                report["version"] = version
                report["config"] = {"params": params, "cost": cost,
                                    "max_hold": max_hold, "years": years,
                                    "interval": interval}
                self._send_json(report)
                return

            # /api/compute: 优先用前端缓存的 rows (改参数时不重复拉行情)
            params = clamp_params(req.get("params") or {})
            version = _norm_version(req.get("version"))
            source = req.get("source")
            if source == "rows" or "rows" in req:
                if source not in (None, "rows"):
                    raise ValueError("K线数据源与 rows 冲突")
                df = df_from_rows(req.get("rows"))
            elif source == "csv":
                df = parse_csv_text(req.get("csv") or "")
            elif source in (None, "sample"):
                df = make_sample_data(900, seed=int(req.get("seed") or 7))
            else:
                raise ValueError(f"未知K线数据源: {source}")
            _validate_frame(df)
            self._send_json(build_payload(df, params, version))
        except Exception as exc:  # noqa: BLE001 - 统一映射 HTTP 边界
            self._send_exception(exc)
        finally:
            _API_SLOTS.release()


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, *args, **kwargs):
        self._thread_slots = threading.BoundedSemaphore(MAX_HTTP_THREADS)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._thread_slots.acquire(blocking=False):
            try:
                body = b"service busy"
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Type: text/plain\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
            finally:
                request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._thread_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._thread_slots.release()


def main():
    ap = argparse.ArgumentParser(description="KK2 EHOPT10 指标前端 UI")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址；默认仅本机访问")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--allow-remote", action="store_true",
                    help="显式允许非回环地址监听（仍应置于可信网络/反向代理之后）")
    ap.add_argument("--allowed-host", action="append", default=[],
                    help="远程监听允许的 Host 名称/IP；可重复指定")
    args = ap.parse_args()

    if not (WEBUI / "index.html").exists():
        raise SystemExit(f"缺少前端文件: {WEBUI / 'index.html'}")

    if not _is_loopback(args.host) and not args.allow_remote:
        raise SystemExit("拒绝直接对外监听；如确认运行在可信网络，请显式添加 --allow-remote")
    if args.host in {"0.0.0.0", "::"} and args.allow_remote and not args.allowed_host:
        raise SystemExit("通配地址监听必须至少指定一个 --allowed-host")

    server = AppServer((args.host, args.port), UiHandler)
    server.bound_host = args.host
    server.allowed_hosts = (None if _is_loopback(args.host) else
                            {str(host).strip().strip("[]").lower()
                             for host in (args.allowed_host or [args.host])})
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}"
    print(f"KK2 EHOPT10 指标 UI 已启动: {url}  (Ctrl+C 退出)")
    print(f"K线本地缓存目录: {DATA_DIR}  (每日自动刷新已开启)")
    threading.Thread(target=_auto_refresh_loop, daemon=True).start()
    # 机会雷达: 阈值股票池日K每小时增量巡检; 每天 09:00 扫描并发送邮件
    threading.Thread(target=radar_engine.warm_loop, daemon=True).start()
    threading.Thread(target=radar_scheduler.daily_radar_loop, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
