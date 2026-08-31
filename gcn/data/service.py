# -*- coding: utf-8 -*-
"""GCN 数据服务: 多数据源 (Futu->Yahoo 回退) + 本地落盘缓存 + 每日自动刷新。

稳定性设计
----------
- 缓存命中优先, 仅在数据陈旧时发起在线请求 (周末/假日不触发);
- 在线更新失败时降级使用本地缓存, 保证服务可用;
- 后台守护线程每 6 小时强制刷新默认标的, 保证每日数据及时。
"""
from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import DEFAULT_SYMBOLS
from gcn.core.tdx import _as_bool

# 仓库根目录 (gcn/ 的上一级)
ROOT = Path(__file__).resolve().parents[2]

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
    "1d": ("K_DAY", "1d", "10y"),
    "1wk": ("K_WEEK", "1wk", "max"),
    "60m": ("K_60M", "60m", "2y"),
    "15m": ("K_15M", "15m", "60d"),
    "5m": ("K_5M", "5m", "60d"),
}
DEFAULT_COUNT = 2500

# ==========================================================================
# K线本地落盘缓存: data/<SYMBOL>_<interval>.csv
#   - 在线抓取后合并落盘, 避免重复请求
#   - 缓存陈旧 (最后一根K线早于今天, 或今天尚未刷新过) 时才重新请求
#   - 服务启动后由后台线程每日自动刷新, 保证数据及时
# ==========================================================================
DATA_DIR = ROOT / "data"


def _cache_path(symbol: str, interval: str):
    return DATA_DIR / f"{symbol.replace('.', '_')}_{interval}.csv"


def _load_cache(symbol: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df["date"], format="mixed")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception:  # noqa: BLE001 - 缓存损坏时忽略, 重新抓取
        return None


def _save_cache(symbol: str, interval: str, df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    out = df.copy()
    idx = out.index
    out.insert(0, "date", [t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "hour") and (t.hour or t.minute)
                           else t.strftime("%Y-%m-%d") for t in idx])
    out.to_csv(_cache_path(symbol, interval), index=False)


# 每标的读写锁: fetch_quote 的 读缓存-合并-落盘 临界区按 symbol 串行化
_symbol_locks: dict[str, threading.Lock] = {}
_symbol_locks_mu = threading.Lock()


def _symbol_lock(symbol: str) -> threading.Lock:
    with _symbol_locks_mu:
        lock = _symbol_locks.get(symbol)
        if lock is None:
            lock = _symbol_locks[symbol] = threading.Lock()
        return lock


def _cache_is_fresh(cached: pd.DataFrame, interval: str, path) -> bool:
    """缓存是否足够新 (无需在线更新)。

    规则: 今天已尝试刷新过 (文件 mtime) / 已有今天的K线 / 自最后一根K线
    以来没有错失任何交易日 (周末与假日不视为陈旧) —— 三者满足其一即新鲜。
    日内周期永远视为需刷新。
    """
    if interval.endswith("m"):
        return False
    now = pd.Timestamp.now().normalize()
    last = cached.index.max().normalize()
    if last >= now:
        return True
    # mtime 按本地时区取 naive 时间, 与 now 同口径比较 (此前按 UTC 换算,
    # 0点~8点区间会误判陈旧导致重复在线请求)
    if path.exists() and pd.Timestamp.fromtimestamp(
            path.stat().st_mtime).normalize() >= now:
        return True
    missed = pd.bdate_range(last + pd.Timedelta(days=1), now - pd.Timedelta(days=1))
    return len(missed) == 0

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


def fetch_quote(symbol: str, interval: str = "1d", count: int = DEFAULT_COUNT,
                use_cache: bool = True, force: bool = False) -> dict:
    """抓取K线: 本地缓存优先 (陈旧才在线更新), Futu 优先, 失败自动回退 Yahoo。

    force=True 时无条件在线刷新 (供后台每日守护使用)。
    返回 {rows, source, symbol, interval, note}; source 可为 cache/futu/yahoo。
    同一标的的 读缓存-合并-落盘 临界区按 symbol 加锁, 避免雷达扫描/预热/
    自动刷新多线程并发更新同一 CSV 时互相覆盖丢数据; 锁内二次检查新鲜度,
    已被其他线程抢先刷新的线程直接走缓存。
    """
    interval = interval if interval in INTERVALS else "1d"
    count = int(min(max(int(count or DEFAULT_COUNT), 100), 5000))
    symbol = (symbol or "TQQQ").strip().upper()
    if not symbol:
        raise ValueError("股票代码不能为空")

    notes = []
    cpath = _cache_path(symbol, interval)
    with _symbol_lock(symbol):
        cached = _load_cache(symbol, interval) if use_cache else None
        need_fetch = force or cached is None or not _cache_is_fresh(cached, interval, cpath)
        if use_cache and cached is not None and not cached.empty and not need_fetch:
            # 缓存足够新 (含被其他线程抢先刷新的情形), 直接命中
            return {"rows": _rows_from_df(cached)[-count:], "source": "cache",
                    "symbol": to_yahoo_symbol(symbol), "interval": interval,
                    "note": "本地缓存"}

        merged = cached
        try:
            import futu  # noqa: F401 - 检测 futu-api 是否安装
            futu_ready = _opend_reachable()
        except ImportError:
            futu_ready = False

        if futu_ready:
            try:
                df = _fetch_futu(symbol, interval, count)
                merged = df if merged is None or merged.empty else (
                    pd.concat([merged, df])[~pd.concat([merged, df]).index.duplicated(keep="last")]
                    .sort_index())
                source, sym_used = "futu", to_futu_symbol(symbol)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Futu({exc})")
                source, sym_used = None, to_yahoo_symbol(symbol)
        else:
            source, sym_used = None, to_yahoo_symbol(symbol)

        if source is None:
            try:
                rows = _fetch_yahoo(symbol, interval, count)
                df = pd.DataFrame([r[1:] for r in rows],
                                  columns=["open", "high", "low", "close", "volume"],
                                  index=pd.to_datetime([r[0] for r in rows], format="mixed"))
                merged = df if merged is None or merged.empty else (
                    pd.concat([merged, df])[~pd.concat([merged, df]).index.duplicated(keep="last")]
                    .sort_index())
                source = "yahoo"
            except Exception as exc:  # noqa: BLE001
                if merged is not None and not merged.empty:
                    notes.append(f"在线更新失败({exc}), 使用本地缓存")
                    out_rows = _rows_from_df(merged)
                    return {"rows": out_rows[-count:], "source": "cache",
                            "symbol": sym_used, "interval": interval,
                            "note": "; ".join(notes)}
                raise RuntimeError(f"获取 {symbol} 行情失败: {'; '.join(notes + [f'Yahoo({exc})'])}") from exc

        _save_cache(symbol, interval, merged)
        out_rows = _rows_from_df(merged)
        return {"rows": out_rows[-count:], "source": source,
                "symbol": sym_used, "interval": interval, "note": "; ".join(notes)}


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


def _auto_refresh_loop():
    """后台数据守护: 每 6 小时刷新默认关注列表的日K缓存。

    fetch_quote 内部有新鲜度判断 (已有今日K线则跳过请求), 因此该循环
    只在跨日/盘中数据陈旧时才真正发起网络请求, 保证每天及时更新。
    """
    while True:
        for sym in DEFAULT_SYMBOLS:
            try:
                fetch_quote(sym, "1d", DEFAULT_COUNT, use_cache=True, force=True)
            except Exception:  # noqa: BLE001 - 后台刷新失败不影响服务
                pass
        time.sleep(6 * 3600)


CSV_ALIASES = {
    "open": ["open", "o", "开盘价", "开盘"],
    "high": ["high", "h", "最高价", "最高"],
    "low": ["low", "l", "最低价", "最低"],
    "close": ["close", "c", "收盘价", "收盘"],
    "volume": ["volume", "vol", "v", "成交量"],
    "date": ["date", "time", "time_key", "datetime", "timestamp", "时间", "日期", "时间戳"],
}


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


