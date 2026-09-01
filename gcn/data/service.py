# -*- coding: utf-8 -*-
"""GCN 数据服务: 多数据源 (Futu->Yahoo 回退) + 本地落盘缓存 + 每日自动刷新。

稳定性设计
----------
- 缓存命中优先, 仅在数据陈旧或历史长度不足时发起在线请求;
- 在线更新失败时降级使用本地缓存, 保证服务可用;
- 后台守护线程每 6 小时检查默认标的, 仅刷新陈旧/不足的缓存。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import socket
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gcn.backtest.engine import DEFAULT_SYMBOLS

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
#   - 缓存未覆盖所属市场最近已收盘工作日时才重新请求
#   - 服务启动后由后台线程每日自动刷新, 保证数据及时
# ==========================================================================
DATA_DIR = ROOT / "data"

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")


def _normalize_symbol(symbol: str) -> str:
    """Validate the public symbol grammar and return an uppercase spelling."""
    value = str(symbol or "").strip().upper()
    if not _SYMBOL_RE.fullmatch(value):
        raise ValueError("股票代码格式无效")
    return value


def _cache_key(symbol: str) -> str:
    """Collapse Futu/Yahoo/bare aliases to one filesystem-safe cache key."""
    value = _normalize_symbol(symbol)
    futu = to_futu_symbol(value)
    market, code = futu.split(".", 1)
    if market in ("HK", "SH", "SZ", "US"):
        value = code
    else:  # defensive: to_futu_symbol currently always returns a known market
        value = futu
    if "." in value or "_" in value:
        return "x_" + value.encode("ascii").hex()
    return value


def _cache_path(symbol: str, interval: str):
    if interval not in INTERVALS:
        raise ValueError(f"不支持的周期: {interval}")
    return DATA_DIR / f"{_cache_key(symbol)}_{interval}.csv"


def _cache_lock_path(symbol: str, interval: str) -> Path:
    return _cache_path(symbol, interval).with_suffix(".lock")


@contextmanager
def _cache_file_lock(symbol: str, interval: str):
    """Serialize read/merge/write across server and CLI processes."""
    import fcntl
    path = _cache_lock_path(symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep finite, internally consistent OHLCV bars and preserve provenance."""
    columns = ["open", "high", "low", "close", "volume"]
    if not isinstance(df, pd.DataFrame) or any(col not in df for col in columns):
        return pd.DataFrame(columns=columns)
    attrs = dict(df.attrs)
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    values = out.to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    valid &= (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
    valid &= values[:, 1] >= np.maximum.reduce(
        [values[:, 0], values[:, 2], values[:, 3]])
    valid &= values[:, 2] <= np.minimum.reduce(
        [values[:, 0], values[:, 1], values[:, 3]])
    out = out.loc[valid]
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.loc[out.index.notna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs.update(attrs)
    return out


def _load_cache(symbol: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df["date"], format="mixed")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not meta.get("sha256") or meta.get("sha256") == digest:
                df.attrs.update({k: meta[k] for k in ("source", "adjustment") if k in meta})
        return _sanitize_ohlcv(df)
    except Exception:  # noqa: BLE001 - 缓存损坏时忽略, 重新抓取
        return None


def _atomic_write_text(path: Path, text: str):
    """Crash-safe replacement in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _save_cache(symbol: str, interval: str, df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    clean = _sanitize_ohlcv(df)
    if clean.empty:
        raise ValueError("无有效K线可写入缓存")
    out = clean.copy()
    idx = out.index
    out.insert(0, "date", [t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "hour") and (t.hour or t.minute)
                           else t.strftime("%Y-%m-%d") for t in idx])
    path = _cache_path(symbol, interval)
    csv_text = out.to_csv(index=False)
    _atomic_write_text(path, csv_text)
    meta = {k: clean.attrs[k] for k in ("source", "adjustment") if k in clean.attrs}
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta:
        meta["sha256"] = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
        _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False))
    else:
        meta_path.unlink(missing_ok=True)


# 每标的读写锁: fetch_quote 的 读缓存-合并-落盘 临界区按 symbol 串行化
_symbol_locks: dict[str, threading.Lock] = {}
_symbol_locks_mu = threading.Lock()


def _symbol_lock(symbol: str) -> threading.Lock:
    symbol = _cache_key(symbol)
    with _symbol_locks_mu:
        lock = _symbol_locks.get(symbol)
        if lock is None:
            lock = _symbol_locks[symbol] = threading.Lock()
        return lock


def _last_completed_session(symbol: str, now=None) -> pd.Timestamp:
    """Return the latest weekday session whose regular close has passed."""
    market = to_futu_symbol(symbol).split(".", 1)[0]
    tz_name, close_hour = {
        "SH": ("Asia/Shanghai", 15), "SZ": ("Asia/Shanghai", 15),
        "HK": ("Asia/Hong_Kong", 16), "US": ("America/New_York", 16),
    }[market]
    if now is None:
        local = pd.Timestamp.now(tz=ZoneInfo(tz_name))
    else:
        local = pd.Timestamp(now)
        if local.tzinfo is not None:
            local = local.tz_convert(ZoneInfo(tz_name))
    local = local.tz_localize(None) if local.tzinfo is not None else local
    day = local.normalize()
    if day.weekday() >= 5:
        return (day - pd.offsets.BDay()).normalize()
    if local < day + pd.Timedelta(hours=close_hour):
        return (day - pd.offsets.BDay()).normalize()
    return day


def _cache_is_fresh(cached: pd.DataFrame, interval: str, path,
                    symbol: str = "US.AAPL", now=None) -> bool:
    """缓存是否足够新 (无需在线更新)。

    日线以对应市场最近一次已经收盘的工作日为基准，不使用文件 mtime：
    checkout/touch 不能证明数据已刷新。周线要求至少覆盖该交易周；日内周期
    始终在线刷新。交易所特殊休市由在线源的失败退避兜底。
    """
    if interval.endswith("m"):
        return False
    if cached is None or cached.empty or not isinstance(cached.index, pd.DatetimeIndex):
        return False
    last = cached.index.max()
    if pd.isna(last):
        return False
    last = pd.Timestamp(last).tz_localize(None).normalize() \
        if pd.Timestamp(last).tzinfo is not None else pd.Timestamp(last).normalize()
    expected = _last_completed_session(symbol, now=now)
    if interval == "1wk":
        return last.to_period("W-SUN") >= expected.to_period("W-SUN")
    return last >= expected

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
    s = _normalize_symbol(s)
    if s.startswith(("US.", "HK.", "SH.", "SZ.")):
        return s
    if s.endswith(".HK") and s[:-3].isdigit():
        return "HK." + s[:-3].zfill(5)
    if s.endswith(".SS") and s[:-3].isdigit():
        return "SH." + s[:-3].zfill(6)
    if s.endswith(".SZ") and s[:-3].isdigit():
        return "SZ." + s[:-3].zfill(6)
    if s.isdigit():
        if len(s) == 5:
            return "HK." + s
        if len(s) == 6:
            return ("SH." if s[0] in "569" else "SZ.") + s
    return "US." + s


def to_yahoo_symbol(s: str) -> str:
    """US.TQQQ->TQQQ, 00700/HK.00700->0700.HK, 600519->600519.SS, 000001->000001.SZ。"""
    s = _normalize_symbol(s)
    if s.endswith(".HK") and s[:-3].isdigit():
        return f"{int(s[:-3]):04d}.HK"
    if s.endswith((".SS", ".SZ")) and s[:-3].isdigit():
        return s
    if s.startswith(("US.", "HK.", "SH.", "SS.", "SZ.")):
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
    df = _sanitize_ohlcv(df)
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        has_time = bool(((idx.hour != 0) | (idx.minute != 0)).any())
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        dates = [t.strftime(fmt) for t in idx]
    else:
        dates = [str(x) for x in idx]
    return [[d, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)]
            for d, r in zip(dates, df.itertuples(index=False))]


def _parse_datetime_values(values) -> pd.DatetimeIndex:
    """Parse mixed ISO values plus Unix seconds/milliseconds safely."""
    parsed = []
    for raw in values:
        value = str(raw).strip()
        ts = pd.NaT
        try:
            number = float(value)
            digits = value.split(".", 1)[0].lstrip("+-")
            if len(digits) == 8 and 19000101 <= int(number) <= 21001231:
                ts = pd.to_datetime(str(int(number)), format="%Y%m%d", errors="coerce")
            else:
                magnitude = abs(number)
                unit = ("ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14
                        else "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e9
                        else None)
                if unit:
                    ts = pd.to_datetime(number, unit=unit, errors="coerce", utc=True)
        except (TypeError, ValueError, OverflowError):
            ts = pd.to_datetime(value, format="mixed", errors="coerce", utc=True)
        if not pd.isna(ts):
            ts = pd.Timestamp(ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        parsed.append(ts)
    return pd.DatetimeIndex(parsed)


def _merge_market_data(cached: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge only series with a compatible adjustment basis.

    Legacy caches have no provenance, so combining them with adjusted data can
    manufacture a corporate-action jump. Replace those caches on first refresh.
    """
    fresh = _sanitize_ohlcv(fresh)
    if fresh.empty:
        raise ValueError("行情源未返回有效 OHLCV")
    cached = _sanitize_ohlcv(cached) if cached is not None else None
    attrs = dict(fresh.attrs)
    if cached is None or cached.empty:
        out = fresh.copy()
    elif (cached.attrs.get("adjustment") != fresh.attrs.get("adjustment")
          or not fresh.attrs.get("adjustment")):
        out = fresh.copy()
    else:
        out = pd.concat([cached, fresh])
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs.update(attrs)
    return out


def _fetch_futu(symbol: str, interval: str, count: int) -> pd.DataFrame:
    from futu import AuType, KLType, OpenQuoteContext

    kl_name = INTERVALS[interval][0]
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        end = pd.Timestamp.now().normalize()
        if interval == "1wk":
            span_days = max(count * 8, 730)
        elif interval == "1d":
            span_days = max(count * 2, 730)
        else:
            # Futu intraday history availability is finite; request its broad
            # supported window and paginate until count or exhaustion.
            span_days = 730 if interval == "60m" else 60
        start = end - pd.Timedelta(days=span_days)
        frames = []
        page_key = None
        seen_keys = set()
        while sum(len(frame) for frame in frames) < count:
            ret, data, next_key = ctx.request_history_kline(
                to_futu_symbol(symbol), start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"), ktype=getattr(KLType, kl_name),
                autype=AuType.QFQ, max_count=min(1000, count),
                page_req_key=page_key)
            if ret != 0:
                raise RuntimeError(str(data) or "FutuOpenD 请求失败")
            if isinstance(data, pd.DataFrame) and not data.empty:
                frames.append(pd.DataFrame({
                    "open": data["open"].astype(float).to_numpy(),
                    "high": data["high"].astype(float).to_numpy(),
                    "low": data["low"].astype(float).to_numpy(),
                    "close": data["close"].astype(float).to_numpy(),
                    "volume": data["volume"].astype(float).to_numpy(),
                }, index=pd.to_datetime(data["time_key"], format="mixed")))
            if next_key is None or next_key in seen_keys:
                break
            seen_keys.add(next_key)
            page_key = next_key
        if not frames:
            raise RuntimeError("FutuOpenD 返回无有效K线")
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index().iloc[-count:]
        df.attrs.update(source="futu", adjustment="adjusted")
        df = _sanitize_ohlcv(df)
        if df.empty:
            raise RuntimeError("FutuOpenD 返回无有效K线")
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
    adj = ((r0.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
    meta = r0.get("meta") or {}
    gmtoffset = int(meta.get("gmtoffset") or 0)
    exchange_tz = meta.get("exchangeTimezoneName")
    try:
        exchange_zone = ZoneInfo(exchange_tz) if exchange_tz else None
    except (KeyError, ValueError):
        exchange_zone = None
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
        if not interval.endswith("m") and i < len(adj) and adj[i] is not None and c:
            factor = float(adj[i]) / float(c)
            if np.isfinite(factor) and factor > 0:
                o, h, l, c = (float(x) * factor for x in (o, h, l, c))
        values = np.asarray([o, h, l, c, v], dtype=float)
        if (not np.isfinite(values).all() or (values[:4] <= 0).any()
                or values[4] < 0 or values[1] < max(values[0], values[2], values[3])
                or values[2] > min(values[0], values[1], values[3])):
            continue
        utc_dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        dt = (utc_dt.astimezone(exchange_zone).replace(tzinfo=None)
              if exchange_zone else
              datetime.fromtimestamp(int(t) + gmtoffset, tz=timezone.utc).replace(tzinfo=None))
        rows.append([dt.strftime(fmt), float(o), float(h), float(l), float(c), float(v)])
    if len(rows) > count:
        rows = rows[-count:]
    if not rows:
        raise RuntimeError(f"Yahoo 返回 {ysym} 无有效K线")
    return rows


def fetch_quote(symbol: str, interval: str = "1d", count: int = DEFAULT_COUNT,
                use_cache: bool = True, force: bool = False) -> dict:
    """抓取K线: 本地缓存优先 (陈旧才在线更新), Futu 优先, 失败自动回退 Yahoo。

    force=True 时无条件在线刷新 (供显式手动预热使用)。
    返回 {rows, source, symbol, interval, note}; source 可为 cache/futu/yahoo。
    同一标的的 读缓存-合并-落盘 临界区按 symbol 加锁, 避免雷达扫描/预热/
    自动刷新多线程并发更新同一 CSV 时互相覆盖丢数据; 锁内二次检查新鲜度,
    已被其他线程抢先刷新的线程直接走缓存。
    """
    interval = str(interval or "").strip().lower()
    if interval not in INTERVALS:
        raise ValueError(f"不支持的周期: {interval or '(空)'}")
    count = int(min(max(int(count or DEFAULT_COUNT), 100), 5000))
    symbol = (symbol or "TQQQ").strip().upper()
    if not symbol:
        raise ValueError("股票代码不能为空")

    notes = []
    cpath = _cache_path(symbol, interval)
    with _symbol_lock(symbol), _cache_file_lock(symbol, interval):
        cached = _load_cache(symbol, interval) if use_cache else None
        need_fetch = (force or cached is None or len(cached) < count
                      or not _cache_is_fresh(cached, interval, cpath, symbol=symbol))
        if use_cache and cached is not None and not cached.empty and not need_fetch:
            # 缓存足够新 (含被其他线程抢先刷新的情形), 直接命中
            return {"rows": _rows_from_df(cached)[-count:], "source": "cache",
                    "symbol": to_yahoo_symbol(symbol), "interval": interval,
                    "note": "本地缓存", "stale": False,
                    "refresh_failed": False}

        merged = cached
        try:
            import futu  # noqa: F401 - 检测 futu-api 是否安装
            futu_ready = _opend_reachable()
        except ImportError:
            futu_ready = False

        if futu_ready:
            try:
                df = _fetch_futu(symbol, interval, count)
                merged = _merge_market_data(merged, df)
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
                df.attrs.update(source="yahoo", adjustment="adjusted")
                merged = _merge_market_data(merged, df)
                source = "yahoo"
            except Exception as exc:  # noqa: BLE001
                if merged is not None and not merged.empty:
                    notes.append(f"在线更新失败({exc}), 使用本地缓存")
                    out_rows = _rows_from_df(merged)
                    return {"rows": out_rows[-count:], "source": "cache",
                            "symbol": sym_used, "interval": interval,
                            "note": "; ".join(notes), "stale": True,
                            "refresh_failed": True}
                detail = "; ".join(notes + [f"Yahoo({exc})"])
                raise RuntimeError(f"获取 {symbol} 行情失败: {detail}") from exc

        _save_cache(symbol, interval, merged)
        out_rows = _rows_from_df(merged)
        return {"rows": out_rows[-count:], "source": source,
                "symbol": sym_used, "interval": interval, "note": "; ".join(notes),
                "stale": False, "refresh_failed": False}


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
    idx = _parse_datetime_values([r[0] for r in recs])
    if idx.notna().any():
        valid = idx.notna()
        df = df.loc[valid].copy()
        df.index = idx[valid]
    else:
        df.index = pd.RangeIndex(len(recs))
    df = _sanitize_ohlcv(df)
    if df.empty:
        raise ValueError("K线数据无有效行")
    return df


def _auto_refresh_loop():
    """后台数据守护: 每 6 小时检查默认关注列表的日K缓存。

    fetch_quote 按市场最近已收盘工作日和请求历史长度判断新鲜度，因此该循环
    只在缓存确实需要补数时才发起网络请求。
    """
    while True:
        for sym in DEFAULT_SYMBOLS:
            try:
                result = fetch_quote(sym, "1d", DEFAULT_COUNT, use_cache=True, force=False)
                if result.get("refresh_failed"):
                    raise RuntimeError(result.get("note") or "自动刷新失败")
            except Exception as exc:  # noqa: BLE001 - 后台失败不终止守护线程
                print(f"[data-refresh] {sym} 刷新失败: {type(exc).__name__}: {exc}",
                      flush=True)
        time.sleep(6 * 3600)


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
        raise ValueError(
            f"CSV 缺少必要列: {missing} "
            "(支持列名: 开盘价/最高价/最低价/收盘价/成交量 或 "
            "open/high/low/close/volume)")

    out = pd.DataFrame({k: df[find(k)] for k in ("open", "high", "low", "close")})
    vcol = find("volume")
    out["volume"] = df[vcol] if vcol is not None else 0.0

    dcol = find("date")
    if dcol is not None:
        out.index = _parse_datetime_values(df[dcol])
        out = out[out.index.notna()]
    out = (out.apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan).dropna())
    if dcol is not None:
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out = _sanitize_ohlcv(out)
    if out.empty:
        raise ValueError("CSV 无有效数据行")
    return out
